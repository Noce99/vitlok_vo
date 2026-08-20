"""Estimate the metric scale ``s`` and shift ``t`` of a predicted depth map.

This is the estimator at the heart of stage 3. Monocular depth networks return
depth that is metric only up to an unknown affine transform; the one thing we
know for certain about the scene is how far the camera is above the ground. That
single fact, applied to the points CSF segmented as ground, pins down both
unknowns.

Inputs are the non-metric 3D points obtained by projecting the predicted depth
through the camera intrinsics.

Model
-----
For each ground point P_i, take its z-depth d_i = P_i.z and the z-normalized
ray m_i = P_i / d_i (so m_i.z == 1). The metric 3D point is:

    X_i = (s * d_i + t) * m_i

so that the metric z-depth is exactly s * d_i + t -- consistent with applying
the correction to the depth map as sas_depth = depth * s + t.

The ground-plane constraint with unit normal n and camera height h gives:

    (s * d_i + t) * (n . m_i) = h

Let beta_i = n . m_i. Each point yields one linear equation in (s, t):

    s * (d_i * beta_i) + t * beta_i = h

We solve robustly with RANSAC + IRLS (Huber) and quantify uncertainty via
the asymptotic covariance plus a bootstrap.
"""

from __future__ import annotations
import numpy as np
from dataclasses import dataclass


@dataclass
class ScaleShiftResult:
    s: float
    t: float
    sigma_s: float          # asymptotic 1-sigma on s
    sigma_t: float          # asymptotic 1-sigma on t
    cov: np.ndarray         # 2x2 covariance of (s, t)
    s_boot_std: float       # bootstrap std of s
    t_boot_std: float       # bootstrap std of t
    inlier_mask: np.ndarray # bool, shape (N,)
    residuals: np.ndarray   # final residuals on inliers (meters)
    n_inliers: int
    condition_number: float # of A^T A on inliers; high -> s,t weakly identifiable


def solve_scale_shift(
    ground_points: np.ndarray,
    n_plane: np.ndarray,
    camera_height: float = 1.8,
    ransac_iters: int = 200,
    ransac_thresh: float = 0.05,   # meters of plane-distance residual
    huber_delta: float = 0.05,     # meters; transition for Huber loss
    irls_iters: int = 8,
    bootstrap_iters: int = 500,
    rng_seed: int = 0,
) -> ScaleShiftResult:
    """
    Parameters
    ----------
    ground_points : (N, 3) non-metric 3D points from CSF, in camera coordinates.
                    These are the points obtained by projecting the predicted depth through
                    the camera intrinsics; CSF flagged them as ground.
    n_plane : (3,) ground-plane unit normal in CAMERA coordinates, oriented so
              that n . X = +h for points on the ground (i.e. pointing from the
              ground toward the camera). If you got the wrong sign you'll see
              s come out negative -- just flip n.
    camera_height : scalar h in meters (default 1.8).
    ransac_thresh : inlier threshold on plane-distance residual, meters.
    huber_delta : Huber transition, meters. Roughly the noise scale you expect.

    Returns
    -------
    ScaleShiftResult
    """
    P = np.asarray(ground_points, dtype=np.float64)
    if P.ndim != 2 or P.shape[1] != 3:
        raise ValueError(f"ground_points must be (N, 3), got {P.shape}")
    n = np.asarray(n_plane, dtype=np.float64).ravel()
    n = n / np.linalg.norm(n)
    h = float(camera_height)

    N = P.shape[0]
    if N < 10:
        raise ValueError(f"Need at least ~10 ground points, got {N}.")

    # z-depth and z-normalized ray direction for each point.
    # We work in DEPTH space (not range) so the returned (s, t) satisfy
    # metric_depth = s * d + t EXACTLY, matching how the correction is applied
    # downstream (sas_depth = depth * s + t). d_i is the camera z-component;
    # m_i = P_i / d_i is the ray scaled to unit z (m_i[2] == 1), so the metric
    # point is (s*d_i + t) * m_i and its z-component is exactly s*d_i + t.
    d = P[:, 2].astype(np.float64)                      # (N,) z-depth
    good = d > 1e-9
    P, d = P[good], d[good]
    m = P / d[:, None]                                  # (N, 3), m[:, 2] == 1
    N = P.shape[0]

    # beta_i = n . m_i
    beta = m @ n                                        # (N,)

    # Reject grazing rays (beta ~ 0): the equation becomes degenerate there.
    keep = np.abs(beta) > 1e-3
    if keep.sum() < 10:
        raise ValueError("Too few ground rays with non-grazing geometry.")
    idx_all = np.where(keep)[0]
    d_k = d[keep]
    beta_k = beta[keep]

    # Linear system: A [s, t]^T = b, with rows [d_i*beta_i, beta_i] and b_i = h
    A_full = np.stack([d_k * beta_k, beta_k], axis=1)   # (M, 2)
    b_full = np.full(beta_k.shape, h)                   # (M,)

    rng = np.random.default_rng(rng_seed)
    M = A_full.shape[0]

    # ---------- RANSAC ----------
    best_inliers = None
    best_count = -1
    for _ in range(ransac_iters):
        i, j = rng.choice(M, size=2, replace=False)
        A2 = A_full[[i, j]]
        if abs(np.linalg.det(A2)) < 1e-10:
            continue
        try:
            theta = np.linalg.solve(A2, b_full[[i, j]])
        except np.linalg.LinAlgError:
            continue
        s_try, t_try = theta
        resid = (s_try * d_k + t_try) * beta_k - h
        inliers = np.abs(resid) < ransac_thresh
        count = int(inliers.sum())
        if count > best_count:
            best_count = count
            best_inliers = inliers

    if best_inliers is None or best_count < 5:
        best_inliers = np.ones(M, dtype=bool)

    # ---------- IRLS with Huber on RANSAC inliers ----------
    A = A_full[best_inliers]
    b = b_full[best_inliers]

    theta, *_ = np.linalg.lstsq(A, b, rcond=None)
    for _ in range(irls_iters):
        resid = A @ theta - b
        absr = np.abs(resid)
        w = np.where(absr <= huber_delta, 1.0, huber_delta / np.maximum(absr, 1e-12))
        Aw = A * np.sqrt(w)[:, None]
        bw = b * np.sqrt(w)
        theta_new, *_ = np.linalg.lstsq(Aw, bw, rcond=None)
        if np.linalg.norm(theta_new - theta) < 1e-10:
            theta = theta_new
            break
        theta = theta_new

    s_hat, t_hat = float(theta[0]), float(theta[1])

    # Final residuals and inlier mask
    resid_final = A @ theta - b
    final_inliers_local = np.abs(resid_final) < max(3 * huber_delta, ransac_thresh)
    inlier_mask = np.zeros(N, dtype=bool)
    idx_keep = idx_all[best_inliers][final_inliers_local]
    inlier_mask[idx_keep] = True

    # ---------- Asymptotic covariance ----------
    A_in = A[final_inliers_local]
    r_in = resid_final[final_inliers_local]
    dof = max(A_in.shape[0] - 2, 1)
    sigma2 = float(r_in @ r_in) / dof
    AtA = A_in.T @ A_in
    try:
        cov = sigma2 * np.linalg.inv(AtA)
    except np.linalg.LinAlgError:
        cov = sigma2 * np.linalg.pinv(AtA)
    sigma_s = float(np.sqrt(max(cov[0, 0], 0)))
    sigma_t = float(np.sqrt(max(cov[1, 1], 0)))
    cond = float(np.linalg.cond(AtA))

    # ---------- Bootstrap ----------
    s_boots = np.empty(bootstrap_iters)
    t_boots = np.empty(bootstrap_iters)
    m_in = A_in.shape[0]
    b_in = b[final_inliers_local]
    for k in range(bootstrap_iters):
        sel = rng.integers(0, m_in, size=m_in)
        Ak, bk = A_in[sel], b_in[sel]
        try:
            th_k, *_ = np.linalg.lstsq(Ak, bk, rcond=None)
            s_boots[k], t_boots[k] = th_k
        except np.linalg.LinAlgError:
            s_boots[k] = np.nan
            t_boots[k] = np.nan
    s_boot_std = float(np.nanstd(s_boots, ddof=1))
    t_boot_std = float(np.nanstd(t_boots, ddof=1))

    return ScaleShiftResult(
        s=s_hat,
        t=t_hat,
        sigma_s=sigma_s,
        sigma_t=sigma_t,
        cov=cov,
        s_boot_std=s_boot_std,
        t_boot_std=t_boot_std,
        inlier_mask=inlier_mask,
        residuals=r_in,
        n_inliers=int(inlier_mask.sum()),
        condition_number=cond,
    )


# ---------------------------------------------------------------------------
# Synthetic self-check
# ---------------------------------------------------------------------------

def generate_synthetic_ground(
    s_true: float = 0.42,
    t_true: float = 0.15,
    camera_height: float = 1.8,
    n_points: int = 2000,
    noise: float = 0.01,
    outlier_fraction: float = 0.10,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Build a synthetic ground-point cloud with known ``(s, t)``.

    Ground points are sampled in the lower half of a 640x480 frame, placed on a
    tilted plane at ``camera_height``, then pushed back through the inverse of
    the affine depth model so that recovering ``(s_true, t_true)`` is exactly the
    job :func:`solve_scale_shift` has to do. Gaussian noise and a fraction of
    gross outliers are added to exercise the robust path.

    Returns:
        ``(points, plane_normal)`` ready to pass to :func:`solve_scale_shift`.
    """
    rng = np.random.default_rng(seed)

    fx, fy, cx, cy = 600.0, 600.0, 320.0, 240.0
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)

    # Ground-plane normal in the camera frame (camera +y points down).
    n_true = np.array([0.05, -1.0, 0.1])
    n_true /= np.linalg.norm(n_true)

    u = rng.uniform(0, 640, n_points)
    v = rng.uniform(260, 480, n_points)
    pixels = np.stack([u, v, np.ones_like(u)], axis=0)
    rays = np.linalg.inv(K) @ pixels
    rays /= np.linalg.norm(rays, axis=0, keepdims=True)
    alpha = n_true @ rays
    hits = alpha > 0.05                     # ray actually meets the ground ahead
    rays, alpha = rays[:, hits], alpha[hits]

    metric_points = ((camera_height / alpha)[None, :] * rays).T

    # Invert X = (s*r + t) * v_hat to get the "predicted", non-metric points.
    metric_range = np.linalg.norm(metric_points, axis=1)
    relative_range = (metric_range - t_true) / s_true
    points = relative_range[:, None] * rays.T

    points += rng.normal(scale=noise, size=points.shape)

    n_outliers = int(outlier_fraction * points.shape[0])
    outlier_idx = rng.choice(points.shape[0], size=n_outliers, replace=False)
    points[outlier_idx] += rng.normal(scale=2.0, size=(n_outliers, 3))

    return points, n_true


if __name__ == "__main__":
    s_true, t_true, h_true = 0.42, 0.15, 1.8
    points, normal = generate_synthetic_ground(s_true, t_true, h_true)
    res = solve_scale_shift(ground_points=points, n_plane=normal,
                            camera_height=h_true)

    print(f"True   s={s_true:.4f}, t={t_true:.4f}")
    print(f"Est    s={res.s:.4f} +/- {res.sigma_s:.4f} (boot {res.s_boot_std:.4f})")
    print(f"Est    t={res.t:.4f} +/- {res.sigma_t:.4f} (boot {res.t_boot_std:.4f})")
    print(f"Inliers: {res.n_inliers}/{points.shape[0]}")
    print(f"Condition number of A^T A: {res.condition_number:.2e}")
    print(f"Residual RMS (m): {np.sqrt(np.mean(res.residuals**2)):.4f}")
