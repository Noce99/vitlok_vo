"""KITTI-style drift metrics for a 2D trajectory.

The KITTI odometry benchmark reports error not as a single global number but as
drift accumulated over sub-trajectories of fixed length: for every starting pose,
take the stretch of path spanning L metres, and compare where the estimate ended
up relative to where it started with where the ground truth did. Averaging over
all start positions and several lengths gives translational error as a percentage
of distance travelled, and rotational error in degrees per metre -- both
independent of how long the run happened to be, and so comparable across runs.

Everything here works in SE(2): these are ground vehicles and walkers, and
heading is taken from the direction of motion (see :func:`heading_angles` for
when that approximation breaks).

The default sub-trajectory lengths are shorter than KITTI's own 100-800 m, since
these sequences are forest and field walks rather than motorway driving.
"""

import numpy as np


def cumulative_path_length(traj: np.ndarray) -> np.ndarray:
    """
    Cumulative arc length along a trajectory.

    Args:
        traj: (n, 2) array of x, y coordinates.
    Returns:
        (n,) array, cum_dist[i] = path length from traj[0] to traj[i].
    """
    diffs = np.diff(traj, axis=0)
    seg_lengths = np.linalg.norm(diffs, axis=1)
    return np.concatenate([[0.0], np.cumsum(seg_lengths)])


def heading_angles(traj: np.ndarray) -> np.ndarray:
    """
    Heading angle at each pose, derived from direction of motion.

    APPROXIMATION: assumes platform heading == velocity direction.
    Not valid in general for a camera that can rotate independently
    of its translation (e.g. monocular VO looking sideways/up/down
    while moving forward).

    Args:
        traj: (n, 2) array of x, y coordinates.
    Returns:
        (n,) array of headings in radians.
    """
    diffs = np.diff(traj, axis=0)
    headings = np.arctan2(diffs[:, 1], diffs[:, 0])
    return np.concatenate([headings, headings[-1:]])  # repeat last


def pose_se2(traj: np.ndarray, headings: np.ndarray, idx: int) -> np.ndarray:
    """Build a 3x3 SE(2) homogeneous pose matrix at index idx."""
    theta = headings[idx]
    c, s = np.cos(theta), np.sin(theta)
    T = np.eye(3)
    T[:2, :2] = [[c, -s], [s, c]]
    T[:2, 2] = traj[idx]
    return T


def se2_inverse(T: np.ndarray) -> np.ndarray:
    """Invert an SE(2) homogeneous pose matrix."""
    R = T[:2, :2]
    t = T[:2, 2]
    T_inv = np.eye(3)
    T_inv[:2, :2] = R.T
    T_inv[:2, 2] = -R.T @ t
    return T_inv


def relative_pose_error_se2(T_i, T_j, Th_i, Th_j) -> np.ndarray:
    """
    E = (T_i^-1 T_j)^-1 (Th_i^-1 Th_j)

    T_i, T_j   : ground-truth poses at frames i, j
    Th_i, Th_j : estimated poses at frames i, j
    """
    gt_rel = se2_inverse(T_i) @ T_j
    est_rel = se2_inverse(Th_i) @ Th_j
    return se2_inverse(gt_rel) @ est_rel


def se2_translation_error(E: np.ndarray) -> float:
    return float(np.linalg.norm(E[:2, 2]))


def se2_rotation_error_deg(E: np.ndarray) -> float:
    theta = np.arctan2(E[1, 0], E[0, 0])
    return float(np.degrees(np.abs(theta)))


def find_segment_end(cum_dist: np.ndarray, start_idx: int, length: float):
    """
    Smallest j > start_idx such that cum_dist[j] - cum_dist[start_idx] >= length.
    Returns None if the trajectory runs out before reaching that length.
    """
    target = cum_dist[start_idx] + length
    j = int(np.searchsorted(cum_dist, target, side='left'))
    if j >= len(cum_dist):
        return None
    return j


def kitti_metrics(estimate: np.ndarray, gt: np.ndarray,
                   lengths=(100, 200, 300, 400, 500, 600, 700, 800)) -> dict:
    """
    KITTI-style trajectory evaluation (translational %, rotational deg/m),
    averaged over all sub-segments of each fixed length, over all starting
    frames, over all requested lengths.

    Args:
        estimate: (n, 2) estimated x, y trajectory.
        gt:       (n, 2) ground-truth x, y trajectory, same frame indexing.
        lengths:  sub-trajectory lengths in meters.

    Returns:
        dict with:
            'trans_error_pct'      : average translational drift, in %
            'rot_error_deg_per_m'  : average rotational drift, in deg/m
            'num_segments'         : total segments evaluated
            'per_length'           : {length: (trans_pct, rot_deg_per_m, n_segments)}

    Note: see heading_angles() docstring for the rotational-error caveat.
    """
    assert estimate.shape == gt.shape
    assert estimate.shape[1] == 2

    cum_dist = cumulative_path_length(gt)
    gt_headings = heading_angles(gt)
    est_headings = heading_angles(estimate)

    n = len(gt)
    trans_errors, rot_errors = [], []
    per_length = {}

    for l in lengths:
        l_trans, l_rot = [], []
        for i in range(n):
            j = find_segment_end(cum_dist, i, l)
            if j is None:
                break  # trajectory too short for further segments of this length

            T_i = pose_se2(gt, gt_headings, i)
            T_j = pose_se2(gt, gt_headings, j)
            Th_i = pose_se2(estimate, est_headings, i)
            Th_j = pose_se2(estimate, est_headings, j)

            E = relative_pose_error_se2(T_i, T_j, Th_i, Th_j)
            e_trans = se2_translation_error(E) / l
            e_rot = se2_rotation_error_deg(E) / l

            l_trans.append(e_trans)
            l_rot.append(e_rot)
            trans_errors.append(e_trans)
            rot_errors.append(e_rot)

        if l_trans:
            per_length[l] = (float(np.mean(l_trans) * 100),
                              float(np.mean(l_rot)),
                              len(l_trans))
        else:
            per_length[l] = (None, None, 0)

    if not trans_errors:
        raise ValueError(
            "No valid segments for any requested length — "
            "trajectory is too short."
        )

    return {
        'trans_error_pct': float(np.mean(trans_errors) * 100),
        'trans_error_pct_std': float(np.std(trans_errors) * 100),
        'rot_error_deg_per_m': float(np.mean(rot_errors)),
        'rot_error_deg_per_m_std': float(np.std(rot_errors)),
        'num_segments': len(trans_errors),
        'per_length': per_length
    }