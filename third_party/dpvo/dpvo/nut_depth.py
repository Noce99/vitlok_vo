import cv2
import numpy as np


def save_depth_visualization(image, patches, ht, wd, frame_idx, save_dir="."):
    """
    Visualize inverse depth stored in patches[:,:,2] and save as image.
    patches: [N, M, 3, 3, 3]
    """
    # Get patch centroids in feature space (1/4 resolution)
    px = patches[:, :, 0, 1, 1].cpu().numpy()  # [N, M]
    py = patches[:, :, 1, 1, 1].cpu().numpy()  # [N, M]
    inv_depth = patches[:, :, 2, 1, 1].cpu().numpy()  # [N, M]

    # Scale to full resolution
    px = (px * 4).astype(np.int32).clip(0, wd - 1)
    py = (py * 4).astype(np.int32).clip(0, ht - 1)

    # Flatten across N and M
    px = px.reshape(-1)
    py = py.reshape(-1)
    inv_depth = inv_depth.reshape(-1)

    # Build sparse depth image
    depth_img = np.zeros((ht, wd), dtype=np.float32)
    depth_img[py, px] = inv_depth

    # Normalize to [0, 255] for visualization (only over valid patches)
    valid = depth_img > 0
    if valid.any():
        d_min = depth_img[valid].min()
        d_max = depth_img[valid].max()
        depth_norm = np.zeros_like(depth_img)
        depth_norm[valid] = (depth_img[valid] - d_min) / (d_max - d_min + 1e-6)
        depth_vis = (depth_norm * 255).astype(np.uint8)
    else:
        depth_vis = np.zeros((ht, wd), dtype=np.uint8)

    # Apply colormap
    depth_colored = cv2.applyColorMap(depth_vis, cv2.COLORMAP_MAGMA)
    image_to_save = (image.squeeze().permute(1, 2, 0) + 0.5) / 2 * 255

    to_save = np.hstack([image_to_save.cpu().numpy(), depth_colored])

    # Save
    path = f"{save_dir}/depth_{frame_idx:06d}.png"
    cv2.imwrite(path, to_save)
    print(f"Saved depth visualization to {path}")