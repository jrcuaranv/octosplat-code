import numpy as np
import open3d as o3d


def filter_depth_map(depth_image, intrinsics, max_depth=2.0):
    """
    Filters a depth map by removing points that are too far or have invalid values.

    Args:
        depth_map (numpy.ndarray): Input depth map of shape (H, W).
        intrinsics (dict): Camera intrinsics 3x3 matrix
        max_depth (float): Maximum valid depth value in meters.

    Returns:
        numpy.ndarray: Filtered depth map of shape (H, W) with invalid points set to 0.
    """
    fx = intrinsics[0, 0]
    fy = intrinsics[1, 1]
    cx = intrinsics[0, 2]
    cy = intrinsics[1, 2]

    # Remove far points and NaNs
    depth_image[depth_image > 2.0] = 0.0
    depth_image = np.nan_to_num(depth_image, nan=0.0)
    height, width = depth_image.shape
    # Pixel grid
    u, v = np.meshgrid(np.arange(width), np.arange(height))
    z = depth_image
    # Keep only valid depth
    valid_mask = z > 0.0
    z = z[valid_mask]
    u = u[valid_mask]
    v = v[valid_mask]

    # Backproject to 3D
    x = (u - cx) * z / fx
    y = (v - cy) * z / fy

    points = np.stack((x, y, z), axis=-1)

    # Open3D point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)

    pcd, ind = pcd.remove_statistical_outlier(
        nb_neighbors=20,
        std_ratio=1.5
    )

    # Generating filtered depth map
    valid_flat_indices = np.flatnonzero(valid_mask)
    depth_filtered = np.zeros(depth_image.size, dtype=depth_image.dtype)
    depth_filtered[valid_flat_indices[ind]] = z[ind]
    depth_filtered = depth_filtered.reshape(depth_image.shape)

    return depth_filtered
