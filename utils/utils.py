import numpy as np
from scipy.spatial.transform import Rotation
from geometry_msgs.msg import Pose, Twist, Point, Quaternion
import torch
import torch.nn.functional as F

def compute_xyz_vector(width, height, fx, fy, skip_pixel = 5, max_range = 1):
    
    width = int(width)
    height = int(height)
    
    cx = width/2
    cy = height/2
    x_grid, y_grid = np.meshgrid(skip_pixel*np.arange(width//skip_pixel), skip_pixel*np.arange(height//skip_pixel), indexing='xy')
    xx = (x_grid - cx)/fx #backprojection
    yy = (y_grid - cy)/fy #backprojection
    xx = xx.reshape(-1)
    yy = yy.reshape(-1)
    depth_z = max_range*np.ones_like(xx)

    x_vector = xx*depth_z
    y_vector = yy*depth_z

    XYZ_vect_hom = np.vstack([x_vector,y_vector,depth_z, np.ones_like(x_vector)]) # (4, width//skip_pixel, height//skip_pixel)

    
    return XYZ_vect_hom

def dilate_pytorch(binary_mask, kernel_size=5):
    """
    Apply morphological dilation to a binary mask using PyTorch.
    :param binary_mask: (H, W) torch tensor, values in {0, 1} or {0, 255}
    :param kernel_size: size of the dilation kernel
    :return: dilated binary mask
    """
    # Ensure the mask is in {0,1} format
    binary_mask = (binary_mask > 0).float().unsqueeze(0).unsqueeze(0)  # (N, C, H, W)

    # Create a kernel (structuring element)
    kernel = torch.ones((1, 1, kernel_size, kernel_size), dtype=torch.float32, device=binary_mask.device)

    # Apply max pooling (dilation)
    dilated_mask = F.max_pool2d(binary_mask, kernel_size, stride=1, padding=kernel_size // 2)

    return dilated_mask.squeeze(0).squeeze(0)  # Remove batch and channel dims


def ros_pose_to_SE3(rospose):
    # rospose in form Pose(point(), quaternion)
    t = np.array([rospose.position.x, rospose.position.y,
                            rospose.position.z]).reshape(3,1)
    R = Rotation.from_quat([rospose.orientation.x,
                                    rospose.orientation.y,
                                    rospose.orientation.z,
                                    rospose.orientation.w]).as_matrix()
    T = np.block([[R, t],[0.0,0.0,0.0,1.0]])
    return T

def SE3_to_ros_pose(T):
    '''
    T a 4x4 matrix
    '''
    q = Rotation.from_matrix(T[0:3,0:3]).as_quat()
    t = T[0:3,3].squeeze()
    ros_pose = Pose(Point(t[0],t[1],t[2]), Quaternion(q[0],q[1],q[2],q[3])) #x,y,z,qx,qy,qz,qw
    return ros_pose


def add_gaussian_noise(depth_image, mean=0, std=0.02):
    """
    Adds Gaussian noise to a depth image.
    
    Parameters:
        depth_image (numpy.ndarray): Input depth image.
        mean (float): Mean of Gaussian noise.
        std (float): Standard deviation of Gaussian noise.
        
    Returns:
        numpy.ndarray: Depth image with added Gaussian noise.
    """
    noise = np.random.normal(mean, std, depth_image.shape).astype(np.float32)
    noisy_image = depth_image.astype(np.float32) + noise
    
    # Ensure depth values remain within a valid range (assuming depth is positive)
    noisy_image = np.clip(noisy_image, 0, np.max(depth_image))
    
    return noisy_image.astype(depth_image.dtype)