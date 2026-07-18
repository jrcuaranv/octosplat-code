import os
import shutil
import sys
import time
import pandas as pd
from importlib.machinery import SourceFileLoader


import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from utils.recon_helpers import setup_camera
from utils.slam_helpers import (
    transformed_params2rendervar, filter_points_in_image, transformed_params2depth_silhouette_rgbloss, transformed_entropy2rendervar, transformed_params2depthplussilhouette,
    transformed_semantics2rendervar, transformed_rgb_loss_rendervar, transform_to_frame, transform_points_to_frame, l1_loss_v1, matrix_to_quaternion
)
from utils.slam_external import calc_ssim, build_rotation, prune_outlier_semantics, prune_gaussians, densify, prune_aux_gaussians

from diff_gaussian_rasterization import GaussianRasterizer as Renderer


def get_pointcloud(color, depth, confidence_map, intrinsics, w2c, transform_pts=True, mask=None,
                   compute_mean_sq_dist=False, mean_sq_dist_method="projective", device="cuda",
                   load_semantics=False, semantic_id=None, semantic_color=None, rgb_loss = None):
    
    width, height = color.shape[2], color.shape[1]
    CX = intrinsics[0][2]
    CY = intrinsics[1][2]
    FX = intrinsics[0][0]
    FY = intrinsics[1][1]

    # Compute indices of pixels
    x_grid, y_grid = torch.meshgrid(torch.arange(width).to(device).float(), 
                                    torch.arange(height).to(device).float(),
                                    indexing='xy')
    xx = (x_grid - CX)/FX
    yy = (y_grid - CY)/FY
    xx = xx.reshape(-1)
    yy = yy.reshape(-1)
    depth_z = depth[0].reshape(-1)

    # Initialize point cloud
    pts_cam = torch.stack((xx * depth_z, yy * depth_z, depth_z), dim=-1)
    if transform_pts:
        pix_ones = torch.ones(height * width, 1).to(device).float()
        pts4 = torch.cat((pts_cam, pix_ones), dim=1)
        c2w = torch.inverse(w2c)
        pts = (c2w @ pts4.T).T[:, :3]
    else:
        pts = pts_cam

    # Compute mean squared distance for initializing the scale of the Gaussians
    if compute_mean_sq_dist:
        if mean_sq_dist_method == "projective":
            # Projective Geometry (this is fast, farther -> larger radius)
            scale_gaussian = depth_z / ((FX + FY)/2)
            mean3_sq_dist = scale_gaussian**2
        else:
            raise ValueError(f"Unknown mean_sq_dist_method {mean_sq_dist_method}")
        
    # Colorize point cloud
    cols = torch.permute(color, (1, 2, 0)).reshape(-1, 3) # (C, H, W) -> (H, W, C) -> (H * W, C)
    point_cld = torch.cat((pts, cols), -1)
    
    # Concat semantic label if load_semantics=True
    confidence_map = confidence_map.unsqueeze(0)
    if load_semantics:
        semantic_id = torch.permute(semantic_id, (1, 2, 0)).reshape(-1, 1) # (1, H, W) -> (H, W, 1) -> (H * W, 1)
        confidence_map = torch.permute(confidence_map, (1, 2, 0)).reshape(-1, 1)
        semantic_color = torch.permute(semantic_color, (1, 2, 0)).reshape(-1, 3) # (3, H, W) -> (H, W, 3) -> (H * W, 3)
        rgb_loss = torch.permute(rgb_loss, (1, 2, 0)).reshape(-1, 1)
        opt_count = torch.zeros(semantic_color.shape[0]).reshape(-1,1).to(device).float()
        point_cld = torch.cat((point_cld, semantic_id, semantic_color*0+0.5, rgb_loss, opt_count), -1)
        
    
    # sem_mask = (semantic_color[:,0] == 1) & (semantic_color[:,1] == 0) & (semantic_color[:,2] == 0)
    # sem_mask = sem_mask & (confidence_map[:,0] > 0.6) # Keep only high confidence semantic gaussians
    # others_mask = ~sem_mask
    # downsample_mask(others_mask, down_factor=0.7) # donwsampling irrelevant semantics (remove down_factor%)
    # combined mask
    # mask = mask & (sem_mask | others_mask)
    
    # Select points based on mask
    if mask is not None:
        point_cld = point_cld[mask]
        if compute_mean_sq_dist:
            mean3_sq_dist = mean3_sq_dist[mask]

    if compute_mean_sq_dist:
        return point_cld, mean3_sq_dist
    else:
        return point_cld

def downsample_mask(mask, down_factor):
    """
    Downsample a binary mask by removing a fraction of True values.
    
    Args:
        mask (torch.Tensor): Binary mask tensor (0s and 1s or False/True)
        down_factor (float): Fraction of True values to flip (0 < down_factor <= 1)
    
    Returns:
        torch.Tensor: Modified mask with fewer True values
    """
    # Ensure mask is a boolean tensor for efficiency
    mask = mask.bool()
    
    # Get number of True elements
    num_true = mask.sum().item()
    if num_true == 0:
        return mask  # Nothing to downsample
    
    
    # Flatten the mask and get indices of True values
    flat_mask = mask.flatten()
    true_indices = torch.nonzero(flat_mask, as_tuple=False).squeeze()
    
    # Randomly select indices to set to False
    num_to_flip = int(down_factor * num_true)
    flip_indices = true_indices[torch.randperm(num_true)[:num_to_flip]]
    
    # Update mask in one operation
    flat_mask[flip_indices] = False
    
    # Reshape back to original shape
    return flat_mask.view(mask.shape)

def get_initial_pointcloud(w2c, transform_pts = True, compute_mean_sq_dist=False, load_semantics = False, device="cuda"):
    '''
    Create auxiliary gaussians to keep information about unknown space
    '''
    grid_size = (70, 70, 35)  # Number of voxels along x, y, z
    voxel_resolution = 0.05#0.1    # Size of each voxel
    scale_aux_gaussians = 0.01#0.01
    # Generate 3D grid coordinates
    x = torch.linspace(-(grid_size[0]/2) * voxel_resolution, (grid_size[0]/2) * voxel_resolution, grid_size[0]).to(device)
    y = torch.linspace(-(grid_size[1]/2) * voxel_resolution, (grid_size[1]/2) * voxel_resolution, grid_size[1]).to(device)
    z = torch.linspace(0, grid_size[2] * voxel_resolution, grid_size[2]).to(device)
    x_grid, y_grid, z_grid = torch.meshgrid(x, y, z, indexing="ij")

    # Stack the grid coordinates into a single tensor
    voxel_grid = torch.stack((x_grid, y_grid, z_grid), dim=-1)  # Shape: (10, 10, 10, 3)
    pts_cam = voxel_grid.reshape(-1,3) #(-1,3)

    # Compute indices of pixels
    
    # Initialize point cloud
    # if transform_pts:
    #     pix_ones = torch.ones(pts_cam.shape[0], 1).to(device).float()
    #     pts4 = torch.cat((pts_cam, pix_ones), dim=1)
    #     c2w = torch.inverse(w2c)
    #     pts = (c2w @ pts4.T).T[:, :3]
    # else:
        # pts = pts_cam
    pts = pts_cam # Auxiliar points to be added to the world coordinate frame
    # Initialize the gaussians with a constant scale
    if compute_mean_sq_dist:
        scale_gaussian =  scale_aux_gaussians*torch.ones(pts.shape[0]).reshape(-1).to(device).float()
        mean3_sq_dist = scale_gaussian**2
        
    # Colorize point cloud
    # cols = (127.0/255)*torch.ones_like(pts).to(device) # (-1,3)
    cols = torch.zeros_like(pts).to(device) # (-1,3)
    cols[:,2] =1.0 # blue
    point_cld = torch.cat((pts, cols), -1)
    
    # Concat semantic label if load_semantics=True
    if load_semantics:
        semantic_id = 0.555*torch.ones(pts.shape[0]).reshape(-1,1).to(device).float() #just a fixed id for all auxiliar gaussians
        semantic_color = (127.0/255)*torch.ones_like(pts).to(device).float()# (-1,3) # just a fixed semantic color for all gaussians
        rgb_loss = torch.ones(pts.shape[0]).reshape(-1,1).to(device).float()
        opt_count = torch.zeros(semantic_color.shape[0]).reshape(-1,1).to(device).float()
        point_cld = torch.cat((point_cld, semantic_id, semantic_color, rgb_loss, opt_count), -1)
        

    if compute_mean_sq_dist:
        return point_cld, mean3_sq_dist
    else:
        return point_cld

def initialize_params(init_pt_cld, num_frames, mean3_sq_dist, device, load_semantics=False):
    num_pts = init_pt_cld.shape[0]
    # channel 0-2 for 3d axis
    means3D = init_pt_cld[:, :3]
    # channel 3-5 for rgb colors
    rgb_colors = init_pt_cld[:, 3:6]
    unnorm_rots = np.tile([1, 0, 0, 0], (num_pts, 1)) # [num_gaussians, 3]
    logit_opacities = torch.zeros((num_pts, 1), dtype=torch.float, device=device) # sigmoid(zero) = 0.5=opacity
    
    params = {
        'means3D': means3D,
        'rgb_colors': rgb_colors,
        'unnorm_rotations': unnorm_rots,
        'logit_opacities': logit_opacities,
        'log_scales': torch.tile(torch.log(torch.sqrt(mean3_sq_dist))[..., None], (1, 1)),
    }

    params_opt_exclude = set()
    if load_semantics:
        # Exclude semantic_ids from gradient
        params_opt_exclude.add('semantic_ids')
        params_opt_exclude.add('opt_count')
        # channel =6 for semantic id
        params['semantic_ids'] = init_pt_cld[:, 6].view(-1, 1)
        # Channel 7-9 for semantic colors
        params['semantic_colors'] = init_pt_cld[:, 7:10]
        params['rgb_loss'] = init_pt_cld[:, 10].view(-1, 1)
        params['opt_count'] = init_pt_cld[:, 11].view(-1, 1)
        

    # Initialize a single gaussian trajectory to model the camera poses relative to the first frame
    cam_rots = np.tile([1, 0, 0, 0], (1, 1))
    cam_rots = np.tile(cam_rots[:, :, None], (1, 1, num_frames))
    params['cam_unnorm_rots'] = cam_rots
    params['cam_trans'] = np.zeros((1, 3, num_frames))
    
    for k, v in params.items():
        if k not in params_opt_exclude:
            # Check if value is already a torch tensor
            if not isinstance(v, torch.Tensor):
                params[k] = torch.nn.Parameter(torch.tensor(v).to(device).float().contiguous().requires_grad_(True))
            else:
                params[k] = torch.nn.Parameter(v.to(device).float().contiguous().requires_grad_(True))

    variables = {'max_2D_radius': torch.zeros(params['means3D'].shape[0]).to(device).float(),
                 'means2D_gradient_accum': torch.zeros(params['means3D'].shape[0]).to(device).float(),
                 'denom': torch.zeros(params['means3D'].shape[0]).to(device).float(),
                 'timestep': torch.zeros(params['means3D'].shape[0]).to(device).float()}

    return params, variables, params_opt_exclude


def initialize_optimizer(params, params_opt_exclude, lrs_dict, tracking):
    lrs = lrs_dict
    param_groups = [{'params': [v], 'name': k, 'lr': lrs[k]} for k, v in params.items() if k not in params_opt_exclude]
    if tracking:
        return torch.optim.Adam(param_groups)
    else:
        return torch.optim.Adam(param_groups, lr=0.0, eps=1e-15)


def initialize_first_timestep(dataset_0, num_frames, scene_radius_depth_ratio, mean_sq_dist_method, device="cuda",
                              densify_dataset=None, load_semantics=False):
    # Get RGB-D Data & Camera Parameters
    if load_semantics:
        color, depth, intrinsics, pose, semantic_id, semantic_color, confidence_map = dataset_0
    else:
        color, depth, intrinsics, pose = dataset_0

    # Process RGB-D Data
    color = color.permute(2, 0, 1) / 255 # (H, W, C) -> (C, H, W)
    depth = depth.permute(2, 0, 1) # (H, W, 1) -> (1, H, W)
    
    if load_semantics:
        semantic_id = semantic_id.permute(2, 0, 1) # (H, W, 1) -> (1, H, W)
        semantic_color = semantic_color.permute(2, 0, 1) # (H, W, 3) -> (3, H, W)
        rgb_loss = torch.zeros_like(depth)
    else:
        semantic_id = None
        semantic_color = None
    # Process Camera Parameters
    intrinsics = intrinsics[:3, :3]
    # print("pose test:",pose)
    w2c = torch.linalg.inv(pose)
    
    # Setup Camera
    # cam = setup_camera(color.shape[2], color.shape[1], intrinsics.cpu().numpy(),
    #                    w2c.detach().cpu().numpy(), device=device)
    # w2c becomes the extrinsic matrix for the first camera frame. For some reason, it does not
    # work with the actual matrix, but only with the identity. 
    cam = setup_camera(color.shape[2], color.shape[1], intrinsics.cpu().numpy(),
                       np.eye(4), device=device)


    if densify_dataset is not None:
        # Get Densification RGB-D Data & Camera Parameters
        color, depth, densify_intrinsics, _ = densify_dataset[0]
        color = color.permute(2, 0, 1) / 255 # (H, W, C) -> (C, H, W)
        depth = depth.permute(2, 0, 1) # (H, W, 1) -> (1, H, W)
        densify_intrinsics = densify_intrinsics[:3, :3]
        densify_cam = setup_camera(color.shape[2], color.shape[1], densify_intrinsics.cpu().numpy(),
                                   w2c.detach().cpu().numpy(), device=device)
    else:
        densify_intrinsics = intrinsics

    # Get Initial Point Cloud (PyTorch CUDA Tensor)
    mask = (depth > 0) # Mask out invalid depth values
    mask = mask.reshape(-1)
    init_pt_cld1, mean3_sq_dist1 = get_pointcloud(color, depth, confidence_map, densify_intrinsics,
                                                w2c, mask=mask, compute_mean_sq_dist=True, 
                                                mean_sq_dist_method=mean_sq_dist_method, device=device,
                                                load_semantics=load_semantics, semantic_id=semantic_id,
                                                semantic_color=semantic_color, rgb_loss = rgb_loss)

    # Auxiliar point cloud
    # new_pt_cld2, mean3_sq_dist2 = get_initial_pointcloud(w2c, transform_pts=True, compute_mean_sq_dist=True,
    #                                                     load_semantics=load_semantics, device=device)
        
    # init_pt_cld = torch.cat((init_pt_cld1,new_pt_cld2), dim=0)
    # mean3_sq_dist = torch.cat((mean3_sq_dist1,mean3_sq_dist2))

    init_pt_cld = init_pt_cld1
    mean3_sq_dist = mean3_sq_dist1

    # Initialize Parameters
    params, variables, params_opt_exclude = initialize_params(init_pt_cld, num_frames, mean3_sq_dist, device,
                                                              load_semantics)

    # Initialize an estimate of scene radius for Gaussian-Splatting Densification
    variables['scene_radius'] = torch.max(depth)/scene_radius_depth_ratio #TODO remove 10, jrcv

    if densify_dataset is not None:
        return params, variables, intrinsics, w2c, cam, params_opt_exclude, densify_intrinsics, densify_cam
    else:
        return params, variables, intrinsics, w2c, cam, params_opt_exclude

def render_any_cam(params, w2c, height = 480, width = 640,device='cuda', intrinsics = None, render_all = False):
    
    if intrinsics is None:
        cx = width / 2
        cy = height / 2
        fov = 120*np.pi / 180
        fx = width / (2 * np.tan(fov / 2))
        fy = height / (2 * np.tan(fov / 2))
        
        intrinsics = np.array([[fx, 0, cx],
                                [0, fy, cy],
                                [0, 0, 1]])
    
    cam = setup_camera(width, height, intrinsics, np.eye(4), device=device)
    
    w2c_tensor = torch.from_numpy(w2c).to(device).float()
    
    pts = params['means3D'].detach()
    
    # Transform Centers and Unnorm Rots of Gaussians to Camera Frame
    pts_ones = torch.ones(pts.shape[0], 1).to(device).float()
    pts4 = torch.cat((pts, pts_ones), dim=1)
    transformed_pts = (w2c_tensor @ pts4.T).T[:, :3]

    rendervar = transformed_params2rendervar(params, transformed_pts, device=device)
    rgb_torch, _, _, = Renderer(raster_settings=cam)(**rendervar)
    rgb_torch = torch.clip(rgb_torch, 0, 1) # use permute(1, 2, 0).detach().cpu().numpy() to visualize the rgb image
    
    if render_all == True:
        depth_sil_rendervar = transformed_params2depthplussilhouette(params, None,transformed_pts, device=device)
        semantic_rendervar = transformed_semantics2rendervar(params, transformed_pts, device=device)
        depth_sil, _, _, = Renderer(raster_settings=cam)(**depth_sil_rendervar)
        semantics, _, _, = Renderer(raster_settings=cam)(**semantic_rendervar)
        depth_torch = depth_sil[0, :, :].unsqueeze(0)
        silhouette_torch = depth_sil[1, :, :]
        semantics_torch = torch.clip(semantics, 0, 1)
        silhouette_torch = torch.clip(silhouette_torch, 0, 1)
        return rgb_torch, depth_torch, semantics_torch, silhouette_torch
   
    return rgb_torch, None, None, None

def render_cam(params, cam, iter_time_idx, device='cuda'):
    # Transform Centers and Unnorm Rots of Gaussians to Camera Frame
    transformed_pts = transform_to_frame(params, iter_time_idx, gaussians_grad=False,
                                         camera_grad=False, device=device)

    rendervar = transformed_params2rendervar(params, transformed_pts, device=device)
    depth_sil_rendervar = transformed_params2depthplussilhouette(params, None,transformed_pts, device=device)
    semantic_rendervar = transformed_semantics2rendervar(params, transformed_pts, device=device)
        
        
    rgb, _, _, = Renderer(raster_settings=cam)(**rendervar)
    depth_sil, _, _, = Renderer(raster_settings=cam)(**depth_sil_rendervar)
    semantics, _, _, = Renderer(raster_settings=cam)(**semantic_rendervar)

    depth_torch = depth_sil[0, :, :].unsqueeze(0)
    silhouette_torch = depth_sil[1, :, :]

    rgb_torch = torch.clip(rgb, 0, 1)
    semantics_torch = torch.clip(semantics, 0, 1)
    silhouette_torch = torch.clip(silhouette_torch, 0, 1)

    return rgb_torch, depth_torch, semantics_torch, silhouette_torch
    
# original implementation loss
def get_loss(params, curr_data, variables, iter_time_idx, loss_weights, use_sil_for_loss, sil_thres,
             use_l1, ignore_outlier_depth_loss, tracking=False, mapping=False, do_ba=False, device="cuda",
             plot_dir=None, visualize_tracking_loss=False, tracking_iteration=None, load_semantics=False, visualization = False):
    # Initialize Loss Dictionary
    losses = {}

    if tracking:
        # Get current frame Gaussians, where only the camera pose gets gradient
        transformed_pts = transform_to_frame(params, iter_time_idx, gaussians_grad=False,
                                             camera_grad=True, device=device)
    elif mapping:
        if do_ba:
            # Get current frame Gaussians, where both camera pose and Gaussians get gradient
            transformed_pts = transform_to_frame(params, iter_time_idx, gaussians_grad=True,
                                                 camera_grad=True, device=device)
        else:
            # Get current frame Gaussians, where only the Gaussians get gradient
            transformed_pts = transform_to_frame(params, iter_time_idx, gaussians_grad=True,
                                                 camera_grad=False, device=device)
    else:
        # Get current frame Gaussians, where only the Gaussians get gradient
        transformed_pts = transform_to_frame(params, iter_time_idx, gaussians_grad=True,
                                             camera_grad=False, device=device)

    # Initialize Render Variables
    rendervar = transformed_params2rendervar(params, transformed_pts, device=device)
    depth_sil_rendervar = transformed_params2depthplussilhouette(params, curr_data['w2c'],
                                                                 transformed_pts, device=device)
    # filter points in image to update opt_count variable, jrcv
    if mapping:
        point_in_image_mask = filter_points_in_image(transformed_pts, curr_data['intrinsics'], H = curr_data['im'].shape[1], W = curr_data['im'].shape[2])
        params['opt_count'][point_in_image_mask] += 1
    # RGB Rendering
    rendervar['means2D'].retain_grad()
    im, radius, _, = Renderer(raster_settings=curr_data['cam'])(**rendervar)
    variables['means2D'] = rendervar['means2D']  # Gradient only accum from colour render for densification

    # Depth & Silhouette Rendering
    depth_sil, _, _, = Renderer(raster_settings=curr_data['cam'])(**depth_sil_rendervar)
    depth = depth_sil[0, :, :].unsqueeze(0)
    silhouette = depth_sil[1, :, :]
    presence_sil_mask = (silhouette > sil_thres)
    depth_sq = depth_sil[2, :, :].unsqueeze(0)
    uncertainty = depth_sq - depth**2
    uncertainty = uncertainty.detach()

    # Semantic colors Rendering
    if load_semantics:
        semantic_rendervar = transformed_semantics2rendervar(params, transformed_pts, device=device)
        rendered_seg, _, _, = Renderer(raster_settings=curr_data['cam'])(**semantic_rendervar)

    # Mask with valid depth values (accounts for outlier depth values)
    nan_mask = (~torch.isnan(depth)) & (~torch.isnan(uncertainty))
    if ignore_outlier_depth_loss:
        depth_error = torch.abs(curr_data['depth'] - depth) * (curr_data['depth'] > 0)
        mask = (depth_error < 10*depth_error.median())
        mask = mask & (curr_data['depth'] > 0)
    else:
        mask = (curr_data['depth'] > 0)
    mask = mask & nan_mask
    # Mask with presence silhouette mask (accounts for empty space)
    if tracking and use_sil_for_loss:
        mask = mask & presence_sil_mask

    # Depth loss
    if use_l1:
        mask = mask.detach()
        if tracking:
            losses['depth'] = torch.abs(curr_data['depth'] - depth)[mask].sum()
        else:
            losses['depth'] = torch.abs(curr_data['depth'] - depth)[mask].mean()
    
    # RGB Loss
    if tracking and (use_sil_for_loss or ignore_outlier_depth_loss):
        color_mask = torch.tile(mask, (3, 1, 1))
        color_mask = color_mask.detach()
        losses['im'] = torch.abs(curr_data['im'] - im)[color_mask].sum()
        if load_semantics:
            losses['seg'] = torch.abs(curr_data['semantic_color'] - rendered_seg)[color_mask].sum()
    elif tracking:
        losses['im'] = torch.abs(curr_data['im'] - im).sum()
        if load_semantics:
            losses['seg'] = torch.abs(curr_data['semantic_color'] - rendered_seg).sum()
    else:
        losses['im'] = 0.8 * l1_loss_v1(im, curr_data['im']) + 0.2 * (1.0 - calc_ssim(im, curr_data['im']))
        if load_semantics:
            losses['seg'] = 0.8 * l1_loss_v1(rendered_seg, curr_data['semantic_color']) \
                + 0.2 * (1.0 - calc_ssim(rendered_seg, curr_data['semantic_color']))

    # Visualize the Diff Images
    if tracking and visualize_tracking_loss:
        fig, ax = plt.subplots(2, 4, figsize=(12, 6))
        weighted_render_im = im * color_mask
        weighted_im = curr_data['im'] * color_mask
        weighted_render_depth = depth * mask
        weighted_depth = curr_data['depth'] * mask
        diff_rgb = torch.abs(weighted_render_im - weighted_im).mean(dim=0).detach().cpu()
        diff_depth = torch.abs(weighted_render_depth - weighted_depth).mean(dim=0).detach().cpu()
        viz_img = torch.clip(weighted_im.permute(1, 2, 0).detach().cpu(), 0, 1)
        ax[0, 0].imshow(viz_img)
        ax[0, 0].set_title("Weighted GT RGB")
        viz_render_img = torch.clip(weighted_render_im.permute(1, 2, 0).detach().cpu(), 0, 1)
        ax[1, 0].imshow(viz_render_img)
        ax[1, 0].set_title("Weighted Rendered RGB")
        ax[0, 1].imshow(weighted_depth[0].detach().cpu(), cmap="jet", vmin=0, vmax=6)
        ax[0, 1].set_title("Weighted GT Depth")
        ax[1, 1].imshow(weighted_render_depth[0].detach().cpu(), cmap="jet", vmin=0, vmax=6)
        ax[1, 1].set_title("Weighted Rendered Depth")
        ax[0, 2].imshow(diff_rgb, cmap="jet", vmin=0, vmax=0.8)
        ax[0, 2].set_title(f"Diff RGB, Loss: {torch.round(losses['im'])}")
        ax[1, 2].imshow(diff_depth, cmap="jet", vmin=0, vmax=0.8)
        ax[1, 2].set_title(f"Diff Depth, Loss: {torch.round(losses['depth'])}")
        ax[0, 3].imshow(presence_sil_mask.detach().cpu(), cmap="gray")
        ax[0, 3].set_title("Silhouette Mask")
        ax[1, 3].imshow(mask[0].detach().cpu(), cmap="gray")
        ax[1, 3].set_title("Loss Mask")
        # Turn off axis
        for i in range(2):
            for j in range(4):
                ax[i, j].axis('off')
        # Set Title
        fig.suptitle(f"Tracking Iteration: {tracking_iteration}", fontsize=16)
        # Figure Tight Layout
        fig.tight_layout()
        os.makedirs(plot_dir, exist_ok=True)
        plt.savefig(os.path.join(plot_dir, f"tmp.png"), bbox_inches='tight')
        plt.close()
        plot_img = cv2.imread(os.path.join(plot_dir, f"tmp.png"))
        cv2.imshow('Diff Images', plot_img)
        cv2.waitKey(1)
        ## Save Tracking Loss Viz
        # save_plot_dir = os.path.join(plot_dir, f"tracking_%04d" % iter_time_idx)
        # os.makedirs(save_plot_dir, exist_ok=True)
        # plt.savefig(os.path.join(save_plot_dir, f"%04d.png" % tracking_iteration), bbox_inches='tight')
        # plt.close()

    weighted_losses = {k: v * loss_weights[k] for k, v in losses.items()}
    loss = sum(weighted_losses.values())

    seen = radius > 0
    variables['max_2D_radius'][seen] = torch.max(radius[seen], variables['max_2D_radius'][seen])
    variables['seen'] = seen
    weighted_losses['loss'] = loss

    return loss, variables, weighted_losses

# new loss
def get_loss_new(params, curr_data, variables, iter_time_idx, loss_weights, use_sil_for_loss, sil_thres,
             use_l1, ignore_outlier_depth_loss, tracking=False, mapping=False, do_ba=False, device="cuda",
             plot_dir=None, visualize_tracking_loss=False, tracking_iteration=None, load_semantics=False, visualization = False):
    # Initialize Loss Dictionary
    losses = {}
    
    
    if tracking:
        # print("******tracking********")
        # Get current frame Gaussians, where only the camera pose gets gradient
        transformed_pts = transform_to_frame(params, iter_time_idx, gaussians_grad=False,
                                             camera_grad=True, device=device)
        
    elif mapping:
        # print("****MAPPING*******")
        
        if do_ba:
            # Get current frame Gaussians, where both camera pose and Gaussians get gradient
            transformed_pts = transform_to_frame(params, iter_time_idx, gaussians_grad=True,
                                                 camera_grad=True, device=device)
        else:
            # Get current frame Gaussians, where only the Gaussians get gradient
            transformed_pts = transform_to_frame(params, iter_time_idx, gaussians_grad=True,
                                                 camera_grad=False, device=device)
    else:
        # print("*****Other case*****")
        # Get current frame Gaussians, where only the Gaussians get gradient
        transformed_pts = transform_to_frame(params, iter_time_idx, gaussians_grad=True,
                                             camera_grad=False, device=device)

    # Initialize Render Variables
    rendervar = transformed_params2rendervar(params, transformed_pts, device=device)
    depth_sil_rendervar = transformed_params2depthplussilhouette(params, curr_data['w2c'],transformed_pts, device=device)
    
    # filter points in image to update opt_count variable
    if mapping:
        point_in_image_mask = filter_points_in_image(transformed_pts, curr_data['intrinsics'], H = curr_data['im'].shape[1], W = curr_data['im'].shape[2])
        params['opt_count'][point_in_image_mask] += 1

    # RGB Rendering
    # current_datetime = datetime.now()
    # time_prefix = current_datetime.strftime("%Y-%m-%d-%H-%M-%S")
    # os.makedirs(plot_dir, exist_ok=True)
    
    rendervar['means2D'].retain_grad()
    im, radius, _, = Renderer(raster_settings=curr_data['cam'])(**rendervar)
    variables['means2D'] = rendervar['means2D']  # Gradient only accum from colour render for densification

    # img1 = torch.clip(im.permute(1, 2, 0).detach().cpu(), 0, 1)
    # plt.figure(1)
    # plt.imshow(img1)
    # plt.savefig(os.path.join(plot_dir, time_prefix+'_img1.png'), bbox_inches='tight')

    # curr_data['cam']
    # im2, radius2, _, = Renderer(raster_settings=curr_data['cam'])(**rendervar)
    # variables['means2D'] = rendervar['means2D']  # Gradient only accum from colour render for densification


    # Depth & Silhouette Rendering
    depth_sil, _, _, = Renderer(raster_settings=curr_data['cam'])(**depth_sil_rendervar)
    depth = depth_sil[0, :, :].unsqueeze(0)
    silhouette = depth_sil[1, :, :]
    presence_sil_mask = (silhouette > sil_thres)
    depth_sq = depth_sil[2, :, :].unsqueeze(0)
    uncertainty = depth_sq - depth**2
    uncertainty = uncertainty.detach()
    # entropy = -silhouette*torch.log2(silhouette) - (1-silhouette)*torch.log2(1-silhouette) #jrcv
    # entropy = torch.nan_to_num(entropy, nan=0.0) #jrcv

    # Semantic colors Rendering
    rgb_fg_mask = None
    if load_semantics:
        semantic_rendervar = transformed_semantics2rendervar(params, transformed_pts, device=device)
        # rgb_loss_rendervar = transformed_rgb_loss_rendervar(params, transformed_pts, device=device)
        # entropy_rendervar = transformed_entropy2rendervar(params, transformed_pts, device=device)
        rendered_seg, _, _, = Renderer(raster_settings=curr_data['cam'])(**semantic_rendervar)
        # rendered_rgb_loss, _, _, = Renderer(raster_settings=curr_data['cam'])(**rgb_loss_rendervar)
        # rendered_entropy, _, _, =Renderer(raster_settings=curr_data['cam'])(**entropy_rendervar)
    
    # forground mask
    confidende_map = curr_data['confidence_map']
    gt_semantic = curr_data['semantic_color']
    is_background = gt_semantic.sum(dim=0, keepdim=True) == 0
    high_confidence = confidende_map > 0.4
    rgb_fg_mask = ((~is_background) & high_confidence).float()
    gt_depth = curr_data['depth']
    rgb_fg_mask[gt_depth > 1.0] = 0
    

    # Mask with valid depth values (accounts for outlier depth values)

    nan_mask = (~torch.isnan(depth)) & (~torch.isnan(uncertainty))
    
    # if ignore_outlier_depth_loss:
    #     depth_error = torch.abs(curr_data['depth'] - depth) * (curr_data['depth'] > 0)
    #     mask = (depth_error < 10*depth_error.median())
    #     mask = mask & (curr_data['depth'] > 0)
    # else:
    #     mask = (curr_data['depth'] > 0)

    # originally, considering mask for curr_data[depth]>0 (meaning ignoring free space),
    # results in floating gaussians that are never optimized. Any gaussian that after some optimization
    # step falls in the empty space, will stay there for ever.
    # mask = mask & nan_mask # commented, jrcv
    
    mask = nan_mask & (curr_data['depth'] > 0) & (rgb_fg_mask[0] > 0) # 
    
    # Mask with presence silhouette mask (accounts for empty space)
    if tracking and use_sil_for_loss:
        mask = mask & presence_sil_mask
    # else: # if mapping
    #     # trying to solve the problem of floating gaussians during mapping
    #     # this consideres all depth values
    #     # At this point, nan values are also zero.
    #     mask = (curr_data['depth'] >= 0) # this seems to work # jrcv, added TODO: further experiments might be necessary

    # Depth loss
    if use_l1:
        mask = mask.detach()
        if tracking:
            losses['depth'] = torch.abs(curr_data['depth'] - depth)[mask].sum()
        else:
            losses['depth'] = torch.abs(curr_data['depth'] - depth)[mask].mean() #original TODO, uncomment
            # losses['depth'] = torch.abs(curr_data['depth'] - depth).mean()
    # test quality loss
    # current_rgb_loss = torch.abs(im - curr_data['im']).detach()
    # losses['quality'] = torch.abs(current_rgb_loss - rendered_rgb_loss).mean()

    # RGB Loss
    if tracking and (use_sil_for_loss or ignore_outlier_depth_loss):
        color_mask = torch.tile(mask, (3, 1, 1))
        color_mask = color_mask.detach()
        losses['im'] = torch.abs(curr_data['im'] - im)[color_mask].sum()
        if load_semantics:
            losses['seg'] = torch.abs(curr_data['semantic_color'] - rendered_seg)[color_mask].sum()
    elif tracking:
        losses['im'] = torch.abs(curr_data['im'] - im).sum()
        if load_semantics:
            losses['seg'] = torch.abs(curr_data['semantic_color'] - rendered_seg).sum()
    else:
        # losses['im'] =  0.8 * l1_loss_v1(im, curr_data['im']) + 0.2 * (1.0 - calc_ssim(im, curr_data['im']))
        n_valid = rgb_fg_mask.sum().clamp(min=1)
        rgb_Ll1_loss = (torch.abs(im - curr_data['im']) * rgb_fg_mask).sum() / (n_valid * im.shape[0])
        rgb_ssim_loss = (1.0 - calc_ssim(im * rgb_fg_mask, curr_data['im'] * rgb_fg_mask))
        losses['im'] = 0.8 * rgb_Ll1_loss + 0.2 * rgb_ssim_loss
        if load_semantics:
            # losses['seg'] = 0.8 * l1_loss_v1(rendered_seg, curr_data['semantic_color']) \
            #     + 0.2 * (1.0 - calc_ssim(rendered_seg, curr_data['semantic_color']))
            # losses['seg'] = l1_loss_v1(rendered_seg, curr_data['semantic_color']) # okay
            losses['seg'] = torch.abs(curr_data['confidence_map']**3*(rendered_seg - curr_data['semantic_color'])).mean() # okay

    
    # Visualize the Diff Images
    if visualization:
        

        color_mask = torch.tile(mask, (3, 1, 1))
        color_mask = color_mask.detach()

        # fig, ax = plt.subplots(nrows=2, ncols=4, num=101, figsize=(12, 6))

        fig, ax = plt.subplots(2, 4, figsize=(12, 6))
        weighted_render_im = im * color_mask
        weighted_im = curr_data['im'] #* color_mask
        weighted_render_depth = depth * mask
        weighted_depth = curr_data['depth'] * mask
        diff_rgb = torch.abs(weighted_render_im - weighted_im).mean(dim=0).detach().cpu()
        diff_depth = torch.abs(weighted_render_depth - weighted_depth).mean(dim=0).detach().cpu()
        # rendered_entropy = torch.abs(rendered_entropy).mean(dim=0).detach().cpu()
        viz_img = torch.clip(weighted_im.permute(1, 2, 0).detach().cpu(), 0, 1)
        ax[0, 0].imshow(viz_img)
        ax[0, 0].set_title("Weighted GT RGB")
        viz_render_img_weighted = torch.clip(weighted_render_im.permute(1, 2, 0).detach().cpu(), 0, 1)
        viz_render_img = torch.clip(im.permute(1, 2, 0).detach().cpu(), 0, 1)
        viz_render_seg = torch.clip(rendered_seg.permute(1, 2, 0).detach().cpu(), 0, 1)
        ax[1, 0].imshow(viz_render_img_weighted)
        ax[1, 0].set_title("Weighted Rendered RGB")
        ax[0, 1].imshow(weighted_depth[0].detach().cpu(), cmap="jet", vmin=0, vmax=2)
        ax[0, 1].set_title("Weighted GT Depth")
        ax[1, 1].imshow(weighted_render_depth[0].detach().cpu(), cmap="jet", vmin=0, vmax=2)
        ax[1, 1].set_title("Weighted Rendered Depth")
        # ax[0, 2].imshow(diff_rgb, cmap="jet", vmin=0, vmax=0.6)
        # ax[0, 2].set_title(f"Diff RGB, Loss: {torch.round(losses['im'])}")
        ax[0, 2].imshow(viz_render_img)
        ax[0, 2].set_title(f"Rendered RGB")
        
        ax[0, 3].imshow(viz_render_seg)
        ax[0, 3].set_title("Rendered sem.")
        
        # ax[1, 2].imshow(diff_depth, cmap="jet", vmin=0, vmax=0.8)
        # ax[1, 2].set_title(f"Diff Depth, Loss: {torch.round(losses['depth'])}")
        ax[1, 2].imshow(silhouette.detach().cpu())
        ax[1, 2].set_title("Silhoutte")

        ax[1, 3].imshow(mask.squeeze().cpu())
        ax[1, 3].set_title("mask")
        

        # ax[0, 3].imshow(presence_sil_mask.detach().cpu(), cmap="gray")
        # ax[0, 3].set_title("Silhouette Mask")
        # ax[1, 3].imshow(mask[0].detach().cpu(), cmap="gray")
        # ax[1, 3].set_title("Loss Mask")
        # ax[1, 3].imshow(silhouette.detach().cpu(), cmap="jet")
        # ax[1, 3].set_title("Silhoutte")
        # ax[1, 3].imshow(entropy.detach().cpu()) #, cmap="jet")
        # ax[1, 3].set_title("Entropy")

        # vis_rend_rgb_loss = torch.clip(rendered_rgb_loss.mean(dim=0).detach().cpu(), 0, 1)
        # ax[1, 3].imshow(vis_rend_rgb_loss, cmap="jet", vmin=0, vmax=0.6)
        # ax[1, 3].set_title("rend_rgb_loss")
        
        # ax[1, 3].imshow(rendered_entropy)
        # ax[1, 3].set_title("rend_entropy")
        

        
        

        mask_cpu = mask.squeeze().cpu()
        # plt.figure(10)
        # plt.imshow(diff_rgb*mask_cpu, cmap="jet", vmin=0, vmax=0.6)
        # plt.title("dif rgb")
        # plt.figure(11)
        # plt.imshow(vis_rend_rgb_loss*mask_cpu, cmap="jet", vmin=0, vmax=0.6)
        # plt.title("rendered rgb loss")
        # plt.show()

        # plt.figure(10)
        # plt.imshow(entropy.detach().cpu(), cmap="jet", vmin=0, vmax=1.0)
        # plt.title("Entropy of Silloutte")

        # plt.figure(11)
        # plt.imshow(rendered_entropy.detach().cpu(), cmap="jet", vmin=0, vmax=1.0)
        # plt.title("Rendered entropy")
        # plt.show()
        
        # plt.figure(14)
        # plt.imshow(curr_data['confidence_map'].detach().cpu().numpy())
        # plt.title("confidence map")

        # weighted_sem_img = curr_data['semantic_color'] #* color_mask # gt semantic image
        # viz_sem_img = torch.clip(weighted_sem_img.permute(1, 2, 0).detach().cpu(), 0, 1)

        # plt.figure(15)
        # plt.imshow(viz_sem_img)
        # plt.title("Semantics")
        # plt.show()

        ## Save Tracking Loss Viz
        # save_plot_dir = os.path.join(plot_dir, f"tracking_%04d" % iter_time_idx)
        # os.makedirs(save_plot_dir, exist_ok=True)
        # plt.savefig(os.path.join(save_plot_dir, f"%04d.png" % tracking_iteration), bbox_inches='tight')
        # plt.close()

        # cam_rot = F.normalize(params['cam_unnorm_rots'][..., iter_time_idx].detach())
        # cam_tran = params['cam_trans'][..., iter_time_idx].detach()
        # rel_w2c = torch.eye(4).to(device).float()
        # rel_w2c[:3, :3] = build_rotation(cam_rot)
        # rel_w2c[:3, 3] = cam_tran
        # print("iter time index:", iter_time_idx)
        # print("relative_w2c:\n")
        # print(rel_w2c)
        # input("Press enter:")
        
        # Turn off axis
        for i in range(2):
            for j in range(4):
                ax[i, j].axis('off')
        # Set Title
        # fig.suptitle(f"Tracking Iteration: {tracking_iteration}", fontsize=16)
        # Figure Tight Layout
        # fig.tight_layout()
        plt.tight_layout()
        plt.show()
        

    weighted_losses = {k: v * loss_weights[k] for k, v in losses.items()}
    loss = sum(weighted_losses.values())

    seen = radius > 0
    variables['max_2D_radius'][seen] = torch.max(radius[seen], variables['max_2D_radius'][seen])
    variables['seen'] = seen
    weighted_losses['loss'] = loss

    return loss, variables, weighted_losses


def initialize_new_params(new_pt_cld, mean3_sq_dist, device, load_semantics=False,
                          params_opt_exclude=None):
    num_pts = new_pt_cld.shape[0]
    means3D = new_pt_cld[:, :3] # [num_gaussians, 3]
    unnorm_rots = np.tile([1, 0, 0, 0], (num_pts, 1)) # [num_gaussians, 3]
    logit_opacities = torch.zeros((num_pts, 1), dtype=torch.float, device=device)
    params = {
        'means3D': means3D,
        'rgb_colors': new_pt_cld[:, 3:6],
        'unnorm_rotations': unnorm_rots,
        'logit_opacities': logit_opacities,
        'log_scales': torch.tile(torch.log(torch.sqrt(mean3_sq_dist))[..., None], (1, 1)),
    }

    if load_semantics:
        params['semantic_ids'] = new_pt_cld[:, 6].view(-1, 1)
        params['semantic_colors'] = new_pt_cld[:, 7:10]
        params['rgb_loss'] = new_pt_cld[:, 10].view(-1, 1)
        params['opt_count'] = new_pt_cld[:, 11].view(-1, 1)


    for k, v in params.items():
        if k not in params_opt_exclude:
            # Check if value is already a torch tensor
            if not isinstance(v, torch.Tensor):
                params[k] = torch.nn.Parameter(torch.tensor(v).to(device).float().contiguous().requires_grad_(True))
            else:
                params[k] = torch.nn.Parameter(v.to(device).float().contiguous().requires_grad_(True))

    return params

def fill_zeros_nearest(depth_map, max_iter=1000):
    """
    Replace zero values in a 2D depth map using nearest non-zero neighbors.

    Args:
        depth_map: (H, W) torch tensor
        max_iter: maximum propagation iterations

    Returns:
        Filled depth map
    """

    depth = depth_map.clone()

    # Mask of missing pixels
    missing = depth == 0

    # 4-neighborhood kernel
    kernel = torch.tensor([
        [0, 1, 0],
        [1, 0, 1],
        [0, 1, 0]
    ], dtype=torch.float32, device=depth.device).view(1, 1, 3, 3)

    depth = depth.unsqueeze(0).unsqueeze(0)  # (1,1,H,W)
    missing = missing.unsqueeze(0).unsqueeze(0)

    for _ in range(max_iter):

        if not missing.any():
            break

        # Neighbor valid mask
        valid = (~missing).float()

        neighbor_count = F.conv2d(valid, kernel, padding=1)

        # Sum neighboring depth values
        neighbor_sum = F.conv2d(depth * valid, kernel, padding=1)

        # Average neighboring values
        avg_neighbor = neighbor_sum / (neighbor_count + 1e-6)

        # Pixels that can now be filled
        fillable = missing & (neighbor_count > 0)

        # Fill them
        depth[fillable] = avg_neighbor[fillable]

        # Update mask
        missing = depth == 0

    return depth.squeeze(0).squeeze(0)

def add_new_gaussians(params, params_opt_exclude, variables, curr_data, sil_thres, time_idx,
                      mean_sq_dist_method, device="cuda", load_semantics=False):
    # Silhouette Rendering
    transformed_pts = transform_to_frame(params, time_idx, gaussians_grad=False,
                                         camera_grad=False, device=device)
    depth_sil_rendervar = transformed_params2depthplussilhouette(params, curr_data['w2c'],
                                                                 transformed_pts, device=device)
    
    fill_foreground_holes = False # added by jrcv
            
    depth_sil, _, _, = Renderer(raster_settings=curr_data['cam'])(**depth_sil_rendervar)
    silhouette = depth_sil[1, :, :]
    non_presence_sil_mask = (silhouette < sil_thres)
    gt_depth = curr_data['depth'][0, :, :]
    
    if fill_foreground_holes:
        gt_seg = curr_data['semantic_color']
        background_color = torch.tensor([0, 0, 0],device=device, dtype=gt_seg.dtype)
        background_mask = torch.all(gt_seg == background_color[:, None, None], dim=0)
        gt_depth_filled = fill_zeros_nearest(gt_depth, max_iter=1000)
        gt_depth[~background_mask] = gt_depth_filled[~background_mask] # fill only relevant semantics (foreground)
    
    # Check for new foreground objects by using GT depth
    render_depth = depth_sil[0, :, :]
    depth_error = torch.abs(gt_depth - render_depth) * (gt_depth > 0)
    # print("Depth error median", depth_error.median())
    # non_presence_depth_mask = (render_depth > gt_depth) * (depth_error > 50*depth_error.median()) # depth_error.median() converges to zero over time :(
    non_presence_depth_mask = (render_depth > gt_depth) * (depth_error > 0.03) #jrcv TODO add threshold parameter
    # Determine non-presence mask
    non_presence_mask = non_presence_sil_mask | non_presence_depth_mask
    non_presence_mask_vis = non_presence_mask
    # Flatten mask
    non_presence_mask = non_presence_mask.reshape(-1)
    visualize = False
    if visualize:

        semantic_rendervar = transformed_semantics2rendervar(params, transformed_pts, device=device)
        rendered_seg, _, _, = Renderer(raster_settings=curr_data['cam'])(**semantic_rendervar)
    
        
        rgb = torch.clip(curr_data['im'].permute(1, 2, 0).detach().cpu(), 0, 1)
        rendered_seg = torch.clip(rendered_seg.permute(1, 2, 0).detach().cpu(), 0, 1)
        depth = curr_data['depth'].squeeze().detach().cpu()
        fig, ax = plt.subplots(2, 4, figsize=(12, 6))
        
        ax[0, 0].imshow(rgb)
        ax[0, 0].set_title("RGB")
        
        ax[0, 1].imshow(depth)
        ax[0, 1].set_title("GT depth")
        
        ax[0, 2].imshow(non_presence_sil_mask.detach().cpu())
        ax[0, 2].set_title("NonPres.Sil.mask")
        
        ax[0, 3].imshow(non_presence_depth_mask.detach().cpu())
        ax[0, 3].set_title("NonPres.DepthMask")
        
        ax[1, 0].imshow(non_presence_mask_vis.detach().cpu())
        ax[1, 0].set_title("NonPresMask")

        ax[1, 1].imshow(depth_error.squeeze().detach().cpu())
        ax[1, 1].set_title("Dept error")

        ax[1, 2].imshow(rendered_seg)
        ax[1, 2].set_title("Rendered Sem.")
        
        
        # Turn off axis
        for i in range(2):
            for j in range(4):
                ax[i, j].axis('off')
        plt.tight_layout()
        plt.show()

    # Get the new frame Gaussians based on the Silhouette
    if torch.sum(non_presence_mask) > 0:
        # Get the new pointcloud in the world frame
        curr_cam_rot = torch.nn.functional.normalize(params['cam_unnorm_rots'][..., time_idx].detach())
        curr_cam_tran = params['cam_trans'][..., time_idx].detach()
        curr_w2c = torch.eye(4).to(device).float()
        curr_w2c[:3, :3] = build_rotation(curr_cam_rot)
        curr_w2c[:3, 3] = curr_cam_tran
        valid_depth_mask = gt_depth > 0 #(curr_data['depth'][0, :, :] > 0)
        non_presence_mask = non_presence_mask & valid_depth_mask.reshape(-1)

        if load_semantics:
            semantic_id = curr_data['semantic_id']
            semantic_color = curr_data['semantic_color']
            rgb_loss = torch.zeros_like(curr_data['depth'])
        else:
            semantic_id = None
            semantic_color = None

        new_pt_cld, mean3_sq_dist = get_pointcloud(curr_data['im'], gt_depth.unsqueeze(0), curr_data['confidence_map'],curr_data['intrinsics'],
                                                   curr_w2c, mask=non_presence_mask, compute_mean_sq_dist=True,
                                                   mean_sq_dist_method=mean_sq_dist_method, device=device,
                                                   load_semantics=load_semantics, semantic_id=semantic_id,
                                                   semantic_color=semantic_color, rgb_loss = rgb_loss)
        new_params = initialize_new_params(new_pt_cld, mean3_sq_dist, device, load_semantics=load_semantics,
                                           params_opt_exclude=params_opt_exclude)
        for k, v in new_params.items():
            if k not in params_opt_exclude:
                params[k] = torch.nn.Parameter(torch.cat((params[k], v), dim=0).requires_grad_(True))
            else:
                params[k] = torch.cat((params[k], v), dim=0)
        # TEST clipping, jrcv
        params['semantic_colors'] = torch.nn.Parameter(params['semantic_colors'].detach().clip(0,1).requires_grad_(True))
        # np.savetxt("params_opt_count.txt", params['opt_count'].detach().cpu().numpy())
        # input("Press enter to continue...")
        num_pts = params['means3D'].shape[0]
        variables['means2D_gradient_accum'] = torch.zeros(num_pts, device=device).float()
        variables['denom'] = torch.zeros(num_pts, device=device).float()
        variables['max_2D_radius'] = torch.zeros(num_pts, device=device).float()
        new_timestep = time_idx*torch.ones(new_pt_cld.shape[0],device=device).float()
        variables['timestep'] = torch.cat((variables['timestep'],new_timestep),dim=0)
        
    return params, variables


def initialize_camera_pose(params, curr_time_idx, forward_prop, rel_w2c_initial_guess = None):
    '''
    Initial guess can come from robot forward kinematics or odometry 
    '''
    with torch.no_grad():
        if rel_w2c_initial_guess is None:
            if curr_time_idx > 1 and forward_prop:
                # Initialize the camera pose for the current frame based on a constant velocity model
                # Rotation
                prev_rot1 = F.normalize(params['cam_unnorm_rots'][..., curr_time_idx-1].detach())
                prev_rot2 = F.normalize(params['cam_unnorm_rots'][..., curr_time_idx-2].detach())
                new_rot = F.normalize(prev_rot1 + (prev_rot1 - prev_rot2))
                params['cam_unnorm_rots'][..., curr_time_idx] = new_rot.detach()
                # Translation
                prev_tran1 = params['cam_trans'][..., curr_time_idx-1].detach()
                prev_tran2 = params['cam_trans'][..., curr_time_idx-2].detach()
                new_tran = prev_tran1 + (prev_tran1 - prev_tran2)
                params['cam_trans'][..., curr_time_idx] = new_tran.detach()
            else:
                # Initialize the camera pose for the current frame
                params['cam_unnorm_rots'][..., curr_time_idx] = params['cam_unnorm_rots'][..., curr_time_idx-1].detach()
                params['cam_trans'][..., curr_time_idx] = params['cam_trans'][..., curr_time_idx-1].detach()
        else:

            # Get the ground truth pose relative to frame 0
            rel_w2c = torch.from_numpy(rel_w2c_initial_guess)
            rel_w2c_rot = rel_w2c[:3, :3].unsqueeze(0)
            rel_w2c_rot_quat = matrix_to_quaternion(rel_w2c_rot)
            rel_w2c_tran = rel_w2c[:3, 3]
            # Update the camera parameters
            params['cam_unnorm_rots'][..., curr_time_idx] = rel_w2c_rot_quat
            params['cam_trans'][..., curr_time_idx] = rel_w2c_tran


    return params


def convert_params_to_store(params):
    params_to_store = {}
    for k, v in params.items():
        if isinstance(v, torch.Tensor):
            params_to_store[k] = v.detach().clone()
        else:
            params_to_store[k] = v
    return params_to_store
