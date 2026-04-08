import argparse
import os
import shutil
import sys
import time
import pandas as pd
from importlib.machinery import SourceFileLoader

_BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _BASE_DIR)

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
import wandb

import rospy
from gazebo_msgs.msg import LinkState, LinkStates
from geometry_msgs.msg import Pose, Twist, Point, Quaternion
from sensor_msgs.msg import Image, CompressedImage
from std_msgs.msg import Float32
from cv_bridge import CvBridge, CvBridgeError
import tf2_ros
import tf
import math
from scipy.spatial.transform import Rotation
import copy
from utils.utils_active_mapping import get_semantic_image, SE3_to_ros_pose, ros_pose_to_SE3, dbscan_clustering
from utils.utils_data import load_gt_data

from datasets.gradslam_datasets import (
    load_dataset_config,
    ICLDataset,
    ReplicaDataset,
    ReplicaV2Dataset,
    EruvaeDataset,
    AzureKinectDataset,
    ScannetDataset,
    Ai2thorDataset,
    Record3DDataset,
    RealsenseDataset,
    TUMDataset,
    ScannetPPDataset,
    NeRFCaptureDataset
)
from utils.common_utils import seed_everything, save_params_ckpt, save_params
from utils.eval_helpers import report_loss, report_progress, eval
from utils.keyframe_selection import keyframe_selection_overlap
from utils.recon_helpers import setup_camera
from utils.slam_helpers import (
    transformed_params2rendervar, transformed_params2depthplussilhouette,
    transformed_semantics2rendervar, transform_to_frame, transform_points_to_frame, l1_loss_v1, matrix_to_quaternion
)
from utils.slam_external import calc_ssim, build_rotation, prune_gaussians, densify

from diff_gaussian_rasterization import GaussianRasterizer as Renderer


def get_dataset(config_dict, basedir, sequence, **kwargs):
    if config_dict["dataset_name"].lower() in ["icl"]:
        return ICLDataset(config_dict, basedir, sequence, **kwargs)
    elif config_dict["dataset_name"].lower() in ["replica"]:
        return ReplicaDataset(config_dict, basedir, sequence, **kwargs)
    elif config_dict["dataset_name"].lower() in ["eruvae"]:
        return EruvaeDataset(config_dict, basedir, sequence, **kwargs)
    elif config_dict["dataset_name"].lower() in ["replicav2"]:
        return ReplicaV2Dataset(config_dict, basedir, sequence, **kwargs)
    elif config_dict["dataset_name"].lower() in ["azure", "azurekinect"]:
        return AzureKinectDataset(config_dict, basedir, sequence, **kwargs)
    elif config_dict["dataset_name"].lower() in ["scannet"]:
        return ScannetDataset(config_dict, basedir, sequence, **kwargs)
    elif config_dict["dataset_name"].lower() in ["ai2thor"]:
        return Ai2thorDataset(config_dict, basedir, sequence, **kwargs)
    elif config_dict["dataset_name"].lower() in ["record3d"]:
        return Record3DDataset(config_dict, basedir, sequence, **kwargs)
    elif config_dict["dataset_name"].lower() in ["realsense"]:
        return RealsenseDataset(config_dict, basedir, sequence, **kwargs)
    elif config_dict["dataset_name"].lower() in ["tum"]:
        return TUMDataset(config_dict, basedir, sequence, **kwargs)
    elif config_dict["dataset_name"].lower() in ["scannetpp"]:
        return ScannetPPDataset(basedir, sequence, **kwargs)
    elif config_dict["dataset_name"].lower() in ["nerfcapture"]:
        return NeRFCaptureDataset(basedir, sequence, **kwargs)
    else:
        raise ValueError(f"Unknown dataset name {config_dict['dataset_name']}")


def evaluate_viewpoint(params, w2c, curr_data, fx, fy, sampling_r, box3d_size, device = 'cuda'):
    '''
    w2c is the target camera pose wrt to first frame (inverse) (w is cam0)
    curr_data['cam'] included intrinsics, H, W...
    curr_data['w2c'] seems to be the pose of the first frame wrt the wold (inverse)
    '''
    transformed_pts = transform_points_to_frame(params, w2c, device=device)
    depth_sil_rendervar = transformed_params2depthplussilhouette(params, curr_data['w2c'],
                                                                 transformed_pts, device=device)
    depth_sil, _, _, = Renderer(raster_settings=curr_data['cam'])(**depth_sil_rendervar)
    silhouette = depth_sil[1, :, :]
    H, W = silhouette.shape
    # Check for new foreground objects by using GT depth
    render_depth = depth_sil[0, :, :]

    entropy = -silhouette*torch.log2(silhouette) - (1-silhouette)*torch.log2(1-silhouette) #jrcv
    entropy = torch.nan_to_num(entropy, nan=0.0) #jrcv
    box2d_size_x = box3d_size*fx/sampling_r
    box2d_size_y = box3d_size*fy/sampling_r
    total_entropy = entropy[int((H-box2d_size_y)/2):int((H+box2d_size_y)/2), int((W-box2d_size_x)/2):int((W+box2d_size_x)/2)].sum().item()

    return total_entropy

def get_pointcloud(color, depth, intrinsics, w2c, transform_pts=True, mask=None,
                   compute_mean_sq_dist=False, mean_sq_dist_method="projective", device="cuda",
                   load_semantics=False, semantic_id=None, semantic_color=None):
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
    if load_semantics:
        semantic_id = torch.permute(semantic_id, (1, 2, 0)).reshape(-1, 1) # (1, H, W) -> (H, W, 1) -> (H * W, 1)
        semantic_color = torch.permute(semantic_color, (1, 2, 0)).reshape(-1, 3) # (3, H, W) -> (H, W, 3) -> (H * W, 3)
        point_cld = torch.cat((point_cld, semantic_id, semantic_color), -1)

    # Select points based on mask
    if mask is not None:
        point_cld = point_cld[mask]
        if compute_mean_sq_dist:
            mean3_sq_dist = mean3_sq_dist[mask]

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
        # channel =6 for semantic id
        params['semantic_ids'] = init_pt_cld[:, 6]
        # Channel 7-9 for semantic colors
        params['semantic_colors'] = init_pt_cld[:, 7:10]

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
        color, depth, intrinsics, pose, semantic_id, semantic_color = dataset_0
    else:
        color, depth, intrinsics, pose = dataset_0

    # Process RGB-D Data
    color = color.permute(2, 0, 1) / 255 # (H, W, C) -> (C, H, W)
    depth = depth.permute(2, 0, 1) # (H, W, 1) -> (1, H, W)
    
    if load_semantics:
        semantic_id = semantic_id.permute(2, 0, 1) # (H, W, 1) -> (1, H, W)
        semantic_color = semantic_color.permute(2, 0, 1) # (H, W, 3) -> (3, H, W)
    else:
        semantic_id = None
        semantic_color = None
    # Process Camera Parameters
    intrinsics = intrinsics[:3, :3]
    # print("pose test:",pose)
    w2c = torch.linalg.inv(pose)

    # Setup Camera
    cam = setup_camera(color.shape[2], color.shape[1], intrinsics.cpu().numpy(),
                       w2c.detach().cpu().numpy(), device=device)

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
    init_pt_cld, mean3_sq_dist = get_pointcloud(color, depth, densify_intrinsics,
                                                w2c, mask=mask, compute_mean_sq_dist=True, 
                                                mean_sq_dist_method=mean_sq_dist_method, device=device,
                                                load_semantics=load_semantics, semantic_id=semantic_id,
                                                semantic_color=semantic_color)

    # Initialize Parameters
    params, variables, params_opt_exclude = initialize_params(init_pt_cld, num_frames, mean3_sq_dist, device,
                                                              load_semantics)

    # Initialize an estimate of scene radius for Gaussian-Splatting Densification
    variables['scene_radius'] = torch.max(depth)/scene_radius_depth_ratio

    if densify_dataset is not None:
        return params, variables, intrinsics, w2c, cam, params_opt_exclude, densify_intrinsics, densify_cam
    else:
        return params, variables, intrinsics, w2c, cam, params_opt_exclude


def get_loss(params, curr_data, variables, iter_time_idx, loss_weights, use_sil_for_loss, sil_thres,
             use_l1, ignore_outlier_depth_loss, tracking=False, mapping=False, do_ba=False, device="cuda",
             plot_dir=None, visualize_tracking_loss=False, tracking_iteration=None, load_semantics=False):
    # Initialize Loss Dictionary
    losses = {}

    # Transform gaussians to current camera frame
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
    # print("curr_data w2c",curr_data['w2c']) # identity, jrcv
    depth_sil_rendervar = transformed_params2depthplussilhouette(params, curr_data['w2c'],
                                                                 transformed_pts, device=device)
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
    entropy = -silhouette*torch.log2(silhouette) - (1-silhouette)*torch.log2(1-silhouette) #jrcv
    entropy = torch.nan_to_num(entropy, nan=0.0) #jrcv

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
        viz_render_seg = torch.clip(rendered_seg.permute(1, 2, 0).detach().cpu(), 0, 1)
        ax[1, 0].imshow(viz_render_img)
        ax[1, 0].set_title("Weighted Rendered RGB")
        ax[0, 1].imshow(weighted_depth[0].detach().cpu(), cmap="jet", vmin=0, vmax=2)
        ax[0, 1].set_title("Weighted GT Depth")
        ax[1, 1].imshow(weighted_render_depth[0].detach().cpu(), cmap="jet", vmin=0, vmax=2)
        ax[1, 1].set_title("Weighted Rendered Depth")
        ax[0, 2].imshow(diff_rgb, cmap="jet", vmin=0, vmax=0.8)
        ax[0, 2].set_title(f"Diff RGB, Loss: {torch.round(losses['im'])}")
        ax[1, 2].imshow(viz_render_seg)
        ax[1, 2].set_title("Rendered sem.")
        
        # ax[1, 2].imshow(diff_depth, cmap="jet", vmin=0, vmax=0.8)
        # ax[1, 2].set_title(f"Diff Depth, Loss: {torch.round(losses['depth'])}")
        ax[0, 3].imshow(silhouette.detach().cpu())
        ax[0, 3].set_title("Silhoutte")
        ax[1, 3].imshow(presence_sil_mask.detach().cpu(), cmap="gray")
        ax[1, 3].set_title("Silhouette Mask")
        # ax[1, 3].imshow(mask[0].detach().cpu(), cmap="gray")
        # ax[1, 3].set_title("Loss Mask")
        # ax[1, 3].imshow(silhouette.detach().cpu(), cmap="jet")
        # ax[1, 3].set_title("Silhoutte")
        # ax[1, 3].imshow(entropy.detach().cpu(), cmap="jet")
        # ax[1, 3].set_title("entropy")
        
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
    # input("Press enter to continue:")
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
        params['semantic_ids'] = new_pt_cld[:, 6]
        params['semantic_colors'] = new_pt_cld[:, 7:10]

    for k, v in params.items():
        if k not in params_opt_exclude:
            # Check if value is already a torch tensor
            if not isinstance(v, torch.Tensor):
                params[k] = torch.nn.Parameter(torch.tensor(v).to(device).float().contiguous().requires_grad_(True))
            else:
                params[k] = torch.nn.Parameter(v.to(device).float().contiguous().requires_grad_(True))

    return params


def add_new_gaussians(params, params_opt_exclude, variables, curr_data, sil_thres, time_idx,
                      mean_sq_dist_method, device="cuda", load_semantics=False):
    # Silhouette Rendering
    transformed_pts = transform_to_frame(params, time_idx, gaussians_grad=False,
                                         camera_grad=False, device=device)
    depth_sil_rendervar = transformed_params2depthplussilhouette(params, curr_data['w2c'],
                                                                 transformed_pts, device=device)
    depth_sil, _, _, = Renderer(raster_settings=curr_data['cam'])(**depth_sil_rendervar)
    silhouette = depth_sil[1, :, :]
    non_presence_sil_mask = (silhouette < sil_thres)
    # Check for new foreground objects by using GT depth
    gt_depth = curr_data['depth'][0, :, :]
    render_depth = depth_sil[0, :, :]
    depth_error = torch.abs(gt_depth - render_depth) * (gt_depth > 0)
    non_presence_depth_mask = (render_depth > gt_depth) * (depth_error > 50*depth_error.median())
    # Determine non-presence mask
    non_presence_mask = non_presence_sil_mask | non_presence_depth_mask
    # Flatten mask
    non_presence_mask = non_presence_mask.reshape(-1)

    # Get the new frame Gaussians based on the Silhouette
    if torch.sum(non_presence_mask) > 0:
        # Get the new pointcloud in the world frame
        curr_cam_rot = torch.nn.functional.normalize(params['cam_unnorm_rots'][..., time_idx].detach())
        curr_cam_tran = params['cam_trans'][..., time_idx].detach()
        curr_w2c = torch.eye(4).to(device).float()
        curr_w2c[:3, :3] = build_rotation(curr_cam_rot)
        curr_w2c[:3, 3] = curr_cam_tran
        valid_depth_mask = (curr_data['depth'][0, :, :] > 0)
        non_presence_mask = non_presence_mask & valid_depth_mask.reshape(-1)

        if load_semantics:
            semantic_id = curr_data['semantic_id']
            semantic_color = curr_data['semantic_color']
        else:
            semantic_id = None
            semantic_color = None

        new_pt_cld, mean3_sq_dist = get_pointcloud(curr_data['im'], curr_data['depth'], curr_data['intrinsics'],
                                                   curr_w2c, mask=non_presence_mask, compute_mean_sq_dist=True,
                                                   mean_sq_dist_method=mean_sq_dist_method, device=device,
                                                   load_semantics=load_semantics, semantic_id=semantic_id,
                                                   semantic_color=semantic_color)
        new_params = initialize_new_params(new_pt_cld, mean3_sq_dist, device, load_semantics=load_semantics,
                                           params_opt_exclude=params_opt_exclude)
        for k, v in new_params.items():
            if k not in params_opt_exclude:
                params[k] = torch.nn.Parameter(torch.cat((params[k], v), dim=0).requires_grad_(True))
            else:
                params[k] = torch.cat((params[k], v), dim=0)
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





class ActiveSLAM:
    def __init__(self, config):
        
        rospy.init_node('active_slam', anonymous=True)
        self.bridge = CvBridge()

         # Subscribers
        rospy.Subscriber('/camera2/color/image_raw/compressed', CompressedImage, self.callback_image_raw)
        rospy.Subscriber('/camera2/depth/image_raw', Image, self.callback_depth_topic)
        rospy.Subscriber('/gazebo/link_states', LinkStates, self.callback_link_states)
        rospy.Subscriber("/new_image_data", Float32, self.callback_new_image_data) # new image data from nbv planning
        self.tf_listener = tf.TransformListener()
        
        # Publisher
        self.pub_link_state = rospy.Publisher('/gazebo/set_link_state', LinkState, queue_size=10)
        self.pub_sgs_status = rospy.Publisher('/sgs_status', Float32, queue_size = 10)
        
        self.cv_image = None
        self.cv_depth_image = None
        self.crop_size = 400
        self.gt_pose_w_camlink = None
        self.T_w_camframe = None
        self.T_camlink_camframe = np.array([[0.0, 0.0, 1.0, 0.0],
                                            [-1.0, 0.0, 0.0, 0.0],
                                            [0.0, -1.0, 0.0, 0.0],
                                            [0.0, 0.0, 0.0, 1.0]])
        self.fx, self.fy, self.cx, self.cy = 381.36246688113556, 381.36246688113556, self.crop_size/2 , self.crop_size/2
        # self.fx, self.fy, self.cx, self.cy = 381.36246688113556, 381.36246688113556, 320.5, 240.5 # 640x480
        self.K = np.array([[self.fx, 0, self.cx],
                            [0, self.fy, self.cy],
                            [0, 0, 1.0]])
        self.sampling_r = 0.4
        self.T_w_c0 = None
        self.initialized  = False
        self.camlink_viewpoints, self.camframe_viewpoints = self.compute_viewpoints()
        self.new_image_data = None
        self.rgbd_slam(config)
        

        rate = rospy.Rate(0.5)
        
        while not rospy.is_shutdown():
            rate.sleep()
            print("Running sgs_slam ros node...")
            

    def compute_viewpoints(self):
        plant_id = 4
        plant_height = 1.0
        
        self.gt_centroids, self.xyz_plant_origin, self.gt_pointcloud_path = load_gt_data(plant_id)
        plant_centroid = copy.deepcopy(self.xyz_plant_origin)
        plant_centroid[2] = plant_height/2
        # camlink poses for different samples
        camlink_poses, camframe_poses = self.gen_cam_poses(0, 360, 45, 60, centroid=plant_centroid, theta_n_grid=36, phi_n_grid = 2, r=self.sampling_r)    

        return camlink_poses, camframe_poses    

    def execute_single_camlink_goal(self, pose):
        self.publish_link_pose(pose,'link_kinect')
    
    def publish_link_pose(self, pose, link_name):
        link_state_msg = LinkState()
        link_state_msg.link_name = link_name  # Replace with the name of your link
        link_state_msg.pose = pose  # Set the desired pose
        # link_state_msg.twist = Twist()  # Set the desired twist
        self.pub_link_state.publish(link_state_msg)
        
    def callback_depth_topic(self, data):
        try:
            self.cv_depth_image = self.bridge.imgmsg_to_cv2(data)
            self.cv_depth_image = self.crop_center_square(self.cv_depth_image, self.crop_size)
        
        except Exception as e:
            rospy.logerr("Error converting depth Image to cv2: %s", e)
            return
    def callback_image_raw(self, data):
        try:
            self.cv_image = self.bridge.compressed_imgmsg_to_cv2(data)
            self.cv_image = self.crop_center_square(self.cv_image, self.crop_size)
        except Exception as e:
            rospy.logerr("Error converting compressed image to cv2: %s", e)
            return
    def callback_link_states(self, data):
        self.gt_pose_w_camlink = data.pose[0] # 0 corresponds to link_kinect
        T_w_camlink = ros_pose_to_SE3(self.gt_pose_w_camlink)
        self.T_w_camframe = np.matmul(T_w_camlink, self.T_camlink_camframe)
    
    def callback_new_image_data(self, msg):
        self.new_image_data = msg.data

    def crop_center_square(self, image, size):
        h, w = image.shape[:2]
        top = (h - size)//2
        bottom = top + size
        left = (w - size) // 2
        right = left + size
        cropped_img = image[top:bottom, left:right]
        return cropped_img
    
    def get_transform(self, parent_frame, child_frame):
        # Wait for the transformation to become available
        
        self.tf_listener.waitForTransform(parent_frame, child_frame, rospy.Time(), rospy.Duration(4.0))
        # Get the transformation
        try:
            (trans, rot) = self.tf_listener.lookupTransform(parent_frame, child_frame, rospy.Time(0))
            pose = Pose()
            pose.position.x = trans[0]
            pose.position.y = trans[1]
            pose.position.z = trans[2]
            pose.orientation.x = rot[0]
            pose.orientation.y = rot[1]
            pose.orientation.z = rot[2]
            pose.orientation.w = rot[3]
            return pose 

        except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException) as e:
            print(f"Error: {e}")
    
    def gen_cam_poses(self, theta_start, theta_end, phi_start, phi_end, centroid, theta_n_grid=5, phi_n_grid=5, r = 1): # values in degrees
        '''
        It generates viewpoints around a centroid, distributed on a sphere with radius r
        Theta and phi in degrees. n_greed determines the number of samples for Theta and phi to create a grid
        '''
        theta_list = ((np.pi/180)*np.linspace(theta_start, theta_end, theta_n_grid)).tolist()
        phi_list = ((np.pi/180)*np.linspace(phi_start, phi_end, phi_n_grid)).tolist()
        #https://mathworld.wolfram.com/SphericalCoordinates.html
        
        poses_cam_link = [] # poses of camera2_link (that we can control)
        poses_camframe = [] # poses of camera frame wrt world frame
        for phi in phi_list:
            for theta in theta_list:
                x = r*math.cos(theta)*math.sin(phi)
                y = r*math.sin(theta)*math.sin(phi)
                z = r*math.cos(phi)
                t_pg = np.array([[x],[y],[z]])
                R_pg = Rotation.from_euler('xyz',[np.pi,phi+np.pi/2,theta]).as_matrix()
                T_pg = np.block([[R_pg, t_pg],[0.0, 0.0, 0.0, 1.0]]) # camera2link to plant_frame
                
                T_wp = np.eye(4)
                T_wp[0,3] = centroid[0]
                T_wp[1,3] = centroid[1]
                T_wp[2,3] = centroid[2] # Twp: plant_centroid to world frame
                T_wg = np.matmul(T_wp, T_pg) # camera2_link to world frame
                
                R_wg = T_wg[0:3,0:3]
                q_wg = Rotation.from_matrix(R_wg).as_quat()
                t_wg = T_wg[:,3].tolist() 
                pose_wg = Pose(Point(t_wg[0],t_wg[1],t_wg[2]),
                                Quaternion(q_wg[0],q_wg[1],q_wg[2],q_wg[3]))
                poses_cam_link.append(pose_wg)

                T_g_camframe = self.T_camlink_camframe
                T_w_camframe = np.matmul(T_wg, T_g_camframe)

                R_w_camframe = T_w_camframe[0:3, 0:3]
                q_w_camframe = Rotation.from_matrix(R_w_camframe).as_quat()
                t_w_camframe = T_w_camframe[:,3].tolist()
                pose_w_camframe = Pose(Point(t_w_camframe[0],t_w_camframe[1],t_w_camframe[2]),
                                Quaternion(q_w_camframe[0],q_w_camframe[1],q_w_camframe[2],q_w_camframe[3]))
                
                poses_camframe.append(pose_w_camframe)
        return poses_cam_link, poses_camframe

    def get_sample_data(self, device="cuda:0", dtype = torch.float):
        intrinsics = torch.from_numpy(self.K)
        if (self.initialized == False):
            self.T_w_c0 = self.T_w_camframe
            T_c0_c = np.eye(4) #c0 is first frame
            self.initialized = True
            
        else:
            # relative to first frame c0
            T_c0_c = np.matmul(np.linalg.inv(self.T_w_c0), self.T_w_camframe)
        T_c0_c = torch.from_numpy(T_c0_c)

        
        bgr_image = copy.deepcopy(self.cv_image)
        rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
        rgb_image = rgb_image.astype(float) #/255
        rgb_image = torch.from_numpy(rgb_image)

        semantic_img_uint8_bgr, _, __ = get_semantic_image(bgr_image, add_seg_noise = False)
        semantic_img_uint8_rgb = cv2.cvtColor(semantic_img_uint8_bgr, cv2.COLOR_BGR2RGB)
        semantic_img_float = semantic_img_uint8_rgb.astype(float) #/255
        semantic_img = torch.from_numpy(semantic_img_float)

        semantic_id = cv2.cvtColor(semantic_img_uint8_bgr, cv2.COLOR_BGR2GRAY)
        semantic_id = semantic_id.astype(float)
        semantic_id = np.expand_dims(semantic_id, -1)#(h,w,1)
        semantic_id = torch.from_numpy(semantic_id)

        depth = self.cv_depth_image.astype(float) # m
        depth = np.expand_dims(depth, -1) #(h,w,1)
        depth = torch.from_numpy(depth)
        # depth = torch.nan_to_num(depth, nan=0.0)
        depth = torch.clamp(depth, 0, 1.5)

        return_data = (
            rgb_image.to(device).type(dtype),
            depth.to(device).type(dtype),
            intrinsics.to(device).type(dtype),
            T_c0_c.to(device).type(dtype),
            semantic_id.to(device).type(dtype),
            semantic_img.to(device).type(dtype),
        )
        self.new_image_data = None
        return return_data

    def rgbd_slam(self, config: dict):
        # Print Config
        print("Loaded Config:")
        if "use_depth_loss_thres" not in config['tracking']:
            config['tracking']['use_depth_loss_thres'] = False
            config['tracking']['depth_loss_thres'] = 100000
        if "visualize_tracking_loss" not in config['tracking']:
            config['tracking']['visualize_tracking_loss'] = False
        print(f"{config}")

        # Create Output Directories
        output_dir = os.path.join(config["workdir"], config["run_name"])
        eval_dir = os.path.join(output_dir, "eval")
        os.makedirs(eval_dir, exist_ok=True)
        
        # Init WandB
        if config['use_wandb']:
            wandb_time_step = 0
            wandb_tracking_step = 0
            wandb_mapping_step = 0
            wandb_run = wandb.init(project=config['wandb']['project'],
                                entity=config['wandb']['entity'],
                                group=config['wandb']['group'],
                                name=config['wandb']['name'],
                                config=config)

        # Get Device
        device = torch.device(config["primary_device"])
        if config["primary_device"].startswith("cuda:"):
            device_id = int(config["primary_device"].split(':')[1])
            torch.cuda.set_device(device_id)

        # Load Dataset
        print("Loading Dataset ...")
        dataset_config = config["data"]
        if "gradslam_data_cfg" not in dataset_config:
            gradslam_data_cfg = {}
            gradslam_data_cfg["dataset_name"] = dataset_config["dataset_name"]
        else:
            gradslam_data_cfg = load_dataset_config(dataset_config["gradslam_data_cfg"])
        if "ignore_bad" not in dataset_config:
            dataset_config["ignore_bad"] = False
        if "use_train_split" not in dataset_config:
            dataset_config["use_train_split"] = True
        if "densification_image_height" not in dataset_config:
            dataset_config["densification_image_height"] = dataset_config["desired_image_height"]
            dataset_config["densification_image_width"] = dataset_config["desired_image_width"]
            seperate_densification_res = False
        else:
            if dataset_config["densification_image_height"] != dataset_config["desired_image_height"] or \
                dataset_config["densification_image_width"] != dataset_config["desired_image_width"]:
                seperate_densification_res = True
            else:
                seperate_densification_res = False
        if "tracking_image_height" not in dataset_config:
            dataset_config["tracking_image_height"] = dataset_config["desired_image_height"]
            dataset_config["tracking_image_width"] = dataset_config["desired_image_width"]
            seperate_tracking_res = False
        else:
            if dataset_config["tracking_image_height"] != dataset_config["desired_image_height"] or \
                dataset_config["tracking_image_width"] != dataset_config["desired_image_width"]:
                seperate_tracking_res = True
            else:
                seperate_tracking_res = False
        if "load_semantics" not in dataset_config:
            load_semantics = False
            num_semantic_classes = 0
        else:
            load_semantics = dataset_config["load_semantics"]
            num_semantic_classes = dataset_config["num_semantic_classes"]
        # Poses are relative to the first frame
        dataset = get_dataset(
            config_dict=gradslam_data_cfg,
            basedir=dataset_config["basedir"],
            sequence=os.path.basename(dataset_config["sequence"]),
            start=dataset_config["start"],
            end=dataset_config["end"],
            stride=dataset_config["stride"],
            desired_height=dataset_config["desired_image_height"],
            desired_width=dataset_config["desired_image_width"],
            device=device,
            relative_pose=True,
            ignore_bad=dataset_config["ignore_bad"],
            use_train_split=dataset_config["use_train_split"],
            load_semantics=load_semantics,
            num_semantic_classes=num_semantic_classes,
        )
        num_frames = dataset_config["num_frames"]
        if num_frames == -1:
            num_frames = len(dataset)


        # Initialize Parameters & Canoncial Camera parameters
            
        while (self.new_image_data == None):
            print("wating for first sample image")
            time.sleep(0.5)
        
        dataset_0 = self.get_sample_data()
        params, variables, intrinsics, first_frame_w2c, cam, \
            params_opt_exclude = initialize_first_timestep(dataset_0, num_frames, config['scene_radius_depth_ratio'],
                                                        config['mean_sq_dist_method'], device=device,
                                                        load_semantics=load_semantics)
    
        # Initialize list to keep track of Keyframes
        keyframe_list = []
        keyframe_time_indices = []
        timestamp_keyframes = []
        
        # Init Variables to keep track of ground truth poses and runtimes
        gt_w2c_all_frames = []
        tracking_iter_time_sum = 0
        tracking_iter_time_count = 0
        mapping_iter_time_sum = 0
        mapping_iter_time_count = 0
        tracking_frame_time_sum = 0
        tracking_frame_time_count = 0
        mapping_frame_time_sum = 0
        mapping_frame_time_count = 0

        # Load Checkpoint
        
        checkpoint_time_idx = 0
        
        # Iterate over Scan
        for time_idx in tqdm(range(checkpoint_time_idx, num_frames)):
            # Load RGBD frames incrementally instead of all frames
            print("Current time idx:", time_idx)
            print("Number of gaussians:", params['means3D'].shape[0])
            time_idx_active = 2
            if time_idx>0:
                # self.pub_sgs_status.publish(Float32(1.0)) #ready to receive data
                while (self.new_image_data == None):
                    print("wating for new image data")
                    time.sleep(0.5)
            
        
            if load_semantics:
                color, depth, _, gt_pose, semantic_id, semantic_color = self.get_sample_data() #dataset[time_idx], jrcv
            else:
                color, depth, _, gt_pose = dataset[time_idx]
            # Process poses
            gt_w2c = torch.linalg.inv(gt_pose)
            # Process RGB-D Data
            color = color.permute(2, 0, 1) / 255
            depth = depth.permute(2, 0, 1)
            gt_w2c_all_frames.append(gt_w2c)
            curr_gt_w2c = gt_w2c_all_frames
            # Optimize only current time step for tracking
            iter_time_idx = time_idx
            # Initialize Mapping Data for selected frame
            curr_data = {'cam': cam, 'im': color, 'depth': depth, 'id': iter_time_idx, 'intrinsics': intrinsics,
                        'w2c': first_frame_w2c, 'iter_gt_w2c_list': curr_gt_w2c}
            
            if load_semantics:
                semantic_id = semantic_id.permute(2, 0, 1)
                semantic_color = semantic_color.permute(2, 0, 1) / 255
                curr_data['semantic_id'] = semantic_id
                curr_data['semantic_color'] = semantic_color
            
            tracking_curr_data = curr_data

            # Optimization Iterations
            num_iters_mapping = config['mapping']['num_iters']
            
            # Initialize the camera pose for the current frame
            if time_idx > 0:
                with torch.no_grad():
                    # initialization based on constant velocity model
                    params = initialize_camera_pose(params, time_idx,
                                                forward_prop=config['tracking']['forward_prop'])
                    # # intialize with ground truth for now
                    # params = initialize_camera_pose(params, time_idx,
                    #                             forward_prop=config['tracking']['forward_prop'],
                    #                             rel_w2c_initial_guess = curr_gt_w2c[-1].detach().cpu().numpy())
                
            # Step 1: Tracking
            tracking_start_time = time.time()
            if time_idx > 0 and not config['tracking']['use_gt_poses']:
                # Reset Optimizer & Learning Rates for tracking
                optimizer = initialize_optimizer(params, params_opt_exclude, config['tracking']['lrs'], tracking=True)
                # Keep Track of Best Candidate Rotation & Translation
                candidate_cam_unnorm_rot = params['cam_unnorm_rots'][..., time_idx].detach().clone()
                candidate_cam_tran = params['cam_trans'][..., time_idx].detach().clone()
                current_min_loss = float(1e20)
                # Tracking Optimization
                iter = 0
                do_continue_slam = False
                num_iters_tracking = config['tracking']['num_iters']
                progress_bar = tqdm(range(num_iters_tracking), desc=f"Tracking Time Step: {time_idx}")
                while True:
                    iter_start_time = time.time()
                    # Loss for current frame
                    loss, variables, losses = get_loss(params, tracking_curr_data, variables, iter_time_idx, config['tracking']['loss_weights'],
                                                    config['tracking']['use_sil_for_loss'], config['tracking']['sil_thres'],
                                                    config['tracking']['use_l1'], config['tracking']['ignore_outlier_depth_loss'],
                                                    tracking=True, device=device, plot_dir=eval_dir,
                                                    visualize_tracking_loss=config['tracking']['visualize_tracking_loss'],
                                                    tracking_iteration=iter, load_semantics=load_semantics)
                    if config['use_wandb']:
                        # Report Loss
                        wandb_tracking_step = report_loss(losses, wandb_run, wandb_tracking_step, tracking=True, load_semantics=load_semantics)
                    # Backprop
                    loss.backward()
                    # Optimizer Update
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                    with torch.no_grad():
                        # Save the best candidate rotation & translation
                        if loss < current_min_loss:
                            current_min_loss = loss
                            candidate_cam_unnorm_rot = params['cam_unnorm_rots'][..., time_idx].detach().clone()
                            candidate_cam_tran = params['cam_trans'][..., time_idx].detach().clone()
                            print("loss< current_min_loss")
                            # update normalized rotation, jrcv
                            # params['cam_unnorm_rots'][..., time_idx] = F.normalize(params['cam_unnorm_rots'][..., time_idx].detach())
                            
                        # Report Progress
                        if config['report_iter_progress']:
                            if config['use_wandb']:
                                report_progress(params, tracking_curr_data, iter+1, progress_bar, iter_time_idx, sil_thres=config['tracking']['sil_thres'],
                                                tracking=True, device=device, load_semantics=load_semantics, wandb_run=wandb_run, wandb_step=wandb_tracking_step,
                                                wandb_save_qual=config['wandb']['save_qual'])
                            else:
                                report_progress(params, tracking_curr_data, iter+1, progress_bar, iter_time_idx, sil_thres=config['tracking']['sil_thres'],
                                                tracking=True, device=device, load_semantics=load_semantics)
                        else:
                            progress_bar.update(1)
                    # Update the runtime numbers
                    iter_end_time = time.time()
                    tracking_iter_time_sum += iter_end_time - iter_start_time
                    tracking_iter_time_count += 1
                    # Check if we should stop tracking
                    iter += 1
                    if iter == num_iters_tracking:
                        if losses['depth'] < config['tracking']['depth_loss_thres'] and config['tracking']['use_depth_loss_thres']:
                            break
                        elif config['tracking']['use_depth_loss_thres'] and not do_continue_slam:
                            do_continue_slam = True
                            progress_bar = tqdm(range(num_iters_tracking), desc=f"Tracking Time Step: {time_idx}")
                            num_iters_tracking = 2*num_iters_tracking
                            if config['use_wandb']:
                                wandb_run.log({"Tracking/Extra Tracking Iters Frames": time_idx,
                                            "Tracking/step": wandb_time_step})
                        else:
                            break

                progress_bar.close()
                # Copy over the best candidate rotation & translation
                with torch.no_grad():
                    # pass #TODO REMOVE
                    params['cam_unnorm_rots'][..., time_idx] = candidate_cam_unnorm_rot
                    params['cam_trans'][..., time_idx] = candidate_cam_tran
            elif time_idx > 0 and config['tracking']['use_gt_poses']:
                with torch.no_grad():
                    # Get the ground truth pose relative to frame 0
                    rel_w2c = curr_gt_w2c[-1]
                    rel_w2c_rot = rel_w2c[:3, :3].unsqueeze(0).detach()
                    rel_w2c_rot_quat = matrix_to_quaternion(rel_w2c_rot)
                    rel_w2c_tran = rel_w2c[:3, 3].detach()
                    # Update the camera parameters
                    params['cam_unnorm_rots'][..., time_idx] = rel_w2c_rot_quat
                    params['cam_trans'][..., time_idx] = rel_w2c_tran
            # Update the runtime numbers
            tracking_end_time = time.time()
            tracking_frame_time_sum += tracking_end_time - tracking_start_time
            tracking_frame_time_count += 1

            if time_idx == 0 or (time_idx+1) % config['report_global_progress_every'] == 0:
                try:
                    # Report Final Tracking Progress
                    progress_bar = tqdm(range(1), desc=f"Tracking Result Time Step: {time_idx}")
                    with torch.no_grad():
                        if config['use_wandb']:
                            report_progress(params, tracking_curr_data, 1, progress_bar, iter_time_idx, sil_thres=config['tracking']['sil_thres'],
                                            tracking=True, device=device, load_semantics=load_semantics, wandb_run=wandb_run, wandb_step=wandb_time_step,
                                            wandb_save_qual=config['wandb']['save_qual'], global_logging=True)
                        else:
                            report_progress(params, tracking_curr_data, 1, progress_bar, iter_time_idx, sil_thres=config['tracking']['sil_thres'],
                                            tracking=True, device=device, load_semantics=load_semantics)
                    progress_bar.close()
                except:
                    ckpt_output_dir = os.path.join(config["workdir"], config["run_name"])
                    save_params_ckpt(params, ckpt_output_dir, time_idx)
                    print('Failed to evaluate trajectory.')

            # Step 2: Densification & KeyFrame-based Mapping
            if time_idx == 0 or (time_idx+1) % config['map_every'] == 0:
                # Densification for time_idx > 0
                if config['mapping']['add_new_gaussians'] and time_idx > 0:
                    # Setup Data for Densification
                    densify_curr_data = curr_data

                    # Add new Gaussians to the scene based on the Silhouette
                    params, variables = add_new_gaussians(params, params_opt_exclude, variables, densify_curr_data, 
                                                        config['mapping']['sil_thres'], time_idx, config['mean_sq_dist_method'],
                                                        device, load_semantics=load_semantics)
                    post_num_pts = params['means3D'].shape[0]
                    if config['use_wandb']:
                        wandb_run.log({"Mapping/Number of Gaussians": post_num_pts,
                                    "Mapping/step": wandb_time_step})
                
                # Update keyframes for gaussian mapping
                with torch.no_grad():
                    # Get the current estimated rotation & translation
                    curr_cam_rot = F.normalize(params['cam_unnorm_rots'][..., time_idx].detach())
                    curr_cam_tran = params['cam_trans'][..., time_idx].detach()
                    curr_w2c = torch.eye(4).to(device).float()
                    curr_w2c[:3, :3] = build_rotation(curr_cam_rot)
                    curr_w2c[:3, 3] = curr_cam_tran
                    # Select Keyframes for Mapping
                    num_keyframes = config['mapping_window_size']-2
                    selected_keyframes = keyframe_selection_overlap(depth, curr_w2c, intrinsics, keyframe_list[:-1],
                                                                    num_keyframes, device=device)
                    selected_time_idx = [keyframe_list[frame_idx]['id'] for frame_idx in selected_keyframes]
                    if len(keyframe_list) > 0:
                        # Add last keyframe to the selected keyframes
                        selected_time_idx.append(keyframe_list[-1]['id'])
                        selected_keyframes.append(len(keyframe_list)-1)
                    # Add current frame to the selected keyframes
                    selected_time_idx.append(time_idx)
                    selected_keyframes.append(-1)
                    # Print the selected keyframes
                    print(f"\nSelected Keyframes at Frame {time_idx}: {selected_time_idx}")
                    timestamp_keyframes.append(selected_time_idx)

                # Reset Optimizer & Learning Rates for Full Map Optimization
                optimizer = initialize_optimizer(params, params_opt_exclude, config['mapping']['lrs'], tracking=False) 

                # Mapping
                mapping_start_time = time.time()
                if num_iters_mapping > 0:
                    progress_bar = tqdm(range(num_iters_mapping), desc=f"Mapping Time Step: {time_idx}")
                for iter in range(num_iters_mapping):
                    iter_start_time = time.time()
                    # Randomly select a frame until current time step amongst keyframes
                    rand_idx = np.random.randint(0, len(selected_keyframes))
                    selected_rand_keyframe_idx = selected_keyframes[rand_idx]
                    if selected_rand_keyframe_idx == -1:
                        # Use Current Frame Data
                        iter_time_idx = time_idx
                        iter_color = color
                        iter_depth = depth
                    else:
                        # Use Keyframe Data
                        iter_time_idx = keyframe_list[selected_rand_keyframe_idx]['id']
                        iter_color = keyframe_list[selected_rand_keyframe_idx]['color']
                        iter_depth = keyframe_list[selected_rand_keyframe_idx]['depth']
                    iter_gt_w2c = gt_w2c_all_frames[:iter_time_idx+1]
                    iter_data = {'cam': cam, 'im': iter_color, 'depth': iter_depth, 'id': iter_time_idx, 
                                'intrinsics': intrinsics, 'w2c': first_frame_w2c, 'iter_gt_w2c_list': iter_gt_w2c}
                    # Add semantic id and colors
                    if load_semantics:
                        if selected_rand_keyframe_idx == -1:
                            iter_data['semantic_id'] = semantic_id
                            iter_data['semantic_color'] = semantic_color
                        else:
                            iter_data['semantic_id'] = keyframe_list[selected_rand_keyframe_idx]['semantic_id']
                            iter_data['semantic_color'] = keyframe_list[selected_rand_keyframe_idx]['semantic_color']
                    # Loss for current frame
                    loss, variables, losses = get_loss(params, iter_data, variables, iter_time_idx, config['mapping']['loss_weights'],
                                                    config['mapping']['use_sil_for_loss'], config['mapping']['sil_thres'],
                                                    config['mapping']['use_l1'], config['mapping']['ignore_outlier_depth_loss'],
                                                    mapping=True, device=device, load_semantics=load_semantics)
                    if config['use_wandb']:
                        # Report Loss
                        wandb_mapping_step = report_loss(losses, wandb_run, wandb_mapping_step, mapping=True, load_semantics=load_semantics)
                    # Backprop
                    loss.backward()
                    with torch.no_grad():
                        # Prune Gaussians
                        if config['mapping']['prune_gaussians']:
                            params, variables = prune_gaussians(params, params_opt_exclude, variables, optimizer, iter, config['mapping']['pruning_dict'])
                            if config['use_wandb']:
                                wandb_run.log({"Mapping/Number of Gaussians - Pruning": params['means3D'].shape[0],
                                            "Mapping/step": wandb_mapping_step})
                        # Gaussian-Splatting's Gradient-based Densification
                        if config['mapping']['use_gaussian_splatting_densification']:
                            params, variables = densify(params, variables, optimizer, iter, config['mapping']['densify_dict'], params_opt_exclude, device=device)
                            if config['use_wandb']:
                                wandb_run.log({"Mapping/Number of Gaussians - Densification": params['means3D'].shape[0],
                                            "Mapping/step": wandb_mapping_step})
                        # Optimizer Update
                        optimizer.step()
                        optimizer.zero_grad(set_to_none=True)
                        # Report Progress
                        if config['report_iter_progress']:
                            if config['use_wandb']:
                                report_progress(params, iter_data, iter+1, progress_bar, iter_time_idx, sil_thres=config['mapping']['sil_thres'], 
                                                wandb_run=wandb_run, wandb_step=wandb_mapping_step, wandb_save_qual=config['wandb']['save_qual'],
                                                mapping=True, device=device, load_semantics=load_semantics, online_time_idx=time_idx)
                            else:
                                report_progress(params, iter_data, iter+1, progress_bar, iter_time_idx, sil_thres=config['mapping']['sil_thres'], 
                                                mapping=True, device=device, load_semantics=load_semantics, online_time_idx=time_idx)
                        else:
                            progress_bar.update(1)
                    # Update the runtime numbers
                    iter_end_time = time.time()
                    mapping_iter_time_sum += iter_end_time - iter_start_time
                    mapping_iter_time_count += 1
                if num_iters_mapping > 0:
                    progress_bar.close()
                # Update the runtime numbers
                mapping_end_time = time.time()
                mapping_frame_time_sum += mapping_end_time - mapping_start_time
                mapping_frame_time_count += 1

                if time_idx == 0 or (time_idx+1) % config['report_global_progress_every'] == 0:
                    try:
                        # Report Mapping Progress
                        progress_bar = tqdm(range(1), desc=f"Mapping Result Time Step: {time_idx}")
                        with torch.no_grad():
                            if config['use_wandb']:
                                report_progress(params, curr_data, 1, progress_bar, time_idx, sil_thres=config['mapping']['sil_thres'], 
                                                wandb_run=wandb_run, wandb_step=wandb_time_step, wandb_save_qual=config['wandb']['save_qual'],
                                                mapping=True, device=device, load_semantics=load_semantics, online_time_idx=time_idx, global_logging=True)
                            else:
                                report_progress(params, curr_data, 1, progress_bar, time_idx, sil_thres=config['mapping']['sil_thres'], 
                                                mapping=True, device=device, load_semantics=load_semantics, online_time_idx=time_idx)
                        progress_bar.close()
                    except:
                        ckpt_output_dir = os.path.join(config["workdir"], config["run_name"])
                        save_params_ckpt(params, ckpt_output_dir, time_idx)
                        print('Failed to evaluate trajectory.')
            
            # Add frame to keyframe list
            if ((time_idx == 0) or ((time_idx+1) % config['keyframe_every'] == 0) or \
                        (time_idx == num_frames-2)) and (not torch.isinf(curr_gt_w2c[-1]).any()) and (not torch.isnan(curr_gt_w2c[-1]).any()):
                with torch.no_grad():
                    # Get the current estimated rotation & translation
                    curr_cam_rot = F.normalize(params['cam_unnorm_rots'][..., time_idx].detach())
                    curr_cam_tran = params['cam_trans'][..., time_idx].detach()
                    curr_w2c = torch.eye(4).to(device).float()
                    curr_w2c[:3, :3] = build_rotation(curr_cam_rot)
                    curr_w2c[:3, 3] = curr_cam_tran
                    # Initialize Keyframe Info
                    curr_keyframe = {'id': time_idx, 'est_w2c': curr_w2c, 'color': color, 'depth': depth}
                    if load_semantics:
                        curr_keyframe['semantic_id'] = semantic_id
                        curr_keyframe['semantic_color'] = semantic_color
                    # Add to keyframe list
                    keyframe_list.append(curr_keyframe)
                    keyframe_time_indices.append(time_idx)
            
            # Checkpoint every iteration
            if time_idx % config["checkpoint_interval"] == 0 and config['save_checkpoints']:
                ckpt_output_dir = os.path.join(config["workdir"], config["run_name"])
                save_params_ckpt(params, ckpt_output_dir, time_idx)
                np.save(os.path.join(ckpt_output_dir, f"keyframe_time_indices{time_idx}.npy"), np.array(keyframe_time_indices))
            
            # Increment WandB Time Step
            if config['use_wandb']:
                wandb_time_step += 1

            torch.cuda.empty_cache()
            self.pub_sgs_status.publish(Float32(1.0)) #ready to receive data

        if config['save_timestamp_keyframes']:
            # Save keyframes selected at each timestamp
            max_length = max(len(inner) for inner in timestamp_keyframes)
            # Insert -1 for placeholder
            timestamp_keyframes_df = pd.DataFrame([inner + [-1 for _ in range(max_length - len(inner))] \
                                                for inner in timestamp_keyframes])
            timestamp_keyframes_df.to_csv(os.path.join(eval_dir, f"timestamp_keyframes.csv"), \
                                        index=False, header=False, na_rep='-1')

        # Compute Average Runtimes
        if tracking_iter_time_count == 0:
            tracking_iter_time_count = 1
            tracking_frame_time_count = 1
        if mapping_iter_time_count == 0:
            mapping_iter_time_count = 1
            mapping_frame_time_count = 1
        tracking_iter_time_avg = tracking_iter_time_sum / tracking_iter_time_count
        tracking_frame_time_avg = tracking_frame_time_sum / tracking_frame_time_count
        mapping_iter_time_avg = mapping_iter_time_sum / mapping_iter_time_count
        mapping_frame_time_avg = mapping_frame_time_sum / mapping_frame_time_count
        print(f"\nAverage Tracking/Iteration Time: {tracking_iter_time_avg*1000} ms")
        print(f"Average Tracking/Frame Time: {tracking_frame_time_avg} s")
        print(f"Average Mapping/Iteration Time: {mapping_iter_time_avg*1000} ms")
        print(f"Average Mapping/Frame Time: {mapping_frame_time_avg} s")
        if config['use_wandb']:
            wandb_run.log({"Final Stats/Average Tracking Iteration Time (ms)": tracking_iter_time_avg*1000,
                        "Final Stats/Average Tracking Frame Time (s)": tracking_frame_time_avg,
                        "Final Stats/Average Mapping Iteration Time (ms)": mapping_iter_time_avg*1000,
                        "Final Stats/Average Mapping Frame Time (s)": mapping_frame_time_avg,
                        "Final Stats/step": 1})
        
        # Evaluate Final Parameters
        with torch.no_grad():
            if config['use_wandb']:
                eval(dataset, params, num_frames, eval_dir, sil_thres=config['mapping']['sil_thres'],
                    wandb_run=wandb_run, wandb_save_qual=config['wandb']['eval_save_qual'],
                    mapping_iters=config['mapping']['num_iters'], add_new_gaussians=config['mapping']['add_new_gaussians'],
                    device=device, load_semantics=load_semantics, eval_every=config['eval_every'], save_frames=True)
            else:
                eval(dataset, params, num_frames, eval_dir, sil_thres=config['mapping']['sil_thres'],
                    mapping_iters=config['mapping']['num_iters'], add_new_gaussians=config['mapping']['add_new_gaussians'],
                    device=device, load_semantics=load_semantics, eval_every=config['eval_every'], save_frames=True)

        # Add Camera Parameters to Save them
        params['timestep'] = variables['timestep']
        params['intrinsics'] = intrinsics.detach().cpu().numpy()
        params['w2c'] = first_frame_w2c.detach().cpu().numpy()
        params['org_width'] = dataset_config["desired_image_width"]
        params['org_height'] = dataset_config["desired_image_height"]
        params['gt_w2c_all_frames'] = []
        for gt_w2c_tensor in gt_w2c_all_frames:
            params['gt_w2c_all_frames'].append(gt_w2c_tensor.detach().cpu().numpy())
        params['gt_w2c_all_frames'] = np.stack(params['gt_w2c_all_frames'], axis=0)
        params['keyframe_time_indices'] = np.array(keyframe_time_indices)

        if load_semantics:
            params['semantic_ids'] = params['semantic_ids'].type(torch.uint8)
        
        # Save Parameters
        save_params(params, output_dir)

        # Close WandB Run
        if config['use_wandb']:
            wandb.finish()
def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("experiment", type=str, help="Path to experiment file")

    args = parser.parse_args()

    experiment = SourceFileLoader(
        os.path.basename(args.experiment), args.experiment
    ).load_module()

    # Set Experiment Seed
    seed_everything(seed=experiment.config['seed'])
    
    # Create Results Directory and Copy Config
    results_dir = os.path.join(
        experiment.config["workdir"], experiment.config["run_name"]
    )
    if not experiment.config['load_checkpoint']:
        os.makedirs(results_dir, exist_ok=True)
        shutil.copy(args.experiment, os.path.join(results_dir, "config.py"))
    return experiment.config

if __name__ == '__main__':
    try:
        config = main()
        
        node = ActiveSLAM(config)
        rospy.spin()
    except rospy.ROSInterruptException:
        pass

