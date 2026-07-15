import cv2
import os
import torch
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

from datasets.gradslam_datasets.geometryutils import relative_transformation
from utils.recon_helpers import setup_camera
from utils.slam_external import build_rotation,calc_psnr
from utils.slam_helpers import (transform_to_frame, transformed_params2rendervar,
                                transformed_params2depthplussilhouette,
                                transformed_semantics2rendervar)

from diff_gaussian_rasterization import GaussianRasterizer as Renderer
from pytorch_msssim import ms_ssim
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

# loss_fn_alex = LearnedPerceptualImagePatchSimilarity(net_type='alex', normalize=True).cuda()

def align(model, data):
    """Align two trajectories using the method of Horn (closed-form).

    Args:
        model -- first trajectory (3xn)
        data -- second trajectory (3xn)

    Returns:
        rot -- rotation matrix (3x3)
        trans -- translation vector (3x1)
        trans_error -- translational error per point (1xn)

    """
    np.set_printoptions(precision=3, suppress=True)
    model_zerocentered = model - model.mean(1).reshape((3,-1))
    data_zerocentered = data - data.mean(1).reshape((3,-1))

    W = np.zeros((3, 3))
    for column in range(model.shape[1]):
        W += np.outer(model_zerocentered[:,
                         column], data_zerocentered[:, column])
    U, d, Vh = np.linalg.linalg.svd(W.transpose())
    S = np.matrix(np.identity(3))
    if (np.linalg.det(U) * np.linalg.det(Vh) < 0):
        S[2, 2] = -1
    rot = U*S*Vh
    trans = data.mean(1).reshape((3,-1)) - rot * model.mean(1).reshape((3,-1))

    model_aligned = rot * model + trans
    alignment_error = model_aligned - data

    trans_error = np.sqrt(np.sum(np.multiply(
        alignment_error, alignment_error), 0)).A[0]

    return rot, trans, trans_error


def recolor_semantic_img(rendered_seg, gt_seg, color_map=None):
    """Adjust the semantic color by assigning to the closest color refer to
       the ground truth semantic image or color dict.
    """
    rendered_seg = rendered_seg.permute(1, 2, 0) # (3, H, W) -> (H, W, 3)
    gt_seg = gt_seg.permute(1, 2, 0)
    img_shape = gt_seg.shape
    rendered_seg = rendered_seg.reshape(-1, 1, 3).type(torch.float32) # (H*W, 1, 3)

    if color_map is None:
        gt_seg = gt_seg.reshape(-1, 3)
        # Find unique colors
        color_map, _ = torch.unique(gt_seg, dim=0, return_inverse=True)
    refer_color = color_map.reshape(1, -1, 3).type(torch.float32).to(gt_seg.device) # (1, H*W, 3)

    # l1_distances = torch.sum(torch.abs(rendered_seg - refer_color), axis=2)
    l1_distances = torch.sqrt(torch.sum((rendered_seg - refer_color) ** 2, axis=2))
    # Find the index of the minimum distance for each pixel
    closest_indices = torch.argmin(l1_distances, axis=1)
    del l1_distances

    # Assign the closest color to the rendered semantic image
    rendered_seg[:, 0, :] = refer_color.squeeze(0)[closest_indices]
    rendered_seg = rendered_seg.reshape(img_shape) # (H*W, 1, 3) -> (H, W, 3)
    rendered_seg = rendered_seg.permute(2, 0, 1) # (H, W, 3) -> (3, H, W)
    
    return rendered_seg


def evaluate_label_miou(pred_label, gt_label):
    """
    Input : 
        pred_label: torch tensor of the predicted semantic label, shape (1, H, W)
        gt_label: torch tensor of the semantic label, shape (1, H, W)
    """
    gt_flat = gt_label.view(-1)
    pred_flat = pred_label.view(-1)

    unique_labels = torch.unique(gt_flat)
    iou_per_label = []

    for label in unique_labels:
        # Skip unlabeled class if necessary (e.g., label == 0)
        if label == 0:
            continue

        gt_label = (gt_flat == label)
        pred_label = (pred_flat == label)
        intersection = torch.logical_and(gt_label, pred_label).sum().item()
        union = torch.logical_or(gt_label, pred_label).sum().item()

        if union == 0:
            continue

        iou = intersection / union
        iou_per_label.append(iou)

    # Mean IoU
    miou = sum(iou_per_label) / len(iou_per_label) if iou_per_label else 0
    return miou


def evaluate_miou(recolored_img, gt_img, valid_mask=None):
    """
    Input :
        recolored_img: torch tensor of the colored semantic image, shape (C, H, W)
        gt_img: torch tensor of the colored semantic image, shape (C, H, W)
        valid_mask: optional (H, W) boolean tensor. When given, this decides
            which pixels are scored instead of inferring it from color, and
            [0, 0, 0] is treated as a real class (e.g. "background") rather
            than "unlabeled". Needed whenever [0, 0, 0] doubles as both a
            legitimate class and the "excluded" fill value -- inferring
            validity from color alone would then also drop real instances
            of that class, and could never register a false-positive
            prediction of another class landing on top of it.
            When omitted, falls back to the original color-based behavior.
    """
    gt_flat = gt_img.permute(1, 2, 0).view(-1, 3)
    pred_flat = recolored_img.permute(1, 2, 0).view(-1, 3)

    if valid_mask is not None:
        labeled_pixels = valid_mask.view(-1).bool()
    else:
        # Filter out [0, 0, 0] (unlabeled) pixels
        labeled_pixels = (gt_flat != torch.tensor([0, 0, 0], dtype=torch.uint8).cuda()).any(dim=1)
    gt_flat = gt_flat[labeled_pixels]
    pred_flat = pred_flat[labeled_pixels]

    unique_colors = torch.unique(gt_flat, dim=0)
    iou_per_color = []

    for color in unique_colors:
        # Skip the unlabeled color. Only applies when valid_mask wasn't
        # given, since with an explicit valid_mask, [0, 0, 0] is a real
        # class and should be scored like any other.
        if valid_mask is None and torch.equal(color, torch.tensor([0, 0, 0], dtype=torch.uint8).cuda()):
            continue

        gt_matches = torch.all(gt_flat == color, dim=1)
        pred_matches = torch.all(pred_flat == color, dim=1)

        # Calculate intersection and union
        intersection = torch.logical_and(gt_matches, pred_matches).sum().item()
        union = torch.logical_or(gt_matches, pred_matches).sum().item()

        if union == 0:
            continue

        iou = intersection / union
        iou_per_color.append(iou)

    # Calculate mean IoU
    miou = sum(iou_per_color) / len(iou_per_color) if iou_per_color else 0
    return miou


def evaluate_ate(gt_traj, est_traj):
    """
    Input : 
        gt_traj: list of 4x4 matrices 
        est_traj: list of 4x4 matrices
        len(gt_traj) == len(est_traj)
    """
    gt_traj_pts = [gt_traj[idx][:3,3] for idx in range(len(gt_traj))]
    est_traj_pts = [est_traj[idx][:3,3] for idx in range(len(est_traj))]

    gt_traj_pts  = torch.stack(gt_traj_pts).detach().cpu().numpy().T
    est_traj_pts = torch.stack(est_traj_pts).detach().cpu().numpy().T

    _, _, trans_error = align(gt_traj_pts, est_traj_pts)

    avg_trans_error = trans_error.mean()

    return avg_trans_error


def report_loss(losses, wandb_run, wandb_step, tracking=False, mapping=False, load_semantics=False):
    # Update loss dict
    loss_dict = {'Loss': losses['loss'].item(),
                 'Image Loss': losses['im'].item(),
                 'Depth Loss': losses['depth'].item(),}
    if load_semantics:
        loss_dict['Semantic Loss'] = losses['seg'].item()
    
    if tracking:
        tracking_loss_dict = {}
        for k, v in loss_dict.items():
            tracking_loss_dict[f"Per Iteration Tracking/{k}"] = v
        tracking_loss_dict['Per Iteration Tracking/step'] = wandb_step
        wandb_run.log(tracking_loss_dict)
    elif mapping:
        mapping_loss_dict = {}
        for k, v in loss_dict.items():
            mapping_loss_dict[f"Per Iteration Mapping/{k}"] = v
        mapping_loss_dict['Per Iteration Mapping/step'] = wandb_step
        wandb_run.log(mapping_loss_dict)
    else:
        frame_opt_loss_dict = {}
        for k, v in loss_dict.items():
            frame_opt_loss_dict[f"Per Iteration Current Frame Optimization/{k}"] = v
        frame_opt_loss_dict['Per Iteration Current Frame Optimization/step'] = wandb_step
        wandb_run.log(frame_opt_loss_dict)
    
    # Increment wandb step
    wandb_step += 1
    return wandb_step
        

def plot_rgbd_silhouette(color, depth, rastered_color, rastered_depth, presence_sil_mask, diff_depth_l1,
                         psnr, depth_l1, fig_title, plot_dir=None, plot_name=None, save_plot=False, seg=None,
                         rastered_seg=None, wandb_run=None, wandb_step=None, wandb_title=None, diff_rgb=None):
    # Determine Plot Aspect Ratio
    aspect_ratio = color.shape[2] / color.shape[1]
    fig_height = 8
    fig_width = 14/1.55
    # Adjust number of subplots and figure size based on 'seg' variable
    num_cols = 4 if seg is not None else 3
    # Scale width for additional column if seg is not None
    fig_width = fig_width * aspect_ratio * num_cols / 3
    # Plot the Ground Truth and Rasterized RGB & Depth,
    # along with Diff Depth & Silhouette, and semantic image
    fig, axs = plt.subplots(2, num_cols, figsize=(fig_width, fig_height))
    axs[0, 0].imshow(color.cpu().permute(1, 2, 0))
    axs[0, 0].set_title("Ground Truth RGB")
    axs[0, 1].imshow(depth[0, :, :].cpu(), cmap='jet', vmin=0, vmax=6)
    axs[0, 1].set_title("Ground Truth Depth")
    rastered_color = torch.clamp(rastered_color, 0, 1)
    axs[1, 0].imshow(rastered_color.cpu().permute(1, 2, 0))
    axs[1, 0].set_title("Rasterized RGB, PSNR: {:.2f}".format(psnr))
    axs[1, 1].imshow(rastered_depth[0, :, :].cpu(), cmap='jet', vmin=0, vmax=6)
    axs[1, 1].set_title("Rasterized Depth, L1: {:.2f}".format(depth_l1))
    if diff_rgb is not None:
        axs[0, 2].imshow(diff_rgb.cpu(), cmap='jet', vmin=0, vmax=6)
        axs[0, 2].set_title("Diff RGB L1")
    else:
        axs[0, 2].imshow(presence_sil_mask, cmap='gray')
        axs[0, 2].set_title("Rasterized Silhouette")
    diff_depth_l1 = diff_depth_l1.cpu().squeeze(0)
    axs[1, 2].imshow(diff_depth_l1, cmap='jet', vmin=0, vmax=6)
    axs[1, 2].set_title("Diff Depth L1")
    
    if seg is not None:
        rastered_seg = recolor_semantic_img(rastered_seg, seg)
        miou = evaluate_miou(rastered_seg, seg)
        axs[0, 3].imshow(seg.cpu().permute(1, 2, 0))
        axs[0, 3].set_title("Ground Truth Semantic Map")
        axs[1, 3].imshow(rastered_seg.cpu().permute(1, 2, 0))
        axs[1, 3].set_title("Rasterized Semantic Map, IOU: {:.4f}".format(miou))
        
    for ax in axs.flatten():
        ax.axis('off')
    fig.suptitle(fig_title, y=0.95, fontsize=16)
    fig.tight_layout()
    if save_plot:
        save_path = os.path.join(plot_dir, f"{plot_name}.png")
        plt.savefig(save_path, bbox_inches='tight')
    if wandb_run is not None:
        if wandb_step is None:
            wandb_run.log({wandb_title: fig})
        else:
            wandb_run.log({wandb_title: fig}, step=wandb_step)
    plt.close()


def report_progress(params, data, i, progress_bar, iter_time_idx, sil_thres, every_i=1, qual_every_i=1, 
                    tracking=False, mapping=False, device="cuda", load_semantics=False, wandb_run=None,
                    wandb_step=None, wandb_save_qual=False, online_time_idx=None, global_logging=True):
    if i % every_i == 0 or i == 1:
        if wandb_run is not None:
            if tracking:
                stage = "Tracking"
            elif mapping:
                stage = "Mapping"
            else:
                stage = "Current Frame Optimization"
        if not global_logging:
            stage = "Per Iteration " + stage

        if tracking:
            # Get list of gt poses
            gt_w2c_list = data['iter_gt_w2c_list']
            valid_gt_w2c_list = []
            
            # Get latest trajectory
            latest_est_w2c = data['w2c']
            latest_est_w2c_list = []
            latest_est_w2c_list.append(latest_est_w2c)
            valid_gt_w2c_list.append(gt_w2c_list[0])
            for idx in range(1, iter_time_idx+1):
                # Check if gt pose is not nan for this time step
                if torch.isnan(gt_w2c_list[idx]).sum() > 0:
                    continue
                interm_cam_rot = F.normalize(params['cam_unnorm_rots'][..., idx].detach())
                interm_cam_trans = params['cam_trans'][..., idx].detach()
                intermrel_w2c = torch.eye(4).to(device).float()
                intermrel_w2c[:3, :3] = build_rotation(interm_cam_rot)
                intermrel_w2c[:3, 3] = interm_cam_trans
                latest_est_w2c = intermrel_w2c
                latest_est_w2c_list.append(latest_est_w2c)
                valid_gt_w2c_list.append(gt_w2c_list[idx])

            # Get latest gt pose
            gt_w2c_list = valid_gt_w2c_list
            iter_gt_w2c = gt_w2c_list[-1]
            # Get euclidean distance error between latest and gt pose
            iter_pt_error = torch.sqrt((latest_est_w2c[0,3] - iter_gt_w2c[0,3])**2 + (latest_est_w2c[1,3] - iter_gt_w2c[1,3])**2 + (latest_est_w2c[2,3] - iter_gt_w2c[2,3])**2)
            if iter_time_idx > 0:
                # Calculate relative pose error
                rel_gt_w2c = relative_transformation(gt_w2c_list[-2], gt_w2c_list[-1])
                rel_est_w2c = relative_transformation(latest_est_w2c_list[-2], latest_est_w2c_list[-1])
                rel_pt_error = torch.sqrt((rel_gt_w2c[0,3] - rel_est_w2c[0,3])**2 + (rel_gt_w2c[1,3] - rel_est_w2c[1,3])**2 + (rel_gt_w2c[2,3] - rel_est_w2c[2,3])**2)
            else:
                rel_pt_error = torch.zeros(1).float()
            
            # Calculate ATE RMSE
            ate_rmse = evaluate_ate(gt_w2c_list, latest_est_w2c_list)
            ate_rmse = np.round(ate_rmse, decimals=6)
            if wandb_run is not None:
                tracking_log = {f"{stage}/Latest Pose Error":iter_pt_error, 
                               f"{stage}/Latest Relative Pose Error":rel_pt_error,
                               f"{stage}/ATE RMSE":ate_rmse}

        # Get current frame Gaussians
        transformed_pts = transform_to_frame(params, iter_time_idx, 
                                             gaussians_grad=False,
                                             camera_grad=False,
                                             device=device)

        # Initialize Render Variables
        rendervar = transformed_params2rendervar(params, transformed_pts, device=device)
        depth_sil_rendervar = transformed_params2depthplussilhouette(params, data['w2c'], 
                                                                     transformed_pts, device=device)
        depth_sil, _, _, = Renderer(raster_settings=data['cam'])(**depth_sil_rendervar)
        rastered_depth = depth_sil[0, :, :].unsqueeze(0)
        valid_depth_mask = (data['depth'] > 0)
        silhouette = depth_sil[1, :, :]
        presence_sil_mask = (silhouette > sil_thres)

        im, _, _, = Renderer(raster_settings=data['cam'])(**rendervar)

        if load_semantics:
            semantic_rendervar = transformed_semantics2rendervar(params, transformed_pts, device=device)
            rastered_seg, _, _, = Renderer(raster_settings=data['cam'])(**semantic_rendervar)
            gt_seg = data['semantic_color']
            # seg_psnr = calc_psnr(seg, data['semantic_color']).mean()
            rastered_seg = recolor_semantic_img(rastered_seg, gt_seg)
            miou = evaluate_miou(rastered_seg, gt_seg)
        else:
            rastered_seg = None
            gt_seg = None
            miou = 0

        if tracking:
            psnr = calc_psnr(im * presence_sil_mask, data['im'] * presence_sil_mask).mean()
        else:
            psnr = calc_psnr(im, data['im']).mean()

        if tracking:
            diff_depth_rmse = torch.sqrt((((rastered_depth - data['depth']) * presence_sil_mask) ** 2))
            diff_depth_rmse = diff_depth_rmse * valid_depth_mask
            rmse = diff_depth_rmse.sum() / valid_depth_mask.sum()
            diff_depth_l1 = torch.abs((rastered_depth - data['depth']) * presence_sil_mask)
            diff_depth_l1 = diff_depth_l1 * valid_depth_mask
            depth_l1 = diff_depth_l1.sum() / valid_depth_mask.sum()
        else:
            diff_depth_rmse = torch.sqrt((((rastered_depth - data['depth'])) ** 2))
            diff_depth_rmse = diff_depth_rmse * valid_depth_mask
            rmse = diff_depth_rmse.sum() / valid_depth_mask.sum()
            diff_depth_l1 = torch.abs((rastered_depth - data['depth']))
            diff_depth_l1 = diff_depth_l1 * valid_depth_mask
            depth_l1 = diff_depth_l1.sum() / valid_depth_mask.sum()

        if not (tracking or mapping):
            progress_bar.set_postfix({f"Time-Step: {iter_time_idx} | PSNR: {psnr:.{7}} | Depth RMSE: {rmse:.{7}} | mIoU: {miou:.{7}} | L1": f"{depth_l1:.{7}}"})
            progress_bar.update(every_i)
        elif tracking:
            progress_bar.set_postfix({f"Time-Step: {iter_time_idx} | Rel Pose Error: {rel_pt_error.item():.{7}} | Pose Error: {iter_pt_error.item():.{7}} | ATE RMSE": f"{ate_rmse.item():.{7}}"})
            progress_bar.update(every_i)
        elif mapping:
            progress_bar.set_postfix({f"Time-Step: {online_time_idx} | Frame {data['id']} | PSNR: {psnr:.{7}} | Depth RMSE: {rmse:.{7}} | mIoU: {miou:.{7}} | L1": f"{depth_l1:.{7}}"})
            progress_bar.update(every_i)
        
        if wandb_run is not None:
            wandb_log = {f"{stage}/PSNR": psnr,
                         f"{stage}/Depth RMSE": rmse,
                         f"{stage}/Depth L1": depth_l1,
                         f"{stage}/mIoU": miou,
                         f"{stage}/step": wandb_step}
            if tracking:
                wandb_log = {**wandb_log, **tracking_log}
            wandb_run.log(wandb_log)
        
        if wandb_save_qual and (i % qual_every_i == 0 or i == 1):
            # Silhouette Mask
            presence_sil_mask = presence_sil_mask.detach().cpu().numpy()

            # Log plot to wandb
            if not mapping:
                fig_title = f"Time-Step: {iter_time_idx} | Iter: {i} | Frame: {data['id']}"
            else:
                fig_title = f"Time-Step: {online_time_idx} | Iter: {i} | Frame: {data['id']}"
            plot_rgbd_silhouette(data['im'], data['depth'], im, rastered_depth, presence_sil_mask, diff_depth_l1,
                                 psnr, depth_l1, fig_title, seg=gt_seg, rastered_seg=rastered_seg, wandb_run=wandb_run,
                                 wandb_step=wandb_step, wandb_title=f"{stage} Qual Viz")


def eval_online(dataset, all_params, num_frames, eval_online_dir, sil_thres, mapping_iters,
                add_new_gaussians, device="cuda", load_semantics=False, wandb_run=None,
                wandb_save_qual=False, eval_every=1):
    print("Evaluating Online Final Parameters...")
    psnr_list = []
    rmse_list = []
    l1_list = []
    miou_list = []
    plot_dir = os.path.join(eval_online_dir, "plots")
    os.makedirs(plot_dir, exist_ok=True)

    for time_idx in tqdm(range(num_frames)):
        if time_idx != 0 and (time_idx+1) % eval_every != 0:
            continue
        # Get Params for current frame
        params = all_params[time_idx]

        if load_semantics:
            color, depth, intrinsics, pose, semantic_id, semantic_color = dataset[time_idx]
            semantic_id = semantic_id.permute(2, 0, 1) # (H, W, 1) -> (1, H, W)
            semantic_color = semantic_color.permute(2, 0, 1) / 255 # (H, W, C) -> (C, H, W)
        else:
            color, depth, intrinsics, pose = dataset[time_idx]

        # Get Camera Parameters
        intrinsics = intrinsics[:3, :3]

        # Process RGB-D Data
        color = color.permute(2, 0, 1) / 255 # (H, W, C) -> (C, H, W)
        depth = depth.permute(2, 0, 1) # (H, W, C) -> (C, H, W)

        if time_idx == 0:
            # Process Camera Parameters
            first_frame_w2c = torch.linalg.inv(pose)
            # Setup Camera
            cam = setup_camera(color.shape[2], color.shape[1], intrinsics.cpu().numpy(),
                               first_frame_w2c.detach().cpu().numpy(), device=device)
        
        # Define current frame data
        curr_data = {'cam': cam, 'im': color, 'depth': depth, 'id': time_idx, 'intrinsics': intrinsics, 'w2c': first_frame_w2c}

        if load_semantics:
            curr_data['semantic_id'] = semantic_id
            curr_data['semantic_color'] = semantic_color

        # Get current frame Gaussians
        transformed_pts = transform_to_frame(params, time_idx, 
                                             gaussians_grad=False,
                                             camera_grad=False,
                                             device=device)

        # Initialize Render Variables
        rendervar = transformed_params2rendervar(params, transformed_pts, device=device)
        depth_sil_rendervar = transformed_params2depthplussilhouette(params, first_frame_w2c,
                                                                     transformed_pts, device=device)
        
        # Render Depth & Silhouette
        depth_sil, _, _, = Renderer(raster_settings=curr_data['cam'])(**depth_sil_rendervar)
        rastered_depth = depth_sil[0, :, :].unsqueeze(0)
        valid_depth_mask = (curr_data['depth'] > 0)
        silhouette = depth_sil[1, :, :]
        presence_sil_mask = (silhouette > sil_thres)
        
        # Render RGB and Calculate PSNR
        im, radius, _, = Renderer(raster_settings=curr_data['cam'])(**rendervar)
        if mapping_iters==0 and not add_new_gaussians:
            psnr = calc_psnr(im * presence_sil_mask, curr_data['im'] * presence_sil_mask).mean()
        else:
            psnr = calc_psnr(im, curr_data['im']).mean()
        psnr_list.append(psnr.cpu().numpy())

        # Compute Depth RMSE
        if mapping_iters==0 and not add_new_gaussians:
            diff_depth_rmse = torch.sqrt((((rastered_depth - curr_data['depth']) * presence_sil_mask) ** 2))
            diff_depth_rmse = diff_depth_rmse * valid_depth_mask
            rmse = diff_depth_rmse.sum() / valid_depth_mask.sum()
            diff_depth_l1 = torch.abs((rastered_depth - curr_data['depth']) * presence_sil_mask)
            diff_depth_l1 = diff_depth_l1 * valid_depth_mask
            depth_l1 = diff_depth_l1.sum() / valid_depth_mask.sum()
        else:
            diff_depth_rmse = torch.sqrt((((rastered_depth - curr_data['depth'])) ** 2))
            diff_depth_rmse = diff_depth_rmse * valid_depth_mask
            rmse = diff_depth_rmse.sum() / valid_depth_mask.sum()
            diff_depth_l1 = torch.abs((rastered_depth - curr_data['depth']))
            diff_depth_l1 = diff_depth_l1 * valid_depth_mask
            depth_l1 = diff_depth_l1.sum() / valid_depth_mask.sum()
        rmse_list.append(rmse.cpu().numpy())
        l1_list.append(depth_l1.cpu().numpy())

        if load_semantics:
            # Render semantic color map
            semantic_rendervar = transformed_semantics2rendervar(params, transformed_pts, device=device)
            rastered_seg, _, _, = Renderer(raster_settings=curr_data['cam'])(**semantic_rendervar)
            gt_seg = curr_data['semantic_color']

            # Calcualte mIoU scores
            rastered_seg = recolor_semantic_img(rastered_seg, gt_seg)
            miou = evaluate_miou(rastered_seg, gt_seg)
            miou_list.append(miou)
        else:
            rastered_seg = None
            gt_seg = None

        # Plot the Ground Truth and Rasterized RGB & Depth, along with Silhouette
        fig_title = "Time Step: {}".format(time_idx)
        plot_name = "%04d" % time_idx
        presence_sil_mask = presence_sil_mask.detach().cpu().numpy()
        if wandb_run is None:
            plot_rgbd_silhouette(color, depth, im, rastered_depth, presence_sil_mask, diff_depth_l1,
                                 psnr, depth_l1, fig_title, plot_dir, plot_name=plot_name, save_plot=True,
                                 seg=gt_seg, rastered_seg=rastered_seg)
        elif wandb_save_qual:
            plot_rgbd_silhouette(color, depth, im, rastered_depth, presence_sil_mask, diff_depth_l1,
                                 psnr, depth_l1, fig_title, plot_dir, plot_name=plot_name, save_plot=True,
                                 seg=gt_seg, rastered_seg=rastered_seg, wandb_run=wandb_run, wandb_step=None, 
                                 wandb_title="Online Eval/Qual Viz")
    
    # Compute Average Metrics
    psnr_list = np.array(psnr_list)
    rmse_list = np.array(rmse_list)
    l1_list = np.array(l1_list)
    miou_list = np.array(miou_list)
    avg_psnr = psnr_list.mean()
    avg_rmse = rmse_list.mean()
    avg_l1 = l1_list.mean()
    avg_miou = miou_list.mean() if miou_list.size > 0 else 0
    print("Online Average PSNR: {:.2f}".format(avg_psnr))
    print("Online Average Depth RMSE: {:.2f}".format(avg_rmse))
    print("Online Average Depth L1: {:.2f}".format(avg_l1))
    print("Average mIoU: {:.4f}".format(avg_miou))

    if wandb_run is not None:
        wandb_run.log({"Final Stats/Online Average PSNR": avg_psnr, 
                       "Final Stats/Online Average Depth RMSE": avg_rmse,
                       "Final Stats/Online Average Depth L1": avg_l1,
                       "Final Stats/step": 1,
                       "Final Stats/Average mIoU": avg_miou})

    # Save metric lists as text files
    np.savetxt(os.path.join(eval_online_dir, "online_psnr.txt"), psnr_list)
    np.savetxt(os.path.join(eval_online_dir, "online_rmse.txt"), rmse_list)
    np.savetxt(os.path.join(eval_online_dir, "online_l1.txt"), l1_list)

    if load_semantics:
        np.savetxt(os.path.join(eval_dir, "miou.txt"), miou_list)

        fig, axs = plt.subplots(1, 3, figsize=(18, 4))
        axs[2].plot(np.arange(len(miou_list)), miou_list)
        axs[2].set_title("mIoU")
        axs[2].set_xlabel("Time Step")
        axs[2].set_ylabel("mIoU")
    else:
        fig, axs = plt.subplots(1, 2, figsize=(12, 4))

    # Plot PSNR & L1 as line plots
    axs[0].plot(np.arange(len(psnr_list)), psnr_list)
    axs[0].set_title("RGB PSNR")
    axs[0].set_xlabel("Time Step")
    axs[0].set_ylabel("PSNR")
    axs[1].plot(np.arange(len(l1_list)), l1_list)
    axs[1].set_title("Depth L1")
    axs[1].set_xlabel("Time Step")
    axs[1].set_ylabel("L1")
    fig.suptitle("Average PSNR: {:.2f}, Average Depth L1: {:.2f}, Average mIoU: {:.4f}".format(avg_psnr, avg_l1, avg_miou),
                 y=1.05, fontsize=16)
    plt.savefig(os.path.join(eval_online_dir, "online_metrics.png"), bbox_inches='tight')
    if wandb_run is not None:
        wandb_run.log({"Online Eval/Metrics": fig})
    plt.close()


def eval(dataset, final_params, num_frames, eval_dir, sil_thres, mapping_iters,
         add_new_gaussians, device="cuda", load_semantics=False, wandb_run=None,
         wandb_save_qual=False, eval_every=1, save_frames=False):
    print("Evaluating Final Parameters ...")
    psnr_list = []
    rmse_list = []
    l1_list = []
    lpips_list = []
    ssim_list = []
    miou_list = []
    plot_dir = os.path.join(eval_dir, "plots")
    os.makedirs(plot_dir, exist_ok=True)
    if save_frames:
        render_rgb_dir = os.path.join(eval_dir, "rendered_rgb")
        os.makedirs(render_rgb_dir, exist_ok=True)
        render_depth_dir = os.path.join(eval_dir, "rendered_depth")
        os.makedirs(render_depth_dir, exist_ok=True)
        # rgb_dir = os.path.join(eval_dir, "rgb")
        # os.makedirs(rgb_dir, exist_ok=True)
        # depth_dir = os.path.join(eval_dir, "depth")
        # os.makedirs(depth_dir, exist_ok=True)
        
        if load_semantics:
            render_seg_dir = os.path.join(eval_dir, "rendered_seg")
            os.makedirs(render_seg_dir, exist_ok=True)

    gt_w2c_list = []
    for time_idx in tqdm(range(num_frames)):
         # Get RGB-D Data & Camera Parameters
        if load_semantics:
            color, depth, intrinsics, pose, semantic_id, semantic_color = dataset[time_idx]
            semantic_id = semantic_id.permute(2, 0, 1) # (H, W, 1) -> (1, H, W)
            semantic_color = semantic_color.permute(2, 0, 1) / 255 # (H, W, C) -> (C, H, W)
        else:
            color, depth, intrinsics, pose = dataset[time_idx]
        gt_w2c = torch.linalg.inv(pose)
        gt_w2c_list.append(gt_w2c)
        intrinsics = intrinsics[:3, :3]

        # Process RGB-D Data
        color = color.permute(2, 0, 1) / 255 # (H, W, C) -> (C, H, W)
        depth = depth.permute(2, 0, 1) # (H, W, C) -> (C, H, W)

        if time_idx == 0:
            # Process Camera Parameters
            first_frame_w2c = torch.linalg.inv(pose)
            # Setup Camera
            cam = setup_camera(color.shape[2], color.shape[1], intrinsics.cpu().numpy(),
                               first_frame_w2c.detach().cpu().numpy(), device=device)
        
        # Skip frames if not eval_every
        if time_idx != 0 and (time_idx+1) % eval_every != 0:
            continue

        # Get current frame Gaussians
        transformed_pts = transform_to_frame(final_params, time_idx, 
                                             gaussians_grad=False,
                                             camera_grad=False,
                                             device=device)
 
        # Define current frame data
        curr_data = {'cam': cam, 'im': color, 'depth': depth, 'id': time_idx, 'intrinsics': intrinsics, 'w2c': first_frame_w2c}

        if load_semantics:
            curr_data['semantic_id'] = semantic_id
            curr_data['semantic_color'] = semantic_color

        # Initialize Render Variables
        rendervar = transformed_params2rendervar(final_params, transformed_pts, device=device)
        depth_sil_rendervar = transformed_params2depthplussilhouette(final_params, curr_data['w2c'],
                                                                     transformed_pts, device=device)

        # Render Depth & Silhouette
        depth_sil, _, _, = Renderer(raster_settings=curr_data['cam'])(**depth_sil_rendervar)
        rastered_depth = depth_sil[0, :, :].unsqueeze(0)
        # Mask invalid depth in GT
        valid_depth_mask = (curr_data['depth'] > 0)
        rastered_depth_viz = rastered_depth.detach()
        rastered_depth = rastered_depth * valid_depth_mask
        silhouette = depth_sil[1, :, :]
        presence_sil_mask = (silhouette > sil_thres)
        
        # Render RGB and Calculate PSNR
        im, radius, _, = Renderer(raster_settings=curr_data['cam'])(**rendervar)
        if mapping_iters==0 and not add_new_gaussians:
            weighted_im = im * presence_sil_mask * valid_depth_mask
            weighted_gt_im = curr_data['im'] * presence_sil_mask * valid_depth_mask
        else:
            weighted_im = im * valid_depth_mask
            weighted_gt_im = curr_data['im'] * valid_depth_mask
        psnr = calc_psnr(weighted_im, weighted_gt_im).mean()
        ssim = ms_ssim(weighted_im.unsqueeze(0).cpu(), weighted_gt_im.unsqueeze(0).cpu(),
                       data_range=1.0, size_average=True)
        loss_fn_alex.to(device)
        lpips_score = loss_fn_alex(torch.clamp(weighted_im.unsqueeze(0), 0.0, 1.0),
                                    torch.clamp(weighted_gt_im.unsqueeze(0), 0.0, 1.0)).item()

        psnr_list.append(psnr.cpu().numpy())
        ssim_list.append(ssim.cpu().numpy())
        lpips_list.append(lpips_score)

        # Compute Depth RMSE
        if mapping_iters==0 and not add_new_gaussians:
            diff_depth_rmse = torch.sqrt((((rastered_depth - curr_data['depth']) * presence_sil_mask) ** 2))
            diff_depth_rmse = diff_depth_rmse * valid_depth_mask
            rmse = diff_depth_rmse.sum() / valid_depth_mask.sum()
            diff_depth_l1 = torch.abs((rastered_depth - curr_data['depth']) * presence_sil_mask)
            diff_depth_l1 = diff_depth_l1 * valid_depth_mask
            depth_l1 = diff_depth_l1.sum() / valid_depth_mask.sum()
        else:
            diff_depth_rmse = torch.sqrt((((rastered_depth - curr_data['depth'])) ** 2))
            diff_depth_rmse = diff_depth_rmse * valid_depth_mask
            rmse = diff_depth_rmse.sum() / valid_depth_mask.sum()
            diff_depth_l1 = torch.abs((rastered_depth - curr_data['depth']))
            diff_depth_l1 = diff_depth_l1 * valid_depth_mask
            depth_l1 = diff_depth_l1.sum() / valid_depth_mask.sum()
        rmse_list.append(rmse.cpu().numpy())
        l1_list.append(depth_l1.cpu().numpy())

        if load_semantics:
            # Render semantic color map
            semantic_rendervar = transformed_semantics2rendervar(final_params, transformed_pts, device=device)
            rastered_seg, _, _, = Renderer(raster_settings=curr_data['cam'])(**semantic_rendervar)
            gt_seg = curr_data['semantic_color']

            # Calcualte mIoU scores
            rastered_seg = recolor_semantic_img(rastered_seg, gt_seg)
            miou = evaluate_miou(rastered_seg, gt_seg)
            miou_list.append(miou)
        else:
            rastered_seg = None
            gt_seg = None

        if save_frames:
            # Save Rendered RGB and Depth
            viz_render_im = torch.clamp(im, 0, 1)
            viz_render_im = viz_render_im.detach().cpu().permute(1, 2, 0).numpy()
            vmin = 0
            vmax = 6
            viz_render_depth = rastered_depth_viz[0].detach().cpu().numpy()
            normalized_depth = np.clip((viz_render_depth - vmin) / (vmax - vmin), 0, 1)
            depth_colormap = cv2.applyColorMap((normalized_depth * 255).astype(np.uint8), cv2.COLORMAP_JET)
            cv2.imwrite(os.path.join(render_rgb_dir, "gs_{:04d}.png".format(time_idx)), cv2.cvtColor(viz_render_im*255, cv2.COLOR_RGB2BGR))
            cv2.imwrite(os.path.join(render_depth_dir, "gs_{:04d}.png".format(time_idx)), depth_colormap)

            if load_semantics:
                viz_render_seg = torch.clamp(rastered_seg, 0, 1)
                viz_render_seg = viz_render_seg.detach().cpu().permute(1, 2, 0).numpy()
                cv2.imwrite(os.path.join(render_seg_dir, "gs_{:04d}.png".format(time_idx)), cv2.cvtColor(viz_render_seg*255, cv2.COLOR_RGB2BGR))
        
        # Plot the Ground Truth and Rasterized RGB & Depth, along with Silhouette
        fig_title = "Time Step: {}".format(time_idx)
        plot_name = "%04d" % time_idx
        presence_sil_mask = presence_sil_mask.detach().cpu().numpy()
        if wandb_run is None:
            plot_rgbd_silhouette(color, depth, im, rastered_depth_viz, presence_sil_mask, diff_depth_l1,
                                 psnr, depth_l1, fig_title, plot_dir, plot_name=plot_name, save_plot=True,
                                 seg=gt_seg, rastered_seg=rastered_seg)
        elif wandb_save_qual:
            plot_rgbd_silhouette(color, depth, im, rastered_depth_viz, presence_sil_mask, diff_depth_l1,
                                 psnr, depth_l1, fig_title, plot_dir, plot_name=plot_name, save_plot=True,
                                 seg=gt_seg, rastered_seg=rastered_seg, wandb_run=wandb_run, wandb_step=None, 
                                 wandb_title="Eval/Qual Viz")

    try:
        # Compute the final ATE RMSE
        # Get the final camera trajectory
        num_frames = final_params['cam_unnorm_rots'].shape[-1]
        latest_est_w2c = first_frame_w2c
        latest_est_w2c_list = []
        latest_est_w2c_list.append(latest_est_w2c)
        valid_gt_w2c_list = []
        valid_gt_w2c_list.append(gt_w2c_list[0])
        for idx in range(1, num_frames):
            # Check if gt pose is not nan for this time step
            if torch.isnan(gt_w2c_list[idx]).sum() > 0:
                continue
            interm_cam_rot = F.normalize(final_params['cam_unnorm_rots'][..., idx].detach())
            interm_cam_trans = final_params['cam_trans'][..., idx].detach()
            intermrel_w2c = torch.eye(4).to(device).float()
            intermrel_w2c[:3, :3] = build_rotation(interm_cam_rot)
            intermrel_w2c[:3, 3] = interm_cam_trans
            latest_est_w2c = intermrel_w2c
            latest_est_w2c_list.append(latest_est_w2c)
            valid_gt_w2c_list.append(gt_w2c_list[idx])
        gt_w2c_list = valid_gt_w2c_list
        # Calculate ATE RMSE
        ate_rmse = evaluate_ate(gt_w2c_list, latest_est_w2c_list)
        print("Final Average ATE RMSE: {:.2f} cm".format(ate_rmse*100))
        if wandb_run is not None:
            wandb_run.log({"Final Stats/Avg ATE RMSE": ate_rmse,
                        "Final Stats/step": 1})
    except:
        ate_rmse = 100.0
        print('Failed to evaluate trajectory with alignment.')
    
    # Compute Average Metrics
    psnr_list = np.array(psnr_list)
    rmse_list = np.array(rmse_list)
    l1_list = np.array(l1_list)
    ssim_list = np.array(ssim_list)
    lpips_list = np.array(lpips_list)
    miou_list = np.array(miou_list)

    avg_psnr = psnr_list.mean()
    avg_rmse = rmse_list.mean()
    avg_l1 = l1_list.mean()
    avg_ssim = ssim_list.mean()
    avg_lpips = lpips_list.mean()
    avg_miou = miou_list.mean() if miou_list.size > 0 else 0
    avg_miou_stride10 = miou_list[::10].mean() if miou_list.size > 0 else 0
    print("Average PSNR: {:.2f}".format(avg_psnr))
    print("Average Depth RMSE: {:.2f} cm".format(avg_rmse*100))
    print("Average Depth L1: {:.2f} cm".format(avg_l1*100))
    print("Average MS-SSIM: {:.3f}".format(avg_ssim))
    print("Average LPIPS: {:.3f}".format(avg_lpips))
    print("Average mIoU: {:.4f}".format(avg_miou))
    print("Average mIoU (stride 10): {:.4f}".format(avg_miou_stride10))

    if wandb_run is not None:
        wandb_run.log({"Final Stats/Average PSNR": avg_psnr,
                        "Final Stats/Average Depth RMSE": avg_rmse,
                        "Final Stats/Average Depth L1": avg_l1,
                        "Final Stats/Average MS-SSIM": avg_ssim,
                        "Final Stats/Average LPIPS": avg_lpips,
                        "Final Stats/step": 1,
                        "Final Stats/Average mIoU": avg_miou,
                        "Final Stats/Average mIoU (stride 10)": avg_miou_stride10})

    # Save metric lists as text files
    np.savetxt(os.path.join(eval_dir, "psnr.txt"), psnr_list)
    np.savetxt(os.path.join(eval_dir, "rmse.txt"), rmse_list)
    np.savetxt(os.path.join(eval_dir, "l1.txt"), l1_list)
    np.savetxt(os.path.join(eval_dir, "ssim.txt"), ssim_list)
    np.savetxt(os.path.join(eval_dir, "lpips.txt"), lpips_list)

    if load_semantics:
        np.savetxt(os.path.join(eval_dir, "miou.txt"), miou_list)
        fig, axs = plt.subplots(1, 3, figsize=(18, 4))
        axs[2].plot(np.arange(len(miou_list)), miou_list)
        axs[2].set_title("mIoU")
        axs[2].set_xlabel("Time Step")
        axs[2].set_ylabel("mIoU")
    else:
        fig, axs = plt.subplots(1, 2, figsize=(12, 4))

    axs[0].plot(np.arange(len(psnr_list)), psnr_list)
    axs[0].set_title("RGB PSNR")
    axs[0].set_xlabel("Time Step")
    axs[0].set_ylabel("PSNR")

    axs[1].plot(np.arange(len(l1_list)), l1_list*100)
    axs[1].set_title("Depth L1")
    axs[1].set_xlabel("Time Step")
    axs[1].set_ylabel("L1 (cm)")

    fig.suptitle("Average PSNR: {:.2f}, Average Depth L1: {:.2f} cm, ATE RMSE: {:.2f} cm, Average mIoU: {:.4f}".format(
        avg_psnr, avg_l1*100, ate_rmse*100, avg_miou), y=1.05, fontsize=16)

    plt.savefig(os.path.join(eval_dir, "metrics.png"), bbox_inches='tight')
    if wandb_run is not None:
        wandb_run.log({"Eval/Metrics": fig})
    plt.close()


def eval_nvs(dataset, final_params, num_frames, eval_dir, sil_thres, mapping_iters, add_new_gaussians,
             device="cuda", load_semantics=False, wandb_run=None, wandb_save_qual=False, eval_every=1, save_frames=False):
    print("Evaluating Final Parameters for Novel View Synthesis ...")
    psnr_list = []
    rmse_list = []
    l1_list = []
    lpips_list = []
    ssim_list = []
    valid_nvs_frames = []
    miou_list = []
    plot_dir = os.path.join(eval_dir, "plots")
    os.makedirs(plot_dir, exist_ok=True)
    if save_frames:
        render_rgb_dir = os.path.join(eval_dir, "rendered_rgb")
        os.makedirs(render_rgb_dir, exist_ok=True)
        render_depth_dir = os.path.join(eval_dir, "rendered_depth")
        os.makedirs(render_depth_dir, exist_ok=True)
        # rgb_dir = os.path.join(eval_dir, "rgb")
        # os.makedirs(rgb_dir, exist_ok=True)
        # depth_dir = os.path.join(eval_dir, "depth")
        # os.makedirs(depth_dir, exist_ok=True)

        if load_semantics:
            render_seg_dir = os.path.join(eval_dir, "rendered_semantic")
            os.makedirs(render_seg_dir, exist_ok=True)
            seg_dir = os.path.join(eval_dir, "semantic")
            os.makedirs(seg_dir, exist_ok=True)

    for time_idx in tqdm(range(num_frames)):
         # Get RGB-D Data & Camera Parameters
        if load_semantics:
            color, depth, intrinsics, pose, semantic_id, semantic_color = dataset[time_idx]
            semantic_id = semantic_id.permute(2, 0, 1) # (H, W, 1) -> (1, H, W)
            semantic_color = semantic_color.permute(2, 0, 1) / 255 # (H, W, C) -> (C, H, W)
        else:
            color, depth, intrinsics, pose = dataset[time_idx]

        gt_w2c = torch.linalg.inv(pose)
        intrinsics = intrinsics[:3, :3]

        # Process RGB-D Data
        color = color.permute(2, 0, 1) / 255 # (H, W, C) -> (C, H, W)
        depth = depth.permute(2, 0, 1) # (H, W, C) -> (C, H, W)

        if time_idx == 0:
            # Process Camera Parameters
            first_frame_w2c = torch.linalg.inv(pose)
            # Setup Camera
            cam = setup_camera(color.shape[2], color.shape[1], intrinsics.cpu().numpy(),
                               first_frame_w2c.detach().cpu().numpy(), device=device)
            # Skip first train frame eval for NVS
            continue
        
        # Skip frames if not eval_every (indexing accounts for first training frame)
        test_time_idx = time_idx - 1
        if test_time_idx != 0 and (test_time_idx+1) % eval_every != 0:
            continue

        # Transform Centers of Gaussians to Camera Frame
        pts = final_params['means3D'].detach()
        pts_ones = torch.ones(pts.shape[0], 1).to(device).float()
        pts4 = torch.cat((pts, pts_ones), dim=1)
        transformed_pts = (gt_w2c @ pts4.T).T[:, :3]
 
        # Define current frame data
        curr_data = {'cam': cam, 'im': color, 'depth': depth, 'id': time_idx, 'intrinsics': intrinsics, 'w2c': first_frame_w2c}
        if load_semantics:
            curr_data['semantic_id'] = semantic_id
            curr_data['semantic_color'] = semantic_color

        # Initialize Render Variables
        rendervar = transformed_params2rendervar(final_params, transformed_pts, device=device)
        depth_sil_rendervar = transformed_params2depthplussilhouette(final_params, curr_data['w2c'],
                                                                     transformed_pts, device=device)

        # Render Depth & Silhouette
        depth_sil, _, _, = Renderer(raster_settings=curr_data['cam'])(**depth_sil_rendervar)
        rastered_depth = depth_sil[0, :, :].unsqueeze(0)
        # Mask invalid depth in GT
        valid_depth_mask = (curr_data['depth'] > 0)
        rastered_depth_viz = rastered_depth.detach()
        rastered_depth = rastered_depth * valid_depth_mask
        silhouette = depth_sil[1, :, :]
        presence_sil_mask = (silhouette > sil_thres)

        # Check if Novel View is Valid based on Silhouette & Valid Depth Mask
        valid_region_mask = presence_sil_mask | ~valid_depth_mask
        percent_holes = (~valid_region_mask).sum() / valid_region_mask.numel() * 100
        if percent_holes > 0.1:
            valid_nvs_frames.append(False)
        else:
            valid_nvs_frames.append(True)
        
        # Render RGB and Calculate PSNR
        im, radius, _, = Renderer(raster_settings=curr_data['cam'])(**rendervar)
        if mapping_iters==0 and not add_new_gaussians:
            weighted_im = im * presence_sil_mask * valid_depth_mask
            weighted_gt_im = curr_data['im'] * presence_sil_mask * valid_depth_mask
        else:
            weighted_im = im * valid_depth_mask
            weighted_gt_im = curr_data['im'] * valid_depth_mask
        diff_rgb = torch.abs(weighted_im - weighted_gt_im).mean(dim=0).detach()
        psnr = calc_psnr(weighted_im, weighted_gt_im).mean()
        ssim = ms_ssim(weighted_im.unsqueeze(0).cpu(), weighted_gt_im.unsqueeze(0).cpu(), 
                        data_range=1.0, size_average=True)
        loss_fn_alex.to(device)
        lpips_score = loss_fn_alex(torch.clamp(weighted_im.unsqueeze(0), 0.0, 1.0),
                                    torch.clamp(weighted_gt_im.unsqueeze(0), 0.0, 1.0)).item()

        psnr_list.append(psnr.cpu().numpy())
        ssim_list.append(ssim.cpu().numpy())
        lpips_list.append(lpips_score)

        # Compute Depth RMSE
        if mapping_iters==0 and not add_new_gaussians:
            diff_depth_rmse = torch.sqrt((((rastered_depth - curr_data['depth']) * presence_sil_mask) ** 2))
            diff_depth_rmse = diff_depth_rmse * valid_depth_mask
            rmse = diff_depth_rmse.sum() / valid_depth_mask.sum()
            diff_depth_l1 = torch.abs((rastered_depth - curr_data['depth']) * presence_sil_mask)
            diff_depth_l1 = diff_depth_l1 * valid_depth_mask
            depth_l1 = diff_depth_l1.sum() / valid_depth_mask.sum()
        else:
            diff_depth_rmse = torch.sqrt((((rastered_depth - curr_data['depth'])) ** 2))
            diff_depth_rmse = diff_depth_rmse * valid_depth_mask
            rmse = diff_depth_rmse.sum() / valid_depth_mask.sum()
            diff_depth_l1 = torch.abs((rastered_depth - curr_data['depth']))
            diff_depth_l1 = diff_depth_l1 * valid_depth_mask
            depth_l1 = diff_depth_l1.sum() / valid_depth_mask.sum()
        rmse_list.append(rmse.cpu().numpy())
        l1_list.append(depth_l1.cpu().numpy())

        if load_semantics:
            # Render semantic color map
            semantic_rendervar = transformed_semantics2rendervar(final_params, transformed_pts, device=device)
            rastered_seg, _, _, = Renderer(raster_settings=curr_data['cam'])(**semantic_rendervar)
            gt_seg = curr_data['semantic_color']

            # Calcualte mIoU scores
            rastered_seg = recolor_semantic_img(rastered_seg, gt_seg)
            miou = evaluate_miou(rastered_seg, gt_seg)
            miou_list.append(miou)
        else:
            rastered_seg = None
            gt_seg = None

        if save_frames:
            # Save Rendered RGB and Depth
            viz_render_im = torch.clamp(im, 0, 1)
            viz_render_im = viz_render_im.detach().cpu().permute(1, 2, 0).numpy()
            vmin = 0
            vmax = 6
            viz_render_depth = rastered_depth_viz[0].detach().cpu().numpy()
            normalized_depth = np.clip((viz_render_depth - vmin) / (vmax - vmin), 0, 1)
            depth_colormap = cv2.applyColorMap((normalized_depth * 255).astype(np.uint8), cv2.COLORMAP_JET)
            cv2.imwrite(os.path.join(render_rgb_dir, "rgb_{:04d}.png".format(test_time_idx)), cv2.cvtColor(viz_render_im*255, cv2.COLOR_RGB2BGR))
            cv2.imwrite(os.path.join(render_depth_dir, "depth_{:04d}.png".format(test_time_idx)), depth_colormap)

            # Save GT RGB and Depth
            # viz_gt_im = torch.clamp(curr_data['im'], 0, 1)
            # viz_gt_im = viz_gt_im.detach().cpu().permute(1, 2, 0).numpy()
            # viz_gt_depth = curr_data['depth'][0].detach().cpu().numpy()
            # normalized_depth = np.clip((viz_gt_depth - vmin) / (vmax - vmin), 0, 1)
            # depth_colormap = cv2.applyColorMap((normalized_depth * 255).astype(np.uint8), cv2.COLORMAP_JET)
            # cv2.imwrite(os.path.join(rgb_dir, "gt_{:04d}.png".format(test_time_idx)), cv2.cvtColor(viz_gt_im*255, cv2.COLOR_RGB2BGR))
            # cv2.imwrite(os.path.join(depth_dir, "gt_{:04d}.png".format(test_time_idx)), depth_colormap)

            if load_semantics:
                viz_render_seg = torch.clamp(rastered_seg, 0, 1)
                viz_render_seg = viz_render_seg.detach().cpu().permute(1, 2, 0).numpy()
                cv2.imwrite(os.path.join(render_seg_dir, "seg_{:04d}.png".format(test_time_idx)), cv2.cvtColor(viz_render_seg*255, cv2.COLOR_RGB2BGR))
                # Save GT
                # viz_gt_seg = torch.clamp(gt_seg, 0, 1)
                # viz_gt_seg = viz_gt_seg.detach().cpu().permute(1, 2, 0).numpy()
                # cv2.imwrite(os.path.join(seg_dir, "gt_{:04d}.png".format(test_time_idx)), cv2.cvtColor(viz_gt_seg*255, cv2.COLOR_RGB2BGR))
        
        # Plot the Ground Truth and Rasterized RGB & Depth, along with Silhouette
        fig_title = "Time Step: {}".format(test_time_idx)
        plot_name = "%04d" % test_time_idx
        presence_sil_mask = presence_sil_mask.detach().cpu().numpy()
        if wandb_run is None:
            plot_rgbd_silhouette(color, depth, im, rastered_depth_viz, presence_sil_mask, diff_depth_l1,
                                 psnr, depth_l1, fig_title, plot_dir, plot_name=plot_name, save_plot=True,
                                 seg=gt_seg, rastered_seg=rastered_seg)
        elif wandb_save_qual:
            plot_rgbd_silhouette(color, depth, im, rastered_depth_viz, presence_sil_mask, diff_depth_l1,
                                 psnr, depth_l1, fig_title, plot_dir, plot_name=plot_name, save_plot=True,
                                 seg=gt_seg, rastered_seg=rastered_seg, wandb_run=wandb_run, wandb_step=None, 
                                 wandb_title="Eval/Qual Viz")

    # Compute Average Metrics based on valid NVS frames
    psnr_list = np.array(psnr_list)
    rmse_list = np.array(rmse_list)
    l1_list = np.array(l1_list)
    ssim_list = np.array(ssim_list)
    lpips_list = np.array(lpips_list)
    valid_nvs_frames = np.array(valid_nvs_frames)
    miou_list = np.array(miou_list)

    avg_psnr = psnr_list[valid_nvs_frames].mean()
    avg_rmse = rmse_list[valid_nvs_frames].mean()
    avg_l1 = l1_list[valid_nvs_frames].mean()
    avg_ssim = ssim_list[valid_nvs_frames].mean()
    avg_lpips = lpips_list[valid_nvs_frames].mean()
    avg_miou = miou_list.mean() if miou_list.size > 0 else 0
    print("Average PSNR: {:.2f}".format(avg_psnr))
    print("Average Depth RMSE: {:.2f} cm".format(avg_rmse*100))
    print("Average Depth L1: {:.2f} cm".format(avg_l1*100))
    print("Average MS-SSIM: {:.3f}".format(avg_ssim))
    print("Average LPIPS: {:.3f}".format(avg_lpips))
    print("Average mIoU: {:.4f}".format(avg_miou))

    if wandb_run is not None:
        wandb_run.log({"Final Stats/Average PSNR": avg_psnr, 
                       "Final Stats/Average Depth RMSE": avg_rmse,
                       "Final Stats/Average Depth L1": avg_l1,
                       "Final Stats/Average MS-SSIM": avg_ssim, 
                       "Final Stats/Average LPIPS": avg_lpips,
                       "Final Stats/step": 1,
                       "Final Stats/Average mIoU": avg_miou})

    # Save metric lists as text files
    np.savetxt(os.path.join(eval_dir, "psnr.txt"), psnr_list)
    np.savetxt(os.path.join(eval_dir, "rmse.txt"), rmse_list)
    np.savetxt(os.path.join(eval_dir, "l1.txt"), l1_list)
    np.savetxt(os.path.join(eval_dir, "ssim.txt"), ssim_list)
    np.savetxt(os.path.join(eval_dir, "lpips.txt"), lpips_list)

    # Save metadata for valid NVS frames
    np.save(os.path.join(eval_dir, "valid_nvs_frames.npy"), valid_nvs_frames)

    if load_semantics:
        np.savetxt(os.path.join(eval_dir, "miou.txt"), miou_list)

        fig, axs = plt.subplots(1, 3, figsize=(18, 4))
        axs[2].plot(np.arange(len(miou_list)), miou_list)
        axs[2].set_title("mIoU")
        axs[2].set_xlabel("Time Step")
        axs[2].set_ylabel("mIoU")
    else:
        fig, axs = plt.subplots(1, 2, figsize=(12, 4))

    # Plot PSNR & L1 as line plots
    axs[0].plot(np.arange(len(psnr_list)), psnr_list)
    axs[0].set_title("RGB PSNR")
    axs[0].set_xlabel("Time Step")
    axs[0].set_ylabel("PSNR")
    axs[1].plot(np.arange(len(l1_list)), l1_list*100)
    axs[1].set_title("Depth L1")
    axs[1].set_xlabel("Time Step")
    axs[1].set_ylabel("L1 (cm)")
    fig.suptitle("Average PSNR: {:.2f}, Average Depth L1: {:.2f} cm, Average mIoU: {:.4f}".format(avg_psnr, avg_l1*100, avg_miou),
                  y=1.05, fontsize=16)
    plt.savefig(os.path.join(eval_dir, "metrics.png"), bbox_inches='tight')
    if wandb_run is not None:
        wandb_run.log({"Eval/Metrics": fig})
    plt.close()

def _gaussian_window_1d(size, sigma):
    coords = torch.arange(size, dtype=torch.float)
    coords -= size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g /= g.sum()
    return g.unsqueeze(0).unsqueeze(0)


def _gaussian_blur(x, win):
    # Separable Gaussian blur with valid (no) padding, mirroring
    # pytorch_msssim's internal gaussian_filter. Shrinks H and W by
    # (win_size - 1) each, since no padding is applied.
    win = win.to(x.device, dtype=x.dtype)
    C = x.shape[1]
    out = x
    out = F.conv2d(out, weight=win, stride=1, padding=0, groups=C)
    out = F.conv2d(out, weight=win.transpose(2, 3), stride=1, padding=0, groups=C)
    return out


def calc_masked_psnr(img1, img2, mask):
    """PSNR computed only over pixels where mask is nonzero.
    img1, img2: (C, H, W) tensors. mask: (H, W) tensor (bool or 0/1 float).
    Returns a (C,) tensor of per-channel PSNR, matching calc_psnr's shape.
    """
    mask = mask.float()
    num_valid = mask.sum().clamp(min=1.0)
    diff_sq = (img1 - img2) ** 2 * mask.unsqueeze(0)
    mse = diff_sq.view(img1.shape[0], -1).sum(dim=1) / num_valid
    return 20 * torch.log10(1.0 / torch.sqrt(mse))


def masked_ssim(X, Y, mask, data_range=1.0, win_size=11, win_sigma=1.5, K=(0.01, 0.03)):
    """Single-scale SSIM computed only over pixels where mask is nonzero.
    X, Y: (N, C, H, W) tensors. mask: (H, W) tensor (bool or 0/1 float), in
    the same pre-blur resolution as X/Y.

    pytorch_msssim's ssim/ms_ssim average the SSIM map over the whole frame,
    so zeroing out irrelevant pixels before calling them just makes the
    (trivially perfect, 0-vs-0) masked-out region dilute the score. This
    instead crops the mask to match the valid-convolution output of the
    Gaussian window and averages only the masked entries of the SSIM map.
    """
    K1, K2 = K
    C1 = (K1 * data_range) ** 2
    C2 = (K2 * data_range) ** 2

    win = _gaussian_window_1d(win_size, win_sigma)
    win = win.repeat([X.shape[1], 1, 1, 1]).to(X.device, dtype=X.dtype)

    mu1 = _gaussian_blur(X, win)
    mu2 = _gaussian_blur(Y, win)
    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2
    sigma1_sq = _gaussian_blur(X * X, win) - mu1_sq
    sigma2_sq = _gaussian_blur(Y * Y, win) - mu2_sq
    sigma12 = _gaussian_blur(X * Y, win) - mu1_mu2

    cs_map = (2 * sigma12 + C2) / (sigma1_sq + sigma2_sq + C2)
    ssim_map = ((2 * mu1_mu2 + C1) / (mu1_sq + mu2_sq + C1)) * cs_map  # (N, C, H', W')

    pad = win_size // 2
    mask = mask.float()
    while mask.dim() < ssim_map.dim():
        mask = mask.unsqueeze(0)
    mask_cropped = mask[..., pad:mask.shape[-2] - pad, pad:mask.shape[-1] - pad]
    mask_cropped = mask_cropped.expand_as(ssim_map)

    num_valid = mask_cropped.sum()
    if num_valid == 0:
        return torch.tensor(float('nan'), device=X.device, dtype=X.dtype)
    return (ssim_map * mask_cropped).sum() / num_valid


def eval_single_frame(gt_rgb, gt_depth, gt_seg, gt_confidence_map, rendered_rgb, rendered_depth, rendered_seg, device="cuda"):
    
    # rgb and seg have shape (C, H, W)
    background_color = torch.tensor([0, 0, 0],device=device, dtype=gt_seg.dtype)
    fruit_color = torch.tensor([1, 0, 0],device=device, dtype=gt_seg.dtype)
    
    background_mask = torch.all(gt_seg == background_color[:, None, None], dim=0)
    fruit_mask = torch.all(gt_seg == fruit_color[:, None, None], dim=0)
    
    no_surface_mask = (rendered_depth[0] == 0)  # no gaussian rendered at this pixel; captured before rendered_depth is overwritten below

    valid_confidence_mask = (gt_confidence_map > 0.4)*(~background_mask) * fruit_mask
    valid_confidence_mask[no_surface_mask] = 0 # also mask out pixels where rendered depth is 0 (no surface rendered)
    valid_depth_mask = (gt_depth > 0)*(gt_depth < 1.0)*valid_confidence_mask
    rendered_depth = rendered_depth * valid_depth_mask
    nan = float('nan')

    if valid_confidence_mask.sum() == 0:
        print("No valid pixels for evaluation based on confidence map. Returning NaN for all metrics.")
        return nan, nan, nan, nan, nan, nan
    # Render RGB and Calculate PSNR, SSIM, LPIPS
    # PSNR and SSIM are computed only over valid_confidence_mask pixels
    # (not just zeroed-out-then-averaged-over-the-whole-frame, which would
    # dilute both metrics by however much of the frame is masked out).

    psnr = calc_masked_psnr(rendered_rgb, gt_rgb, valid_confidence_mask).mean()
    ssim = masked_ssim(rendered_rgb.unsqueeze(0).cuda(), gt_rgb.unsqueeze(0).cuda(),
                        valid_confidence_mask, data_range=1.0)
    # loss_fn_alex.to(device)
    # lpips_score = loss_fn_alex(torch.clamp(weighted_rend_im.unsqueeze(0), 0.0, 1.0),
                                # torch.clamp(weighted_gt_im.unsqueeze(0), 0.0, 1.0)).item()

    # plotting gt_rgb, weighted_gt_im, and gt_seg side by side for debugging
    # gt_rgb_np = gt_rgb.permute(1, 2, 0).cpu().numpy()
    # weighted_gt_im_np = weighted_gt_im.permute(1, 2, 0).cpu().numpy()
    # gt_seg_np = gt_seg.permute(1, 2, 0).cpu().numpy()
    # fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    # axs[0].imshow(gt_rgb_np)
    # axs[0].set_title("GT RGB")
    # axs[1].imshow(weighted_gt_im_np)
    # axs[1].set_title("Weighted GT RGB")
    # axs[2].imshow(gt_seg_np)
    # axs[2].set_title("GT Segmentation")
    # plt.savefig("debug_gt_rgb.png")
    # plt.close()
    # input("Press enter to continue...")

    lpips_score = 0.0
    # Compute Depth Metrics: RMSE and L1
    n_depth = valid_depth_mask.sum()
    if n_depth == 0:
        rmse = nan
        depth_l1 = nan
    else:
        diff_sq = (rendered_depth - gt_depth) ** 2
        diff_sq = diff_sq * valid_depth_mask
        rmse = torch.sqrt(diff_sq.sum() / valid_depth_mask.sum()).item()
        diff_depth_l1 = torch.abs((rendered_depth - gt_depth))
        diff_depth_l1 = diff_depth_l1 * valid_depth_mask
        depth_l1 = (diff_depth_l1.sum() / valid_depth_mask.sum()).item()
    
    # Compute metrics for semantics.
    # Deliberately NOT using valid_confidence_mask here: that mask requires
    # gt_seg == fruit_color, so masking gt_seg with it turns every
    # background/leaf pixel into [0, 0, 0], and evaluate_miou's own
    # unlabeled-pixel filtering would then drop them from consideration
    # entirely -- silently discarding any case where the render hallucinates
    # fruit/leaf color over background (or misses a leaf) instead of
    # counting it as an error. Use a mask based only on label confidence and
    # render coverage instead, so all three classes (background, fruit,
    # leaves) stay distinguishable and are scored, including [0, 0, 0].
    miou_valid_mask = (gt_confidence_map > 0.4)*(~background_mask)
    miou_valid_mask[no_surface_mask] = 0
    if miou_valid_mask.sum() == 0:
        miou = nan
    else:
        rendered_seg_recolored = recolor_semantic_img(rendered_seg, gt_seg)
        miou = evaluate_miou(rendered_seg_recolored, gt_seg, valid_mask=miou_valid_mask)
    

    return psnr.detach().cpu().item(), ssim.detach().cpu().item(), lpips_score, rmse, depth_l1, miou

def depth_colormap(img, cmap='jet', color_bar = True):
    
    W, H = img.shape[:2]
    dpi = 300
    fig, ax = plt.subplots(1, figsize=(H/dpi, W/dpi), dpi=dpi)
    im = ax.imshow(img, cmap=cmap)
    ax.set_axis_off()
    if color_bar:
        fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.canvas.draw()
    data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    img = torch.from_numpy(data / 255.).float().permute(2,0,1)
    plt.close()
    return img
    
    