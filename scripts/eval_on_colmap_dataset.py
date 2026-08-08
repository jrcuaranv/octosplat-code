"""
Evaluate a saved Gaussian Splatting model (as produced by
scripts/active_slam_xarm_colmap.py, e.g. a "*_sgs_output*" folder containing
a "*_params.npz" file) on a held-out COLMAP-format evaluation dataset folder
(images/, depth/, semantics/, confidences/, poses/, intrinsics.txt).

This is a standalone, ROS-free counterpart to ActiveSLAM.eval_test() in
active_slam_xarm_colmap.py, meant for re-evaluating a model after changes to
eval_single_frame() / eval_helpers.py without having to re-run the full
active mapping session.

Example:
    conda activate sgs_splatting
    python scripts/eval_on_colmap_dataset.py \
        --model_dir /mnt/ssd2T/datasets/gaussian_splat_data/active_mapping_evaluation_2026/sgs_gazebo_noisy_seg_noisy_depth/2026-06-28-20-07-04_sgs_output_g1_row1 \
        --eval_data_dir /mnt/ssd2T/datasets/gaussian_splat_data/active_mapping_evaluation_2026/eval_data_folders/greenhouse_1/row_1
"""
import argparse
import glob
import os
import sys
from datetime import datetime

_BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _BASE_DIR)

import cv2
import numpy as np
import torch
from PIL import Image as PILImage
from tqdm import tqdm

from utils.eval_helpers import eval_single_frame, depth_colormap
from utils.utils_sgs_slam import render_any_cam


def find_params_file(model_dir, params_file=None):
    if params_file is not None:
        return params_file
    candidates = sorted(glob.glob(os.path.join(model_dir, "*params.npz")))
    if len(candidates) == 0:
        raise FileNotFoundError(f"No *params.npz file found in {model_dir}")
    if len(candidates) > 1:
        print(f"Found {len(candidates)} params files in {model_dir}, using the most recent one: {candidates[-1]}")
    return candidates[-1]


def load_params(params_path, device="cuda"):
    raw_params = dict(np.load(params_path, allow_pickle=True))
    params = {k: torch.tensor(v).to(device).float() for k, v in raw_params.items()}
    return params


def filter_depth_map(depth_image, intrinsics, max_depth=2.0):
    # Standalone copy of utils.utils_active_mapping.filter_depth_map, inlined
    # here to avoid pulling in that module's ROS (geometry_msgs) dependency.
    import open3d as o3d

    fx, fy, cx, cy = intrinsics[0, 0], intrinsics[1, 1], intrinsics[0, 2], intrinsics[1, 2]
    depth_image = depth_image.copy()
    depth_image[depth_image > max_depth] = 0.0
    depth_image = np.nan_to_num(depth_image, nan=0.0)
    height, width = depth_image.shape
    u, v = np.meshgrid(np.arange(width), np.arange(height))
    z = depth_image
    valid_mask = z > 0.0
    z_valid, u_valid, v_valid = z[valid_mask], u[valid_mask], v[valid_mask]
    x = (u_valid - cx) * z_valid / fx
    y = (v_valid - cy) * z_valid / fy
    points = np.stack((x, y, z_valid), axis=-1)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd, ind = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=1.5)

    valid_flat_indices = np.flatnonzero(valid_mask)
    depth_filtered = np.zeros(depth_image.size, dtype=depth_image.dtype)
    depth_filtered[valid_flat_indices[ind]] = z_valid[ind]
    return depth_filtered.reshape(depth_image.shape)


def load_colmap_sample(eval_data_dir, file_name, intrinsics, max_depth,
                        apply_depth_median_filter=False,
                        apply_statistical_outlier_filter=False, device="cuda"):
    # Mirrors ActiveSLAM.get_colmap_sample_data() in active_slam_xarm_colmap.py.
    rgb_path = os.path.join(eval_data_dir, "images", file_name)
    semantic_path = os.path.join(eval_data_dir, "semantics", file_name)
    confidence_path = os.path.join(eval_data_dir, "confidences", file_name)
    depth_path = os.path.join(eval_data_dir, "depth", file_name)
    pose_path = os.path.join(eval_data_dir, "poses", file_name.replace(".png", ".txt"))

    for path in (rgb_path, semantic_path, confidence_path, depth_path, pose_path):
        if not os.path.exists(path):
            print(f"Missing file, skipping {file_name}: {path}")
            return None

    c2w = np.loadtxt(pose_path)

    bgr_image = cv2.imread(rgb_path)  # (h,w,3) BGR
    rgb_image = bgr_image[:, :, ::-1].astype(float)  # RGB, values in [0,255]

    semantic_bgr = cv2.imread(semantic_path)  # (h,w,3) BGR
    semantic_rgb = semantic_bgr[:, :, ::-1].astype(float)

    confidence_map = cv2.imread(confidence_path, cv2.IMREAD_UNCHANGED).astype(float) / 255.0

    depth = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED).astype(float) / 1000.0  # uint16 mm -> meters
    if apply_depth_median_filter:
        depth = cv2.medianBlur(depth.astype(np.float32), 5)
    if apply_statistical_outlier_filter:
        depth = filter_depth_map(depth, intrinsics, max_depth=max_depth)
    depth = np.nan_to_num(depth, nan=0.0)
    depth[depth > max_depth] = 0.0

    rgb_torch = torch.from_numpy(rgb_image).to(device).float()
    semantic_torch = torch.from_numpy(semantic_rgb).to(device).float()
    confidence_torch = torch.from_numpy(confidence_map).to(device).float()
    depth_torch = torch.from_numpy(depth).unsqueeze(-1).to(device).float()
    c2w_torch = torch.from_numpy(c2w).to(device).float()

    return rgb_torch, depth_torch, semantic_torch, confidence_torch, c2w_torch


def save_qualitative(images_dir, file_name, gt_color, gt_depth, gt_semantics,
                      rendered_color, rendered_depth, rendered_semantics, rendered_silhouette,
                      c2w, max_depth):
    # Mirrors the qualitative-dump block inside ActiveSLAM.eval_test().
    gt_color_np = (gt_color.permute(1, 2, 0).detach().cpu().numpy() * 255).astype(np.uint8)
    gt_semantics_np = (gt_semantics.permute(1, 2, 0).detach().cpu().numpy() * 255).astype(np.uint8)

    rendered_color_np = (rendered_color.permute(1, 2, 0).detach().cpu().numpy() * 255).astype(np.uint8)
    rendered_semantics_np = (rendered_semantics.permute(1, 2, 0).detach().cpu().numpy() * 255).astype(np.uint8)
    rendered_silhouette_np = rendered_silhouette.detach().cpu().numpy()

    valid_mask = rendered_silhouette_np > 0.2
    rendered_semantics_np[~valid_mask] = 0

    file_prefix = file_name.replace(".png", "")
    np.savetxt(os.path.join(images_dir, f"{file_prefix}_pose.txt"), c2w.detach().cpu().numpy())

    gt_depth_vis_src = torch.clamp(gt_depth, 0.0, max_depth).clone()
    gt_depth_vis_src[0, 0] = max_depth  # force colormap range to span [0, max_depth]
    gt_depth_vis_src[0, 1] = 0.0
    gt_depth_vis = depth_colormap((gt_depth_vis_src / max_depth).detach().cpu().numpy()[0], cmap="turbo", color_bar=False)
    gt_depth_vis_img = PILImage.fromarray((gt_depth_vis.clamp(0.0, 1.0).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8))

    rendered_depth_vis_src = torch.clamp(rendered_depth, 0.0, max_depth).clone()
    rendered_depth_vis_src[0, 0] = max_depth
    rendered_depth_vis_src[0, 1] = 0.0
    rendered_depth_vis = depth_colormap((rendered_depth_vis_src / max_depth).detach().cpu().numpy()[0], cmap="turbo", color_bar=False)
    rendered_depth_vis_img = PILImage.fromarray((rendered_depth_vis.clamp(0.0, 1.0).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8))

    rendered_silhouette_vis = (rendered_silhouette_np * 255).astype(np.uint8)
    rendered_silhouette_vis_color = cv2.applyColorMap(rendered_silhouette_vis, cv2.COLORMAP_JET)

    cv2.imwrite(os.path.join(images_dir, f"{file_prefix}_color_gt.png"), cv2.cvtColor(gt_color_np, cv2.COLOR_RGB2BGR))
    gt_depth_vis_img.save(os.path.join(images_dir, f"{file_prefix}_depth_gt.png"))
    cv2.imwrite(os.path.join(images_dir, f"{file_prefix}_semantics_gt.png"), cv2.cvtColor(gt_semantics_np, cv2.COLOR_RGB2BGR))
    cv2.imwrite(os.path.join(images_dir, f"{file_prefix}_color_rendered.png"), cv2.cvtColor(rendered_color_np, cv2.COLOR_RGB2BGR))
    rendered_depth_vis_img.save(os.path.join(images_dir, f"{file_prefix}_depth_rendered.png"))
    cv2.imwrite(os.path.join(images_dir, f"{file_prefix}_semantics_rendered.png"), cv2.cvtColor(rendered_semantics_np, cv2.COLOR_RGB2BGR))
    cv2.imwrite(os.path.join(images_dir, f"{file_prefix}_silhouette_rendered.png"), rendered_silhouette_vis_color)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model_dir", required=True,
                         help="SLAM output folder produced by active_slam_xarm_colmap.py, e.g. .../sgs_output_g1_row1")
    parser.add_argument("--eval_data_dir", required=True,
                         help="COLMAP-format eval dataset folder (images/, depth/, semantics/, confidences/, poses/, intrinsics.txt)")
    parser.add_argument("--params_file", default=None,
                         help="Explicit path to a *_params.npz file, overrides auto-discovery in model_dir")
    parser.add_argument("--output_dir", default=None,
                         help="Where to write metrics/qualitative results. Defaults to a new timestamped folder under model_dir")
    parser.add_argument("--max_depth", type=float, default=1.0, help="Depth values above this (in meters) are ignored")
    parser.add_argument("--apply_depth_median_filter", action="store_true")
    parser.add_argument("--apply_statistical_outlier_filter", action="store_true")
    parser.add_argument("--save_qualitative_every", type=int, default=1,
                         help="Save GT/rendered qualitative images every N frames, 0 to disable")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    params_path = find_params_file(args.model_dir, args.params_file)
    print(f"Loading Gaussian model from {params_path}")
    params = load_params(params_path, device=args.device)
    print(f"Loaded {params['means3D'].shape[0]} gaussians")

    intrinsics = np.loadtxt(os.path.join(args.eval_data_dir, "intrinsics.txt"))

    image_dir = os.path.join(args.eval_data_dir, "images")
    file_names = sorted(os.listdir(image_dir))
    print(f"Found {len(file_names)} evaluation frames in {image_dir}")

    if args.output_dir is None:
        timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        eval_data_name = os.path.basename(os.path.normpath(args.eval_data_dir))
        args.output_dir = os.path.join(args.model_dir, f"{timestamp}_eval_on_{eval_data_name}")
    os.makedirs(args.output_dir, exist_ok=True)

    images_dir = os.path.join(args.output_dir, "qualitative")
    if args.save_qualitative_every > 0:
        os.makedirs(images_dir, exist_ok=True)

    psnr_list, ssim_list, lpips_list, rmse_list, depth_l1_list, miou_list = [], [], [], [], [], []

    with torch.no_grad():
        for idx, file_name in enumerate(tqdm(file_names, desc="Evaluating")):
            sample = load_colmap_sample(
                args.eval_data_dir, file_name, intrinsics, args.max_depth,
                apply_depth_median_filter=args.apply_depth_median_filter,
                apply_statistical_outlier_filter=args.apply_statistical_outlier_filter,
                device=args.device,
            )
            if sample is None:
                continue
            gt_color, gt_depth, gt_semantics, gt_confidence, c2w = sample
            gt_color = gt_color.permute(2, 0, 1) / 255
            gt_semantics = gt_semantics.permute(2, 0, 1) / 255
            gt_depth = gt_depth.permute(2, 0, 1).float()

            height, width = gt_color.shape[1], gt_color.shape[2]
            w2c = torch.linalg.inv(c2w).detach().cpu().numpy()

            rendered_color, rendered_depth, rendered_semantics, rendered_silhouette = render_any_cam(
                params, w2c, height, width, device=args.device, intrinsics=intrinsics, render_all=True)

            psnr, ssim, lpips_score, rmse, depth_l1, miou = eval_single_frame(
                gt_color, gt_depth, gt_semantics, gt_confidence,
                rendered_color, rendered_depth, rendered_semantics, device=args.device)

            psnr_list.append(psnr)
            ssim_list.append(ssim)
            lpips_list.append(lpips_score)
            rmse_list.append(rmse)
            depth_l1_list.append(depth_l1)
            miou_list.append(miou)

            if args.save_qualitative_every > 0 and idx % args.save_qualitative_every == 0:
                save_qualitative(images_dir, file_name, gt_color, gt_depth, gt_semantics,
                                  rendered_color, rendered_depth, rendered_semantics, rendered_silhouette,
                                  c2w, args.max_depth)

    mean_psnr = np.nanmean(psnr_list)
    mean_ssim = np.nanmean(ssim_list)
    mean_lpips = np.nanmean(lpips_list)
    mean_rmse = np.nanmean(rmse_list)
    mean_depth_l1 = np.nanmean(depth_l1_list)
    mean_miou = np.nanmean(miou_list)

    lines = [
        "Test Evaluation Metrics",
        f"Model: {params_path}",
        f"Eval data: {args.eval_data_dir}",
        f"Number of gaussians: {params['means3D'].shape[0]}",
        f"Number of evaluated frames: {len(psnr_list)}",
        f"Mean PSNR: {mean_psnr}",
        f"Mean SSIM: {mean_ssim}",
        f"Mean LPIPS: {mean_lpips}",
        f"Mean RMSE: {mean_rmse}",
        f"Mean Depth L1: {mean_depth_l1}",
        f"Mean mIoU: {mean_miou}",
    ]
    # deleting any existing file ending with "evaluation_metrics.txt" in the model_dir
    existing_metrics_files = glob.glob(os.path.join(args.model_dir, "*evaluation_metrics.txt"))
    for file_path in existing_metrics_files:
        os.remove(file_path)
        print(f"Deleted existing metrics file: {file_path}")
    
    # metrics_path = os.path.join(args.output_dir, "evaluation_metrics.txt")
    metrics_path = os.path.join(args.model_dir, "evaluation_metrics.txt")
    with open(metrics_path, "w") as f:
        f.write("\n".join(lines) + "\n")

    print()
    print("\n".join(lines))
    print(f"\nMetrics written to {metrics_path}")


if __name__ == "__main__":
    main()
