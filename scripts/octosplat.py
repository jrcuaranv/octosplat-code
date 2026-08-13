'''
SGS-SLAM active mapping over an offline dataset (no ROS) containing
/images, /depth, /semantics, /confidences, /poses and intrinsics.txt

Loads RGB-D + semantics + confidence + pose data directly from a
scene folder (images/, semantics/, confidences/, depth/, poses/, intrinsics.txt),
runs online tracking + Gaussian Splatting mapping, and saves params + evaluation
metrics (training and held-out test split) at the end of the run.
'''
import argparse
import os
import shutil
import sys
import time
import random
from datetime import datetime
from importlib.machinery import SourceFileLoader

_BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _BASE_DIR)

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import yaml
from PIL import Image as PILImage
from tqdm import tqdm

from utils.utils_active_mapping import filter_depth_map
from utils.common_utils import seed_everything, save_params_ckpt, save_params, get_gpu_memory
from utils.keyframe_selection import keyframe_selection_overlap
from utils.eval_helpers import eval_single_frame, depth_colormap
from utils.slam_helpers import get_c2w_from_params, matrix_to_quaternion
from utils.slam_external import build_rotation, densify_v2, prune_background_semantics, prune_gaussians, reset_semantics
from utils.utils_sgs_slam import (render_any_cam, render_cam, initialize_optimizer, initialize_first_timestep,
                                  add_new_gaussians, initialize_camera_pose, get_loss)


class ActiveSLAM:
    def __init__(self, config, train_data_path=None, test_data_path=None):
        self.config = config

        prefix = config['active_mapping']['data_mode']
        self.episode_suffix = ""
        self.train_data_path = train_data_path
        if self.train_data_path is None:
            raise ValueError("train_data_path is not provided. Please provide it as a command line argument.")
        if self.train_data_path.endswith("/"):
            self.train_data_path = self.train_data_path[:-1]

        if test_data_path is not None:
            self.test_data_path = test_data_path
        else:
            default_test_data_path = os.path.join(self.train_data_path, "eval_data")
            if not os.path.isdir(default_test_data_path):
                raise ValueError(
                    f"No test_data_path was provided and no eval_data folder was found at "
                    f"{default_test_data_path}. Please pass --test_data_path explicitly."
                )
            self.test_data_path = default_test_data_path
        if self.test_data_path.endswith("/"):
            self.test_data_path = self.test_data_path[:-1]
        if not os.path.isdir(self.test_data_path):
            raise ValueError(f"Test data path does not exist: {self.test_data_path}")

        print("Train data path:", self.train_data_path)
        print("Test data path:", self.test_data_path)
        self.train_files, self.test_files = self.split_dataset()
        self.intrinsics = np.loadtxt(os.path.join(self.train_data_path, "intrinsics.txt"))
        self.fx, self.fy, self.cx, self.cy = self.intrinsics[0, 0], self.intrinsics[1, 1], self.intrinsics[0, 2], self.intrinsics[1, 2]

        self.output_directory = self.config['active_mapping'][prefix]['output_dir']
        if not os.path.exists(self.output_directory):
            os.makedirs(self.output_directory)

        self.max_depth = self.config['active_mapping'][prefix].get('max_depth')
        self.confidence_threshold = self.config['active_mapping'][prefix].get('confidence_threshold')

        self.init_variables()
        self.rgbd_slam(self.config)

    def init_variables(self):
        self.K = np.array([[self.fx, 0, self.cx],
                            [0, self.fy, self.cy],
                            [0, 0, 1.0]])

        self.params_file_prefix = "octosplat_"
        self.params = None
        self.keyframe_list = []
        self.intrinsics = None
        self.gt_w2c_all_frames = []
        self.first_frame_w2c = None
        self.variables = None
        self.cam = None
        self.device = None
        self.eval_dir = None
        self.episode_dir = None
        self.offline_sample_idx = 0
        self.start_mapping_time = time.time()
        self.end_mapping_time = time.time()
        self.gpu_memory_list = []

    def bgr_to_rgb(self, bgr_image):
        rgb_image = bgr_image[:, :, ::-1]
        return rgb_image

    def bgr_to_gray(self, bgr_image):
        B = bgr_image[:, :, 0]
        G = bgr_image[:, :, 1]
        R = bgr_image[:, :, 2]
        gray_image = 0.114 * B + 0.587 * G + 0.299 * R
        gray_image = gray_image.astype(np.uint8)
        return gray_image

    def get_sample_data(self, dtype=torch.float):
        sample_data = None
        while sample_data is None:
            if self.offline_sample_idx >= len(self.train_files):
                print("All samples from the training dataset have been used")
                return None
            file_name = self.train_files[self.offline_sample_idx]
            sample_data = self.get_sample_data_from_filename(file_name)
            self.offline_sample_idx += 1
        return sample_data

    def split_dataset(self):
        train_image_dir = os.path.join(self.train_data_path, "images")
        train_image_files = sorted(os.listdir(train_image_dir))
        random.shuffle(train_image_files) # to get random samples from the dataset, rather than sequential ones
        test_image_dir = os.path.join(self.test_data_path, "images")
        test_image_files = sorted(os.listdir(test_image_dir))
        print("Number of training files:", len(train_image_files))
        print("Number of test files:", len(test_image_files))
        return train_image_files, test_image_files

    def get_sample_data_from_filename(self, file_name, dtype = torch.float, eval = False):
        if eval:
            data_path = self.test_data_path
        else:
            data_path = self.train_data_path
        image_dir = os.path.join(data_path, "images") # contains both train and test images
        semantics_dir = os.path.join(data_path, "semantics")
        confidences_dir = os.path.join(data_path, "confidences")
        depth_dir = os.path.join(data_path, "depth")
        pose_dir = os.path.join(data_path, "poses")

        rgb_path = os.path.join(image_dir, file_name)
        semantic_path = os.path.join(semantics_dir, file_name)
        confidences_path = os.path.join(confidences_dir, file_name)
        depth_path = os.path.join(depth_dir, file_name)
        pose_path = os.path.join(pose_dir, file_name.replace(".png", ".txt"))

        if not os.path.exists(rgb_path):
            print(f"RGB image not found: {rgb_path}")
            return None
        if not os.path.exists(semantic_path):
            print(f"Semantic image not found: {semantic_path}")
            return None
        if not os.path.exists(confidences_path):
            print(f"Confidence map not found: {confidences_path}")
            return None
        if not os.path.exists(depth_path):
            print(f"Depth image not found: {depth_path}")
            return None
        if not os.path.exists(pose_path):
            print(f"Pose file not found: {pose_path}")
            return None

        intrinsics = torch.from_numpy(self.K)

        T_wc_rel = np.loadtxt(pose_path)
        T_wc_rel = torch.from_numpy(T_wc_rel)

        bgr_image = cv2.imread(rgb_path) #(h,w,3) in BGR format
        rgb_image = self.bgr_to_rgb(bgr_image)
        rgb_image = rgb_image.astype(float) #/255
        rgb_image = torch.from_numpy(rgb_image)

        semantic_img_uint8_bgr = cv2.imread(semantic_path) #(h,w,3) in BGR format
        semantic_img_uint8_rgb = self.bgr_to_rgb(semantic_img_uint8_bgr)
        semantic_img_float = semantic_img_uint8_rgb.astype(float) #/255
        semantic_img = torch.from_numpy(semantic_img_float)

        semantic_id = self.bgr_to_gray(semantic_img_uint8_bgr)
        semantic_id = semantic_id.astype(float)
        semantic_id = np.expand_dims(semantic_id, -1)#(h,w,1)
        semantic_id = torch.from_numpy(semantic_id)

        confidence_map = cv2.imread(confidences_path, cv2.IMREAD_UNCHANGED) #(h,w), confidence in [0,255]
        confidence_map = confidence_map.astype(float) / 255.0
        confidence_map = torch.from_numpy(confidence_map)

        depth = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)/1000.0 # (h,w) in uint16 format, depth in mm
        depth = np.expand_dims(depth, -1) #(h,w,1)
        depth = torch.from_numpy(depth)
        depth = torch.nan_to_num(depth, nan=0.0)
        depth[depth>self.max_depth] = 0.0

        return_data = (
            rgb_image.to(self.device).type(dtype),
            depth.to(self.device).type(dtype),
            intrinsics.to(self.device).type(dtype),
            T_wc_rel.to(self.device).type(dtype),
            semantic_id.to(self.device).type(dtype),
            semantic_img.to(self.device).type(dtype),
            confidence_map.to(self.device).type(dtype),
        )

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
        timestamp_str = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        self.episode_dir = os.path.join(self.output_directory, f"{timestamp_str}_sgs_output{self.episode_suffix}")
        os.makedirs(self.episode_dir, exist_ok=True)
        output_dir = self.episode_dir
        self.eval_dir = os.path.join(output_dir, "eval")

        # saving config file in output dir
        with open(os.path.join(self.episode_dir, "config.yaml"), 'w') as yaml_file:
            yaml.dump(config, yaml_file)

        # Get Device
        self.device = torch.device(config["primary_device"])
        if config["primary_device"].startswith("cuda:"):
            device_id = int(config["primary_device"].split(':')[1])
            torch.cuda.set_device(device_id)
            torch.cuda.reset_peak_memory_stats()

        # Load Dataset
        print("Loading Dataset Config...")
        dataset_config = config["data"]

        load_semantics = True
        num_frames = min(dataset_config["num_frames"], len(self.train_files))

        print("========= Number of frames to process:", num_frames)
        valid_data = False
        while valid_data == False:
            dataset_0 = self.get_sample_data()
            if dataset_0 is None:
                print("No valid sample data...")
            else:
                _, depth_sample, _, _, _, _, _ = dataset_0
                if depth_sample.sum().item() > 0:
                    valid_data = True
                else:
                    print("Depth sample is empty. Waiting for next sample...")

        self.params, self.variables, self.intrinsics, self.first_frame_w2c, self.cam, \
            self.params_opt_exclude = initialize_first_timestep(dataset_0, num_frames, config['scene_radius_depth_ratio'],
                                                        config['mean_sq_dist_method'], device=self.device,
                                                        load_semantics=load_semantics)
        # Initialize list to keep track of Keyframes
        self.keyframe_list = []
        keyframe_time_indices = []
        timestamp_keyframes = []

        # Init Variables to keep track of ground truth poses and runtimes
        self.gt_w2c_all_frames = []
        self.gpu_memory_list = []

        checkpoint_time_idx = 0

        # Iterate over Scan
        for time_idx in tqdm(range(checkpoint_time_idx, num_frames)):

            print("Current time idx:", time_idx)
            print("Number of gaussians:", self.params['means3D'].shape[0])

            if time_idx == 0:
                color, depth, _, gt_pose, semantic_id, semantic_color, confidence_map = dataset_0

            if time_idx > 0:
                if time_idx == num_frames - 1:
                    self.save_params_and_eval()
                    print(f"Checkpoint reached at time idx {time_idx}. Pausing SLAM session for inspection.")
                    return
                if self.offline_sample_idx >= len(self.train_files):
                    print("All samples from the training dataset have been used. Saving and exiting current SLAM session.")
                    self.save_params_and_eval()
                    return

                valid_data = False
                while valid_data == False:
                    sample_data = self.get_sample_data()
                    if sample_data is None:
                        print("No valid sample data...")
                    else:
                        color, depth, _, gt_pose, semantic_id, semantic_color, confidence_map = sample_data
                        if depth.sum().item() > 0:
                            valid_data = True
                        else:
                            print("Depth sample is empty. Waiting for next sample...")

            # Process poses
            gt_w2c = torch.linalg.inv(gt_pose)
            # Process RGB-D Data
            color = color.permute(2, 0, 1) / 255
            depth = depth.permute(2, 0, 1)
            self.gt_w2c_all_frames.append(gt_w2c)
            curr_gt_w2c = self.gt_w2c_all_frames
            # Optimize only current time step for tracking
            iter_time_idx = time_idx
            # Initialize Mapping Data for selected frame
            curr_data = {'cam': self.cam, 'im': color, 'depth': depth, 'id': iter_time_idx, 'intrinsics': self.intrinsics,
                        'w2c': self.first_frame_w2c, 'iter_gt_w2c_list': curr_gt_w2c}

            semantic_id = semantic_id.permute(2, 0, 1)
            semantic_color = semantic_color.permute(2, 0, 1) / 255
            curr_data['semantic_id'] = semantic_id
            curr_data['semantic_color'] = semantic_color
            curr_data['confidence_map'] = confidence_map

            # Initialize Data for Tracking
            tracking_curr_data = curr_data

            # Optimization Iterations
            self.num_iters_mapping = config['mapping']['num_iters']
            if time_idx == num_frames-1:
                self.num_iters_mapping = 2*self.num_iters_mapping #jrcv, to refine the optimization in the last frame

            if time_idx >= 0:
                with torch.no_grad():
                    # initialization based on constant velocity model
                    w2c_init = curr_gt_w2c[-1].detach().cpu().numpy()
                    self.params = initialize_camera_pose(self.params, time_idx,
                                                forward_prop=config['tracking']['forward_prop'],
                                                rel_w2c_initial_guess = w2c_init)
            # Step 1: Tracking (Pose refinement )
            if time_idx > 0 and not config['tracking']['use_gt_poses']:
                # Reset Optimizer & Learning Rates for tracking
                optimizer = initialize_optimizer(self.params, self.params_opt_exclude, config['tracking']['lrs'], tracking=True)
                # Keep Track of Best Candidate Rotation & Translation
                candidate_cam_unnorm_rot = self.params['cam_unnorm_rots'][..., time_idx].detach().clone()
                candidate_cam_tran = self.params['cam_trans'][..., time_idx].detach().clone()
                current_min_loss = float(1e20)
                # Tracking Optimization
                iter = 0
                do_continue_slam = False
                num_iters_tracking = config['tracking']['num_iters']
                progress_bar = tqdm(range(num_iters_tracking), desc=f"Tracking Time Step: {time_idx}")
                while True:
                    # Loss for current frame
                    loss, self.variables, losses = get_loss(self.params, tracking_curr_data, self.variables, iter_time_idx, config['tracking']['loss_weights'],
                                                    config['tracking']['use_sil_for_loss'], config['tracking']['sil_thres'],
                                                    config['tracking']['use_l1'], config['tracking']['ignore_outlier_depth_loss'],
                                                    tracking=True, device=self.device, plot_dir=self.eval_dir,
                                                    visualize_tracking_loss=config['tracking']['visualize_tracking_loss'],
                                                    tracking_iteration=iter, load_semantics=load_semantics, running_baseline=config['running_baseline'], confidence_threshold=self.confidence_threshold, max_depth=self.max_depth)
                    # Backprop
                    loss.backward()
                    # Optimizer Update
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)

                    with torch.no_grad():
                        # Save the best candidate rotation & translation
                        if loss < current_min_loss:
                            current_min_loss = loss
                            candidate_cam_unnorm_rot = self.params['cam_unnorm_rots'][..., time_idx].detach().clone()
                            candidate_cam_tran = self.params['cam_trans'][..., time_idx].detach().clone()
                        # Report Progress
                        if not config['report_iter_progress']:
                            progress_bar.update(1)
                    # Check if we should stop tracking
                    iter += 1
                    if iter == num_iters_tracking:
                        if losses['depth'] < config['tracking']['depth_loss_thres'] and config['tracking']['use_depth_loss_thres']:
                            break
                        elif config['tracking']['use_depth_loss_thres'] and not do_continue_slam:
                            do_continue_slam = True
                            progress_bar = tqdm(range(num_iters_tracking), desc=f"Tracking Time Step: {time_idx}")
                            num_iters_tracking = 2*num_iters_tracking
                        else:
                            break

                progress_bar.close()
                # Copy over the best candidate rotation & translation
                with torch.no_grad():
                    self.params['cam_unnorm_rots'][..., time_idx] = candidate_cam_unnorm_rot
                    self.params['cam_trans'][..., time_idx] = candidate_cam_tran
            elif time_idx > 0: #and config['tracking']['use_gt_poses']: #TODO change
                with torch.no_grad():
                    # Get the ground truth pose relative to frame 0
                    rel_w2c = curr_gt_w2c[-1]
                    rel_w2c_rot = rel_w2c[:3, :3].unsqueeze(0).detach()
                    rel_w2c_rot_quat = matrix_to_quaternion(rel_w2c_rot)
                    rel_w2c_tran = rel_w2c[:3, 3].detach()
                    # Update the camera parameters
                    self.params['cam_unnorm_rots'][..., time_idx] = rel_w2c_rot_quat
                    self.params['cam_trans'][..., time_idx] = rel_w2c_tran

            print("Densification step...")
            # Step 2: Densification & KeyFrame-based Mapping
            if time_idx == 0 or (time_idx+1) % config['map_every'] == 0:
                # Densification
                if config['mapping']['add_new_gaussians'] and time_idx >0:
                    # Setup Data for Densification
                    densify_curr_data = curr_data

                    # Add new Gaussians to the scene based on the Silhouette
                    self.params, self.variables = add_new_gaussians(self.params, self.params_opt_exclude, self.variables, densify_curr_data,
                                                        config['mapping']['sil_thres'], time_idx, config['mean_sq_dist_method'],
                                                        self.device, load_semantics=load_semantics, fill_depth_holes=config['mapping']['fill_depth_holes'])

                # Update keyframes for gaussian mapping
                with torch.no_grad():
                    # Get the current estimated rotation & translation
                    curr_cam_rot = F.normalize(self.params['cam_unnorm_rots'][..., time_idx].detach())
                    curr_cam_tran = self.params['cam_trans'][..., time_idx].detach()
                    curr_w2c = torch.eye(4).to(self.device).float()
                    curr_w2c[:3, :3] = build_rotation(curr_cam_rot)
                    curr_w2c[:3, 3] = curr_cam_tran

                    # Select Keyframes for Mapping
                    num_keyframes = config['mapping_window_size']-2
                    print("Keyframe selection overlap...")
                    selected_keyframes = keyframe_selection_overlap(depth, curr_w2c, self.intrinsics, self.keyframe_list[:-1],
                                                                    num_keyframes, device=self.device)
                    selected_time_idx = [self.keyframe_list[frame_idx]['id'] for frame_idx in selected_keyframes]
                    if len(self.keyframe_list) > 0:
                        # Add last keyframe to the selected keyframes
                        selected_time_idx.append(self.keyframe_list[-1]['id'])
                        selected_keyframes.append(len(self.keyframe_list)-1)
                    # Add current frame to the selected keyframes
                    selected_time_idx.append(time_idx)
                    selected_keyframes.append(-1)
                    # Print the selected keyframes
                    print(f"\nSelected Keyframes at Frame {time_idx}: {selected_time_idx}")
                    timestamp_keyframes.append(selected_time_idx)

                # Reset Optimizer & Learning Rates for Full Map Optimization
                optimizer = initialize_optimizer(self.params, self.params_opt_exclude, config['mapping']['lrs'], tracking=False)

                # Mapping
                print("Mapping...")
                mapping_start_time = time.time()
                if self.num_iters_mapping > 0:
                    progress_bar = tqdm(range(self.num_iters_mapping), desc=f"Mapping Time Step: {time_idx}")
                loss_compute_times = []
                prune_compute_times = []
                for iter in range(self.num_iters_mapping):
                    # Randomly select a frame until current time step amongst keyframes
                    rand_idx = np.random.randint(0, len(selected_keyframes))
                    selected_rand_keyframe_idx = selected_keyframes[rand_idx]
                    if selected_rand_keyframe_idx == -1:
                        # Use Current Frame Data
                        iter_time_idx = time_idx
                        iter_color = color
                        iter_depth = depth
                        iter_confidence_map = confidence_map
                    else:
                        # Use Keyframe Data
                        iter_time_idx = self.keyframe_list[selected_rand_keyframe_idx]['id']
                        iter_color = self.keyframe_list[selected_rand_keyframe_idx]['color']
                        iter_depth = self.keyframe_list[selected_rand_keyframe_idx]['depth']
                        iter_confidence_map = self.keyframe_list[selected_rand_keyframe_idx]['confidence_map']
                    iter_gt_w2c = self.gt_w2c_all_frames[:iter_time_idx+1]
                    iter_data = {'cam': self.cam, 'im': iter_color, 'depth': iter_depth, 'confidence_map': iter_confidence_map, 'id': iter_time_idx,
                                'intrinsics': self.intrinsics, 'w2c': self.first_frame_w2c, 'iter_gt_w2c_list': iter_gt_w2c}
                    # Add semantic id and colors
                    if selected_rand_keyframe_idx == -1:
                        iter_data['semantic_id'] = semantic_id
                        iter_data['semantic_color'] = semantic_color
                    else:
                        iter_data['semantic_id'] = self.keyframe_list[selected_rand_keyframe_idx]['semantic_id']
                        iter_data['semantic_color'] = self.keyframe_list[selected_rand_keyframe_idx]['semantic_color']

                    visualization = False
                    loss_start_time = time.time()
                    loss, self.variables, losses = get_loss(self.params, iter_data, self.variables, iter_time_idx, config['mapping']['loss_weights'],
                                                    config['mapping']['use_sil_for_loss'], config['mapping']['sil_thres'],
                                                    config['mapping']['use_l1'], config['mapping']['ignore_outlier_depth_loss'],
                                                    mapping=True, device=self.device, plot_dir = self.eval_dir, load_semantics=load_semantics, visualization = visualization, running_baseline=config['running_baseline'], confidence_threshold=self.confidence_threshold, max_depth=self.max_depth)
                    loss_end_time = time.time()
                    loss_compute_times.append(loss_end_time - loss_start_time)
                    # Backprop
                    loss.backward()
                    with torch.no_grad():
                        # Prune Gaussians
                        if config['mapping']['use_gaussian_splatting_densification']:
                            self.params, self.variables = densify_v2(self.params, self.variables, optimizer, iter, config['mapping']['densify_dict'], self.params_opt_exclude, device=self.device)
                        # Optimizer Update
                        optimizer.step()
                        optimizer.zero_grad(set_to_none=True)

                        prune_start_time = time.time()
                        if config['mapping']['prune_gaussians']:
                            self.params, self.variables = prune_gaussians(self.params, self.params_opt_exclude, self.variables, optimizer, iter, config['mapping']['pruning_dict'])
                        if config['mapping']['prune_background_gaussians']:
                            self.params, self.variables = prune_background_semantics(self.params, self.params_opt_exclude, self.variables, optimizer, iter, config['mapping']['pruning_dict'])
                        prune_end_time = time.time()
                        prune_compute_times.append(prune_end_time - prune_start_time)

                        # Report Progress
                        if not config['report_iter_progress']:
                            progress_bar.update(1)

                if self.num_iters_mapping > 0:
                    progress_bar.close()
                mapping_end_time = time.time()
                gpu_memory = get_gpu_memory()
                if gpu_memory is not None:
                    self.gpu_memory_list.append(gpu_memory)

                print(f"Total loss compute time: {sum(loss_compute_times)}")
                print(f"Total prune compute time: {sum(prune_compute_times)}")
                print(f"Total mapping time: {mapping_end_time - mapping_start_time}")

            # Add frame to keyframe list
            if ((time_idx == 0) or ((time_idx+1) % config['keyframe_every'] == 0) or \
                        (time_idx == num_frames-2)) and (not torch.isinf(curr_gt_w2c[-1]).any()) and (not torch.isnan(curr_gt_w2c[-1]).any()):
                with torch.no_grad():
                    # Get the current estimated rotation & translation
                    curr_cam_rot = F.normalize(self.params['cam_unnorm_rots'][..., time_idx].detach())
                    curr_cam_tran = self.params['cam_trans'][..., time_idx].detach()
                    curr_w2c = torch.eye(4).to(self.device).float()
                    curr_w2c[:3, :3] = build_rotation(curr_cam_rot)
                    curr_w2c[:3, 3] = curr_cam_tran
                    # Initialize Keyframe Info
                    curr_keyframe = {'id': time_idx, 'est_w2c': curr_w2c, 'color': color, 'depth': depth}
                    curr_keyframe['semantic_id'] = semantic_id
                    curr_keyframe['semantic_color'] = semantic_color
                    curr_keyframe['confidence_map'] = confidence_map
                    # Add to keyframe list
                    self.keyframe_list.append(curr_keyframe)
                    keyframe_time_indices.append(time_idx)

            # Checkpoint every iteration
            if time_idx % config["checkpoint_interval"] == 0 and config['save_checkpoints']:
                ckpt_output_dir = os.path.join(config["workdir"], config["run_name"])
                save_params_ckpt(self.params, ckpt_output_dir, time_idx)
                np.save(os.path.join(ckpt_output_dir, f"keyframe_time_indices{time_idx}.npy"), np.array(keyframe_time_indices))

            torch.cuda.empty_cache()

        if config['save_timestamp_keyframes']:
            # Save keyframes selected at each timestamp
            max_length = max(len(inner) for inner in timestamp_keyframes)
            # Insert -1 for placeholder
            timestamp_keyframes_df = pd.DataFrame([inner + [-1 for _ in range(max_length - len(inner))] \
                                                for inner in timestamp_keyframes])
            timestamp_keyframes_df.to_csv(os.path.join(self.eval_dir, f"timestamp_keyframes.csv"), \
                                        index=False, header=False, na_rep='-1')

        # Add Camera Parameters to Save them
        self.params['timestep'] = self.variables['timestep']
        self.params['intrinsics'] = self.intrinsics.detach().cpu().numpy()
        self.params['w2c'] = self.first_frame_w2c.detach().cpu().numpy()
        self.params['org_width'] = dataset_config["desired_image_width"]
        self.params['org_height'] = dataset_config["desired_image_height"]
        self.params['gt_w2c_all_frames'] = []
        for gt_w2c_tensor in self.gt_w2c_all_frames:
            self.params['gt_w2c_all_frames'].append(gt_w2c_tensor.detach().cpu().numpy())
        self.params['gt_w2c_all_frames'] = np.stack(self.params['gt_w2c_all_frames'], axis=0)
        self.params['keyframe_time_indices'] = np.array(keyframe_time_indices)

        self.params['semantic_ids'] = self.params['semantic_ids'].type(torch.uint8)
        self.save_params_and_eval()

    def full_map_optimization(self, number_steps):
        print(f"Running full map optimization on all keyframes for {number_steps} iterations...")
        # Reset Optimizer & Learning Rates for Full Map Optimization
        optimizer = initialize_optimizer(self.params, self.params_opt_exclude, self.config['mapping']['lrs'], tracking=False)

        for iter in range(number_steps):
            if iter % 100 == 0:
                print(f"Full Map Optimization Iteration {iter}/{number_steps}. Number of gaussians: {self.params['means3D'].shape[0]}")
            # Randomly select a frame until current time step amongst keyframes
            rand_idx = np.random.randint(0, len(self.keyframe_list))
            # Use Keyframe Data
            iter_time_idx = self.keyframe_list[rand_idx]['id']
            iter_color = self.keyframe_list[rand_idx]['color']
            iter_depth = self.keyframe_list[rand_idx]['depth']
            iter_confidence_map = self.keyframe_list[rand_idx]['confidence_map']
            iter_gt_w2c = self.gt_w2c_all_frames[:iter_time_idx+1]
            iter_data = {'cam': self.cam, 'im': iter_color, 'depth': iter_depth, 'confidence_map': iter_confidence_map, 'id': iter_time_idx,
                        'intrinsics': self.intrinsics, 'w2c': self.first_frame_w2c, 'iter_gt_w2c_list': iter_gt_w2c}
            # Add semantic id and colors
            iter_data['semantic_id'] = self.keyframe_list[rand_idx]['semantic_id']
            iter_data['semantic_color'] = self.keyframe_list[rand_idx]['semantic_color']
            # Loss for current frame
            visualization = False
            loss, self.variables, losses = get_loss(self.params, iter_data, self.variables, iter_time_idx, self.config['mapping']['loss_weights'],
                                            self.config['mapping']['use_sil_for_loss'], self.config['mapping']['sil_thres'],
                                            self.config['mapping']['use_l1'], self.config['mapping']['ignore_outlier_depth_loss'],
                                            mapping=True, device=self.device, plot_dir = self.eval_dir, load_semantics=True, visualization = visualization, running_baseline=self.config['running_baseline'], confidence_threshold=self.confidence_threshold, max_depth=self.max_depth)
            # Backprop
            loss.backward()
            with torch.no_grad():
                self.params, self.variables = densify_v2(self.params, self.variables, optimizer, iter, self.config['mapping']['densify_dict'], self.params_opt_exclude, device=self.device)
                # Optimizer Update
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

                # Prune Gaussians
                if self.config['mapping']['prune_gaussians']:
                    self.params, self.variables = prune_gaussians(self.params, self.params_opt_exclude, self.variables, optimizer, iter, self.config['mapping']['pruning_dict'])
                if self.config['mapping']['prune_background_gaussians']:
                    self.params, self.variables = prune_background_semantics(self.params, self.params_opt_exclude, self.variables, optimizer, iter, self.config['mapping']['pruning_dict'])

    def save_params_and_eval(self):
        self.end_mapping_time = time.time()
        self.full_map_optimization(200) # a few additional optimization steps for refinement
        print("Saving parameters...")
        now = datetime.now()
        timestamp = now.strftime("%Y-%m-%d-%H-%M-%S")
        file_prefix = timestamp + '_' + self.params_file_prefix
        save_params(self.params, self.episode_dir, save_ply=True, file_prefix=file_prefix)
        self.eval_training(timestamp_str=timestamp)
        self.eval_test(timestamp_str=timestamp)

    def eval_training(self, timestamp_str=''):
        # also computes evaluation metrics
        images_dir = os.path.join(self.episode_dir, timestamp_str + "_train_images")
        os.makedirs(images_dir, exist_ok=True)
        psnr_list = []
        ssmi_list = []
        lpips_list = []
        rmse_list = []
        depth_l1_list = []
        miou_list = []
        for frame_idx in range(len(self.keyframe_list)):
            curr_time_idx = self.keyframe_list[frame_idx]['id']
            gt_color_torch = self.keyframe_list[frame_idx]['color']
            gt_semantics_torch = self.keyframe_list[frame_idx]['semantic_color']
            gt_depth_torch = self.keyframe_list[frame_idx]['depth'].float()
            gt_confidence_map_torch = self.keyframe_list[frame_idx]['confidence_map']
            rendered_color_torch, rendered_depth_torch, rendered_semantics_torch, rendered_silhouette_torch = render_cam(self.params, self.cam, curr_time_idx)

            # computing evaluation metric
            psnr, ssim, lpips_score, rmse, depth_l1, miou = eval_single_frame(gt_color_torch, gt_depth_torch, gt_semantics_torch, gt_confidence_map_torch, rendered_color_torch, rendered_depth_torch, rendered_semantics_torch, confidence_threshold=self.confidence_threshold, max_depth=self.max_depth)
            psnr_list.append(psnr)
            ssmi_list.append(ssim)
            lpips_list.append(lpips_score)
            rmse_list.append(rmse)
            depth_l1_list.append(depth_l1)
            miou_list.append(miou)

            if frame_idx % 10 == 0:
                gt_color_torch = gt_color_torch.permute(1,2,0).float()
                gt_color_numpy  = (gt_color_torch.detach().cpu().numpy() * 255).astype(np.uint8)
                gt_semantics_torch = gt_semantics_torch.permute(1,2,0).float()
                gt_semantics_numpy  = (gt_semantics_torch.detach().cpu().numpy() * 255).astype(np.uint8)

                rendered_color_numpy  = (rendered_color_torch.permute(1, 2, 0).detach().cpu().numpy() * 255).astype(np.uint8)
                rendered_semantics_numpy = (rendered_semantics_torch.permute(1, 2, 0).detach().cpu().numpy() * 255).astype(np.uint8)
                rendered_silhouette_numpy = rendered_silhouette_torch.detach().cpu().numpy()

                valid_mask = rendered_silhouette_numpy > 0.9
                rendered_color_numpy[~valid_mask] = 0
                rendered_semantics_numpy[~valid_mask] = 0

                c2w_torch = get_c2w_from_params(self.params, curr_time_idx)
                c2w_numpy = c2w_torch.detach().cpu().numpy()
                np.savetxt(os.path.join(images_dir, f"frame_{curr_time_idx}_pose.txt"), c2w_numpy)

                gt_depth = torch.clamp(gt_depth_torch, 0.0, self.max_depth)
                gt_depth[0, 0] = self.max_depth # to ensure the colormap is scaled correctly from min depth to max depth
                gt_depth[0, 1] = 0.0
                gt_depth_vis = depth_colormap((gt_depth / self.max_depth).detach().cpu().numpy()[0], cmap='turbo', color_bar=False)
                gt_depth_vis_color = (gt_depth_vis.clamp(0.0, 1.0).permute(1, 2, 0).cpu().numpy()*255).astype(np.uint8)
                gt_depth_vis_color = PILImage.fromarray(gt_depth_vis_color)

                rendered_depth = torch.clamp(rendered_depth_torch, 0.0, self.max_depth)
                rendered_depth[0, 0] = self.max_depth # to ensure the colormap is scaled correctly from min depth to max depth
                rendered_depth[0, 1] = 0.0
                rendered_depth_vis = depth_colormap((rendered_depth / self.max_depth).detach().cpu().numpy()[0], cmap='turbo', color_bar=False)
                rendered_depth_vis_color = (rendered_depth_vis.clamp(0.0, 1.0).permute(1, 2, 0).cpu().numpy()*255).astype(np.uint8)
                rendered_depth_vis_color = PILImage.fromarray(rendered_depth_vis_color)

                rendered_silhouette_vis = (rendered_silhouette_numpy * 255).astype(np.uint8)
                rendered_silhouette_vis_color = cv2.applyColorMap(rendered_silhouette_vis, cv2.COLORMAP_JET)

                # Save GT and Rendered Images
                cv2.imwrite(os.path.join(images_dir, f"frame_{curr_time_idx}_color_gt.png"), cv2.cvtColor(gt_color_numpy, cv2.COLOR_RGB2BGR))
                gt_depth_vis_color.save(os.path.join(images_dir, f"frame_{curr_time_idx}_depth_gt.png"))
                cv2.imwrite(os.path.join(images_dir, f"frame_{curr_time_idx}_semantics_gt.png"), cv2.cvtColor(gt_semantics_numpy, cv2.COLOR_RGB2BGR))
                cv2.imwrite(os.path.join(images_dir, f"frame_{curr_time_idx}_color_rendered.png"), cv2.cvtColor(rendered_color_numpy, cv2.COLOR_RGB2BGR))
                rendered_depth_vis_color.save(os.path.join(images_dir, f"frame_{curr_time_idx}_depth_rendered.png"))
                cv2.imwrite(os.path.join(images_dir, f"frame_{curr_time_idx}_semantics_rendered.png"), cv2.cvtColor(rendered_semantics_numpy, cv2.COLOR_RGB2BGR))
                cv2.imwrite(os.path.join(images_dir, f"frame_{curr_time_idx}_silhouette_rendered.png"), rendered_silhouette_vis_color)
        mean_psnr = np.nanmean(psnr_list)
        mean_ssim = np.nanmean(ssmi_list)
        mean_lpips = np.nanmean(lpips_list)
        mean_rmse = np.nanmean(rmse_list)
        mean_depth_l1 = np.nanmean(depth_l1_list)
        mean_miou = np.nanmean(miou_list)
        metrics_output_path = os.path.join(self.episode_dir, timestamp_str+"_evaluation_metrics.txt")
        with open(metrics_output_path, 'w') as f:
            f.write(f"Training Evaluation Metrics\n")
            f.write(f"Number of gaussians: {self.params['means3D'].shape[0]}\n")
            f.write(f"Mean PSNR: {mean_psnr}\n")
            f.write(f"Mean SSIM: {mean_ssim}\n")
            f.write(f"Mean LPIPS: {mean_lpips}\n")
            f.write(f"Mean RMSE: {mean_rmse}\n")
            f.write(f"Mean Depth L1: {mean_depth_l1}\n")
            f.write(f"Mean mIoU: {mean_miou}\n")
            f.write(f"Number of keyframes: {len(self.keyframe_list)}\n")
            f.write(f"Total time: {self.end_mapping_time - self.start_mapping_time}\n")
            f.write(f"Time per frame: {(self.end_mapping_time - self.start_mapping_time)/len(self.keyframe_list)}\n")
            f.write(f"Max GPU Memory: {max(self.gpu_memory_list) if len(self.gpu_memory_list) > 0 else 0}\n")
        # printing metrics
        print(f"\nTraining Evaluation Metrics:")
        print(f"\nNumber of gaussians: {self.params['means3D'].shape[0]}")
        print(f"Mean PSNR: {mean_psnr}")
        print(f"Mean SSIM: {mean_ssim}")
        print(f"Mean LPIPS: {mean_lpips}")
        print(f"Mean RMSE: {mean_rmse}")
        print(f"Mean Depth L1: {mean_depth_l1}")
        print(f"Mean mIoU: {mean_miou}")

    def eval_test(self, timestamp_str=''):
        # also computes evaluation metrics
        images_dir = os.path.join(self.episode_dir, timestamp_str + "_test_images")
        os.makedirs(images_dir, exist_ok=True)
        psnr_list = []
        ssmi_list = []
        lpips_list = []
        rmse_list = []
        depth_l1_list = []
        miou_list = []
        curr_time_idx = 0

        for file_name in self.test_files:
            sample_data = self.get_sample_data_from_filename(file_name, eval=True)
            if sample_data is None:
                continue
            gt_color_torch, gt_depth_torch, intrinsics, c2w_torch, _, gt_semantics_torch, gt_confidence_map_torch = sample_data
            gt_color_torch = gt_color_torch.permute(2,0,1)/255
            gt_semantics_torch = gt_semantics_torch.permute(2,0,1)/255
            gt_depth_torch = gt_depth_torch.permute(2,0,1).float()

            height, width = gt_color_torch.shape[1], gt_color_torch.shape[2]
            w2c_torch = torch.linalg.inv(c2w_torch)
            w2c_numpy = w2c_torch.detach().cpu().numpy()
            rendered_color_torch, rendered_depth_torch, rendered_semantics_torch, rendered_silhouette_torch = render_any_cam(self.params, w2c_numpy, height, width, intrinsics=intrinsics, render_all = True)

            # computing evaluation metric
            psnr, ssim, lpips_score, rmse, depth_l1, miou = eval_single_frame(gt_color_torch, gt_depth_torch, gt_semantics_torch, gt_confidence_map_torch, rendered_color_torch, rendered_depth_torch, rendered_semantics_torch, confidence_threshold=self.confidence_threshold, max_depth=self.max_depth)

            psnr_list.append(psnr)
            ssmi_list.append(ssim)
            lpips_list.append(lpips_score)
            rmse_list.append(rmse)
            depth_l1_list.append(depth_l1)
            miou_list.append(miou)

            if curr_time_idx % 10 == 0:

                gt_color_torch = gt_color_torch.permute(1,2,0).float()
                gt_color_numpy  = (gt_color_torch.detach().cpu().numpy() * 255).astype(np.uint8)
                gt_semantics_torch = gt_semantics_torch.permute(1,2,0).float()
                gt_semantics_numpy  = (gt_semantics_torch.detach().cpu().numpy() * 255).astype(np.uint8)

                rendered_color_numpy  = (rendered_color_torch.permute(1, 2, 0).detach().cpu().numpy() * 255).astype(np.uint8)
                rendered_semantics_numpy = (rendered_semantics_torch.permute(1, 2, 0).detach().cpu().numpy() * 255).astype(np.uint8)
                rendered_silhouette_numpy = rendered_silhouette_torch.detach().cpu().numpy()

                valid_mask = rendered_silhouette_numpy > 0.2 #0.9
                rendered_semantics_numpy[~valid_mask] = 0

                c2w_numpy = c2w_torch.detach().cpu().numpy()
                np.savetxt(os.path.join(images_dir, f"{file_name}_pose.txt"), c2w_numpy)

                gt_depth = torch.clamp(gt_depth_torch, 0.0, self.max_depth)
                gt_depth[0, 0] = self.max_depth # to ensure the colormap is scaled correctly from min depth to max depth
                gt_depth[0, 1] = 0.0
                gt_depth_vis = depth_colormap((gt_depth / self.max_depth).detach().cpu().numpy()[0], cmap='turbo', color_bar=False)
                gt_depth_vis_color = (gt_depth_vis.clamp(0.0, 1.0).permute(1, 2, 0).cpu().numpy()*255).astype(np.uint8)
                gt_depth_vis_color = PILImage.fromarray(gt_depth_vis_color)

                rendered_depth = torch.clamp(rendered_depth_torch, 0.0, self.max_depth)
                rendered_depth[0, 0] = self.max_depth # to ensure the colormap is scaled correctly from min depth to max depth
                rendered_depth[0, 1] = 0.0
                rendered_depth_vis = depth_colormap((rendered_depth / self.max_depth).detach().cpu().numpy()[0], cmap='turbo', color_bar=False)
                rendered_depth_vis_color = (rendered_depth_vis.clamp(0.0, 1.0).permute(1, 2, 0).cpu().numpy()*255).astype(np.uint8)
                rendered_depth_vis_color = PILImage.fromarray(rendered_depth_vis_color)

                rendered_silhouette_vis = (rendered_silhouette_numpy * 255).astype(np.uint8)
                rendered_silhouette_vis_color = cv2.applyColorMap(rendered_silhouette_vis, cv2.COLORMAP_JET)

                file_prefix = file_name.replace(".png", "")
                # Save GT and Rendered Images
                cv2.imwrite(os.path.join(images_dir, f"{file_prefix}_color_gt.png"), cv2.cvtColor(gt_color_numpy, cv2.COLOR_RGB2BGR))
                gt_depth_vis_color.save(os.path.join(images_dir, f"{file_prefix}_depth_gt.png"))
                cv2.imwrite(os.path.join(images_dir, f"{file_prefix}_semantics_gt.png"), cv2.cvtColor(gt_semantics_numpy, cv2.COLOR_RGB2BGR))
                cv2.imwrite(os.path.join(images_dir, f"{file_prefix}_color_rendered.png"), cv2.cvtColor(rendered_color_numpy, cv2.COLOR_RGB2BGR))
                rendered_depth_vis_color.save(os.path.join(images_dir, f"{file_prefix}_depth_rendered.png"))
                cv2.imwrite(os.path.join(images_dir, f"{file_prefix}_semantics_rendered.png"), cv2.cvtColor(rendered_semantics_numpy, cv2.COLOR_RGB2BGR))
                cv2.imwrite(os.path.join(images_dir, f"{file_prefix}_silhouette_rendered.png"), rendered_silhouette_vis_color)
            curr_time_idx += 1
        mean_psnr = np.nanmean(psnr_list)
        mean_ssim = np.nanmean(ssmi_list)
        mean_lpips = np.nanmean(lpips_list)
        mean_rmse = np.nanmean(rmse_list)
        mean_depth_l1 = np.nanmean(depth_l1_list)
        mean_miou = np.nanmean(miou_list)
        metrics_output_path = os.path.join(self.episode_dir, timestamp_str+"_evaluation_metrics.txt")
        with open(metrics_output_path, 'a') as f:
            f.write(f"\nTest Evaluation Metrics\n")
            f.write(f"Number of gaussians: {self.params['means3D'].shape[0]}\n")
            f.write(f"Mean PSNR: {mean_psnr}\n")
            f.write(f"Mean SSIM: {mean_ssim}\n")
            f.write(f"Mean LPIPS: {mean_lpips}\n")
            f.write(f"Mean RMSE: {mean_rmse}\n")
            f.write(f"Mean Depth L1: {mean_depth_l1}\n")
            f.write(f"Mean mIoU: {mean_miou}\n")
            f.write(f"Max GPU Memory: {max(self.gpu_memory_list) if len(self.gpu_memory_list) > 0 else 0}\n")

        # printing metrics
        print(f"\nTest Evaluation Metrics:")
        print(f"Number of gaussians: {self.params['means3D'].shape[0]}")
        print(f"Mean PSNR: {mean_psnr}")
        print(f"Mean SSIM: {mean_ssim}")
        print(f"Mean LPIPS: {mean_lpips}")
        print(f"Mean RMSE: {mean_rmse}")
        print(f"Mean Depth L1: {mean_depth_l1}")
        print(f"Mean mIoU: {mean_miou}")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("experiment", type=str, help="Path to experiment file")
    parser.add_argument("--train_data_path", type=str, help="Path to training dataset for a single scene")
    parser.add_argument("--test_data_path", type=str, default=None,
                         help="Path to a held-out  eval dataset folder (images/, semantics/, "
                              "confidences/, depth/, poses/). Defaults to <train_data_path>/eval_data if it "
                              "exists; raises an error if omitted and that folder is missing.")

    args = parser.parse_args()
    experiment = SourceFileLoader(
        os.path.basename(args.experiment), args.experiment
    ).load_module()

    # Set Experiment Seed
    seed_everything(seed=experiment.config['seed'])

    
    return experiment.config, args.train_data_path, args.test_data_path

if __name__ == '__main__':
    config, train_data_path, test_data_path = main()
    ActiveSLAM(config, train_data_path, test_data_path)