import os
from os.path import join as p_join
import getpass
username = getpass.getuser()
print(f"Running as user: {username}")
if username == "temp":
    output_dir = '/mnt/ssd2T/sgs_results_mapping/'
elif username == "jose":
    output_dir = '/media/jose/SSD1G/results_mapping/'
elif username == "companion":
    output_dir = '/mnt/ssd1T/results_mapping/'
else:
    raise ValueError(f"Unknown username: {username}. Please set the output_dir for this user in configs/octosplat/slam.py")

RUNNING_BASELINE = False
NUM_FRAMES = 999999999 # to limit the number of frames to a subset of the dataset. Otherwise, keep it large to use the entire dataset. 
primary_device="cuda:0"
seed = 0

map_every = 1
keyframe_every = 1 # 5 for slam, 1 for active mapping, jrcv
mapping_window_size = 24
tracking_iters = 40 #40
mapping_iters = 60 #100

config = dict(
    running_baseline=RUNNING_BASELINE,
    seed=seed,
    primary_device=primary_device,
    map_every=map_every, # Mapping every nth frame
    keyframe_every=keyframe_every, # Keyframe every nth frame
    mapping_window_size=mapping_window_size, # Mapping window size
    report_global_progress_every=500, # Report Global Progress every nth frame
    eval_every=5, # Evaluate every nth frame (at end of SLAM)
    scene_radius_depth_ratio=3, # Max First Frame Depth to Scene Radius Ratio (For Pruning/Densification)
    mean_sq_dist_method="projective", # ["projective", "knn"] (Type of Mean Squared Distance Calculation for Scale of Gaussians)
    report_iter_progress=False,
    load_checkpoint=False,
    checkpoint_time_idx=0,
    save_checkpoints=False, # Save Checkpoints
    checkpoint_interval=500, # Checkpoint Interval
    save_timestamp_keyframes=False,
    data=dict(
        desired_image_height=400,
        desired_image_width=400,
        num_frames= NUM_FRAMES,
    ),
    tracking=dict(
        visualize_tracking_loss = True,
        use_gt_poses=True, #True, # Use GT Poses for Tracking, saving tracking time
        forward_prop=True, # Forward Propagate Poses
        num_iters=tracking_iters,
        use_sil_for_loss=True,
        sil_thres=0.9,
        use_l1=True,
        ignore_outlier_depth_loss=False, # Not working.
        loss_weights=dict(
            im=0.5,
            depth=1.0,
            seg=0.05,
            quality = 0.0,
            depth_2 = 0.5,
        ),
        lrs=dict(
            means3D=0.0,
            rgb_colors=0.0,
            unnorm_rotations=0.0,
            logit_opacities=0.0,
            log_scales=0.0,
            cam_unnorm_rots=0.0004,
            cam_trans=0.002,
            semantic_colors=0.0,
            rgb_loss = 0.0,
            means3D_2 = 0.0001,
            unnorm_rotations_2=0.0,
            logit_opacities_2=0.0,
            log_scales_2=0.0,
            
        ),
    ),
    mapping=dict(
        num_iters=mapping_iters,
        add_new_gaussians=True, #TODO CHECK
        fill_depth_holes=False, # Didn't really help. Keep false. Fill Depth Holes in the GT Depth Map before adding new Gaussians
        sil_thres=0.5, #0.5 For Addition of new Gaussians. Densify areas with silh. lower than this
        
        use_l1=True,
        use_sil_for_loss=False,
        ignore_outlier_depth_loss= False, #Not working.
        loss_weights=dict(
            im=1.0, #1.0, #0.5
            depth= 0.5, #0.5, #0.5, #0.25, #0.5,#1.0,
            seg=0.1,#0.1
            quality=0.1, #0.1,
            depth_2 = 0.5,
        ),
        lrs=dict(
            means3D= 0.00001,#0.0001,
            rgb_colors= 0.0025, #0.0025,
            unnorm_rotations=0.001,
            logit_opacities= 0.01,#0.05,
            log_scales=0.009,#0.001,
            cam_unnorm_rots=0.0000,
            cam_trans=0.0000,
            semantic_colors= 0.02,#0.02,#0.008,#0.0025,
            rgb_loss = 0.001,#0.0025,
            means3D_2 = 0.0001,
            unnorm_rotations_2=0.001,
            logit_opacities_2=0.05,
            log_scales_2=0.001,
        ),
        prune_gaussians=True, # Prune Gaussians during Mapping
        prune_background_gaussians= not RUNNING_BASELINE, # Prune Background Gaussians during Mapping
        pruning_dict=dict( # Needs to be updated based on the number of mapping iterations
            start_after=1,
            remove_big_after=0,
            stop_after=20000, #20,
            prune_every=59,
            removal_opacity_threshold=0.4,
            final_removal_opacity_threshold=0.4,
            reset_opacities=False,
            reset_opacities_every=500, # Doesn't consider iter 0
        ),
        use_gaussian_splatting_densification=True, # Use Gaussian Splatting-based Densification during Mapping
        densify_dict=dict( # Needs to be updated based on the number of mapping iterations
            start_after=1, #500,
            remove_big_after=0,
            stop_after=9999999,
            densify_every= 55,#100,
            grad_thresh=0.0004 if RUNNING_BASELINE else 0.0002, # limiting densification for baseline to avoid memory issuess
            num_to_split_into=2,
            removal_opacity_threshold=0.4,
            final_removal_opacity_threshold=0.4,
            reset_opacities=False,
            reset_opacities_every=3000, # Doesn't consider iter 0
        ),
    ),
    active_mapping=dict(
        data_mode='octosplat_dataset', # Adjusts Intrinsics and other data-related parameters based on the dataset being used for active mapping
        octosplat_dataset=dict(
            output_dir= output_dir,
            max_depth=1.0,
            confidence_threshold=0.4, # reliable confidence threshold

        ),
    ),
)