import os
from os.path import join as p_join
import getpass
username = getpass.getuser()
print(f"Running as user: {username}")
if username == "temp":
    output_dir = '/mnt/ssd2T/sgs_results_mapping/'
    eval_data_folders = '/mnt/ssd2T/datasets/gaussian_splat_data/active_mapping_evaluation_2026/eval_data_folders'
elif username == "jose":
    output_dir = '/media/jose/SSD1G/results_mapping/'
    eval_data_folders = None
elif username == "companion":
    output_dir = '/mnt/ssd1T/results_mapping/'
    eval_data_folders = '/mnt/ssd1T/active_mapping_evaluation_2026/eval_data_folders'
else:
    raise ValueError(f"Unknown username: {username}. Please set the output_dir and eval_data_folders for this user.")

scenes = ["plant3"] #No used anymore. can add more scenes to this list, jrcv



primary_device="cuda:0"
seed = 0
scene_name = "plant3"

map_every = 1
keyframe_every = 1 #5 # 5 for slam, 1 for active mapping, jrcv
mapping_window_size = 24
tracking_iters = 40 #40
mapping_iters = 60 #100 #60

group_name = "eruvae"
run_name = f"{scene_name}_{seed}"

config = dict(
    workdir=f"./experiments/{group_name}",
    run_name=run_name,
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
    use_wandb=True,
    wandb=dict(
        entity="jrcuaranv-uiuc", # Please change the entity name
        project="SGS-SLAM",
        group=group_name,
        name=run_name,
        save_qual=False,
        eval_save_qual=True,
    ),
    data=dict(
        basedir="/media/jose/SSD1G/datasets/eruvae_gazebo_dataset",
        gradslam_data_cfg="./configs/data/eruvae.yaml",
        sequence=scene_name,
        desired_image_height=400,
        desired_image_width=400,
        start=0,
        end=-1,
        stride=1,
        num_frames= 80,#Set to -1 to use all frames
        load_semantics=True,
        num_semantic_classes=3 # background, fruit, leaves
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
        sil_thres=0.5, #0.5 For Addition of new Gaussians. Densify areas with silh. lower than this
        use_l1=True,
        use_sil_for_loss=False,
        ignore_outlier_depth_loss= False, #Not working.
        # loss_weights=dict(
        #     im=0.5, #0.5
        #     depth= 1.0,#1.0,
        #     seg=0.1,#0.1
        #     quality=0.1, #0.1,
        #     depth_2 = 0.5,
        # ),
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
        # prune_gaussians=True, # Prune Gaussians during Mapping
        # pruning_dict=dict( # Needs to be updated based on the number of mapping iterations
        #     start_after=0,
        #     remove_big_after=0,
        #     stop_after=20000, #20,
        #     prune_every=20,
        #     removal_opacity_threshold=0.005,
        #     final_removal_opacity_threshold=0.005,
        #     reset_opacities=False,
        #     reset_opacities_every=500, # Doesn't consider iter 0
        # ),
        prune_gaussians=True, # Prune Gaussians during Mapping
        prune_background_gaussians=False,
        reset_semantics=False, # Reset all semantic colors to 0.5 periodically during Mapping
        reset_semantics_every=200, # Doesn't consider iter 0
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
        # use_gaussian_splatting_densification=False, # Use Gaussian Splatting-based Densification during Mapping
        # densify_dict=dict( # Needs to be updated based on the number of mapping iterations
        #     start_after=50, #500,
        #     remove_big_after=3000,
        #     stop_after=5000,
        #     densify_every= 50,#100,
        #     grad_thresh=0.0002,
        #     num_to_split_into=2,
        #     removal_opacity_threshold=0.005,
        #     final_removal_opacity_threshold=0.005,
        #     reset_opacities=True,
        #     reset_opacities_every=3000, # Doesn't consider iter 0
        # ),
        use_gaussian_splatting_densification=True, # Use Gaussian Splatting-based Densification during Mapping
        densify_dict=dict( # Needs to be updated based on the number of mapping iterations
            start_after=1, #500,
            remove_big_after=0,
            stop_after=5000,
            densify_every= 55,#100,
            grad_thresh=0.00005,
            num_to_split_into=2,
            removal_opacity_threshold=0.4,
            final_removal_opacity_threshold=0.4,
            reset_opacities=False,
            reset_opacities_every=3000, # Doesn't consider iter 0
        ),
    ),
    viz=dict(
        render_mode='color', # ['color', 'depth', 'centers', 'semantic_color']
        offset_first_viz_cam=True, # Offsets the view camera back by 0.5 units along the view direction (For Final Recon Viz)
        show_sil=False, # Show Silhouette instead of RGB
        visualize_cams=True, # Visualize Camera Frustums and Trajectory
        viz_w=600, viz_h=340,
        viz_near=0.01, viz_far=100.0,
        view_scale=2,
        viz_fps=5, # FPS for Online Recon Viz
        enter_interactive_post_online=True, # Enter Interactive Mode after Online Recon Viz
        scene_name=scene_name,
        load_semantics=True, # Whether load semantic information
    ),
    active_mapping=dict(
        data_mode='colmap_dataset', # ['real_robot', 'gazebo_robot', 'colmap_dataset', 'rosbag_file'] (Adjusts Intrinsics and other data-related parameters based on the dataset being used for active mapping)
        real_robot=dict(
            output_dir= output_dir,
            apply_depth_median_filter = False,
            apply_statistical_outlier_filter = True,
            max_depth=1.5,
            crop_size = None,
            fx = 381.97095128993436,
            fy = 382.44965399519884,
            cx = 316.09723809,
            cy = 235.35205365,
            T_link6_camframe = [[ 0.01791203, -1.00004949,  0.00200584,  0.00739383],
                                [ 0.99971913,  0.01790737, -0.00507818, -0.00849351],
                                [ 0.00504143,  0.00209712,  0.99988083,  0.05245934],
                                [ 0.,          0.,          0.,          1.        ]]
        ),
        gazebo_robot=dict(
            output_dir= output_dir, # for jose user
            apply_depth_median_filter = False,
            apply_statistical_outlier_filter = False,
            max_depth=1.5,
            crop_size = None,
            fx = 381.3624572753906,
            fy = 381.3624572753906,
            cx = 320.0, #200.0,
            cy = 240.0, #200.0,
            T_link6_camframe = [[0.0, -1.0, 0.0, 0.0],
                                [1.0, 0.0, 0.0, 0.01],
                                [0.0, 0.0, 1.0, 0.053],
                                [0.0, 0.0, 0.0, 1.0]]
        ),
        # adjust intrinsics depending on the dataset being used for active mapping
        colmap_dataset=dict(
            output_dir= output_dir,
            eval_data_folders = eval_data_folders,
            # datasets in temp user's directory
            # colmap_scene_path = '/mnt/ssd2T/datasets/gaussian_splat_data/2026-05-13-11-03-47_greenhouse',
            # colmap_scene_path = '/mnt/ssd2T/datasets/gaussian_splat_data/2026-06-18_12-53-29_gazebo_peppers',
            colmap_scene_path = None, # This will be set from the command line argument in the script
            
            # datasets in jose user's directory
            # colmap_scene_path = '/media/jose/SSD1G/datasets/active_mapping_2025_data/zed2i_lab_data/2026-05-09-12-28-20_lab_trees_postprocessed',
            # colmap_scene_path = '/media/jose/SSD1G/datasets/active_mapping_2025_data/zed2i_lab_data/2026-05-13-11-03-47_greenhouse',
            # colmap_scene_path = '/media/jose/SSD1G/datasets/active_mapping_2025_data/zed2i_lab_data/2026-05-27-12-38-08_greenhouse',
            # colmap_scene_path = '/media/jose/SSD1G/datasets/active_mapping_results/2026-05-25_10-46-08',
            # colmap_scene_path = '/media/jose/SSD1G/datasets/active_mapping_results/2026-05-26_10-38-11',
            # colmap_scene_path = '/media/jose/SSD1G/datasets/active_mapping_results/2026-05-27_11-01-28_greenhouse_robot',
            apply_depth_median_filter = False,
            apply_statistical_outlier_filter = False,
            max_depth=1.0,
            crop_size = None,
            T_link6_camframe = None, # Not needed for colmap dataset since poses are already in the camera frame
        ),
        # adjust intrinsics depending on the dataset being used for active mapping
        rosbag_file=dict(
            output_dir= output_dir,
            apply_depth_median_filter = False,
            apply_statistical_outlier_filter = True,
            max_depth=1.5,
            crop_size = None,
            fx = 381.97095128993436,
            fy = 382.44965399519884,
            cx = 316.09723809,
            cy = 235.35205365,
            T_link6_camframe = [[ 0.01791203, -1.00004949,  0.00200584,  0.00739383],
                                [ 0.99971913,  0.01790737, -0.00507818, -0.00849351],
                                [ 0.00504143,  0.00209712,  0.99988083,  0.05245934],
                                [ 0.,          0.,          0.,          1.        ]]
        ),
    ),
)