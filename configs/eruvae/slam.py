import os
from os.path import join as p_join

scenes = ["plant3"] #can add more scenes to this list, jrcv



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
        num_frames= 100, # Set to -1 to use all frames
        load_semantics=True,
        num_semantic_classes=3 # background, fruit, leaves
    ),
    tracking=dict(
        visualize_tracking_loss = True,
        use_gt_poses=False, #True, # Use GT Poses for Tracking, saving tracking time
        forward_prop=True, # Forward Propagate Poses
        num_iters=tracking_iters,
        use_sil_for_loss=True,
        sil_thres=0.9,
        use_l1=True,
        ignore_outlier_depth_loss=False,
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
        ignore_outlier_depth_loss= False, #False,
        loss_weights=dict(
            im=0.5, #0.5
            depth= 1.0,#1.0,
            seg=0.1,#0.1
            quality=0.1, #0.1,
            depth_2 = 0.5,
        ),
        lrs=dict(
            means3D= 0.0001,#0.0001,
            rgb_colors= 0.0025, #0.0025,
            unnorm_rotations=0.001,
            logit_opacities= 0.05,#0.05,
            log_scales=0.001,
            cam_unnorm_rots=0.0000,
            cam_trans=0.0000,
            semantic_colors= 0.008,#0.0025,
            rgb_loss = 0.001,#0.0025,
            means3D_2 = 0.0001,
            unnorm_rotations_2=0.001,
            logit_opacities_2=0.05,
            log_scales_2=0.001,
        ),
        prune_gaussians=True, # Prune Gaussians during Mapping
        pruning_dict=dict( # Needs to be updated based on the number of mapping iterations
            start_after=0,
            remove_big_after=0,
            stop_after=20,
            prune_every=20,
            removal_opacity_threshold=0.005,
            final_removal_opacity_threshold=0.005,
            reset_opacities=False,
            reset_opacities_every=500, # Doesn't consider iter 0
        ),
        use_gaussian_splatting_densification=False, # Use Gaussian Splatting-based Densification during Mapping
        densify_dict=dict( # Needs to be updated based on the number of mapping iterations
            start_after=50, #500,
            remove_big_after=3000,
            stop_after=5000,
            densify_every= 50,#100,
            grad_thresh=0.0002,
            num_to_split_into=2,
            removal_opacity_threshold=0.005,
            final_removal_opacity_threshold=0.005,
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
        using_real_robot=True,
        output_dir= '/home/companion/Documents/results_mapping/',
        real_robot=dict(
            rgb_topic = '/camera/color/image_raw/compressed',
            depth_topic = '/camera/depth/image_rect_raw',
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
            rgb_topic = '/realsense_gazebo_camera/color/image_raw/compressed',
            depth_topic = '/realsense_gazebo_camera/aligned_depth_to_color/image_raw',
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
    ),
)