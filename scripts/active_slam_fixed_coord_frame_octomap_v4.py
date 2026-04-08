'''
Tests using the entropy of the silhouette (works fine)
Test rendering RGB losses (works fine)
Test rendering entropy from opacities. (they are overestimated :())
Fixing coordinate frames.
Integrate octomap for active mapping.
Compute quality loss in terms of rgb_loss and sillouete
Compute all loses (reconstruction and quality loss) directly from sgs slam
Compute viewpoint candidates directly from this script
Prune outlier semantics (in slam_external) after mapping iterations (This part was commented)
Add confidence of segmentation masks in loss function to tackle noise
Prune outlier semantics just at the end
Initialize params['semantic_colors'] in 0.5 (rather than current semantics) (check get_pointcloud())to reduce the effect of noise. After some optim. steps it should converge to some value
Clip params[semantic_colors'] between [0,1] (in add_new_gaussians)as other values dont make sense and reduce inertia during optimization.
Added a fixed depth error threshold parameters in add_new_gaussians for robustness
Distance threshold in osamcep changed to 0.15 and box_size_on_frame = 0.3 //leverage more information around clusters
Octomap was edited to keep all rays, regarding the color voxel, as this issue does not affect the new viewp. evaluation
Include number of times each gaussian has been optimized in the loss function, to detect stable and unstable gaussians.
SGS_slam functions were moved to a utils script (utils_sgs_slam.py)
Includes evaluation of the surface coverage
'''
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


import rospy
from gazebo_msgs.msg import LinkState, LinkStates
from geometry_msgs.msg import Pose, Twist, Point, Quaternion
import geometry_msgs.msg
from sensor_msgs.msg import Image, CompressedImage
from std_msgs.msg import Float32
from cv_bridge import CvBridge, CvBridgeError
import tf2_ros
import tf
import math
from scipy.spatial.transform import Rotation
import copy
from utils.utils_active_mapping import ViewpointEvaluation, get_semantic_image, SE3_to_ros_pose, ros_pose_to_SE3, dbscan_clustering
from utils.utils_data import load_gt_data
from datetime import datetime
from semantic_octomap.srv import *
from utils.utils import dilate_pytorch, add_gaussian_noise
from utils.utils_evaluation import compute_surface_coverage

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
    transformed_params2rendervar, filter_points_in_image, transformed_params2depth_silhouette_rgbloss, transformed_entropy2rendervar, transformed_params2depthplussilhouette,
    transformed_semantics2rendervar, transformed_rgb_loss_rendervar, transform_to_frame, transform_points_to_frame, l1_loss_v1, matrix_to_quaternion
)
from utils.slam_external import calc_ssim, build_rotation, prune_outlier_semantics, prune_gaussians, densify, prune_aux_gaussians

from diff_gaussian_rasterization import GaussianRasterizer as Renderer

from utils.utils_sgs_slam import (get_pointcloud, downsample_mask, get_initial_pointcloud, initialize_params,
                                  initialize_optimizer, initialize_first_timestep, initialize_new_params,
                                  add_new_gaussians, convert_params_to_store, initialize_camera_pose, get_loss)


class ActiveSLAM:
    def __init__(self, config):
        
        rospy.init_node('active_slam', anonymous=True)
        self.config = config
        self.bridge = CvBridge()
        self.T_camlink_camframe = np.array([[0.0, 0.0, 1.0, 0.0],
                                            [-1.0, 0.0, 0.0, 0.0],
                                            [0.0, -1.0, 0.0, 0.0],
                                            [0.0, 0.0, 0.0, 1.0]])

         # Subscribers
        rospy.Subscriber('/camera2/color/image_raw/compressed', CompressedImage, self.callback_image_raw)
        rospy.Subscriber('/camera2/depth/image_raw', Image, self.callback_depth_topic)
        rospy.Subscriber('/gazebo/link_states', LinkStates, self.callback_link_states)
        rospy.Subscriber("/new_image_data", Float32, self.callback_new_image_data) # new image data from nbv planning
        rospy.Subscriber("/octomap_status", Float32, self.callback_octomap_status)
        self.tf_listener = tf.TransformListener()
        
        # Publisher
        self.pub_link_state = rospy.Publisher('/gazebo/set_link_state', LinkState, queue_size=10)
        self.pub_semantic_image = rospy.Publisher('/camera2/color/semantics', Image, queue_size = 10)
        self.pub_rgb_image = rospy.Publisher('/camera2/color/rgb', Image, queue_size = 10)
        self.pub_depth_image = rospy.Publisher('/camera2/color/depth', Image, queue_size = 10)
        self.pub_reset_octomap = rospy.Publisher('/reset_octomap', Float32, queue_size = 10)
        

        self.cv_image = None
        self.cv_depth_image = None
        self.crop_size = 400
        self.gt_pose_w_camlink = None
        self.T_w_camframe = None
        self.semantic_img = None
        self.confidence_map = None
        self.depth_img = None
        
        
        
        self.fx, self.fy, self.cx, self.cy = 381.36246688113556, 381.36246688113556, self.crop_size/2 , self.crop_size/2
        # self.fx, self.fy, self.cx, self.cy = 381.36246688113556, 381.36246688113556, 320.5, 240.5 # 640x480
        self.K = np.array([[self.fx, 0, self.cx],
                            [0, self.fy, self.cy],
                            [0, 0, 1.0]])
        self.new_image_data = None
        self.sampling_r = 0.4
        self.map_res = 0.05
        self.T_wc_0 = None
        self.initialized  = False
        self.plant_ids = [3,4,5,6,7,8]
        self.plant_height = 1.0
        self.octomap_status = None
        self.bridge = CvBridge()
        self.add_seg_noise = False
        self.add_depth_noise = False
        self.viewpoint_count = 0
        self.best_viewpoints_list= []
        self.output_directory = '/home/jose/results_mapping/'

        rospy.wait_for_service('querry_RLE')
        try:
            self.RLE_query = rospy.ServiceProxy('querry_RLE', GetRLE, persistent=True)
            print("********************* RLE query Test succesfull **********************")
        except rospy.ServiceException as e:
            print("*********************RLE Service initialization failed: %s"%e)
        
        self.vp_evaluation = ViewpointEvaluation(self.RLE_query, phi=-0.1, psi=1.0)
        
        
        self.run_mapping_multiple_plants()
        # self.rgbd_slam(config)
        

        rate = rospy.Rate(0.5)
        
        while not rospy.is_shutdown():
            rate.sleep()
            print("Running...")
            

    
    def octomap_warmap(self):
        print("Octomap_warmap...")
        plant_centroid = [0,0,0.5]
        camlink_poses, camframe_poses = self.gen_cam_poses(0, 90, 30, 120, centroid=plant_centroid, theta_n_grid=2, phi_n_grid = 1, r=self.sampling_r)
        for pose in camlink_poses:
            print("Octomap warmap...")
            self.execute_single_camlink_goal(pose)
        self.pub_reset_octomap.publish(Float32(1))
        print("Resetting octomap")
        time.sleep(5)
        
    def callback_octomap_status(self, msg):
        self.octomap_status = msg.data

    def run_mapping_multiple_plants(self):
        
        self.max_iterations_per_plant = 5
        self.dist_threshold = 0.01 # for surface coverage evaluation

        #initial "warmup" for octomap
        self.octomap_warmap()
        
        
        for plant_id in self.plant_ids:
            self.plant_id = plant_id
            self.gt_centroids, self.xyz_plant_origin, self.gt_pointcloud_path = load_gt_data(plant_id)
            self.gt_pointcloud = np.loadtxt(self.gt_pointcloud_path)
            plant_centroid = copy.deepcopy(self.xyz_plant_origin)
            plant_centroid[2] = self.plant_height/2
            
            # Sample viewpoints aroung a plant for initialization
            self.camlink_viewpoints, _ = self.gen_cam_poses(0, 360, 45, 135, centroid=plant_centroid, theta_n_grid=10, phi_n_grid = 5, r=self.sampling_r)    
            
            method = "OSAMCEP"
            for i in range(self.max_iterations_per_plant):
                print("Current Plant:", plant_id, " pose:", i, " Method:", method)
                params_file_prefix = 'plant' + str(self.plant_id) + '_pose' + str(i) + '_' + method # for params file
                self.score_method = method
                self.previous_surface_coverage = 0
                self.initial_pose_id = i
                self.viewpoint_count = 0
                self.surface_coverage = []
                self.rgbd_slam(self.config, params_file_prefix)

                self.pub_reset_octomap.publish(Float32(1))
                time.sleep(5)


    def execute_single_camlink_goal(self, pose):
        # self.show_visibility_map(pose)

        self.publish_link_pose(pose,'link_kinect')
        time.sleep(1)
        # input("Press enter to publish main topics:")

        image2 = copy.deepcopy(self.cv_image)
        depth_image2 = copy.deepcopy(self.cv_depth_image)
        depth_image2 = np.where(np.isnan(depth_image2), 3.0, depth_image2)
        current_pose = copy.deepcopy(self.gt_pose_w_camlink)
        self.octomap_status = None
        self.publish_main_topics(image2, depth_image2, current_pose)
        max_time_octomap = 4 # TODO to finetune
        wait_time = 0
        # while ((self.octomap_status==None or self.sgs_status == None) and wait_time < max_time_octomap):
        while (self.octomap_status==None and wait_time < max_time_octomap):
            wait_time += 0.3
            time.sleep(0.3) # to update octomap and sgs slam
            print("waiting for octomap or sgs update")
        print("viewpoint count:", self.viewpoint_count)
        self.viewpoint_count += 1

    def publish_main_topics(self, rgb, depth, pose_w_camlink):
        current_time_stamp = rospy.Time.now()
        
        rgb_img_msg = self.bridge.cv2_to_imgmsg(rgb, encoding='bgr8')
        rgb_img_msg.header.stamp = current_time_stamp
        rgb_img_msg.header.frame_id = 'camera2_frame'
        self.pub_rgb_image.publish(rgb_img_msg)

        self.semantic_img, _, __ , self.confidence_map = get_semantic_image(rgb, self.add_seg_noise, conf=True)
        semantic_img_msg = self.bridge.cv2_to_imgmsg(self.semantic_img, encoding='bgr8')
        semantic_img_msg.header.stamp = current_time_stamp
        semantic_img_msg.header.frame_id = 'camera2_frame'
        self.pub_semantic_image.publish(semantic_img_msg)

        if self.add_depth_noise:
            self.depth_img = add_gaussian_noise(depth, mean=0, std=0.02) #TODO add parameter
            self.depth_img = cv2.medianBlur(self.depth_img.astype(np.float32), 5)
        else:
            self.depth_img = depth
        depth_img_msg = self.bridge.cv2_to_imgmsg(depth, encoding='passthrough')
        depth_img_msg.header.stamp = current_time_stamp
        depth_img_msg.header.frame_id = 'camera2_frame'
        self.pub_depth_image.publish(depth_img_msg)

        T_w_camlink = ros_pose_to_SE3(pose_w_camlink)
        T_w_camframe = np.matmul(T_w_camlink, self.T_camlink_camframe)
        pose_w_camframe = SE3_to_ros_pose(T_w_camframe)

        # self.pub_new_image_data.publish(Float32(1.0))
        self.publish_tf(pose_w_camframe, current_time_stamp)

    def publish_tf(self, pose_w_camframe, current_time_stamp):

        tf_broadcaster = tf2_ros.TransformBroadcaster()
        transform_msg = geometry_msgs.msg.TransformStamped()
        transform_msg.header.frame_id = "world"
        transform_msg.child_frame_id = "camera2_frame"
        transform_msg.transform.translation = pose_w_camframe.position
        transform_msg.transform.rotation = pose_w_camframe.orientation
        transform_msg.header.stamp = current_time_stamp
        tf_broadcaster.sendTransform(transform_msg)

    def evaluate_viewpoint(self, params, centroid, curr_data, T_cw, device = 'cuda', visualize = False):
        '''
        T_cw (Numpy SE3) is the target camera pose wrt to first frame (inverse)
        curr_data['cam'] included intrinsics, H, W...
        curr_data['w2c'] seems to be the pose of the first frame wrt the wold (inverse)
        '''
        T_wc = np.linalg.inv(T_cw)
        # transformed_pts = transform_points_to_frame(params, T_cw, device=device)
        
        with torch.no_grad():
            
            # TODO: curr_data['w2c'] is not used. Remove
            # depth_sil_rgbloss_rendervar = transformed_params2depth_silhouette_rgbloss(params, curr_data['w2c'],
            #                                                         transformed_pts, device=device)
            # rgb_loss_rendervar = transformed_rgb_loss_rendervar(params, transformed_pts, device=device)
            # rgb_rendervar = transformed_params2rendervar(params, transformed_pts, device=device)
            # semantic_rendervar = transformed_semantics2rendervar(params, transformed_pts, device=device)
        
            
            # depth_sil, _, _, = Renderer(raster_settings=curr_data['cam'])(**depth_sil_rgbloss_rendervar)
            # rendered_depth = depth_sil[0, :, :]
            # silhouette = depth_sil[1, :, :]
            # rendered_rgb_loss = depth_sil[2, :, :]
            # H, W = silhouette.shape
            # rendered_rgb_loss, _, _, = Renderer(raster_settings=curr_data['cam'])(**rgb_loss_rendervar)
            # rendered_rgb_loss = rendered_rgb_loss.mean(dim=0)
            # rgb, _, _, = Renderer(raster_settings=curr_data['cam'])(**rgb_rendervar)
            # rgb = torch.clip(rgb.permute(1, 2, 0), 0, 1)
            
            # rendered_seg, _, _, = Renderer(raster_settings=curr_data['cam'])(**semantic_rendervar)
            # rendered_seg = torch.clip(rendered_seg.permute(1, 2, 0), 0, 1)

            # This raycasting line takes so long. TODO check
            # visibility_map = self.vp_evaluation.get_visibility_map(T_w_camframe=T_wc, fx = self.fx, fy=self.fy,
            #                                                     sampling_r=self.sampling_r+0.5, max_range=1.2, map_res=self.map_res,
            #                                                     box_size_on_frame=self.crop_size)
            # visibility_map_adjusted = cv2.resize(visibility_map, (W,H), interpolation = cv2.INTER_NEAREST)
            # visibility_map_adjusted = torch.from_numpy(visibility_map_adjusted).to(device)
            # visibility_map_adjusted = torch.ones_like(silhouette).to(device)
            # Reconstruction gain
            reconstruction_gain = self.vp_evaluation.compute_viewpoint_score_v2(T_w_camframe=T_wc, centroid=centroid,
                                        method='OSAMCEP', sampling_rad=self.sampling_r, box3d_size=0.3,
                                        sample_resol_3d=self.map_res, fx=self.fx, fy=self.fy)

            # # semantic target max
            # sem_target = torch.tensor([1,0,0]).to(device)
            # sem_error = torch.linalg.norm(rendered_seg - sem_target, axis=2)/math.sqrt(3)
            # sem_mask = sem_error < 0.01 # TODO add parameter
            # sem_mask = sem_mask.to(torch.float32)

            # sem_mask = dilate_pytorch(sem_mask, kernel_size=7)

            # # target aware information
            # # silhouette > 0 condition to discard free space as potential information
            # rgb_info = (rendered_rgb_loss * sem_mask * visibility_map_adjusted).sum().item()
            # silhouette_info = ((1-silhouette)*sem_mask*visibility_map_adjusted*(silhouette>0)).sum().item()
            
            rgb_info = None
            silhouette_info = None
            # if visualize:
            #     fig, ax = plt.subplots(2, 4, figsize=(12, 6))
            #     ax[0, 0].imshow(rgb.cpu())
            #     ax[0, 0].set_title("RGB")
                
            #     ax[0, 1].imshow(rendered_rgb_loss.cpu())
            #     ax[0, 1].set_title("RGB LOSS")
                
            #     ax[0, 2].imshow(((1-silhouette)*(silhouette>0)).cpu())
            #     ax[0, 2].set_title("1-Silhouette")
                
            #     ax[0, 3].imshow(visibility_map_adjusted.cpu())
            #     ax[0, 3].set_title("Vis.Map")
                
            #     ax[1, 0].imshow(rendered_depth.cpu())
            #     ax[1, 0].set_title("Rend.Depth")

                
            #     ax[1, 1].imshow(rendered_seg.cpu())
            #     ax[1, 1].set_title("Rend.Semantics")

            #     ax[1, 2].imshow(sem_error.cpu())
            #     ax[1, 2].set_title("sem. Error")

            #     ax[1, 3].imshow(sem_mask.cpu())
            #     ax[1, 3].set_title("Dil.sem.mask")
                
                
            #     # Turn off axis
            #     for i in range(2):
            #         for j in range(4):
            #             ax[i, j].axis('off')
            #     plt.tight_layout()
            #     plt.show()
        
        return reconstruction_gain, rgb_info, silhouette_info

    def evaluation(self, data_points, dist_threshold):
        output_dir = self.output_directory
        file_suffix = '_plant' + str(self.plant_id) + '_pose' + str(self.initial_pose_id) + '_' + self.score_method + '.txt'
        if data_points is not None:
            if data_points.shape[0] > 0:
                surface_coverage = compute_surface_coverage(self.gt_pointcloud, data_points, offset = self.xyz_plant_origin, dist_threshold=dist_threshold)
            # np.savetxt(output_dir + 'pointcloud' + file_suffix, data_points, fmt='%.8f')
            else:
                surface_coverage = self.previous_surface_coverage
        else:
            surface_coverage = self.previous_surface_coverage
        
        self.surface_coverage.append(surface_coverage)
        self.previous_surface_coverage = surface_coverage
        
        np.savetxt(output_dir + 'surface_coverage' + file_suffix, np.array(self.surface_coverage), fmt='%.8f')
        
        print("current plant model:", file_suffix)
        print("Current surf. coverage:", surface_coverage)
    
    def compute_viewpoints(self):
        plant_id = 5 #7
        plant_height = 1.0
        
        self.gt_centroids, self.xyz_plant_origin, self.gt_pointcloud_path = load_gt_data(plant_id)
        self.gt_pointcloud = np.loadtxt(self.gt_pointcloud_path)
        plant_centroid = copy.deepcopy(self.xyz_plant_origin)
        plant_centroid[2] = plant_height/2
        # camlink poses for different samples
        camlink_poses, camframe_poses = self.gen_cam_poses(0, 360, 45, 60, centroid=plant_centroid, theta_n_grid=36, phi_n_grid = 2, r=self.sampling_r)    

        return camlink_poses, camframe_poses    

  
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
    def callback_new_image_data(self, msg):
        self.new_image_data = msg.data

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
            self.T_wc_0 = self.T_w_camframe
            self.initialized = True
        T_wc_rel = self.T_w_camframe
        
        T_wc_rel = torch.from_numpy(T_wc_rel)

        bgr_image = copy.deepcopy(self.cv_image)
        rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
        rgb_image = rgb_image.astype(float) #/255
        rgb_image = torch.from_numpy(rgb_image)

        # semantic_img_uint8_bgr, _, __ = get_semantic_image(bgr_image, add_seg_noise = self.add_seg_noise)
        semantic_img_uint8_bgr = self.semantic_img
        semantic_img_uint8_rgb = cv2.cvtColor(semantic_img_uint8_bgr, cv2.COLOR_BGR2RGB)
        semantic_img_float = semantic_img_uint8_rgb.astype(float) #/255
        semantic_img = torch.from_numpy(semantic_img_float)

        semantic_id = cv2.cvtColor(semantic_img_uint8_bgr, cv2.COLOR_BGR2GRAY)
        semantic_id = semantic_id.astype(float)
        print("Semantic ids:", np.unique(semantic_id))
        semantic_id = np.expand_dims(semantic_id, -1)#(h,w,1)
        semantic_id = torch.from_numpy(semantic_id)

        # depth = self.cv_depth_image.astype(float) # m
        depth = self.depth_img
        # depth = np.where(np.isnan(depth), 0.0, depth)
        depth = np.expand_dims(depth, -1) #(h,w,1)
        depth = torch.from_numpy(depth)
        depth = torch.nan_to_num(depth, nan=0.0)
        depth[depth>1.0] = 0.0

        confidence_map = torch.from_numpy(self.confidence_map)
        return_data = (
            rgb_image.to(device).type(dtype),
            depth.to(device).type(dtype),
            intrinsics.to(device).type(dtype),
            T_wc_rel.to(device).type(dtype),
            semantic_id.to(device).type(dtype),
            semantic_img.to(device).type(dtype),
            confidence_map.to(device).type(dtype),
        )
        self.new_image_data = None
        return return_data

    def rgbd_slam(self, config: dict, output_file_prefix: str):
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
        
        # Get Device
        device = torch.device(config["primary_device"])
        if config["primary_device"].startswith("cuda:"):
            device_id = int(config["primary_device"].split(':')[1])
            torch.cuda.set_device(device_id)

        # Load Dataset
        print("Loading Dataset Config...")
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
        
        load_semantics = True
        num_frames = dataset_config["num_frames"]
        
        random_viewpoint_id = np.random.randint(0, len(self.camlink_viewpoints))
        self.execute_single_camlink_goal(self.camlink_viewpoints[random_viewpoint_id])
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

        checkpoint_time_idx = 0
        
        # Iterate over Scan
        for time_idx in tqdm(range(checkpoint_time_idx, num_frames)):
            # Load RGBD frames incrementally instead of all frames
            # print("Semantic colors:", params['semantic_colors'][0:100, :])
            print("Current time idx:", time_idx)
            print("Number of gaussians:", params['means3D'].shape[0])
            target_gaussians_3D = None
            if time_idx>0:
                # self.execute_single_camlink_goal(self.camlink_viewpoints[time_idx])
                # time.sleep(0.5)
            
                sem_target = torch.tensor([1.0,0,0]).to(device) #red
                rmse = torch.linalg.norm(sem_target - params['semantic_colors'].clip(0,1), axis=1)/math.sqrt(3)
                # cos_similarity = F.cosine_similarity(sem_target, params['semantic_colors'],dim=1)
                
                sem_mask = (rmse < 0.01) # TODO add parameters
                stable_gaussians_mask = (params['opt_count'] > 200) # TODO add parameters
                target_sem_mask = sem_mask & stable_gaussians_mask

                n_sem_gaussians = sem_mask.sum().item()
                n_stable_sem_gaussians = target_sem_mask.sum().item()
                
                print("number of sem gaussians:", n_sem_gaussians)
                print("Number of stable sem gaussians:", n_stable_sem_gaussians)

                target_gaussians_3D = params['means3D'][target_sem_mask].detach().cpu().numpy()
                # np.savetxt("/home/jose/gaussians.txt", target_gaussians_3D)

                if target_gaussians_3D.shape[0] > 0:
                    # evaluate surface coverage
                    # surface_coverage = compute_surface_coverage(self.gt_pointcloud, target_gaussians_3D,
                    #                                             offset = self.xyz_plant_origin, dist_threshold= 0.01)

                    sem_centroids = dbscan_clustering(target_gaussians_3D, eps_= 0.02, min_samples=100) # 250
                    sem_centroids_w = sem_centroids
                    print("world sem centroids:\n", sem_centroids_w)
                    # generate candidate viewpoints
                    cand_camframe_poses = [] #wrt world frame
                    cand_camlink_poses = []
                    centroids_list = []
                    if (sem_centroids_w.shape[0] > 0):
                        
                        for sem_centroid in sem_centroids_w.tolist():
                            camlink_poses, camframe_poses = self.gen_cam_poses(0, 360, 30, 135,
                                                            centroid=sem_centroid, theta_n_grid=12, phi_n_grid = 5, r=self.sampling_r)    
                            cand_camframe_poses = cand_camframe_poses + camframe_poses
                            cand_camlink_poses = cand_camlink_poses + camlink_poses
                            for i in range(len(camframe_poses)):
                                centroids_list.append(sem_centroid)
                        scores = []
                        rec_info_list = []
                        rgb_info_list = []
                        silhouette_info_list = []
                        start_time = time.time()
                        for centroid, cand_camframe_pose in zip(centroids_list, cand_camframe_poses):
                            T_w_cx = ros_pose_to_SE3(cand_camframe_pose)
                            T_cw = np.linalg.inv(T_w_cx)
                            reconstruction_info, rgb_info, silhouette_info = self.evaluate_viewpoint(params, centroid, curr_data=curr_data, T_cw = T_cw, visualize = False)
                            rec_info_list.append(reconstruction_info)
                            rgb_info_list.append(rgb_info)
                            silhouette_info_list.append(silhouette_info)
                            end_time = time.time()
                        
                        
                        print("Total evaluation time:", end_time - start_time)
                        
                        rec_info = np.array(rec_info_list)
                        scores = rec_info
                        sorted_indices = np.argsort(scores)[::-1].tolist()
                        sorted_scores = scores[sorted_indices]
                        print("\nNumber of evaluated viewpoints:", len(sorted_indices))
                        print("Absolute best recon scores:", np.max(rec_info))
                        print("***Best scores:", sorted_scores[0:3])
                        #visualize best viewpoint
                        T_wc = ros_pose_to_SE3(cand_camframe_poses[sorted_indices[0]])
                        centroid = centroids_list[sorted_indices[0]]
                        T_cw = np.linalg.inv(T_wc)
                        _, _, _ = self.evaluate_viewpoint(params, centroid, curr_data=curr_data, T_cw = T_cw, visualize = False)
                        ########### end visualization

                        sorted_camlink_poses = [cand_camlink_poses[i] for i in sorted_indices]
                        sorted_camframe_poses = [cand_camframe_poses[i] for i in sorted_indices]
                        self.execute_single_camlink_goal(sorted_camlink_poses[0])
                        time.sleep(0.5)
                    else:
                        print("No sem centroids found. Executing random viewpoint")
                        random_viewpoint_id = np.random.randint(0, len(self.camlink_viewpoints))
                        self.execute_single_camlink_goal(self.camlink_viewpoints[random_viewpoint_id])
                        time.sleep(0.5)
                else:
                    print("No semantic targets found. Executing random viewpoint")
                    random_viewpoint_id = np.random.randint(0, len(self.camlink_viewpoints))
                    self.execute_single_camlink_goal(self.camlink_viewpoints[random_viewpoint_id])
                    time.sleep(0.5)
            # input("Press enter to continue")
            self.evaluation(target_gaussians_3D, dist_threshold=self.dist_threshold)
                         

            color, depth, _, gt_pose, semantic_id, semantic_color, confidence_map = self.get_sample_data() #dataset[time_idx], jrcv
            
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
            
            # T_cw = gt_w2c.detach().cpu().numpy()
            # self.evaluate_viewpoint(params, curr_data, T_cw)

            
            semantic_id = semantic_id.permute(2, 0, 1)
            semantic_color = semantic_color.permute(2, 0, 1) / 255
            curr_data['semantic_id'] = semantic_id
            curr_data['semantic_color'] = semantic_color
            curr_data['confidence_map'] = confidence_map
            
            # Initialize Data for Tracking
            tracking_curr_data = curr_data

            # Optimization Iterations
            
            num_iters_mapping = config['mapping']['num_iters']
            if time_idx == num_frames-1: 
                num_iters_mapping = 2*num_iters_mapping #jrcv, to refine the optimization in the last frame
            
            if time_idx >= 0:
                with torch.no_grad():
                    # initialization based on constant velocity model
                    # params = initialize_camera_pose(params, time_idx,
                    #                             forward_prop=config['tracking']['forward_prop'])
                    # # intialize with ground truth for now # TODO comment
                    w2c_init = curr_gt_w2c[-1].detach().cpu().numpy()
                    SE3_error = np.eye(4)
                    
                    # euler_error = ((5*np.pi/180)*np.random.rand(3)).tolist()
                    # trans_error = 0.05*np.random.rand(3)
                    # rot_error = Rotation.from_euler('xyz',euler_error).as_matrix()
                    # SE3_error[0:3,0:3] = rot_error
                    # SE3_error[0:3,3] = trans_error
                    w2c_init_plus_error = np.matmul(w2c_init, SE3_error)
                    params = initialize_camera_pose(params, time_idx,
                                                forward_prop=config['tracking']['forward_prop'],
                                                rel_w2c_initial_guess = w2c_init)
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
                    # Backprop
                    loss.backward()
                    # Optimizer Update
                    
                    # optimizer.step() # TODO uncomment this, jrcv
                    optimizer.zero_grad(set_to_none=True) 
                    
                    with torch.no_grad():
                        # Save the best candidate rotation & translation
                        if loss < current_min_loss:
                            current_min_loss = loss
                            candidate_cam_unnorm_rot = params['cam_unnorm_rots'][..., time_idx].detach().clone()
                            candidate_cam_tran = params['cam_trans'][..., time_idx].detach().clone()
                        # Report Progress
                        if config['report_iter_progress']:
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
                        else:
                            break

                progress_bar.close()
                # Copy over the best candidate rotation & translation
                with torch.no_grad():
                    pass #TODO REMOVE
                    # params['cam_unnorm_rots'][..., time_idx] = candidate_cam_unnorm_rot
                    # params['cam_trans'][..., time_idx] = candidate_cam_tran
            elif time_idx > 0: #and config['tracking']['use_gt_poses']: #TODO change
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
                        report_progress(params, tracking_curr_data, 1, progress_bar, iter_time_idx, sil_thres=config['tracking']['sil_thres'],
                                            tracking=True, device=device, load_semantics=load_semantics)
                    progress_bar.close()
                except:
                    ckpt_output_dir = os.path.join(config["workdir"], config["run_name"])
                    save_params_ckpt(params, ckpt_output_dir, time_idx)
                    print('Failed to evaluate trajectory.')

            # Step 2: Densification & KeyFrame-based Mapping
            if time_idx == 0 or (time_idx+1) % config['map_every'] == 0:
                # Densification
                if config['mapping']['add_new_gaussians'] and time_idx >0:
                    # Setup Data for Densification
                    densify_curr_data = curr_data

                    # Add new Gaussians to the scene based on the Silhouette
                    params, variables = add_new_gaussians(params, params_opt_exclude, variables, densify_curr_data, 
                                                        config['mapping']['sil_thres'], time_idx, config['mean_sq_dist_method'],
                                                        device, load_semantics=load_semantics)
                    
                    
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
                    # if time_idx ==0 and iter > 5: # jrcv test
                    #     break
                    iter_start_time = time.time()
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
                        iter_time_idx = keyframe_list[selected_rand_keyframe_idx]['id']
                        iter_color = keyframe_list[selected_rand_keyframe_idx]['color']
                        iter_depth = keyframe_list[selected_rand_keyframe_idx]['depth']
                        iter_confidence_map = keyframe_list[selected_rand_keyframe_idx]['confidence_map']
                    iter_gt_w2c = gt_w2c_all_frames[:iter_time_idx+1]
                    iter_data = {'cam': cam, 'im': iter_color, 'depth': iter_depth, 'confidence_map': iter_confidence_map, 'id': iter_time_idx, 
                                'intrinsics': intrinsics, 'w2c': first_frame_w2c, 'iter_gt_w2c_list': iter_gt_w2c}
                    # Add semantic id and colors
                    
                    if selected_rand_keyframe_idx == -1:
                        iter_data['semantic_id'] = semantic_id
                        iter_data['semantic_color'] = semantic_color
                    else:
                        iter_data['semantic_id'] = keyframe_list[selected_rand_keyframe_idx]['semantic_id']
                        iter_data['semantic_color'] = keyframe_list[selected_rand_keyframe_idx]['semantic_color']
                    # Loss for current frame
                    
                    if iter % 10 == 0:
                        visualization = True
                    else:
                        visualization = False
                    
                    # visualization = False
                    loss, variables, losses = get_loss(params, iter_data, variables, iter_time_idx, config['mapping']['loss_weights'],
                                                    config['mapping']['use_sil_for_loss'], config['mapping']['sil_thres'],
                                                    config['mapping']['use_l1'], config['mapping']['ignore_outlier_depth_loss'],
                                                    mapping=True, device=device, plot_dir = eval_dir, load_semantics=load_semantics, visualization = visualization)
                    
                    # Backprop
                    loss.backward()
                    with torch.no_grad():
                        # Prune Gaussians
                        
                        if config['mapping']['prune_gaussians']:
                            params, variables = prune_gaussians(params, params_opt_exclude, variables, optimizer, iter, config['mapping']['pruning_dict'])
                            # if iter == num_iters_mapping - 1:
                            #     params, variables = prune_outlier_semantics(params, params_opt_exclude, variables, optimizer)
                            
                        # Gaussian-Splatting's Gradient-based Densification
                        if config['mapping']['use_gaussian_splatting_densification']:
                            params, variables = densify(params, variables, optimizer, iter, config['mapping']['densify_dict'], params_opt_exclude, device=device)
                            
                        # Optimizer Update

                        optimizer.step()
                        optimizer.zero_grad(set_to_none=True)
                        
                        #test clipping semantic colors jrcv, TODO Check
                        # if  iter % 10 ==0:
                        #     np.savetxt('/home/jose/params.txt',params['semantic_colors'].detach().cpu().numpy())
                        #     np.savetxt('/home/jose/means3D.txt',params['means3D'].detach().cpu().numpy())

                        #     input("Press enter to continue")
                        # Report Progress
                        if config['report_iter_progress']:
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
                    curr_keyframe['semantic_id'] = semantic_id
                    curr_keyframe['semantic_color'] = semantic_color
                    curr_keyframe['confidence_map'] = confidence_map
                    # Add to keyframe list
                    keyframe_list.append(curr_keyframe)
                    keyframe_time_indices.append(time_idx)
            
            # Checkpoint every iteration
            if time_idx % config["checkpoint_interval"] == 0 and config['save_checkpoints']:
                ckpt_output_dir = os.path.join(config["workdir"], config["run_name"])
                save_params_ckpt(params, ckpt_output_dir, time_idx)
                np.save(os.path.join(ckpt_output_dir, f"keyframe_time_indices{time_idx}.npy"), np.array(keyframe_time_indices))
            
            
            torch.cuda.empty_cache()

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
        
        
        # remove auxiliar params, jrcv
        # params, variables = prune_aux_gaussians(params, params_opt_exclude, variables, optimizer)
        print("********TEsting1")
        # params, variables = prune_outlier_semantics(params, params_opt_exclude, variables, optimizer)
        print("********TEsting2")
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

        params['semantic_ids'] = params['semantic_ids'].type(torch.uint8)
        
        # Save Parameters
        # save_params(params, output_dir)
        save_params(params, self.output_directory, save_ply=False, file_prefix=output_file_prefix)
        # input("Press enter finish")
        # Close WandB Run
        
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

