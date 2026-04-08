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
Includes active mapping with husky robot
Derived fro active_slam_husky.py. Adapted to xarm manipulator
Params were added to config file (slam.py)
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
import sensor_msgs.point_cloud2 as pc2
from sensor_msgs.msg import PointCloud2, PointField
from std_srvs.srv import Empty, EmptyResponse
import math
from scipy.spatial.transform import Rotation
import copy
from utils.utils_active_mapping import ViewpointEvaluation, get_semantic_image, SE3_to_ros_pose, ros_pose_to_SE3, dbscan_clustering
from utils.utils_data import load_gt_data
from datetime import datetime
# from semantic_octomap.srv import *
from utils.utils import dilate_pytorch, add_gaussian_noise
from utils.utils_evaluation import compute_surface_coverage
import yaml
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
# from utils.eval_helpers import report_loss, report_progress, eval
# from utils.eval_helpers import report_progress
from utils.keyframe_selection import keyframe_selection_overlap
from utils.recon_helpers import setup_camera
from utils.slam_helpers import (
    transformed_params2rendervar, filter_points_in_image, transformed_params2depth_silhouette_rgbloss, transformed_entropy2rendervar, transformed_params2depthplussilhouette,
    transformed_semantics2rendervar, transformed_rgb_loss_rendervar, transform_to_frame, transform_points_to_frame, l1_loss_v1, matrix_to_quaternion
)
from utils.slam_external import calc_ssim, build_rotation, prune_outlier_semantics, prune_gaussians, densify, prune_aux_gaussians

from diff_gaussian_rasterization import GaussianRasterizer as Renderer

from utils.utils_sgs_slam import (render_any_cam, get_pointcloud, downsample_mask, get_initial_pointcloud, initialize_params,
                                  initialize_optimizer, initialize_first_timestep, initialize_new_params,
                                  add_new_gaussians, convert_params_to_store, initialize_camera_pose, get_loss)

def create_pointcloud2(points, frame_id="map"):
    """
    Create a PointCloud2 message from a Nx3 numpy array.
    """
    fields = [
        PointField(name="x", offset=0, datatype=PointField.FLOAT32, count=1),
        PointField(name="y", offset=4, datatype=PointField.FLOAT32, count=1),
        PointField(name="z", offset=8, datatype=PointField.FLOAT32, count=1)
    ]
    
    header = rospy.Header()
    header.stamp = rospy.Time.now()
    header.frame_id = frame_id
    
    return pc2.create_cloud(header, fields, points)

class ActiveSLAM:
    def __init__(self, config):
        
        rospy.init_node('active_slam', anonymous=True)
        self.config = config
        self.bridge = CvBridge()

        # just a transform for convenience.
        self.T_camlink_camframe = np.array([[0.0, 0.0, 1.0, 0.0],
                                            [-1.0, 0.0, 0.0, 0.0],
                                            [0.0, -1.0, 0.0, 0.0],
                                            [0.0, 0.0, 0.0, 1.0]])

        
        if config['active_mapping']['using_real_robot'] == True:
            prefix = 'real_robot'
        else:
            prefix = 'gazebo_robot'
        
        self.output_directory = config['active_mapping']['output_dir']
        # rgb_topic = config['active_mapping'][prefix]['rgb_topic']
        # depth_topic = config['active_mapping'][prefix]['depth_topic']
        rgb_topic = '/camera2/color/rgb'
        depth_topic = '/camera2/color/depth'
        semantics_topic = '/camera2/color/semantics'
        confidence_topic = '/camera2/color/confidence'
        self.crop_size = config['active_mapping'][prefix]['crop_size']
        self.fx = config['active_mapping'][prefix]['fx']
        self.fy = config['active_mapping'][prefix]['fy']
        self.cx = config['active_mapping'][prefix]['cx']
        self.cy = config['active_mapping'][prefix]['cy']
        self.T_link6_camframe = np.array(config['active_mapping'][prefix]['T_link6_camframe'])

        # rospy.Subscriber(rgb_topic, CompressedImage, self.callback_image_raw)
        # rospy.Subscriber(depth_topic, Image, self.callback_depth_topic)
        rospy.Subscriber(depth_topic, Image, self.callback_depth_topic)
        rospy.Subscriber(rgb_topic, Image, self.callback_rgb_topic)
        rospy.Subscriber(semantics_topic, Image, self.callback_semantic_topic)
        rospy.Subscriber(confidence_topic, Image, self.callback_confidence_topic)
        # rospy.Subscriber('/gazebo/link_states', LinkStates, self.callback_link_states)
        rospy.Subscriber("/new_image_data", Float32, self.callback_new_image_data) # new image data from nbv planning
        rospy.Subscriber("/octomap_status", Float32, self.callback_octomap_status)
        rospy.Subscriber("/camera2/color/pose", Pose, self.callback_camera_pose) # just to trigger the callback and update the transform
        self.tf_listener = tf.TransformListener()
        
        
        # Publisher
        self.pub_link_state = rospy.Publisher('/gazebo/set_link_state', LinkState, queue_size=10)
        self.pub_reset_octomap = rospy.Publisher('/reset_octomap', Float32, queue_size = 10)
        self.pub_sem_centroids = rospy.Publisher("/semantic_centroids_gs", PointCloud2, queue_size=10)
        self.pub_gs_status = rospy.Publisher("/gs_status", Float32, queue_size=10)

        # server to save params
        self.save_params_server = rospy.Service("/sgs/save_params", Empty, self.save_params_callback)
        rospy.loginfo("Service '/sgs/save_params' is ready.")
    
        self.bgr_image = None
        self.depth_image = None
        self.semantic_image = None
        self.confidence_image = None
        self.new_image_data = None
        self.camera_pose = None
        

        self.gt_pose_w_camlink = None
        self.T_w_camframe = None
        
        
        self.K = np.array([[self.fx, 0, self.cx],
                            [0, self.fy, self.cy],
                            [0, 0, 1.0]])
        
        self.octomap_status = None
        self.bridge = CvBridge()
        self.add_seg_noise = False
        self.add_depth_noise = False
        self.viewpoint_count = 0
        self.best_viewpoints_list= []
        self.params_copy = None

        # rospy.wait_for_service('querry_RLE')
        # try:
        #     self.RLE_query = rospy.ServiceProxy('querry_RLE', GetRLE, persistent=True)
        #     print("********************* RLE query Test succesfull **********************")
        # except rospy.ServiceException as e:
        #     print("*********************RLE Service initialization failed: %s"%e)
        
        # self.vp_evaluation = ViewpointEvaluation(self.RLE_query, phi=-0.1, psi=1.0)
        
        
        # self.run_mapping_multiple_plants()
        self.params_file_prefix = "xarm_"
        self.rgbd_slam(self.config, self.params_file_prefix)

        # self.rgbd_slam(config)
        

        rate = rospy.Rate(0.5)
        
        while not rospy.is_shutdown():
            rate.sleep()
            print("Running...")
            

    def save_params_callback(self, req):
        rospy.loginfo("Saving parameters...")
        # Get current time
        now = datetime.now()
        timestamp = now.strftime("%Y-%m-%d-%H-%M-%S")
        file_prefix = timestamp + '_' +self.params_file_prefix
        save_params(self.params_copy, self.output_directory, save_ply=False, file_prefix=file_prefix)
        save_params(self.params_copy, self.output_directory, save_ply=False, file_prefix=self.params_file_prefix)
        return EmptyResponse()
    def callback_octomap_status(self, msg):
        self.octomap_status = msg.data

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
    
    def callback_camera_pose(self, msg):
        self.camera_pose = msg
        
        
    def callback_depth_topic(self, data):
        try:
            # self.depth_image = self.bridge.imgmsg_to_cv2(data) #/1000.0
            self.depth_image = self.depth_message_to_array(data)
            # self.depth_image = self.crop_center_square(self.depth_image, self.crop_size)
            print("Received depth image of shape:", self.depth_image.shape)
        except Exception as e:
            rospy.logerr("Error converting depth Image to cv2: %s", e)
            return
    def callback_new_image_data(self, msg):
        self.new_image_data = msg.data

    # def callback_image_raw(self, data):
    #     try:
    #         self.cv_image = self.bridge.compressed_imgmsg_to_cv2(data)
    #         self.cv_image = self.crop_center_square(self.cv_image, self.crop_size)
    #     except Exception as e:
    #         rospy.logerr("Error converting compressed image to cv2: %s", e)
    #         return
    def callback_rgb_topic(self, data):
        try:
            # self.bgr_image = self.bridge.imgmsg_to_cv2(data)
            self.bgr_image = self.image_message_to_array(data)
            print("Received RGB image of shape:", self.bgr_image.shape)
        except Exception as e:
            rospy.logerr("Error converting RGB Image to cv2: %s", e)
            return
    def callback_semantic_topic(self, data):
        try:
            # self.semantic_image = self.bridge.imgmsg_to_cv2(data)
            self.semantic_image = self.image_message_to_array(data)
            print("Received Semantic image of shape:", self.semantic_image.shape)
        except Exception as e:
            rospy.logerr("Error converting semantic Image to cv2: %s", e)
            return
    
    def callback_confidence_topic(self, data):
        try:
            # self.confidence_image = self.bridge.imgmsg_to_cv2(data)
            self.confidence_image = self.image_message_to_array(data)
            print("Received Confidence image of shape:", self.confidence_image.shape)
        except Exception as e:
            rospy.logerr("Error converting confidence Image to cv2: %s", e)
            return
    def depth_message_to_array(self, data):
        """Convert depth message to numpy array"""
        try:
            dtype = np.uint16 if data.encoding == '16UC1' else np.float64
            depth_array = np.frombuffer(data.data, dtype=dtype).reshape(data.height, data.width)
            
            return depth_array
        except Exception as e:
            rospy.logerr("Error converting depth message to array: %s", e)
            return None
    def image_message_to_array(self, data):
        try:
            # Get image height, width, and encoding
            height = data.height
            width = data.width
            encoding = data.encoding  # e.g., 'rgb8', 'bgr8', 'mono8', etc.

            # Convert the raw data to a NumPy array
            np_arr = np.frombuffer(data.data, dtype=np.uint8)

            # Reshape according to channels
            if 'rgb' in encoding or 'bgr' in encoding:
                channels = 3
            elif 'mono' in encoding or 'gray' in encoding:
                channels = 1
            else:
                rospy.logerr("Unsupported encoding: %s", encoding)
                return

            np_arr = np_arr.reshape((height, width, channels))

            # Optional: convert to RGB if needed
            # if encoding == 'bgr8':
            #     np_arr = np_arr[:, :, ::-1]  # BGR -> RGB

            return np_arr.squeeze()  # Remove single-dimensional entries if any

        except Exception as e:
            rospy.logerr("Error converting image msg to numpy array: %s", e)
            return
    # def callback_link_states(self, data):
    #     gt_pose_w_link6 = data.pose[-1] # last state corresponds to link6 xarm
    #     T_w_link6 = ros_pose_to_SE3(gt_pose_w_link6)
    #     self.T_w_camframe = np.matmul(T_w_link6, self.T_link6_camframe)
    
    def upgrade_transforms(self):
        gt_pose_w_link6 = self.get_transform('world', 'link6')
        T_w_link6 = ros_pose_to_SE3(gt_pose_w_link6)
        self.T_w_camframe = np.matmul(T_w_link6, self.T_link6_camframe)

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

    def crop_center_square(self, image, size):
        h, w = image.shape[:2]
        top = (h - size)//2
        bottom = top + size
        left = (w - size) // 2
        right = left + size
        cropped_img = image[top:bottom, left:right]
        return cropped_img
    
    def bgr_to_rgb(self, bgr_image):
        rgb_image = bgr_image[:, :, ::-1]
        return rgb_image
    
    def bgr_to_gray(self, bgr_image):
        # gray_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY)
        # Extract channels
        B = bgr_image[:, :, 0]
        G = bgr_image[:, :, 1]
        R = bgr_image[:, :, 2]

        # Apply grayscale conversion weights
        gray_image = 0.114 * B + 0.587 * G + 0.299 * R

        # Convert to uint8 if needed
        gray_image = gray_image.astype(np.uint8)

        return gray_image
    def get_sample_data(self, device="cuda:0", dtype = torch.float):
        self.upgrade_transforms()
        intrinsics = torch.from_numpy(self.K)
        # gt_pose_w_camframe = self.get_transform('world', 'camera2_frame')
        # T_w_camframe = ros_pose_to_SE3(gt_pose_w_camframe)
        
        while (self.bgr_image is None) or (self.depth_image is None) or (self.semantic_image is None) or (self.confidence_image is None) or (self.camera_pose is None):
            print("waiting for image data........")
            time.sleep(0.5)
        
        T_wc_rel = self.T_w_camframe
        T_wc_rel = torch.from_numpy(T_wc_rel)

        bgr_image = copy.deepcopy(self.bgr_image)
        rgb_image = self.bgr_to_rgb(bgr_image) #cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
        rgb_image = rgb_image.astype(float) #/255
        rgb_image = torch.from_numpy(rgb_image)

        # semantic_img_uint8_bgr, _, _, self.confidence_map = get_semantic_image(bgr_image, add_seg_noise = self.add_seg_noise, conf=True)
        semantic_img_uint8_bgr = copy.deepcopy(self.semantic_image)
        semantic_img_uint8_rgb = self.bgr_to_rgb(semantic_img_uint8_bgr) #cv2.cvtColor(semantic_img_uint8_bgr, cv2.COLOR_BGR2RGB)
        semantic_img_float = semantic_img_uint8_rgb.astype(float) #/255
        semantic_img = torch.from_numpy(semantic_img_float)


        semantic_id = self.bgr_to_gray(semantic_img_uint8_bgr) #cv2.cvtColor(semantic_img_uint8_bgr, cv2.COLOR_BGR2GRAY)
        semantic_id = semantic_id.astype(float)
        # print("Semantic ids:", np.unique(semantic_id))
        semantic_id = np.expand_dims(semantic_id, -1)#(h,w,1)
        semantic_id = torch.from_numpy(semantic_id)

        depth = self.depth_image.astype(float) # m
        # depth = self.depth_img
        # depth = np.where(np.isnan(depth), 0.0, depth)
        depth = np.expand_dims(depth, -1) #(h,w,1)
        depth = torch.from_numpy(depth)
        depth = torch.nan_to_num(depth, nan=0.0)
        depth[depth>1.0] = 0.0

        #confidence map to from np.uint8 to np.float32
        
        confidence_map = copy.deepcopy(self.confidence_image).astype(float) / 255.0
        confidence_map = torch.from_numpy(confidence_map)
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
        self.bgr_image = None
        self.depth_image = None
        self.semantic_image = None
        self.confidence_image = None
        self.camera_pose = None
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
        
        
        valid_depth = False
        while valid_depth == False:
            self.pub_gs_status.publish(Float32(1.0))    
            while (self.new_image_data == None):
                print("waiting for first sample image")
                time.sleep(0.5)
            dataset_0 = self.get_sample_data()
            _, depth_sample, _, _, _, _, _ = dataset_0
            valid_depth = depth_sample.sum().item() > 0
            if (valid_depth == False):
                print("No valid depth map")
                plt.figure(1)
                plt.imshow(self.depth_image)
                plt.show()
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
            
            print("Current time idx:", time_idx)
            print("Number of gaussians:", params['means3D'].shape[0])
            
            if time_idx == 0:
                color, depth, _, gt_pose, semantic_id, semantic_color, confidence_map = dataset_0
            
            target_gaussians_3D = None

            # test plotting any cam
            # Camera pose wrt to world frame
            # R = np.array([[0.0, 0.0, 1.0],
            #                 [-1.0, 0.0, 0.0],
            #                 [0.0, -1.0, 0.0]])
            # t = np.array([[0.0],
            #                 [0.0],
            #                 [0.5]]) # position of camera wrt to world frame
            R = np.array([[-1.0, 0.0, 0.0],
                            [0.0, 0.0, -1.0],
                            [0.0, -1.0, 0.0]])
            t = np.array([[3.0],
                            [0.0],
                            [0.5]]) # position of camera wrt to world frame
            T_wc = np.block([[R, t],
                                    [0.0, 0.0, 0.0, 1.0]])
            T_cw = np.linalg.inv(T_wc)
            # render_any_cam(params, T_cw, device=device)
            if time_idx>0:
                # self.pub_gs_status.publish(Float32(1.0))
                valid_depth = False
                while valid_depth == False:
                    self.pub_gs_status.publish(Float32(1.0))
                
                    while (self.new_image_data == None):
                        print("waiting for first sample image")
                        time.sleep(0.5)
                    color, depth, _, gt_pose, semantic_id, semantic_color, confidence_map = self.get_sample_data()
                    valid_depth = depth.sum().item() > 0
                    if (valid_depth == False):
                        print("No valid depth map")
                sem_target = torch.tensor([1.0,0,0]).to(device) #red
                rmse = torch.linalg.norm(sem_target - params['semantic_colors'].clip(0,1), axis=1)/math.sqrt(3)
                # cos_similarity = F.cosine_similarity(sem_target, params['semantic_colors'],dim=1)
                
                sem_mask = (rmse < 0.01) # TODO add parameters
                stable_gaussians_mask = (params['opt_count'] > 10) # TODO add parameters
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
                        ptc_msg = create_pointcloud2(sem_centroids_w, frame_id='world')
                        rospy.loginfo("Publishing PointCloud2 message")
                        self.pub_sem_centroids.publish(ptc_msg)
                        # for sem_centroid in sem_centroids_w.tolist():
                        #     camlink_poses, camframe_poses = self.gen_cam_poses(0, 360, 30, 135,
                        #                                     centroid=sem_centroid, theta_n_grid=12, phi_n_grid = 5, r=self.sampling_r)    
                        #     cand_camframe_poses = cand_camframe_poses + camframe_poses
                        #     cand_camlink_poses = cand_camlink_poses + camlink_poses
                        #     for i in range(len(camframe_poses)):
                        #         centroids_list.append(sem_centroid)
                        # scores = []
                        # rec_info_list = []
                        # rgb_info_list = []
                        # silhouette_info_list = []
                        # start_time = time.time()
                        # for centroid, cand_camframe_pose in zip(centroids_list, cand_camframe_poses):
                        #     T_w_cx = ros_pose_to_SE3(cand_camframe_pose)
                        #     T_cw = np.linalg.inv(T_w_cx)
                        #     reconstruction_info, rgb_info, silhouette_info = self.evaluate_viewpoint(params, centroid, curr_data=curr_data, T_cw = T_cw, visualize = False)
                        #     rec_info_list.append(reconstruction_info)
                        #     rgb_info_list.append(rgb_info)
                        #     silhouette_info_list.append(silhouette_info)
                        #     end_time = time.time()
                        
                        
                        # print("Total evaluation time:", end_time - start_time)
                        
                        # rec_info = np.array(rec_info_list)
                        # scores = rec_info
                        # sorted_indices = np.argsort(scores)[::-1].tolist()
                        # sorted_scores = scores[sorted_indices]
                        # print("\nNumber of evaluated viewpoints:", len(sorted_indices))
                        # print("Absolute best recon scores:", np.max(rec_info))
                        # print("***Best scores:", sorted_scores[0:3])
                        # #visualize best viewpoint
                        # T_wc = ros_pose_to_SE3(cand_camframe_poses[sorted_indices[0]])
                        # centroid = centroids_list[sorted_indices[0]]
                        # T_cw = np.linalg.inv(T_wc)
                        # _, _, _ = self.evaluate_viewpoint(params, centroid, curr_data=curr_data, T_cw = T_cw, visualize = False)
                        # ########### end visualization

                        # sorted_camlink_poses = [cand_camlink_poses[i] for i in sorted_indices]
                        # sorted_camframe_poses = [cand_camframe_poses[i] for i in sorted_indices]
                        # self.execute_single_camlink_goal(sorted_camlink_poses[0])
                        # time.sleep(0.5)
                    else:
                        print("No sem centroids found. Executing random viewpoint")
                        # random_viewpoint_id = np.random.randint(0, len(self.camlink_viewpoints))
                        # self.execute_single_camlink_goal(self.camlink_viewpoints[random_viewpoint_id])
                        # time.sleep(0.5)
                else:
                    print("No semantic targets found. Executing random viewpoint")
                    # random_viewpoint_id = np.random.randint(0, len(self.camlink_viewpoints))
                    # self.execute_single_camlink_goal(self.camlink_viewpoints[random_viewpoint_id])
                    # time.sleep(0.5)
            # input("Press enter to continue")
            # self.evaluation(target_gaussians_3D, dist_threshold=self.dist_threshold)
                         

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
                            # report_progress(params, tracking_curr_data, iter+1, progress_bar, iter_time_idx, sil_thres=config['tracking']['sil_thres'],
                            #                     tracking=True, device=device, load_semantics=load_semantics)
                            pass
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
                    # pass #TODO REMOVE
                    params['cam_unnorm_rots'][..., time_idx] = candidate_cam_unnorm_rot
                    params['cam_trans'][..., time_idx] = candidate_cam_tran
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
                        pass
                        # report_progress(params, tracking_curr_data, 1, progress_bar, iter_time_idx, sil_thres=config['tracking']['sil_thres'],
                        #                     tracking=True, device=device, load_semantics=load_semantics)
                    progress_bar.close()
                except:
                    ckpt_output_dir = os.path.join(config["workdir"], config["run_name"])
                    save_params_ckpt(params, ckpt_output_dir, time_idx)
                    print('Failed to evaluate trajectory.')
            print("Densification step...")
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
                    print("Keyframe selection overlap...")
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

                ######## test redensifying periodically jrcv ############# TODO delete
                # if time_idx%5 == 0 and time_idx >0:
                    
                #     for idx in range(len(selected_keyframes)-1):
                #         iter_time_idx = keyframe_list[idx]['id']
                #         iter_color = keyframe_list[idx]['color']
                #         iter_depth = keyframe_list[idx]['depth']
                #         iter_confidence_map = keyframe_list[idx]['confidence_map']

                #         iter_gt_w2c = gt_w2c_all_frames[:iter_time_idx+1]
                #         iter_data = {'cam': cam, 'im': iter_color, 'depth': iter_depth, 'confidence_map': iter_confidence_map, 'id': iter_time_idx, 
                #                     'intrinsics': intrinsics, 'w2c': first_frame_w2c, 'iter_gt_w2c_list': iter_gt_w2c}
                #         # Add semantic id and colors
                        
                #         iter_data['semantic_id'] = keyframe_list[idx]['semantic_id']
                #         iter_data['semantic_color'] = keyframe_list[idx]['semantic_color']
                #         # adding new gaussians
                #         print("+++++++++++++++++++++++++ Adding new gaussians test +++++++++++++++++++++++")
                #         print("idx:", idx, " time idx:", iter_time_idx)
                #         params, variables = add_new_gaussians(params, params_opt_exclude, variables, iter_data, 
                #                                             config['mapping']['sil_thres'], iter_time_idx, config['mean_sq_dist_method'],
                #                                             device, load_semantics=load_semantics)
                    
                ######################## end test
                
                # Reset Optimizer & Learning Rates for Full Map Optimization
                optimizer = initialize_optimizer(params, params_opt_exclude, config['mapping']['lrs'], tracking=False) 

                # Mapping
                print("Mapping...")
                mapping_start_time = time.time()
                if num_iters_mapping > 0:
                    progress_bar = tqdm(range(num_iters_mapping), desc=f"Mapping Time Step: {time_idx}")
                loss_compute_times = []
                prune_compute_times = []
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
                    
                    if (iter+1) % 9800 == 0:
                        visualization = True
                    else:
                        visualization = False
                    

                    # visualization = False
                    loss_start_time = time.time()
                    loss, variables, losses = get_loss(params, iter_data, variables, iter_time_idx, config['mapping']['loss_weights'],
                                                    config['mapping']['use_sil_for_loss'], config['mapping']['sil_thres'],
                                                    config['mapping']['use_l1'], config['mapping']['ignore_outlier_depth_loss'],
                                                    mapping=True, device=device, plot_dir = eval_dir, load_semantics=load_semantics, visualization = visualization)
                    loss_end_time = time.time()
                    loss_compute_times.append(loss_end_time - loss_start_time)
                    # Backprop
                    loss.backward()
                    with torch.no_grad():
                        # Prune Gaussians
                        
                        prune_start_time = time.time()
                        if config['mapping']['prune_gaussians']:
                            params, variables = prune_gaussians(params, params_opt_exclude, variables, optimizer, iter, config['mapping']['pruning_dict'])
                            # if iter == num_iters_mapping - 1:
                            #     params, variables = prune_outlier_semantics(params, params_opt_exclude, variables, optimizer)
                        prune_end_time = time.time()
                        prune_compute_times.append(prune_end_time - prune_start_time)
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
                            pass
                            # report_progress(params, iter_data, iter+1, progress_bar, iter_time_idx, sil_thres=config['mapping']['sil_thres'], 
                            #                     mapping=True, device=device, load_semantics=load_semantics, online_time_idx=time_idx)
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
                print(f"Total loss compute time: {sum(loss_compute_times)}")
                print(f"Total prune compute time: {sum(prune_compute_times)}")
                print(f"Total mapping time: {mapping_end_time - mapping_start_time}")

                if time_idx == 0 or (time_idx+1) % config['report_global_progress_every'] == 0:
                    try:
                        # Report Mapping Progress
                        progress_bar = tqdm(range(1), desc=f"Mapping Result Time Step: {time_idx}")
                        with torch.no_grad():
                           pass
                        #    report_progress(params, curr_data, 1, progress_bar, time_idx, sil_thres=config['mapping']['sil_thres'], 
                        #                         mapping=True, device=device, load_semantics=load_semantics, online_time_idx=time_idx)
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
            
            self.params_copy = copy.deepcopy(params) 
            
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
        # if tracking_iter_time_count == 0:
        #     tracking_iter_time_count = 1
        #     tracking_frame_time_count = 1
        # if mapping_iter_time_count == 0:
        #     mapping_iter_time_count = 1
        #     mapping_frame_time_count = 1
        # tracking_iter_time_avg = tracking_iter_time_sum / tracking_iter_time_count
        # tracking_frame_time_avg = tracking_frame_time_sum / tracking_frame_time_count
        # mapping_iter_time_avg = mapping_iter_time_sum / mapping_iter_time_count
        # mapping_frame_time_avg = mapping_frame_time_sum / mapping_frame_time_count
        # print(f"\nAverage Tracking/Iteration Time: {tracking_iter_time_avg*1000} ms")
        # print(f"Average Tracking/Frame Time: {tracking_frame_time_avg} s")
        # print(f"Average Mapping/Iteration Time: {mapping_iter_time_avg*1000} ms")
        # print(f"Average Mapping/Frame Time: {mapping_frame_time_avg} s")
        
        
        # remove auxiliar params, jrcv
        # params, variables = prune_aux_gaussians(params, params_opt_exclude, variables, optimizer)
        # params, variables = prune_outlier_semantics(params, params_opt_exclude, variables, optimizer)
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
        save_params(params, self.output_directory, save_ply=False, file_prefix=output_file_prefix)
        
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

