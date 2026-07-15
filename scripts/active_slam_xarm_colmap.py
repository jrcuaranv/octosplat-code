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
Initialize self.params['semantic_colors'] in 0.5 (rather than current semantics) (check get_pointcloud())to reduce the effect of noise. After some optim. steps it should converge to some value
Clip self.params[semantic_colors'] between [0,1] (in add_new_gaussians)as other values dont make sense and reduce inertia during optimization.
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
import random

import rospy
from gazebo_msgs.msg import LinkState, LinkStates
from geometry_msgs.msg import Pose, Twist, Point, Quaternion
import geometry_msgs.msg
from sensor_msgs.msg import Image, CompressedImage
from PIL import Image as PILImage
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
from utils.utils_active_mapping import ViewpointEvaluation, filter_depth_map, get_semantic_image, SE3_to_ros_pose, ros_pose_to_SE3, dbscan_clustering
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
from utils.eval_helpers import eval_single_frame, depth_colormap
from utils.slam_helpers import (
    get_c2w_from_params,transformed_params2rendervar, filter_points_in_image, transformed_params2depth_silhouette_rgbloss, transformed_entropy2rendervar, transformed_params2depthplussilhouette,
    transformed_semantics2rendervar, transformed_rgb_loss_rendervar, transform_to_frame, transform_points_to_frame, l1_loss_v1, matrix_to_quaternion
)
from utils.slam_external import calc_ssim, build_rotation, densify_v2, prune_outliers_based_on_density_statistics, prune_background_semantics, prune_outlier_semantics, prune_gaussians, densify, prune_aux_gaussians, prune_outliers, reset_semantics

from diff_gaussian_rasterization import GaussianRasterizer as Renderer

from utils.utils_sgs_slam import (render_any_cam, render_cam, get_pointcloud, downsample_mask, get_initial_pointcloud, initialize_params,
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
    def __init__(self, config, alternative_colmap_dataset=None):
        
        rospy.init_node('active_slam', anonymous=True)
        self.config = config
        self.bridge = CvBridge()

        # just a transform for convenience.
        self.T_camlink_camframe = np.array([[0.0, 0.0, 1.0, 0.0],
                                            [-1.0, 0.0, 0.0, 0.0],
                                            [0.0, -1.0, 0.0, 0.0],
                                            [0.0, 0.0, 0.0, 1.0]])

        
        prefix = config['active_mapping']['data_mode']
        self.episode_suffix = ""
        if prefix == 'colmap_dataset':
            self.running_colmap_dataset = True
            self.colmap_scene_path = self.config['active_mapping'][prefix]['colmap_scene_path']
            if self.colmap_scene_path is None:
                self.colmap_scene_path = alternative_colmap_dataset
            if self.colmap_scene_path is None:
                raise ValueError("Colmap scene path is not provided. Please provide it in the config file or as a command line argument.")
            if self.colmap_scene_path.endswith("/"):
                self.colmap_scene_path = self.colmap_scene_path[:-1]
            self.test_data_path = os.path.join(self.colmap_scene_path, "eval_data")
            if not os.path.exists(self.test_data_path):
                print("Colmap scene path:", self.colmap_scene_path)
                colmap_scene_name = os.path.basename(self.colmap_scene_path)
                greenhouse_id = colmap_scene_name.split('_g')[-1].split('_row')[0]
                row_id = colmap_scene_name.split('_row')[-1]
                print("Colmap scene name:", colmap_scene_name)
                print("Greenhouse ID:", greenhouse_id)
                print("Row ID:", row_id)
                self.episode_suffix = f"_g{greenhouse_id}_row{row_id}"
                # ubuntu 20 laptop
                # self.test_data_path = os.path.join('/mnt/ssd2T/datasets/gaussian_splat_data/active_mapping_evaluation_2026/eval_data_folders', f"greenhouse_{greenhouse_id}", f"row_{row_id}")
                # jetson orin
                self.test_data_path = os.path.join('/mnt/ssd1T/active_mapping_evaluation_2026/eval_data_folders', f"greenhouse_{greenhouse_id}", f"row_{row_id}")
            print("Colmap scene path:", self.colmap_scene_path)
            print("Test data path:", self.test_data_path)
            self.train_files, self.test_files = self.split_colmap_dataset()
            self.intrinsics = np.loadtxt(os.path.join(self.colmap_scene_path, "intrinsics.txt"))
            self.fx, self.fy, self.cx, self.cy = self.intrinsics[0, 0], self.intrinsics[1, 1], self.intrinsics[0, 2], self.intrinsics[1, 2]
        else:
            self.running_colmap_dataset = False
            self.fx = self.config['active_mapping'][prefix]['fx']
            self.fy = self.config['active_mapping'][prefix]['fy']
            self.cx = self.config['active_mapping'][prefix]['cx']
            self.cy = self.config['active_mapping'][prefix]['cy']
        
        self.output_directory = self.config['active_mapping'][prefix]['output_dir']
        # creating directory if it does not exist
        if not os.path.exists(self.output_directory):
            os.makedirs(self.output_directory)
        rgb_topic = '/camera2/color/rgb'
        depth_topic = '/camera2/color/depth'
        semantics_topic = '/camera2/color/semantics'
        confidence_topic = '/camera2/color/confidence'
        self.crop_size = self.config['active_mapping'][prefix]['crop_size']
            
        self.apply_depth_median_filter = self.config['active_mapping'][prefix].get('apply_depth_median_filter')
        self.apply_statistical_outlier_filter = self.config['active_mapping'][prefix].get('apply_statistical_outlier_filter')
        self.max_depth = self.config['active_mapping'][prefix].get('max_depth')
        self.T_link6_camframe = np.array(self.config['active_mapping'][prefix]['T_link6_camframe'])

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
        self.rgbd_slam_server = rospy.Service("/sgs/rgbd_slam", Empty, self.rgbd_slam_callback)
        rospy.loginfo("Service '/sgs/save_params' is ready.")
    
        # server to perform full pose and map optimization
        self.full_optimization_server = rospy.Service("/sgs/full_optimization", Empty, self.full_optimization_callback)
        rospy.loginfo("Service '/sgs/full_optimization' is ready.")
        

        # rospy.wait_for_service('querry_RLE')
        # try:
        #     self.RLE_query = rospy.ServiceProxy('querry_RLE', GetRLE, persistent=True)
        #     print("********************* RLE query Test succesfull **********************")
        # except rospy.ServiceException as e:
        #     print("*********************RLE Service initialization failed: %s"%e)
        
        # self.vp_evaluation = ViewpointEvaluation(self.RLE_query, phi=-0.1, psi=1.0)
        
        self.init_variables()
        # self.run_mapping_multiple_plants()
        # self.rgbd_slam(self.config, self.params_file_prefix)
        

        rate = rospy.Rate(0.5)

        if self.running_colmap_dataset == True:
            self.new_rgbd_slam_session = True
        self.terminate_process = False
        while not rospy.is_shutdown() and not self.terminate_process:
            rate.sleep()
            if self.new_rgbd_slam_session:
                time.sleep(5.0) # wait for any ongoing process to finish
                self.init_variables()
                self.rgbd_slam(self.config, self.params_file_prefix)
            
            print("Running...")
        return 0
            
    def init_variables(self):
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
        self.params_file_prefix = "xarm_"
        self.new_rgbd_slam_session = False
        self.params = None
        params_opt_exclude = None
        self.keyframe_list = []
        self.numbers_iters_mapping = None
        self.intrinsics = None
        self.gt_w2c_all_frames = []
        self.first_frame_w2c = None
        self.variables = None
        self.cam = None
        self.device = None
        self.eval_dir = None
        self.full_optimization_requested = False
        self.episode_dir = None
        self.offline_sample_idx = 0
        

    def save_params_callback(self, req):
        
        self.full_map_optimization(200) # a few additional optimization steps for refinenment
        rospy.loginfo("Saving parameters...")
        # Get current time
        now = datetime.now()
        timestamp = now.strftime("%Y-%m-%d-%H-%M-%S")
        file_prefix = timestamp + '_' +self.params_file_prefix
        save_params(self.params, self.episode_dir, save_ply=True, file_prefix=file_prefix)
        # save_params(self.params, self.episode_dir, save_ply=False, file_prefix=self.params_file_prefix)
        self.eval_training(timestamp_str = timestamp)
        if self.running_colmap_dataset:
            self.eval_test(timestamp_str = timestamp)
        return EmptyResponse()
    def callback_octomap_status(self, msg):
        self.octomap_status = msg.data

    def evaluation(self, data_points, dist_threshold):
        output_dir = self.episode_dir
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
            if data.encoding == '16UC1':
                dtype = np.uint16
            elif data.encoding in ('32FC1', '32FC2'):
                dtype = np.float32
            else:
                dtype = np.float64  # Default to float64 for other encodings
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
    
    def get_sample_data(self, dtype = torch.float):
        if self.running_colmap_dataset == True:
            sample_data = None
            while sample_data is None:
                if self.offline_sample_idx >= len(self.train_files):
                    print("All samples from the training dataset have been used")
                    return None
                file_name = self.train_files[self.offline_sample_idx]
                sample_data = self.get_colmap_sample_data(file_name)
                self.offline_sample_idx += 1
            return sample_data
        
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
        if self.apply_depth_median_filter:
            depth = cv2.medianBlur(depth.astype(np.float32), 5)
        if self.apply_statistical_outlier_filter:
            depth = filter_depth_map(depth, self.K, max_depth=self.max_depth)
        # depth = self.depth_img
        # depth = np.where(np.isnan(depth), 0.0, depth)
        depth = np.expand_dims(depth, -1) #(h,w,1)
        depth = torch.from_numpy(depth)
        depth = torch.nan_to_num(depth, nan=0.0)
        depth[depth>self.max_depth] = 0.0

        #confidence map to from np.uint8 to np.float32
        
        confidence_map = copy.deepcopy(self.confidence_image).astype(float) / 255.0
        confidence_map = torch.from_numpy(confidence_map)
        return_data = (
            rgb_image.to(self.device).type(dtype),
            depth.to(self.device).type(dtype),
            intrinsics.to(self.device).type(dtype),
            T_wc_rel.to(self.device).type(dtype),
            semantic_id.to(self.device).type(dtype),
            semantic_img.to(self.device).type(dtype),
            confidence_map.to(self.device).type(dtype),
        )
        self.new_image_data = None
        self.bgr_image = None
        self.depth_image = None
        self.semantic_image = None
        self.confidence_image = None
        self.camera_pose = None
        return return_data
    
    def split_colmap_dataset(self):
        
        if os.path.isdir(self.test_data_path):
            print("Eval data directory already exists. Assuming dataset is already split.")
            train_image_dir = os.path.join(self.colmap_scene_path, "images")
            train_image_files = sorted(os.listdir(train_image_dir))
            random.shuffle(train_image_files) # to get random samples from the dataset, rather than sequential ones
            test_image_dir = os.path.join(self.test_data_path, "images")
            test_image_files = sorted(os.listdir(test_image_dir))
            print("Number of training files:", len(train_image_files))
            print("Number of test files:", len(test_image_files))
            return train_image_files, test_image_files
        
        # splitting data
        self.test_data_path = self.colmap_scene_path # same as train data path
        image_dir = os.path.join(self.colmap_scene_path, "images")
        image_files = sorted(os.listdir(image_dir))
        random.shuffle(image_files) # to get random samples from the dataset, rather than sequential ones
        print("Number of files in the colmap dataset:", len(image_files))
        split_idx = int(0.8 * len(image_files))
        train_files = image_files[:split_idx]
        test_files = image_files[split_idx:]
        print("Number of training files:", len(train_files))
        print("Number of test files:", len(test_files))
        return train_files, test_files
    
    def get_colmap_sample_data(self, file_name, dtype = torch.float, eval = False):
        if eval:
            data_path = self.test_data_path
        else:
            data_path = self.colmap_scene_path
        image_dir = os.path.join(data_path, "images") # contains both train and test images
        semantics_dir = os.path.join(data_path, "semantics")
        confidences_dir = os.path.join(data_path, "confidences")
        depth_dir = os.path.join(data_path, "depth")
        pose_dir = os.path.join(data_path, "poses")
        image_files = sorted(os.listdir(image_dir))
        
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
        rgb_image = self.bgr_to_rgb(bgr_image) #cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
        rgb_image = rgb_image.astype(float) #/255
        rgb_image = torch.from_numpy(rgb_image)

        # semantic_img_uint8_bgr, _, _, self.confidence_map = get_semantic_image(bgr_image, add_seg_noise = self.add_seg_noise, conf=True)
        semantic_img_uint8_bgr = cv2.imread(semantic_path) #(h,w,3) in BGR format
        semantic_img_uint8_rgb = self.bgr_to_rgb(semantic_img_uint8_bgr) #cv2.cvtColor(semantic_img_uint8_bgr, cv2.COLOR_BGR2RGB)
        semantic_img_float = semantic_img_uint8_rgb.astype(float) #/255
        semantic_img = torch.from_numpy(semantic_img_float)


        semantic_id = self.bgr_to_gray(semantic_img_uint8_bgr) #cv2.cvtColor(semantic_img_uint8_bgr, cv2.COLOR_BGR2GRAY)
        semantic_id = semantic_id.astype(float)
        # print("Semantic ids:", np.unique(semantic_id))
        semantic_id = np.expand_dims(semantic_id, -1)#(h,w,1)
        semantic_id = torch.from_numpy(semantic_id)

        # confidence_map = np.ones_like(depth.squeeze()).astype(float) # just for testing
        confidence_map = cv2.imread(confidences_path, cv2.IMREAD_UNCHANGED) #(h,w), confidence in [0,255]
        # confidence_map[confidence_map == 127] = 63 # Chaning background confidence. TODO Remove. Just testing
        confidence_map = confidence_map.astype(float) / 255.0
        confidence_map = torch.from_numpy(confidence_map)
        

        depth = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)/1000.0 # (h,w) in uint16 format, depth in mm
        if self.apply_depth_median_filter:
            depth = cv2.medianBlur(depth.astype(np.float32), 5)
        if self.apply_statistical_outlier_filter:
            depth = filter_depth_map(depth, self.K, max_depth=self.max_depth)
            
        # depth = depth * valid_confidence_mask
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
        self.new_image_data = None
        self.bgr_image = None
        self.depth_image = None
        self.semantic_image = None
        self.confidence_image = None
        self.camera_pose = None

        
        return return_data


    def rgbd_slam_callback(self, req):
        rospy.loginfo("New RGBD SLAM session requested ...")
        self.new_rgbd_slam_session = True
        
        return EmptyResponse()

    def rgbd_slam(self, config: dict, output_file_prefix: str):
        self.new_rgbd_slam_session = False
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
        # output_dir = os.path.join(config["workdir"], config["run_name"])
        output_dir = self.episode_dir
        self.eval_dir = os.path.join(output_dir, "eval")
        # os.makedirs(self.eval_dir, exist_ok=True)
        
        # saving config file in output dir
        with open(os.path.join(self.episode_dir, "config.yaml"), 'w') as yaml_file:
            yaml.dump(config, yaml_file)

        # Get Device
        self.device = torch.device(config["primary_device"])
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
        num_frames = min(dataset_config["num_frames"], len(self.train_files) if self.running_colmap_dataset else float('inf'))
        
        
        valid_data = False
        while valid_data == False:
            self.pub_gs_status.publish(Float32(1.0))    
            while (self.new_image_data == None):
                print("waiting for first sample image")
                if self.new_rgbd_slam_session == True:
                    print("New RGBD SLAM session triggered. Exiting current SLAM session.")
                    return
                if self.running_colmap_dataset == True:
                    print("Running colmap dataset. Getting next sample data...")
                    break
                time.sleep(0.5)
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
            print("Number of gaussians:", self.params['means3D'].shape[0])
            
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
            
            if time_idx>0:
                # self.pub_gs_status.publish(Float32(1.0))
                valid_data = False
                while valid_data == False:
                    self.pub_gs_status.publish(Float32(1.0))

                    while (self.new_image_data == None):
                        print("waiting for next sample image")
                        time.sleep(0.5)
                        if self.new_rgbd_slam_session == True:
                            print("New RGBD SLAM session triggered. Exiting current SLAM session.")
                            return
                        if time_idx == num_frames - 1:
                            self.save_params_callback(None)
                            print(f"Checkpoint reached at time idx {time_idx}. Pausing SLAM session for inspection.")
                            # input("Press Enter to continue...")
                            self.terminate_process = True
                            return
                            
                            
                        
                        if self.running_colmap_dataset == True:
                            if self.offline_sample_idx >= len(self.train_files):
                                print("All samples from the training dataset have been used. Saving and exiting current SLAM session.")
                                self.save_params_callback(None)
                                self.terminate_process = True
                                return
                            print("Running colmap dataset. Getting next sample data...")
                            break
                        while self.full_optimization_requested == True:
                            time.sleep(0.5)
                        
                        # if self.full_optimization_requested == True:
                        #     print("Full optimization requested. Exiting current SLAM session.")
                        #     self.full_pose_optimization()
                        #     self.full_map_optimization()
                        #     self.full_optimization_requested = False
                            
                    sample_data = self.get_sample_data()
                    if sample_data is None:
                        print("No valid sample data...")
                    else:
                        color, depth, _, gt_pose, semantic_id, semantic_color, confidence_map = sample_data
                        if depth.sum().item() > 0:
                            valid_data = True
                        else:
                            print("Depth sample is empty. Waiting for next sample...")
                sem_target = torch.tensor([1.0,0,0]).to(self.device) #red
                rmse = torch.linalg.norm(sem_target - self.params['semantic_colors'].clip(0,1), axis=1)/math.sqrt(3)
                # cos_similarity = F.cosine_similarity(sem_target, self.params['semantic_colors'],dim=1)
                
                sem_mask = (rmse < 0.01) # TODO add parameters
                stable_gaussians_mask = (self.params['opt_count'] > 10) # TODO add parameters
                target_sem_mask = sem_mask.view(-1) & stable_gaussians_mask.view(-1)

                n_sem_gaussians = sem_mask.sum().item()
                n_stable_sem_gaussians = target_sem_mask.sum().item()
                
                print("number of sem gaussians:", n_sem_gaussians)
                print("Number of stable sem gaussians:", n_stable_sem_gaussians)

                target_gaussians_3D = self.params['means3D'][target_sem_mask].detach().cpu().numpy()
                # np.savetxt("/home/jose/gaussians.txt", target_gaussians_3D)

                if target_gaussians_3D.shape[0] > 0:
                    # evaluate surface coverage
                    # surface_coverage = compute_surface_coverage(self.gt_pointcloud, target_gaussians_3D,
                    #                                             offset = self.xyz_plant_origin, dist_threshold= 0.01)
                    start_time = time.time()
                    sem_centroids = dbscan_clustering(target_gaussians_3D, eps_= 0.02, min_samples=100) # 250
                    end_time = time.time()
                    print("DBSCAN clustering time:", end_time - start_time)
                    sem_centroids_w = sem_centroids
                    
                    print(f"Number of semantic centroids: {sem_centroids_w.shape[0]}")
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
            self.gt_w2c_all_frames.append(gt_w2c)
            curr_gt_w2c = self.gt_w2c_all_frames
            # Optimize only current time step for tracking
            iter_time_idx = time_idx
            # Initialize Mapping Data for selected frame
            curr_data = {'cam': self.cam, 'im': color, 'depth': depth, 'id': iter_time_idx, 'intrinsics': self.intrinsics,
                        'w2c': self.first_frame_w2c, 'iter_gt_w2c_list': curr_gt_w2c}
            
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
            
            self.num_iters_mapping = config['mapping']['num_iters']
            if time_idx == num_frames-1: 
                self.num_iters_mapping = 2*self.num_iters_mapping #jrcv, to refine the optimization in the last frame
            
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
                    self.params = initialize_camera_pose(self.params, time_idx,
                                                forward_prop=config['tracking']['forward_prop'],
                                                rel_w2c_initial_guess = w2c_init)
            # Step 1: Tracking (Pose refinement )
            tracking_start_time = time.time()
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
                    iter_start_time = time.time()
                    # Loss for current frame
                    loss, self.variables, losses = get_loss(self.params, tracking_curr_data, self.variables, iter_time_idx, config['tracking']['loss_weights'],
                                                    config['tracking']['use_sil_for_loss'], config['tracking']['sil_thres'],
                                                    config['tracking']['use_l1'], config['tracking']['ignore_outlier_depth_loss'],
                                                    tracking=True, device=self.device, plot_dir=self.eval_dir,
                                                    visualize_tracking_loss=config['tracking']['visualize_tracking_loss'],
                                                    tracking_iteration=iter, load_semantics=load_semantics)
                    # Backprop
                    loss.backward()
                    # Optimizer Update
                    
                    optimizer.step() # TODO uncomment this, jrcv
                    optimizer.zero_grad(set_to_none=True) 
                    
                    with torch.no_grad():
                        # Save the best candidate rotation & translation
                        if loss < current_min_loss:
                            current_min_loss = loss
                            candidate_cam_unnorm_rot = self.params['cam_unnorm_rots'][..., time_idx].detach().clone()
                            candidate_cam_tran = self.params['cam_trans'][..., time_idx].detach().clone()
                        # Report Progress
                        if config['report_iter_progress']:
                            # report_progress(self.params, tracking_curr_data, iter+1, progress_bar, iter_time_idx, sil_thres=config['tracking']['sil_thres'],
                            #                     tracking=True, device=self.device, load_semantics=load_semantics)
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
                        # report_progress(self.params, tracking_curr_data, 1, progress_bar, iter_time_idx, sil_thres=config['tracking']['sil_thres'],
                        #                     tracking=True, device=self.device, load_semantics=load_semantics)
                    progress_bar.close()
                except:
                    ckpt_output_dir = os.path.join(config["workdir"], config["run_name"])
                    save_params_ckpt(self.params, ckpt_output_dir, time_idx)
                    print('Failed to evaluate trajectory.')
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
                                                        self.device, load_semantics=load_semantics)
                    
                    
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

                ######## test redensifying periodically jrcv ############# TODO delete
                # if time_idx%5 == 0 and time_idx >0:
                    
                #     for idx in range(len(selected_keyframes)-1):
                #         iter_time_idx = self.keyframe_list[idx]['id']
                #         iter_color = self.keyframe_list[idx]['color']
                #         iter_depth = self.keyframe_list[idx]['depth']
                #         iter_confidence_map = self.keyframe_list[idx]['confidence_map']

                #         iter_gt_w2c = self.gt_w2c_all_frames[:iter_time_idx+1]
                #         iter_data = {'cam': self.cam, 'im': iter_color, 'depth': iter_depth, 'confidence_map': iter_confidence_map, 'id': iter_time_idx, 
                #                     'intrinsics': self.intrinsics, 'w2c': self.first_frame_w2c, 'iter_gt_w2c_list': iter_gt_w2c}
                #         # Add semantic id and colors
                        
                #         iter_data['semantic_id'] = self.keyframe_list[idx]['semantic_id']
                #         iter_data['semantic_color'] = self.keyframe_list[idx]['semantic_color']
                #         # adding new gaussians
                #         print("+++++++++++++++++++++++++ Adding new gaussians test +++++++++++++++++++++++")
                #         print("idx:", idx, " time idx:", iter_time_idx)
                #         self.params, self.variables = add_new_gaussians(self.params, self.params_opt_exclude, self.variables, iter_data, 
                #                                             config['mapping']['sil_thres'], iter_time_idx, config['mean_sq_dist_method'],
                #                                             self.device, load_semantics=load_semantics)
                    
                ######################## end test
                
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
                    # Loss for current frame
                    
                    if (iter+1) % 200000 == 0:
                        visualization = True
                    else:
                        visualization = False
                    

                    # visualization = False
                    loss_start_time = time.time()
                    loss, self.variables, losses = get_loss(self.params, iter_data, self.variables, iter_time_idx, config['mapping']['loss_weights'],
                                                    config['mapping']['use_sil_for_loss'], config['mapping']['sil_thres'],
                                                    config['mapping']['use_l1'], config['mapping']['ignore_outlier_depth_loss'],
                                                    mapping=True, device=self.device, plot_dir = self.eval_dir, load_semantics=load_semantics, visualization = visualization)
                    loss_end_time = time.time()
                    loss_compute_times.append(loss_end_time - loss_start_time)
                    # Backprop
                    loss.backward()
                    with torch.no_grad():
                        # Prune Gaussians
                        # Gaussian-Splatting's Gradient-based Densification
                        # if time_idx > 30 and (iter == 50 or iter == 55):
                        #     print("Iteration:", iter)
                        #     input("Press enter to continue before densification/pruning step")
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
                        if config['mapping']['reset_semantics']:
                            self.params = reset_semantics(self.params, self.params_opt_exclude, optimizer, iter, config['mapping'])
                        
                            # self.params, self.variables = prune_outliers_based_on_density_statistics(self.params, self.params_opt_exclude, self.variables, optimizer, iter, config['mapping']['pruning_dict'], device=self.device)
                            # if iter == self.num_iters_mapping - 1:
                            #     params, self.variables = prune_outlier_semantics(params, self.params_opt_exclude, self.variables, optimizer)
                        prune_end_time = time.time()
                        prune_compute_times.append(prune_end_time - prune_start_time)
                        
                        # if time_idx > 30 and (iter == 50 or iter == 55):
                        #     print("Iteration:", iter)
                        #     input("Press enter to continue after densification and pruning step")
                       
                        
                        #test clipping semantic colors jrcv, TODO Check
                        # if  iter % 10 ==0:
                        #     np.savetxt('/home/jose/params.txt',self.params['semantic_colors'].detach().cpu().numpy())
                        #     np.savetxt('/home/jose/means3D.txt',self.params['means3D'].detach().cpu().numpy())

                        #     input("Press enter to continue")
                        # Report Progress
                        if config['report_iter_progress']:
                            pass
                            # report_progress(params, iter_data, iter+1, progress_bar, iter_time_idx, sil_thres=config['mapping']['sil_thres'], 
                            #                     mapping=True, device=self.device, load_semantics=load_semantics, online_time_idx=time_idx)
                        else:
                            progress_bar.update(1)
                    # Update the runtime numbers
                    iter_end_time = time.time()
                    mapping_iter_time_sum += iter_end_time - iter_start_time
                    mapping_iter_time_count += 1
                
                if self.num_iters_mapping > 0:
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
                        #    report_progress(self.params, curr_data, 1, progress_bar, time_idx, sil_thres=config['mapping']['sil_thres'], 
                        #                         mapping=True, device=self.device, load_semantics=load_semantics, online_time_idx=time_idx)
                        progress_bar.close()
                    except:
                        ckpt_output_dir = os.path.join(config["workdir"], config["run_name"])
                        save_params_ckpt(self.params, ckpt_output_dir, time_idx)
                        print('Failed to evaluate trajectory.')
            
            if time_idx > 0:
                self.save_side_view_img(time_idx)

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
            
            # self.params_copy = copy.deepcopy(self.params) 
            
            torch.cuda.empty_cache()

        if config['save_timestamp_keyframes']:
            # Save keyframes selected at each timestamp
            max_length = max(len(inner) for inner in timestamp_keyframes)
            # Insert -1 for placeholder
            timestamp_keyframes_df = pd.DataFrame([inner + [-1 for _ in range(max_length - len(inner))] \
                                                for inner in timestamp_keyframes])
            timestamp_keyframes_df.to_csv(os.path.join(self.eval_dir, f"timestamp_keyframes.csv"), \
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
        # self.params, self.variables = prune_aux_gaussians(self.params, self.params_opt_exclude, self.variables, optimizer)
        # self.params, self.variables = prune_outlier_semantics(self.params, self.params_opt_exclude, self.variables, optimizer)
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
        self.save_params_callback(req = None)
        

    def full_pose_optimization(self):
        # Step 1: Tracking (Pose refinement )
        print("Running full pose optimization on all frames...")
        for frame_idx in range(1, len(self.keyframe_list)):
            ##########
            print(f"Optimizing Pose for Frame {frame_idx} / {len(self.keyframe_list)-1}...")
            iter_time_idx = self.keyframe_list[frame_idx]['id']
            iter_color = self.keyframe_list[frame_idx]['color']
            iter_depth = self.keyframe_list[frame_idx]['depth']
            iter_confidence_map = self.keyframe_list[frame_idx]['confidence_map']
            iter_gt_w2c = self.gt_w2c_all_frames[:iter_time_idx+1]
            iter_data = {'cam': self.cam, 'im': iter_color, 'depth': iter_depth, 'confidence_map': iter_confidence_map, 'id': iter_time_idx, 
                        'intrinsics': self.intrinsics, 'w2c': self.first_frame_w2c, 'iter_gt_w2c_list': iter_gt_w2c}
            iter_data['semantic_id'] = self.keyframe_list[frame_idx]['semantic_id']
            iter_data['semantic_color'] = self.keyframe_list[frame_idx]['semantic_color']



            ##########
            # Reset Optimizer & Learning Rates for tracking
            optimizer = initialize_optimizer(self.params, self.params_opt_exclude, self.config['tracking']['lrs'], tracking=True)
            # Keep Track of Best Candidate Rotation & Translation
            candidate_cam_unnorm_rot = self.params['cam_unnorm_rots'][..., frame_idx].detach().clone()
            candidate_cam_tran = self.params['cam_trans'][..., frame_idx].detach().clone()
            current_min_loss = float(1e20)
            # Tracking Optimization
            iter = 0
            num_iters_tracking = self.config['tracking']['num_iters']
            while True:
                # Loss for current frame
                loss, self.variables, losses = get_loss(self.params, iter_data, self.variables, frame_idx, self.config['tracking']['loss_weights'],
                                                self.config['tracking']['use_sil_for_loss'], self.config['tracking']['sil_thres'],
                                                self.config['tracking']['use_l1'], self.config['tracking']['ignore_outlier_depth_loss'],
                                                tracking=True, device=self.device, plot_dir=self.eval_dir,
                                                visualize_tracking_loss=self.config['tracking']['visualize_tracking_loss'],
                                                tracking_iteration=iter, load_semantics=True)
                # Backprop
                loss.backward()
                # Optimizer Update
                
                optimizer.step() 
                optimizer.zero_grad(set_to_none=True) 
                
                with torch.no_grad():
                    # Save the best candidate rotation & translation
                    if loss < current_min_loss:
                        current_min_loss = loss
                        candidate_cam_unnorm_rot = self.params['cam_unnorm_rots'][..., frame_idx].detach().clone()
                        candidate_cam_tran = self.params['cam_trans'][..., frame_idx].detach().clone()
                    
                # Check if we should stop tracking
                iter += 1
                if iter == num_iters_tracking:
                    break

            
            # Copy over the best candidate rotation & translation
            with torch.no_grad():
                self.params['cam_unnorm_rots'][..., frame_idx] = candidate_cam_unnorm_rot
                self.params['cam_trans'][..., frame_idx] = candidate_cam_tran
        
        
    def full_map_optimization(self, number_steps):
        print(f"Running full map optimization on all keyframes for {number_steps} iterations...")
        # Reset Optimizer & Learning Rates for Full Map Optimization
        optimizer = initialize_optimizer(self.params, self.params_opt_exclude, self.config['mapping']['lrs'], tracking=False) 
        # Mapping
        
        # pruning_dict=dict( # Needs to be updated based on the number of mapping iterations
        #     start_after=1,
        #     remove_big_after=1,
        #     stop_after=15000, #20,
        #     prune_every=500,
        #     removal_opacity_threshold=0.4,
        #     final_removal_opacity_threshold=0.4,
        #     reset_opacities=False,
        #     reset_opacities_every=500, # Doesn't consider iter 0
        # )
        # densify_dict=dict( # Needs to be updated based on the number of mapping iterations
        #     start_after=1, #500,
        #     remove_big_after=1,
        #     stop_after=15000,
        #     densify_every= 100,#100,
        #     grad_thresh=0.00004,#0.00005,
        #     num_to_split_into=2,
        #     removal_opacity_threshold=0.4,
        #     final_removal_opacity_threshold=0.4,
        #     reset_opacities=False,
        #     reset_opacities_every=3000, # Doesn't consider iter 0
        # )
        
        for iter in range(number_steps):
            # if iter%1000 == 0:
            #     print("RUnning pose optimization before map optimization...")
            #     self.full_pose_optimization()
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
                                            mapping=True, device=self.device, plot_dir = self.eval_dir, load_semantics=True, visualization = visualization)
            # Backprop
            loss.backward()
            with torch.no_grad():
                self.params, self.variables = densify_v2(self.params, self.variables, optimizer, iter, self.config['mapping']['densify_dict'], self.params_opt_exclude, device=self.device)
                # Optimizer Update
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                
                # Prune Gaussians
                self.params, self.variables = prune_gaussians(self.params, self.params_opt_exclude, self.variables, optimizer, iter, self.config['mapping']['pruning_dict'])
                self.params, self.variables = prune_background_semantics(self.params, self.params_opt_exclude, self.variables, optimizer, iter, self.config['mapping']['pruning_dict'])
                
                # if iter%2000 == 0:
                #     prune_outliers_based_on_density_statistics(self.params, self.params_opt_exclude, self.variables, optimizer, iter, pruning_dict)
                #     prune_outlier_semantics(self.params, self.params_opt_exclude, self.variables, optimizer)
                #     prune_outliers(self.params, self.params_opt_exclude, self.variables, optimizer) # based on density  
            # if iter%1000 == 0:
            #     print(f"Completed {iter} / {number_steps} iterations of full map optimization.")
            #     self.save_params_callback(req = None)
    def full_optimization_callback(self, req):
        self.full_optimization_requested = True
        time.sleep(2)
        self.full_map_optimization(number_steps=15000)
        # if not self.running_colmap_dataset:
        # self.full_pose_optimization()
        # self.full_map_optimization(number_steps=100)
        self.full_optimization_requested = False
        return EmptyResponse()
    
    def eval_training(self, timestamp_str=''):
        # also computes evaluation metrics
        images_dir = os.path.join(self.episode_dir, timestamp_str + "_train_images")
        os.makedirs(images_dir, exist_ok=True)
        self.save_side_view_img(1000) # save a side view image at the end of the episode for visualization
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
            psnr, ssim, lpips_score, rmse, depth_l1, miou = eval_single_frame(gt_color_torch, gt_depth_torch, gt_semantics_torch, gt_confidence_map_torch, rendered_color_torch, rendered_depth_torch, rendered_semantics_torch)
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
                gt_depth_numpy = gt_depth_torch.detach().cpu().numpy()
                
                rendered_color_numpy  = (rendered_color_torch.permute(1, 2, 0).detach().cpu().numpy() * 255).astype(np.uint8)
                rendered_depth_numpy = rendered_depth_torch.squeeze().detach().cpu().numpy()
                rendered_semantics_numpy = (rendered_semantics_torch.permute(1, 2, 0).detach().cpu().numpy() * 255).astype(np.uint8)
                rendered_silhouette_numpy = rendered_silhouette_torch.detach().cpu().numpy()

                valid_mask = rendered_silhouette_numpy > 0.9
                rendered_color_numpy[~valid_mask] = 0
                rendered_depth_numpy[~valid_mask] = 0
                rendered_semantics_numpy[~valid_mask] = 0

                c2w_torch = get_c2w_from_params(self.params, curr_time_idx)
                c2w_numpy = c2w_torch.detach().cpu().numpy()
                np.savetxt(os.path.join(images_dir, f"frame_{curr_time_idx}_pose.txt"), c2w_numpy)

                # converting depth maps to colorable depth maps for visualization
                # depth_min = 0
                # depth_max = 2
                # depth_range = depth_max - depth_min
                # gt_depth_vis = (gt_depth_numpy - depth_min) / depth_range * 255
                # gt_depth_vis = gt_depth_vis.astype(np.uint8)
                # gt_depth_vis_color = cv2.applyColorMap(gt_depth_vis, cv2.COLORMAP_JET)
                # rendered_depth_vis = (rendered_depth_numpy - depth_min) / depth_range * 255
                # rendered_depth_vis = rendered_depth_vis.astype(np.uint8)
                # rendered_depth_vis_color = cv2.applyColorMap(rendered_depth_vis, cv2.COLORMAP_JET)

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
                # cv2.imwrite(os.path.join(images_dir, f"frame_{curr_time_idx}_depth_gt.png"), gt_depth_vis_color)
                gt_depth_vis_color.save(os.path.join(images_dir, f"frame_{curr_time_idx}_depth_gt.png"))
                cv2.imwrite(os.path.join(images_dir, f"frame_{curr_time_idx}_semantics_gt.png"), cv2.cvtColor(gt_semantics_numpy, cv2.COLOR_RGB2BGR))
                cv2.imwrite(os.path.join(images_dir, f"frame_{curr_time_idx}_color_rendered.png"), cv2.cvtColor(rendered_color_numpy, cv2.COLOR_RGB2BGR))
                # cv2.imwrite(os.path.join(images_dir, f"frame_{curr_time_idx}_depth_rendered.png"), rendered_depth_vis_color)
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
            sample_data = self.get_colmap_sample_data(file_name, eval=True)
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
            psnr, ssim, lpips_score, rmse, depth_l1, miou = eval_single_frame(gt_color_torch, gt_depth_torch, gt_semantics_torch, gt_confidence_map_torch, rendered_color_torch, rendered_depth_torch, rendered_semantics_torch)

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
                gt_depth_numpy = gt_depth_torch.detach().cpu().numpy()
                
                rendered_color_numpy  = (rendered_color_torch.permute(1, 2, 0).detach().cpu().numpy() * 255).astype(np.uint8)
                rendered_depth_numpy = rendered_depth_torch.squeeze().detach().cpu().numpy()
                rendered_semantics_numpy = (rendered_semantics_torch.permute(1, 2, 0).detach().cpu().numpy() * 255).astype(np.uint8)
                rendered_silhouette_numpy = rendered_silhouette_torch.detach().cpu().numpy()

                valid_mask = rendered_silhouette_numpy > 0.2 #0.9
                # rendered_color_numpy[~valid_mask] = 0
                rendered_depth_numpy[~valid_mask] = 0
                rendered_semantics_numpy[~valid_mask] = 0

                c2w_numpy = c2w_torch.detach().cpu().numpy()
                np.savetxt(os.path.join(images_dir, f"{file_name}_pose.txt"), c2w_numpy)

                # converting depth maps to colorable depth maps for visualization
                # depth_min = 0
                # depth_max = self.max_depth
                # depth_range = depth_max - depth_min
                # gt_depth_vis = (gt_depth_numpy - depth_min) / depth_range * 255
                # gt_depth_vis = gt_depth_vis.astype(np.uint8)
                # gt_depth_vis_color = cv2.applyColorMap(gt_depth_vis, cv2.COLORMAP_JET)
                # rendered_depth_vis = (rendered_depth_numpy - depth_min) / depth_range * 255
                # rendered_depth_vis = rendered_depth_vis.astype(np.uint8)
                # rendered_depth_vis_color = cv2.applyColorMap(rendered_depth_vis, cv2.COLORMAP_JET)
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
                # cv2.imwrite(os.path.join(images_dir, f"{file_prefix}_depth_gt.png"), gt_depth_vis_color)
                gt_depth_vis_color.save(os.path.join(images_dir, f"{file_prefix}_depth_gt.png"))
                cv2.imwrite(os.path.join(images_dir, f"{file_prefix}_semantics_gt.png"), cv2.cvtColor(gt_semantics_numpy, cv2.COLOR_RGB2BGR))
                cv2.imwrite(os.path.join(images_dir, f"{file_prefix}_color_rendered.png"), cv2.cvtColor(rendered_color_numpy, cv2.COLOR_RGB2BGR))
                # cv2.imwrite(os.path.join(images_dir, f"{file_prefix}_depth_rendered.png"), rendered_depth_vis_color)
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
        # printing metrics
        print(f"\nTest Evaluation Metrics:")
        print(f"Number of gaussians: {self.params['means3D'].shape[0]}")
        print(f"Mean PSNR: {mean_psnr}")
        print(f"Mean SSIM: {mean_ssim}")
        print(f"Mean LPIPS: {mean_lpips}")
        print(f"Mean RMSE: {mean_rmse}")
        print(f"Mean Depth L1: {mean_depth_l1}")
        print(f"Mean mIoU: {mean_miou}")
    
    def save_side_view_img(self, curr_time_idx):
        if self.running_colmap_dataset:
            return # since we dont have the transform from link_base to world frame.
        images_dir = os.path.join(self.episode_dir, "side_images")
        os.makedirs(images_dir, exist_ok=True)
        # Rendering right side view
        R_base_cam = np.array([[-1.0, 0.0, 0.0],
                        [0.0, 0.0, -1.0],
                        [0.0, -1.0, 0.0]])
        
        pose_w_base = self.get_transform('world', 'link_base')
        T_w_base = ros_pose_to_SE3(pose_w_base)
        R_w_base = T_w_base[:3, :3]
        R_w_cam = np.matmul(R_w_base, R_base_cam)
        x_base = T_w_base[0, 3]
        y_base = T_w_base[1, 3]
        z_base = T_w_base[2, 3]
        t = np.array([[x_base],
                        [y_base],
                        [z_base + 0.2]]) # position of camera wrt to world frame
        t = t.reshape(3, 1)
        T_wc = np.block([[R_w_cam, t],[0.0, 0.0, 0.0, 1.0]])
        T_cw = np.linalg.inv(T_wc)
        side_view_torch, _, _, _ = render_any_cam(self.params, T_cw, height = 480, width = 1280)
        side_view_torch = side_view_torch.permute(1, 2, 0).detach().cpu()
        side_view_torch[side_view_torch==0] = 1.0 # set background white
        side_view_numpy = (side_view_torch.numpy() * 255).astype(np.uint8)
        cv2.imwrite(os.path.join(images_dir, f"frame_{curr_time_idx}_side_view_rendered.png"), cv2.cvtColor(side_view_numpy, cv2.COLOR_RGB2BGR))
def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("experiment", type=str, help="Path to experiment file")
    parser.add_argument("alternative_colmap_dataset", type=str, help="Path to alternative colmap dataset")

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
    return experiment.config, args.alternative_colmap_dataset

if __name__ == '__main__':
    try:
        config, alternative_colmap_dataset = main()
        node = ActiveSLAM(config, alternative_colmap_dataset)
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
