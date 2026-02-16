#!/usr/bin/env python3
"""
4DGS-SLAM ROS2 Node
Node that integrates the 4D Gaussian Splatting SLAM system with ROS2
"""

import os
import sys
import yaml
import time
import numpy as np
from typing import Optional, List, Dict

# ROS2 imports
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSHistoryPolicy, QoSReliabilityPolicy
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import Transform, TransformStamped, PoseStamped
from std_msgs.msg import Header, String
from nav_msgs.msg import Odometry
from cv_bridge import CvBridge

# Add parent package to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class SLAMParameters(Node):
    """ROS2 configuration parameters for SLAM system"""
    
    def __init__(self):
        super().__init__('slam_parameters')
        
        # SLAM configuration parameters (will be loaded from config)
        self.config = {
            'slam': {
                'camera_model': 'pinhole',
                'focal_length': 527.0,
                'principal_point': [326.0, 247.0],
                'use_dynamic_segmentation': True,
                'min_num_keyframes': 3,
                'max_keyframes': 20,
                'max_depth': 10.0,
                'min_depth': 0.1,
                'keyframe_distance': 0.5,
                'keyframe_angle': 0.4,
                'dynamic_confidence': 0.7,
                'render_fps': 30,
                'num_gaussians': 100000,
                'gaussian_decay_rate': 0.975,
                'tracking_max_iter': 30,
                'mapping_update_interval': 5,
                'visualization_interval': 1,
                'save_interval': 50,
            },
            'gpu': {
                'cuda_device': 0,
                'batch_size': 1024,
                'use_tensor_cores': True
            },
            'system': {
                'working_directory': '/tmp/4dgs_slam',
                'log_level': 'INFO',
                'enable_gui': False,
                'save_results': True,
                'load_from_checkpoint': '',
            },
            'data': {
                'use_rosbag': True,
                'bag_file': '',
                'image_topic': '/camera/image_raw',
                'depth_topic': '/camera/depth',
                'camera_info_topic': '/camera/camera_info',
            },
            'monitoring': {
                'publish_odometry': True,
                'publish_poses': True,
                'publish_statistics': True,
                'statistics_interval': 1.0,
            }
        }
        
        self.load_config_file()
    
    def load_config_file(self):
        """Load configuration from YAML file or use defaults"""
        config_path = self.get_node_parameters('slam_config_file', 'slam_config.yaml')
        
        if isinstance(config_path, str) and os.path.exists(config_path):
            try:
                with open(config_path, 'r') as f:
                    loaded_config = yaml.safe_load(f)
                    # Update configuration with loaded values
                    self._deep_update(self.config, loaded_config)
                    self.get_logger().info(f'Loaded configuration from {config_path}')
            except Exception as e:
                self.get_logger().warning(f'Failed to load config file: {e}')
        else:
            self.get_logger().info('Using default configuration')
    
    def _deep_update(self, base_dict, update_dict):
        """Recursively update dictionary values"""
        for key, value in update_dict.items():
            if key in base_dict and isinstance(base_dict[key], dict) and isinstance(value, dict):
                self._deep_update(base_dict[key], value)
            else:
                base_dict[key] = value
    
    def get(self, *keys, default=None):
        """Get nested configuration value"""
        value = self.config
        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return default
        return value
    
    def get_node_parameters(self, param_name, default_value=None):
        """Get parameter from ROS2 parameter server"""
        try:
            param_value = self.get_parameter(param_name).value if default_value is None else default_value
            return param_value
        except Exception:
            return default_value