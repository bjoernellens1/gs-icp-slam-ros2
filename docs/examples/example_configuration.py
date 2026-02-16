#!/usr/bin/env python3
"""
Example 3: Real-time SLAM Configuration
Demonstrates advanced configuration for real-time SLAM operations
"""

import yaml
import os
from pathlib import Path


class RealtimeSLAMConfig:
    """Configuration class for real-time SLAM operations"""
    
    DEFAULT_CONFIG = {
        'slam': {
            'camera_model': 'pinhole',
            'focal_length': 527.0,
            'principal_point': [326.0, 247.0],
            'num_gaussians': 50000,
            'tracking_max_iter': 20,
            'keyframe_distance': 1.0,
            'keyframe_angle': 0.6,
            'min_depth': 0.1,
            'max_depth': 10.0,
        },
        'gpu': {
            'cuda_device': 0,
            'batch_size': 2048,
            'use_mixed_precision': True,
        },
        'monitoring': {
            'publish_odometry': True,
            'publish_poses': True,
            'publish_trajectory': True,
            'publish_statistics': False,
        },
        'system': {
            'working_directory': '/tmp/gs_icp_slam_realtime',
            'save_results': True,
            'save_format': 'ply',
        }
    }
    
    def __init__(self, config_file=None):
        """
        Initialize configuration
        
        Args:
            config_file: Optional path to custom configuration file
        """
        self.config = self.DEFAULT_CONFIG.copy()
        
        if config_file and os.path.exists(config_file):
            self.load_config(config_file)
    
    def load_config(self, config_file):
        """Load configuration from file"""
        with open(config_file, 'r') as f:
            loaded_config = yaml.safe_load(f)
            self._deep_update(self.config, loaded_config)
    
    def save_config(self, config_file):
        """Save configuration to file"""
        with open(config_file, 'w') as f:
            yaml.dump(self.config, f, default_flow_style=False)
    
    def _deep_update(self, base_dict, update_dict):
        """Recursively update dictionary values"""
        for key, value in update_dict.items():
            if isinstance(base_dict.get(key), dict) and isinstance(value, dict):
                self._deep_update(base_dict[key], value)
            else:
                base_dict[key] = value
    
    def configure_for_performance(self):
        """Apply performance-focused configuration"""
        self.config['slam']['num_gaussians'] = 30000
        self.config['slam']['tracking_max_iter'] = 15
        self.config['slam']['keyframe_distance'] = 1.2
        self.config['slam']['keyframe_angle'] = 0.7
        self.config['gpu']['batch_size'] = 4096
        self.config['monitoring']['publish_statistics'] = False
        self.config['monitoring']['publish_trajectory'] = False
    
    def configure_for_quality(self):
        """Apply quality-focused configuration"""
        self.config['slam']['num_gaussians'] = 150000
        self.config['slam']['tracking_max_iter'] = 50
        self.config['slam']['keyframe_distance'] = 0.8
        self.config['slam']['keyframe_angle'] = 0.4
        self.config['gpu']['batch_size'] = 1024
        self.config['monitoring']['publish_statistics'] = True
        self.config['monitoring']['publish_trajectory'] = True
    
    def configure_for_large_scenes(self):
        """Apply configuration for large scene mapping"""
        self.config['slam']['num_gaussians'] = 200000
        self.config['slam']['tracking_max_iter'] = 40
        self.config['slam']['keyframe_distance'] = 2.0
        self.config['slam']['keyframe_angle'] = 0.8
        self.config['slam']['dynamic_confidence_threshold'] = 0.8
        self.config['gpu']['batch_size'] = 1024
        self.config['gpu']['cuda_device'] = 1
    
    def get_camera_parameters(self):
        """Get current camera configuration"""
        return self.config['slam']
    
    def get_gpu_parameters(self):
        """Get current GPU configuration"""
        return self.config['gpu']


def create_realtime_config():
    """Create configuration for real-time operation"""
    config = RealtimeSLAMConfig()
    
    # Apply performance configuration
    config.configure_for_performance()
    
    # Save configuration
    config.save_config('config/realtime_performance.yaml')
    
    print("Real-time performance configuration created:")
    print("  - Num Gaussians: 30000")
    print("  - Tracking iterations: 15")
    print("  - Keyframe distance: 1.2m")
    print("  - Batch size: 4096")
    
    return config


def create_test_config():
    """Create configuration for testing purposes"""
    config = RealtimeSLAMConfig()
    
    # Configure for quick testing
    config.configure_for_quality()
    config.save_config('config/test_config.yaml')
    
    return config


def main():
    """Main example runner"""
    
    print("GS-ICP-SLAM Real-time Configuration Examples\n")
    
    # Example 1: Real-time configuration
    print("Example 1: Real-time Performance Configuration")
    print("------ ----------------------------------------")
    realtime_config = create_realtime_config()
    
    # Example 2: Test configuration
    print("\nExample 2: Testing Configuration")
    print("-------- ------------------------")
    test_config = create_test_config()
    
    # Display configuration details
    print("\nConfiguration Details:")
    print(f"Camera model: {config.get_camera_parameters()['camera_model']}")
    print(f"Number of Gaussians: {config.get_camera_parameters()['num_gaussians']}")
    print(f"GPU device: {config.get_gpu_parameters()['cuda_device']}")


if __name__ == '__main__':
    main()