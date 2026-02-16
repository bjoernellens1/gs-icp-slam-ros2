#!/usr/bin/env python3
"""
Test configuration and parameter loading
"""

import os
import sys
import unittest
import yaml
from pathlib import Path

# Add parent package to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from parameters import SLAMParameters


class TestSLAMParameters(unittest.TestCase):
    """Test cases for SLAM parameters"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.params = SLAMParameters()
    
    def test_parameter_initialization(self):
        """Test that parameters are initialized correctly"""
        self.assertIsNotNone(self.params)
        self.assertIsInstance(self.params.config, dict)
    
    def test_camera_parameters(self):
        """Test camera parameter initialization"""
        camera_params = self.params.get('slam', 'camera_model', default='pinhole')
        self.assertEqual(camera_params, 'pinhole')
    
    def test_get_nested_parameter(self):
        """Test getting nested configuration parameters"""
        num_gaussians = self.params.get('slam', 'num_gaussians', default=100000)
        focal_length = self.params.get('slam', 'focal_length', default=527.0)
        
        self.assertEqual(num_gaussians, 100000)
        self.assertEqual(focal_length, 527.0)
    
    def test_default_value(self):
        """Test getting parameters with default values"""
        custom_param = self.params.get('slam', 'custom_param', default='default_value')
        self.assertEqual(custom_param, 'default_value')


class TestConfigurationLoading(unittest.TestCase):
    """Test case for configuration file loading"""
    
    def test_yaml_loading(self):
        """Test YAML configuration file loading"""
        config_path = 'config/slam_config.yaml'
        
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                content = yaml.safe_load(f)
                self.assertIsInstance(content, dict)
                self.assertIn('slam', content)
                self.assertIn('system', content)


if __name__ == '__main__':
    unittest.main()