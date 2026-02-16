#!/usr/bin/env python3
#!/usr/bin/env python3
"""
4D Gaussian Splatting SLAM ROS2 Package
This file initializes the ROS2 Python package
"""

from 4dgs_slam_ros2.node import SLAMNode
from 4dgs_slam_ros2.parameters import SLAMParameters

__version__ = "1.0.0"
__author__ = "Your Name"
__license__ = "MIT"
__email__ = "your.email@example.com"
__maintainer__ = "Your Name"

# Export main classes for easy access
__all__ = ['SLAMNode', 'SLAMParameters']