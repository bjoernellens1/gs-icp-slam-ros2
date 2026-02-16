#!/usr/bin/env python3
"""Launch file for 4DGS-SLAM visualization"""

from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import IncludeLaunchDescription, ExecuteProcess
from launch.substitutions import LaunchConfiguration, PythonExpression
from launch.launch_description_sources import PythonLaunchDescriptionSource
import os


def generate_launch_description():
    """Generate launch description for 4DGS-SLAM visualization"""
    
    # ROS 2 RViz2 visualization
    rviz_node = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        output='screen',
    )
    
    # 4DGS-SLAM node
    slam_node = Node(
        package='gs_icp_slam',
        executable='gs_icp_slam_node',
        name='gs_icp_slam_node',
        output='screen',
    )
    
    # TF visualization node (optional)
    tf_node = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        arguments=['0', '0', '0', '0', '0', '0', 'map', 'camera'],
        name='static_transform_publisher'
    )
    
    return LaunchDescription([
        rviz_node,
        slam_node,
        tf_node,
    ])