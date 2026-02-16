#!/usr/bin/env python3
"""Launch file for basic 4DGS-SLAM node"""
from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    """Generate launch description for 4DGS-SLAM node"""
    
    return LaunchDescription([
        Node(
            package='gs_icp_slam',
            executable='gs_icp_slam_node',
            name='gs_icp_slam_node',
            output='screen',
            parameters=[],
            remappings=[],
        ),
    ])