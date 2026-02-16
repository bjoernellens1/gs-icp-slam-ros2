#!/usr/bin/env python3
"""Launch file for 4DGS-SLAM with ROS bag input"""
from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, ExecuteProcess
from launch.substitutions import LaunchConfiguration, PythonExpression
from launch.launch_description_sources import PythonLaunchDescriptionSource
import os


def generate_launch_description():
    """Generate launch description for 4DGS-SLAM with ROS bag"""
    
    # Declare launch arguments
    bag_file_arg = DeclareLaunchArgument(
        'bag_file',
        default_value='',
        description='Path to ROS bag file to process'
    )
    
    image_topic_arg = DeclareLaunchArgument(
        'image_topic',
        default_value='/camera/image_raw',
        description='Topic to subscribe for camera images'
    )
    
    depth_topic_arg = DeclareLaunchArgument(
        'depth_topic',
        default_value='/camera/depth',
        description='Topic to subscribe for depth images'
    )
    
    config_file_arg = DeclareLaunchArgument(
        'config_file',
        default_value='',
        description='Path to custom configuration file'
    )
    
    start_time_arg = DeclareLaunchArgument(
        'start_time',
        default_value='-0.0',
        description='Start timestamp in the bag file'
    )
    
    end_time_arg = DeclareLaunchArgument(
        'end_time',
        default_value='-0.0',
        description='End timestamp in the bag file'
    )
    
    # 4DGS-SLAM node
    slam_node = Node(
        package='gs_icp_slam',
        executable='gs_icp_slam_node',
        name='gs_icp_slam_node',
        output='screen',
        parameters=[],
        remappings=[
            ('image_raw', LaunchConfiguration('image_topic')),
            ('depth', LaunchConfiguration('depth_topic')),
        ],
    )
    
    # ROS bag player configuration
    bag_config = {
        'bag_file': LaunchConfiguration('bag_file'),
        'start_time': LaunchConfiguration('start_time'),
        'end_time': LaunchConfiguration('end_time'),
    }
    
    return LaunchDescription([
        bag_file_arg,
        image_topic_arg,
        depth_topic_arg,
        config_file_arg,
        start_time_arg,
        end_time_arg,
        slam_node,
    ])