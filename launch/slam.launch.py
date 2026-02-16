from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os

def generate_launch_description():
    # Get package directory
    pkg_share = get_package_share_directory('gs_icp_slam')

    # Launch arguments
    dataset_path_arg = DeclareLaunchArgument(
        'dataset_path', default_value='',
        description='Path to dataset (optional)'
    )
    
    config_path_arg = DeclareLaunchArgument(
        'config_path', default_value='',
        description='Path to camera calibration config file (optional)'
    )
    
    verbose_arg = DeclareLaunchArgument(
        'verbose', default_value='false',
        description='Enable verbose logging'
    )
    
    rerun_viewer_arg = DeclareLaunchArgument(
        'rerun_viewer', default_value='true',
        description='Enable Rerun viewer'
    )
    
    use_external_pose_arg = DeclareLaunchArgument(
        'use_external_pose', default_value='false',
        description='Use external pose topic instead of internal GICP'
    )
    
    camera_topic_arg = DeclareLaunchArgument(
        'camera_topic', default_value='/camera/rgb/image_raw',
        description='RGB image topic'
    )
    
    depth_topic_arg = DeclareLaunchArgument(
        'depth_topic', default_value='/camera/depth/image_raw',
        description='Depth image topic'
    )
    
    camera_info_topic_arg = DeclareLaunchArgument(
        'camera_info_topic', default_value='/camera/camera_info',
        description='Camera info topic'
    )
    
    pose_topic_arg = DeclareLaunchArgument(
        'pose_topic', default_value='/pose',
        description='External pose topic'
    )

    # Node
    slam_node = Node(
        package='gs_icp_slam',
        executable='gs_icp_slam_node',
        name='gs_icp_slam_node',
        output='screen',
        parameters=[{
            'dataset_path': LaunchConfiguration('dataset_path'),
            'config_path': LaunchConfiguration('config_path'),
            'verbose': LaunchConfiguration('verbose'),
            'rerun_viewer': LaunchConfiguration('rerun_viewer'),
            'use_external_pose': LaunchConfiguration('use_external_pose'),
            'camera_topic': LaunchConfiguration('camera_topic'),
            'depth_topic': LaunchConfiguration('depth_topic'),
            'camera_info_topic': LaunchConfiguration('camera_info_topic'),
            'pose_topic': LaunchConfiguration('pose_topic'),
            # Other parameters can be added here with default values
            'keyframe_th': 0.7,
            'knn_maxd': 99999.0,
            'overlapped_th': 0.0005,
            'max_correspondence_distance': 0.02,
            'trackable_opacity_th': 0.05,
            'overlapped_th2': 0.00005,
            'downsample_rate': 10,
        }]
    )

    return LaunchDescription([
        dataset_path_arg,
        config_path_arg,
        verbose_arg,
        rerun_viewer_arg,
        use_external_pose_arg,
        camera_topic_arg,
        depth_topic_arg,
        camera_info_topic_arg,
        pose_topic_arg,
        slam_node
    ])
