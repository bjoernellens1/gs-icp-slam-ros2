#!/usr/bin/env python3
"""
Example script for running 4DGS-SLAM with custom parameters
"""

import subprocess
import argparse


def run_slam_with_custom_config():
    """Run SLAM with custom configuration file"""
    
    parser = argparse.ArgumentParser(description='Run 4DGS-SLAM with custom configuration')
    parser.add_argument('config_file', type=str, help='Path to configuration YAML file')
    parser.add_argument('--bag', type=str, default='', help='Path to ROS bag file')
    parser.add_argument('--image-topic', type=str, default='/camera/image_raw', help='ROS topic for images')
    parser.add_argument('--depth-topic', type=str, default='/camera/depth', help='ROS topic for depth images')
    
    args = parser.parse_args()
    
    # Build launch command
    cmd = ['ros2', 'launch', '4dgs_slam', '4dgs_slam_with_bag.launch.py']
    
    if args.config_file:
        cmd.extend(['--config-file', args.config_file])
    
    if args.bag:
        cmd.extend(['--bag-file', args.bag])
    
    if args.image_topic:
        cmd.extend(['--image-topic', args.image_topic])
    
    if args.depth_topic:
        cmd.extend(['--depth-topic', args.depth_topic])
    
    # Run the launch file
    subprocess.run(cmd)


def run_standalone_slam():
    """Run SLAM in standalone mode"""
    
    parser = argparse.ArgumentParser(description='Run SLAM in standalone mode')
    parser.add_argument('--workspace', type=str, default='/tmp/4dgs_slam', help='Working directory')
    parser.add_argument('--gpu-id', type=int, default=0, help='GPU device ID')
    parser.add_argument('--num-gaussians', type=int, default=100000, help='Number of Gaussians')
    
    args = parser.parse_args()
    
    print(f"Running 4DGS-SLAM with parameters:")
    print(f"  - Workspace: {args.workspace}")
    print(f"  - GPU: {args.gpu_id}")
    print(f"  - Gaussians: {args.num_gaussians}")
    
    # TODO: Implement SLAM launch logic
    # This would start the SLAM node and set up necessary subsystems


def run_visualization():
    """Run SLAM with visualization"""
    
    cmd = ['ros2', 'launch', '4dgs_slam', '4dgs_slam_visualization.launch.py']
    
    subprocess.run(cmd)


def main():
    """Main function to handle different SLAM modes"""
    
    parser = argparse.ArgumentParser(description='4DGS-SLAM ROS2 Package')
    subparsers = parser.add_subparsers(dest='mode', help='Execution mode')
    
    # Configuration mode
    config_parser = subparsers.add_parser('config', help='Run with custom configuration')
    config_parser.add_argument('config_file', type=str, help='Path to configuration YAML file')
    config_parser.add_argument('--bag', type=str, default='', help='ROS bag file')
    
    # Standalone mode
    subparsers.add_parser('standalone', help='Run in standalone mode')
    
    # Visualization mode
    subparsers.add_parser('visualize', help='Run with visualization')
    
    args = parser.parse_args()
    
    if args.mode == 'config':
        run_slam_with_custom_config()
    elif args.mode == 'standalone':
        run_standalone_slam()
    elif args.mode == 'visualize':
        run_visualization()
    else:
        parser.print_help()


if __name__ == '__main__':
    main()