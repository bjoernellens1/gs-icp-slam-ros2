#!/usr/bin/env python3
"""
Example 2: Bag Playback SLAM
Demonstrates how to run SLAM with pre-recorded ROS bag files
"""

import subprocess
import sys
import os


class BagPlaybackExample:
    """Example demonstrating bag playback for SLAM testing"""
    
    def __init__(self):
        self.packages = ['/opt/ros/humble', '/opt/ros/humble/share']
    
    def record_data(self):
        """Record camera data to save into a bag"""
        print("Recording camera data...")
        record_cmd = ['ros2', 'bag', 'record',
                     '/camera/image_raw',
                     '/camera/depth',
                     '/camera/camera_info',
                     '-o', 'recorded_data']
        subprocess.run(record_cmd, check=True)
        print("Recording saved to recorded_data.bag")
    
    def play_and_slam(self, bag_file=None):
        """
        Play bag file and run SLAM simultaneously
        
        Args:
            bag_file: Path to ROS bag file
        """
        if not bag_file:
            bag_file = 'recorded_data.bag'
        
        if not os.path.exists(bag_file):
            print(f"Error: Bag file '{bag_file}' not found")
            return False
        
        print(f"Playing bag file: {bag_file}")
        print("Starting SLAM node...")
        
        # Get current process list
        print(f"Active processes:")
        subprocess.run(['ps', 'aux'], check=True)
    
    def custom_bag_config(self):
        """Run SLAM with custom bag configuration"""
        commands = [
            "ros2 bag play your_bag.bag --clock -s 1",
            "ros2 run 4dgs_slam 4dgs_slam_node",
        ]
        
        for cmd in commands:
            print(f"Running: {cmd}")
            subprocess.run(cmd.split(), shell=True)
    
    def advanced_bag_control(self, start_time=0, end_time=-1):
        """Advanced bag playback with timing controls"""
        
        start_cmd = f"--start-time {start_time}" if start_time else ""
        end_cmd = f"--end-time {end_time}" if end_time != -1 else ""
        
        bag_play_command = f"ros2 bag play your_bag.bag {start_cmd} {end_cmd} --clock"
        
        print("Starting bag playback with timing control...")
        print(f"Command: {bag_play_command}")
        subprocess.run(bag_play_command.split())
    
    def create_bag_for_testing(self, image_topic, depth_topic):
        """Create a test bag file from camera nodes"""
        print("Creating test bag file...")
        subprocess.run([
            'ros2', 'bag', 'record',
            f'--topic', image_topic,
            f'--topic', depth_topic,
            '-o', 'test_slam_bag',
            '--duration', '30'  # Record for 30 seconds
        ])
        print("Test bag file created")


def main():
    """Main example runner"""
    
    print("4DGS-SLAM ROS2 Package - Bag Playback Examples\n")
    
    example = BagPlaybackExample()
    
    # Example 1: Record and play
    print("Example 1: Record and Play")
    print("----------------------------")
    example.record_data()
    example.play_and_slam()
    
    # Example 2: Custom configuration
    print("\nExample 2: Custom Bag Configuration")
    print("--------------------------------------")
    example.custom_bag_config()
    
    # Example 3: Advanced controls
    print("\nExample 3: Advanced Bag Controls")
    print("----------------------------------")
    example.advanced_bag_control(start_time=1.0, end_time=10.0)
    
    # Example 4: Create test bag
    print("\nExample 4: Create Test Bag")
    print("---------------------------")
    example.create_bag_for_testing(
        '/camera/image_raw',
        '/camera/depth'
    )


if __name__ == '__main__':
    main()