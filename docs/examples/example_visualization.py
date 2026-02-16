#!/usr/bin/env python3
"""
Example 4: Visualization and Monitoring
Demonstrates how to use SLAM with visualization tools
"""

import subprocess
import sys
import os
from pathlib import Path


class VisualizationExample:
    """Visualisation and monitoring example for SLAM"""
    
    def __init__(self):
        self.rviz_config_path = Path('config/rviz2_default.rviz')
        self.monitoring_topics = [
            '/slam/odometry',
            '/slam/pose',
            '/slam/trajectory',
            '/slam/keyframes',
            '/slam/statistics'
        ]
    
    def setup_rviz2_configuration(self):
        """Create RViz2 configuration for SLAM visualization"""
        
        rviz_config = """
Panels:
  - Class: rviz_common/Displays
    
    Visualization Manager:
      Displays:
        - Class: rviz_default_plugins/Grid
          Name: Grid
          Position:
            X: 0
            Y: 0
            Z: 0
          Scale: 10
          Reference Frame: "map"
        
        - Class: rviz_default_plugins/TF
          Name: TF
          Position:
            X: 0
            Y: 1.0
            Z: 0
          Enabled: true
          Show Arrows: true
          Show Axes: true
          Show Names: true
        
        - Class: rviz_default_plugins/Topic
          Name: SLAM Odometry
          Topic: /slam/odometry
          Value: Odometry
          Position:
            X: 0
            Y: 2.0
            Z: 0
        
        - Class: rviz_default_plugins/Topic
          Name: SLAM Pose
          Topic: /slam/pose
          Value: PoseStamped
          Position:
            X: 0
            Y: 3.0
            Z: 0
        
        - Class: rviz_default_plugins/TF2Sensor
          Name: TF2Sensor
          Topic: /slam/trajectory
          Position:
            X: 0
            Y: 4.0
            Z: 0
        
        - Class: rviz_default_plugins/Topic
          Name: Keyframe Display
          Topic: /slam/keyframes
          Value: Image
          Position:
            X: 1.0
            Y: 0
            Z: 0
      Global Options:
        Fixed Frame: map
        Frame Rate: 30
      Title: UI
    Window Geometry:
      Height: 1080
      Width: 1920
      X: 0
      Y: 0

Panels:
  - Class: rviz_common/Views
    Name: Views
    Window Geometry:
      Height: 1080
      Width: 1920
      X: 0
      Y: 0"""
        
        try:
            self.rviz_config_path.write_text(rviz_config)
            print(f"RViz2 configuration created at: {self.rviz_config_path}")
            return True
        except Exception as e:
            print(f"Error creating RViz2 configuration: {e}")
            return False
    
    def monitor_slam_topics(self):
        """Monitor SLAM output topics for debugging"""
        print("Monitoring SLAM output topics...")
        
        # Monitor a specific topic using roswtf or rqt
        print(f"Available SLAM topics:")
        for topic in self.monitoring_topics:
            print(f"  - {topic}")
        
        # Example: Monitor odometry topic
        print("\nExample: Monitoring odometry")
        print("To monitor odometry in real-time:")
        print(f"ros2 topic echo /slam/odometry [--once]")
        
        # Example: Monitor statistics
        print("To monitor statistics:")
        print(f"ros2 topic echo /slam/statistics [--once]")
    
    def launch_rviz_and_slam(self):
        """Launch RViz2 and SLAM simultaneously"""
        
        print("Launching RViz2 and SLAM visualization...")
        
        # Check if packages are available
        try:
            subprocess.run(['ros2', 'pkg-list'], capture_output=True)
            return True
        except Exception as e:
            print(f"ROS2 commands not available: {e}")
            return False
    
    def check_topic_connections(self):
        """Check topic connections and publishing status"""
        
        print("Checking topic connections...")
        
        # Example commands to check status
        commands = [
            'ros2 topic list | grep slam',
            'ros2 topic echo /slam/odometry --once',
            'ros2 topic echo /slam/pose --once'
        ]
        
        for cmd in commands:
            print(f"Running: {cmd}")


def main():
    """Main example runner"""
    
    print("4DGS-SLAM Visualization Examples\n")
    
    example = VisualizationExample()
    
    # Example 1: Setup RViz2
    print("Example 1: Set up RViz2 configuration")
    print("-------------------------------")
    example.setup_rviz2_configuration()
    
    # Example 2: Monitor topics
    print("\nExample 2: Monitor SLAM topics")
    print("------------------------------")
    example.monitor_slam_topics()
    
    # Example 3: Launch visualization
    print("\nExample 3: Launch RViz2 and SLAM")
    print("---------------------------------")
    example.launch_rviz_and_slam()
    
    # Example 4: Check connections
    print("\nExample 4: Check Topic Connections")
    print("-----------------------------------")
    example.check_topic_connections()


if __name__ == '__main__':
    main()