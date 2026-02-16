#!/usr/bin/env python3
#!/usr/bin/env python3
"""
Example 1: Basic SLAM Node Usage
Demonstrates how to use the SLAM node with default parameters
"""

import rclpy
from rclpy.node import Node
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Note: This would include the 4dgs_slam_ros2 package imports
# from 4dgs_slam_ros2.node import SLAMNode


class BasicExample(Node):
    """Basic example demonstrating SLAM node usage"""
    
    def __init__(self):
        super().__init__('basic_example')
        self.get_logger().info('Starting basic SLAM example')
        
        # Initialize SLAM node (this would be imported)
        # self.slam_node = SLAMNode()
        
        # Subscribe to data topics
        self.image_sub = self.create_subscription(
            Image,
            '/camera/image_raw',
            self.image_callback,
            10
        )
        
        self.depth_sub = self.create_subscription(
            Image,
            '/camera/depth',
            self.depth_callback,
            10
        )
        
        self.get_logger().info('Basic example ready to process frames')
    
    def image_callback(self, msg):
        """Handle incoming image"""
        pass
    
    def depth_callback(self, msg):
        """Handle incoming depth image"""
        pass
    
    def run_example(self):
        """Run the example"""
        rclpy.spin(self)


def main():
    """Main entry point"""
    rclpy.init()
    example = BasicExample()
    
    try:
        example.run_example()
    except KeyboardInterrupt:
        example.get_logger().info('Example stopped')
    finally:
        example.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()