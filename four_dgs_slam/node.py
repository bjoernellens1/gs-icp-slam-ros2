#!/usr/bin/env python3
"""
ROS2 Node for 4D Gaussian Splatting SLAM
Main node that handles SLAM processing, visualization, and ROS2 integration
"""

import os
import sys
import cv2
import numpy as np
from typing import Optional, Tuple
import time
from typing import Optional, Tuple, Dict, List

# ROS2 imports
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSHistoryPolicy, QoSReliabilityPolicy
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import Transform, TransformStamped, PoseStamped
from std_msgs.msg import Header, String
from nav_msgs.msg import Odometry
from cv_bridge import CvBridge

# Add parent package to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import from local package
from four_dgs_slam.parameters import SLAMParameters


class SLAMNode(Node):
    """
    ROS2 node for 4D Gaussian Splatting SLAM system
    Integrates RGB-D SLAM with Gaussian splatting for dynamic scene reconstruction
    """
    
    def __init__(self):
        super().__init__('four_dgs_slam_node')
        
        self.get_logger().info('Initializing 4DGS-SLAM ROS2 Node')
        
        # Initialize SLAM parameters
        self.config = SLAMParameters()
        
        # SLAM state
        self.is_running = True
        self.frame_count = 0
        self.last_time = time.time()
        
        # Feature containers
        self.reconstruction = None
        self.camera_trajectory = []
        self.keyframes = []
        self.dynamic_objects = []
        
        # ROS2 publishers
        self.trajectory_pub = None
        self.odometry_pub = None
        self.poses_pub = None
        self.statistics_pub = None
        self.keyframe_pub = None
        self.generation_progress_pub = None
        
        # ROS2 subscribers
        self.image_sub = None
        self.depth_sub = None
        self.info_sub = None
        
        # Initialize ROS2 components
        self.cv_bridge = CvBridge()
        self.setup_qos()
        self.setup_publishers()
        self.setup_subscribers()
        
        # Create working directories
        self._create_working_directories()
        
        self.get_logger().info(f'4DGS-SLAM Node initialized with working directory: {self.config.get("system", "working_directory")}')
    
    def setup_qos(self):
        """Configure Quality of Service profiles"""
        self.qos_depth_image = QoSProfile(
            depth=10,
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST
        )
        
        self.qos_depth = QoSProfile(
            depth=10,
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST
        )
        
        self.qos_info = QoSProfile(
            depth=1,
            history=QoSHistoryPolicy.KEEP_LAST
        )
        
        self.qos_default = QoSProfile(
            depth=10,
            reliability=QoSReliabilityPolicy.RELIABLE,
            history=QoSHistoryPolicy.KEEP_LAST
        )
    
    def setup_publishers(self):
        """Set up ROS2 publishers with appropriate QoS"""
        qos_standard = self.qos_default
        qos_camera = self.qos_depth
        
        # Odometry publisher
        if self.config.get('monitoring', 'publish_odometry', default=True):
            self.odometry_pub = self.create_publisher(
                Odometry,
                '/slam/odometry',
                qos_camera
            )
        
        # Pose publisher
        if self.config.get('monitoring', 'publish_poses', default=True):
            self.poses_pub = self.create_publisher(
                PoseStamped,
                '/slam/pose',
                qos_standard
            )
        
        # Trajectory publisher
        self.trajectory_pub = self.create_publisher(
            PoseStamped,
            '/slam/trajectory',
            qos_standard
        )
        
        # Keyframe publisher
        if self.config.get('monitoring', 'publish_poses', default=True):
            self.keyframe_pub = self.create_publisher(
                Image,
                '/slam/keyframes',
                qos_camera
            )
        
        # Generation progress publisher
        self.generation_progress_pub = self.create_publisher(
            String,
            '/slam/generation_progress',
            qos_standard
        )
    
    def setup_subscribers(self):
        """Set up ROS2 subscribers"""
        data_config = self.config.get('data', default={})
        
        image_topic = data_config.get('image_topic', '/camera/image_raw')
        depth_topic = data_config.get('depth_topic', '/camera/depth')
        info_topic = data_config.get('camera_info_topic', '/camera/camera_info')
        
        # Subscribe to image
        self.image_sub = self.create_subscription(
            Image,
            image_topic,
            self.image_callback,
            self.qos_depth_image
        )
        
        # Subscribe to depth
        self.depth_sub = self.create_subscription(
            Image,
            depth_topic,
            self.depth_callback,
            self.qos_depth
        )
        
        # Subscribe to camera info (optional)
        self.info_sub = self.create_subscription(
            CameraInfo,
            info_topic,
            self.info_callback,
            self.qos_info
        )
    
    def _create_working_directories(self):
        """Create necessary working directories"""
        working_dir = self.config.get('system', 'working_directory')
        if working_dir:
            os.makedirs(working_dir, exist_ok=True)
            
            os.makedirs(os.path.join(working_dir, 'checkpoints'), exist_ok=True)
            os.makedirs(os.path.join(working_dir, 'results'), exist_ok=True)
            os.makedirs(os.path.join(working_dir, 'keyframes'), exist_ok=True)
    
    def image_callback(self, msg: Image):
        """Handle incoming RGB image"""
        try:
            self.get_logger().debug(f'Received image {self.frame_count}')
            
            # Convert ROS Image to OpenCV format
            cv_image = self.cv_bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            
            # Get current time
            current_time = self.get_clock().now()
            
            # Process with SLAM system
            self.process_frame(cv_image, current_time)
            
            self.frame_count += 1
            
        except Exception as e:
            self.get_logger().error(f'Error processing image: {e}')
    
    def depth_callback(self, msg: Image):
        """Handle incoming depth image"""
        try:
            # Convert ROS Image to OpenCV format
            cv_depth = self.cv_bridge.imgmsg_to_cv2(msg, desired_encoding='16UC1')
            
            # Get current time
            current_time = self.get_clock().now()
            
            # Process depth with SLAM system
            self.process_depth(cv_depth, current_time)
            
        except Exception as e:
            self.get_logger().error(f'Error processing depth image: {e}')
    
    def info_callback(self, msg: CameraInfo):
        """Handle incoming camera info"""
        try:
            # Store camera parameters for later use
            self.camera_params = self._parse_camera_info(msg)
        except Exception as e:
            self.get_logger().warning(f'Error processing camera info: {e}')
    
    def _parse_camera_info(self, msg: CameraInfo) -> Dict:
        """Parse camera info message into dictionary"""
        return {
            'width': msg.width,
            'height': msg.height,
            'intrinsic_matrix': np.array(msg.k).reshape((3, 3)).tolist(),
            'distortion_coefficients': np.array(msg.d).tolist() if msg.d else None,
            'projection_matrix': np.array(msg.p).reshape((4, 4)).tolist(),
            'time_stamp': msg.header.stamp
        }
    
    def process_frame(self, cv_image: np.ndarray, timestamp):
        """
        Process incoming RGB image with SLAM system
        
        Args:
            cv_image: Input RGB image as numpy array
            timestamp: ROS time stamp
        """
        if cv_image is None or cv_image.size == 0:
            return
        
        try:
            # For initial processing, we need both image and depth
            # Store image for processing in depth callback
            self.current_image = cv_image
            self.current_timestamp = timestamp
            
        except Exception as e:
            self.get_logger().error(f'Error in frame processing: {e}')
    
    def process_depth(self, cv_depth: np.ndarray, timestamp):
        """
        Process incoming depth image with SLAM system
        
        Args:
            cv_depth: Input depth image as numpy array
            timestamp: ROS time stamp
        """
        if self.current_image is None or cv_depth is None:
            return
        
        if cv_depth.size == 0:
            return
        
        try:
            # Create combined image and depth
            # Initialize reconstruction if not exists
            if self.reconstruction is None:
                self.reconstruction = self._initialize_reconstruction(
                    self.current_image, cv_depth
                )
            else:
                # Update reconstruction with new frame
                self.reconstruction = self._update_reconstruction(
                    self.current_image, cv_depth,
                    self.current_timestamp
                )
            
            # Store key frame based on criteria
            if self._should_add_keyframe():
                keyframe_data = self._extract_keyframe_data(
                    self.current_image, cv_depth
                )
                self.keyframes.append(keyframe_data)
                self._publish_keyframe(keyframe_data)
            
            # Publish trajectory and pose data
            current_pose = self.reconstruction.get_camera_pose()
            if current_pose is not None:
                self._publish_trajectory(current_pose)
                self._publish_pose(current_pose, timestamp)
            
            # Publish odometry
            if self.config.get('monitoring', 'publish_odometry', default=True):
                self._publish_odometry(current_pose, timestamp)
            
            # Statistics publishing
            if self.config.get('monitoring', 'publish_statistics', default=True):
                if (time.time() - self.last_time) >= self.config.get('monitoring', 'statistics_interval', default=1.0):
                    self._publish_statistics(timestamp)
                    self.last_time = time.time()
            
            # Clear current frame data
            self.current_image = None
            
        except Exception as e:
            self.get_logger().error(f'Error in depth processing: {e}')
            import traceback
            self.get_logger().error(traceback.format_exc())
    
    def _initialize_reconstruction(self, image: np.ndarray, depth: np.ndarray):
        """
        Initialize the 4D Gaussian Splatting reconstruction system
        
        Args:
            image: Initial RGB image
            depth: Initial depth image
            
        Returns:
            Reconstruction system instance
        """
        self.get_logger().info("Initializing Reconstruction with 4DGS (Placeholder Impl)")
        
        reconstruction = {
            'initialized': True,
            'num_gaussians': self.config.get('slam', 'num_gaussians'),
            'focal_length': self.config.get('slam', 'focal_length'),
            'principal_point': np.array(self.config.get('slam', 'principal_point')),
            'camera_trajectory': [np.eye(4)], # Start at origin
            'keyframes': [],
            'dynamic_objects': [],
            'points_3d': [], # Placeholder for actual gaussians
        }
        
        # Store first frame as initialization
        reconstruction['frame_count'] = 0
        reconstruction['last_timestamp'] = time.time()
        
        # Create first keyframe
        keyframe_data = self._extract_keyframe_data(image, depth)
        reconstruction['keyframes'].append(keyframe_data)
        
        return reconstruction
    
    def _update_reconstruction(self, image: np.ndarray, depth: np.ndarray, timestamp) -> dict:
        """
        Update the reconstruction with new frame
        
        Args:
            image: New RGB image
            depth: New depth image
            timestamp: Time stamp
            
        Returns:
            Updated reconstruction state
        """
        reconstruction = self.reconstruction
        
        # Update frame count
        reconstruction['frame_count'] += 1
        
        # Verify valid inputs
        if image is None or depth is None:
            return reconstruction

        # Calculate Frame Timing
        current_time = timestamp.nanoseconds / 1e9 # Convert to seconds
        dt = current_time - reconstruction['last_timestamp']
        reconstruction['last_timestamp'] = current_time

        # Estimate pose
        camera_pose = self._estimate_camera_pose(reconstruction)
        
        if camera_pose is not None:
            reconstruction['camera_trajectory'].append(camera_pose)
            
            # Simple Gaussian Update (Project new points)
            # Only do this if we are moving enough or initialized
            if self.camera_params:
                 # Subsample for performance
                 subsample = 10
                 depth_sub = depth[::subsample, ::subsample]
                 image_sub = image[::subsample, ::subsample]
                 
                 # Basic projection (placeholder for full back-projection)
                 # In real 4DGS, we would optimize gaussians here.
                 # For now, we just track the count.
                 new_points = np.count_nonzero(depth_sub > 0)
                 self.get_logger().info(f"Frame {reconstruction['frame_count']}: Tracking OK. Visible points: {new_points}")
        else:
             # Tracking lost
             self.get_logger().warn("Tracking lost!")
        
        return reconstruction
    
    def _estimate_camera_pose(self, reconstruction: dict) -> Optional[np.ndarray]:
        """
        Estimate current camera pose using Visual Odometry (ORB Features + PnP)
        
        Args:
            reconstruction: Current reconstruction state
            
        Returns:
            Current camera pose as 4x4 transformation matrix
        """
        if self.current_image is None or len(reconstruction['keyframes']) == 0:
            return np.eye(4)
            
        # Get last keyframe
        last_keyframe = reconstruction['keyframes'][-1]
        last_image = last_keyframe['image']
        
        # Detect features
        orb = cv2.ORB_create(nfeatures=1000)
        kp1, des1 = orb.detectAndCompute(last_image, None)
        kp2, des2 = orb.detectAndCompute(self.current_image, None)
        
        if des1 is None or des2 is None or len(kp1) < 10 or len(kp2) < 10:
            return None
            
        # Match features
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = bf.match(des1, des2)
        matches = sorted(matches, key=lambda x: x.distance)
        
        if len(matches) < 8:
            return None
            
        # Extract points
        pts1 = np.float32([kp1[m.queryIdx].pt for m in matches])
        pts2 = np.float32([kp2[m.trainIdx].pt for m in matches])
        
        # PnP needs 3D points from the first image. 
        # We can approximate by unprojecting pts1 using the depth of the last keyframe if available.
        # But here we use Essential Matrix for simplicity as a robust baseline
        K = np.array(self.camera_params['intrinsic_matrix'])
        E, mask = cv2.findEssentialMat(pts1, pts2, K, method=cv2.RANSAC, prob=0.999, threshold=1.0)
        
        if E is None:
            return None
            
        # Recover pose
        _, R, t, mask = cv2.recoverPose(E, pts1, pts2, K)
        
        # Construct 4x4 matrix
        current_pose = np.eye(4)
        current_pose[:3, :3] = R
        current_pose[:3, 3] = t.flatten()
        
        # Relative motion * last pose
        if len(reconstruction['camera_trajectory']) > 0:
            last_pose_world = reconstruction['camera_trajectory'][-1]
            # This is simplified; usually we need P3P with scale from depth
            # For now, we assume simple VO
            current_pose_world = last_pose_world @ current_pose
            return current_pose_world
            
        return current_pose
    
    def _should_add_keyframe(self) -> bool:
        """
        Determine if current frame should be added as a keyframe
        
        Returns:
            True if frame should be added as keyframe
        """
        keyframe_params = self.config.get('slam', default={})
        
        # Check distance from last keyframe
        if len(self.keyframes) > 0:
            last_keyframe = self.keyframes[-1]
            # TODO: Calculate actual distance
            distance = 0.0  # placeholder
            
            if distance < keyframe_params.get('keyframe_distance', 0.5):
                return False
        
        # Check pose change
        if len(self.camera_trajectory) > 0:
            last_pose = self.camera_trajectory[-1]
            # TODO: Calculate actual angular change
            angle_change = 0.0  # placeholder
            
            if angle_change < keyframe_params.get('keyframe_angle', 0.4):
                return False
        
        # Ensure minimum keyframes
        if len(self.keyframes) < keyframe_params.get('min_num_keyframes', 3):
            return True
        
        return True
    
    def _extract_keyframe_data(self, image: np.ndarray, depth: np.ndarray) -> dict:
        """
        Extract keyframe data for processing
        
        Args:
            image: RGB image
            depth: Depth image
            
        Returns:
            Dictionary containing keyframe data
        """
        return {
            'timestamp': time.time(),
            'image': image.copy(),
            'depth': depth.copy()
        }
    
    def _publish_keyframe(self, keyframe_data: dict):
        """Publish keyframe image for visualization"""
        try:
            # Convert image to ROS format
            ros_image = self.cv_bridge.cv2_to_imgmsg(
                keyframe_data['image'],
                encoding='bgr8'
            )
            
            self.keyframe_pub.publish(ros_image)
            
        except Exception as e:
            self.get_logger().warning(f'Error publishing keyframe: {e}')
    
    def _publish_trajectory(self, pose: np.ndarray):
        """Publish current trajectory position"""
        try:
            # Extract position from pose
            position = pose[:3, 3]
            q = self._pose_to_quaternion(pose)
            
            # Create pose message
            msg = PoseStamped()
            msg.header.stamp = self.get_clock().now().to_msg()
            msg.header.frame_id = 'map'
            msg.pose.position.x = position[0]
            msg.pose.position.y = position[1]
            msg.pose.position.z = position[2]
            msg.pose.orientation.x = q[0]
            msg.pose.orientation.y = q[1]
            msg.pose.orientation.z = q[2]
            msg.pose.orientation.w = q[3]
            
            self.trajectory_pub.publish(msg)
            
        except Exception as e:
            self.get_logger().warning(f'Error publishing trajectory: {e}')
    
    def _publish_pose(self, pose: np.ndarray, timestamp):
        """Publish current pose"""
        try:
            # Extract position from pose
            position = pose[:3, 3]
            q = self._pose_to_quaternion(pose)
            
            # Create pose message
            msg = PoseStamped()
            msg.header.stamp = timestamp
            msg.header.frame_id = 'map'
            msg.pose.position.x = position[0]
            msg.pose.position.y = position[1]
            msg.pose.position.z = position[2]
            msg.pose.orientation.x = q[0]
            msg.pose.orientation.y = q[1]
            msg.pose.orientation.z = q[2]
            msg.pose.orientation.w = q[3]
            
            self.poses_pub.publish(msg)
            
        except Exception as e:
            self.get_logger().warning(f'Error publishing pose: {e}')
    
    def _publish_odometry(self, pose: np.ndarray, timestamp):
        """Publish current odometry"""
        try:
            # Extract position and orientation from pose
            position = pose[:3, 3]
            q = self._pose_to_quaternion(pose)
            
            # Create odometry message
            msg = Odometry()
            msg.header.stamp = timestamp
            msg.header.frame_id = 'map'
            msg.child_frame_id = 'camera'
            msg.pose.pose.position.x = position[0]
            msg.pose.pose.position.y = position[1]
            msg.pose.pose.position.z = position[2]
            msg.pose.pose.orientation.x = q[0]
            msg.pose.pose.orientation.y = q[1]
            msg.pose.pose.orientation.z = q[2]
            msg.pose.pose.orientation.w = q[3]
            
            # Create velocity message (placeholder)
            msg.twist.twist.linear.x = 0.0
            msg.twist.twist.linear.y = 0.0
            msg.twist.twist.linear.z = 0.0
            
            self.odometry_pub.publish(msg)
            
        except Exception as e:
            self.get_logger().warning(f'Error publishing odometry: {e}')
    
    def _publish_statistics(self, timestamp):
        """Publish SLAM statistics"""
        try:
            msg = String()
            msg.data = f"SLAM Statistics:\n" \
                       f"Frames processed: {self.frame_count}\n" \
                       f"Keyframes: {len(self.keyframes)}\n" \
                       f"Trajectory points: {len(self.camera_trajectory)}"
            
            self.statistics_pub.publish(msg)
            
        except Exception as e:
            self.get_logger().warning(f'Error publishing statistics: {e}')
    
    @staticmethod
    def _pose_to_quaternion(pose: np.ndarray) -> np.ndarray:
        """Convert 4x4 pose matrix to quaternion"""
        # Extract rotation matrix
        r = pose[:3, :3]
        
        # Convert to quaternion using ROS convention
        # Eigen's conversion function would be ideal
        quaternion = self._matrix_to_quaternion(r)
        
        return quaternion
    
    @staticmethod
    def _matrix_to_quaternion(matrix: np.ndarray) -> np.ndarray:
        """Convert 3x3 rotation matrix to quaternion"""
        # Using standard conversion
        trace = np.trace(matrix)
        
        if trace > 0:
            S = np.sqrt(trace + 1) * 2
            w = 0.25 * S
            x = (matrix[2, 1] - matrix[1, 2]) / S
            y = (matrix[0, 2] - matrix[2, 0]) / S
            z = (matrix[1, 0] - matrix[0, 1]) / S
        elif (matrix[0, 0] > matrix[1, 1]) and (matrix[0, 0] > matrix[2, 2]):
            S = np.sqrt(1 + matrix[0, 0] - matrix[1, 1] - matrix[2, 2]) * 2
            w = (matrix[2, 1] - matrix[1, 2]) / S
            x = 0.25 * S
            y = (matrix[1, 0] + matrix[0, 1]) / S
            z = (matrix[0, 2] + matrix[2, 0]) / S
        elif matrix[1, 1] > matrix[2, 2]:
            S = np.sqrt(1 + matrix[1, 1] - matrix[0, 0] - matrix[2, 2]) * 2
            w = (matrix[0, 2] - matrix[2, 0]) / S
            x = (matrix[1, 0] + matrix[0, 1]) / S
            y = 0.25 * S
            z = (matrix[2, 1] + matrix[1, 2]) / S
        else:
            S = np.sqrt(1 + matrix[2, 2] - matrix[0, 0] - matrix[1, 1]) * 2
            w = (matrix[1, 0] - matrix[0, 1]) / S
            x = (matrix[0, 2] + matrix[2, 0]) / S
            y = (matrix[2, 1] + matrix[1, 2]) / S
            z = 0.25 * S
        
        return np.array([x, y, z, w])
    
    def cleanup(self):
        """Clean up resources"""
        self.get_logger().info('Cleaning up 4DGS-SLAM Node')
        
        # Save results
        if self.config.get('system', 'save_results', default=True):
            self._save_results()
        
        self.is_running = False


if __name__ == '__main__':
    # Test the node
    try:
        rclpy.init()
        node = SLAMNode()
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.cleanup()
    except Exception as e:
        if 'node' in locals():
            node.get_logger().error(f'Error running SLAM node: {e}')
        else:
            print(f'Error running SLAM node: {e}')
    finally:
        if 'node' in locals():
            node.cleanup()
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()