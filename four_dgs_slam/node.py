import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseStamped, TransformStamped
from tf2_ros import TransformBroadcaster
from cv_bridge import CvBridge
import cv2
import numpy as np
import threading
import queue
import time
import argparse
import os
from scipy.spatial.transform import Rotation

# Assuming GS_ICP_SLAM is importable from the same package
from four_dgs_slam.gs_icp_slam import GS_ICP_SLAM

class GSICPSLAMNode(Node):
    def __init__(self):
        super().__init__('gs_icp_slam_node')

        # Parameters
        self.declare_parameter('dataset_path', '')
        self.declare_parameter('config_path', '') # Path to camera calibration file if not from topic
        self.declare_parameter('verbose', False)
        self.declare_parameter('keyframe_th', 0.7)
        self.declare_parameter('knn_maxd', 99999.0)
        self.declare_parameter('overlapped_th', 0.0005) # 5e-4
        self.declare_parameter('max_correspondence_distance', 0.02)
        self.declare_parameter('trackable_opacity_th', 0.05)
        self.declare_parameter('overlapped_th2', 0.00005) # 5e-5
        self.declare_parameter('downsample_rate', 10)
        self.declare_parameter('test', None)
        self.declare_parameter('save_results', False)
        self.declare_parameter('rerun_viewer', True)
        self.declare_parameter('demo', False)
        self.declare_parameter('use_external_pose', False)
        self.declare_parameter('camera_topic', '/camera/rgb/image_raw')
        self.declare_parameter('depth_topic', '/camera/depth/image_raw')
        self.declare_parameter('camera_info_topic', '/camera/camera_info')
        self.declare_parameter('pose_topic', '/pose') # For external pose
        self.declare_parameter('output_path', os.path.join(os.getcwd(), 'output'))

        # Get params
        self.dataset_path = self.get_parameter('dataset_path').value
        self.config_path = self.get_parameter('config_path').value
        self.verbose = self.get_parameter('verbose').value
        self.use_external_pose = self.get_parameter('use_external_pose').value
        
        # Args object to mock argparse expected by GS_ICP_SLAM
        self.args = argparse.Namespace()
        self.args.dataset_path = self.dataset_path
        self.args.config = self.config_path
        self.args.output_path = self.get_parameter('output_path').value
        self.args.verbose = self.verbose
        self.args.keyframe_th = self.get_parameter('keyframe_th').value
        self.args.knn_maxd = self.get_parameter('knn_maxd').value
        self.args.overlapped_th = self.get_parameter('overlapped_th').value
        self.args.max_correspondence_distance = self.get_parameter('max_correspondence_distance').value
        self.args.trackable_opacity_th = self.get_parameter('trackable_opacity_th').value
        self.args.overlapped_th2 = self.get_parameter('overlapped_th2').value
        self.args.downsample_rate = self.get_parameter('downsample_rate').value
        self.args.test = self.get_parameter('test').value
        self.args.save_results = self.get_parameter('save_results').value
        self.args.rerun_viewer = self.get_parameter('rerun_viewer').value
        self.args.demo = self.get_parameter('demo').value

        # Queues
        self.input_queue = threading.Queue() # Using threading queue for same-process comms if possible, or mp.Queue
        self.output_queue = threading.Queue()

        # NOTE: GS_ICP_SLAM uses multiprocessing.Queue internally if we passed it one, 
        # but here we might be running in the same process or need to bridge.
        # GS_ICP_SLAM.run() spawns processes. 
        # We need a multiprocessing queue for input/output if the tracker/mapper are in separate processes.
        import multiprocessing as mp
        self.input_queue = mp.Queue()
        self.output_queue = mp.Queue()

        # Initialize SLAM
        self.slam = GS_ICP_SLAM(self.args, input_queue=self.input_queue, output_queue=self.output_queue)
        
        # Start SLAM processes
        self.slam_processes = self.slam.run()

        # ROS2 setup
        self.bridge = CvBridge()
        self.tf_broadcaster = TransformBroadcaster(self)
        
        # Subscribers
        self.create_subscription(Image, self.get_parameter('camera_topic').value, self.image_callback, 10)
        self.create_subscription(Image, self.get_parameter('depth_topic').value, self.depth_callback, 10)
        self.create_subscription(CameraInfo, self.get_parameter('camera_info_topic').value, self.camera_info_callback, 10)
        
        if self.use_external_pose:
            self.create_subscription(PoseStamped, self.get_parameter('pose_topic').value, self.pose_callback, 10)

        # Publishers
        self.odom_publisher = self.create_publisher(Odometry, '/odometry', 10)

        # Data buffers
        self.latest_rgb = None
        self.latest_depth = None
        self.latest_rgb_ts = None
        self.latest_depth_ts = None
        self.latest_pose = None
        self.latest_pose_ts = None
        
        # Extrinsics (Camera to Base link) - Identity for now
        self.base_to_camera = np.eye(4)

        # Processing loop
        self.process_timer = self.create_timer(0.01, self.process_loop)
        self.publish_timer = self.create_timer(0.01, self.publish_loop)
        
        self.get_logger().info("GS_ICP_SLAM_Node started")

    def camera_info_callback(self, msg):
        # Update camera intrinsics if not provided via config file
        # This is a bit tricky since GS_ICP_SLAM initializes with fixed intrinsics.
        # Ideally we'd set this before starting the SLAM, allowing us to wait for CamInfo.
        # For now, we assume config is passed or defaults are used.
        pass

    def image_callback(self, msg):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='rgb8')
            self.latest_rgb = cv_image
            self.latest_rgb_ts = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        except Exception as e:
            self.get_logger().error(f"Error processing RGB image: {e}")

    def depth_callback(self, msg):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
            # Ensure depth is float32 (meters) or uint16 (mm) and converted appropriately
            # GS_ICP_SLAM expects float32/scaling factor.
            self.latest_depth = cv_image
            self.latest_depth_ts = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        except Exception as e:
            self.get_logger().error(f"Error processing Depth image: {e}")

    def pose_callback(self, msg):
        # Convert PoseStamped to 4x4 matrix
        t = [msg.pose.position.x, msg.pose.position.y, msg.pose.position.z]
        q = [msg.pose.orientation.x, msg.pose.orientation.y, msg.pose.orientation.z, msg.pose.orientation.w]
        R = Rotation.from_quat(q).as_matrix()
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = t
        self.latest_pose = T
        self.latest_pose_ts = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9

    def process_loop(self):
        # Synchronize and push to queue
        if self.latest_rgb is not None and self.latest_depth is not None:
            # Simple sync: time diff < 0.05s
            if abs(self.latest_rgb_ts - self.latest_depth_ts) < 0.05:
                
                # Check for external pose if enabled
                pose_to_send = None
                if self.use_external_pose and self.latest_pose is not None:
                     # Simple sync for pose: time diff < 0.1s check?
                     # Or just use latest.
                      pose_to_send = self.latest_pose
                
                # Push to queue
                if pose_to_send is not None:
                    self.input_queue.put((self.latest_rgb, self.latest_depth, self.latest_rgb_ts, pose_to_send))
                else:
                    self.input_queue.put((self.latest_rgb, self.latest_depth, self.latest_rgb_ts))
                
                # Clear buffers
                self.latest_rgb = None
                self.latest_depth = None

    def publish_loop(self):
        try:
            while not self.output_queue.empty():
                pose, timestamp = self.output_queue.get(False)
                
                # Publish Odometry
                odom_msg = Odometry()
                odom_msg.header.stamp.sec = int(timestamp)
                odom_msg.header.stamp.nanosec = int((timestamp - int(timestamp)) * 1e9)
                odom_msg.header.frame_id = "odom"
                odom_msg.child_frame_id = "camera_link"
                
                t = pose[:3, 3]
                q = Rotation.from_matrix(pose[:3, :3]).as_quat()
                
                odom_msg.pose.pose.position.x = t[0]
                odom_msg.pose.pose.position.y = t[1]
                odom_msg.pose.pose.position.z = t[2]
                odom_msg.pose.pose.orientation.x = q[0]
                odom_msg.pose.pose.orientation.y = q[1]
                odom_msg.pose.pose.orientation.z = q[2]
                odom_msg.pose.pose.orientation.w = q[3]
                
                self.odom_publisher.publish(odom_msg)
                
                # Publish TF
                t_msg = TransformStamped()
                t_msg.header.stamp = odom_msg.header.stamp
                t_msg.header.frame_id = "odom"
                t_msg.child_frame_id = "camera_link"
                t_msg.transform.translation.x = t[0]
                t_msg.transform.translation.y = t[1]
                t_msg.transform.translation.z = t[2]
                t_msg.transform.rotation.x = q[0]
                t_msg.transform.rotation.y = q[1]
                t_msg.transform.rotation.z = q[2]
                t_msg.transform.rotation.w = q[3]
                
                self.tf_broadcaster.sendTransform(t_msg)
                
        except queue.Empty:
            pass
        except Exception as e:
            self.get_logger().error(f"Error publishing: {e}")

def main(args=None):
    rclpy.init(args=args)
    node = GSICPSLAMNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()