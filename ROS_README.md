# ROS2 Wrapper for GS ICP SLAM

This package provides a ROS2 wrapper for the GS ICP SLAM system, enabling real-time 3D Gaussian Splatting SLAM with RGB-D input.

## Features
- **Real-time SLAM**: Processes live RGB-D streams.
- **ROS2 Integration**: Subscribes to standard `sensor_msgs/Image` and publishes `nav_msgs/Odometry` and TF.
- **External Pose Support**: Can utilize external odometry (e.g., from VIO or wheel odometry) to bypass internal GICP registration.
- **Rerun Integration**: Built-in support for `rerun.io` visualization.

## Architecture
The system consists of a ROS2 node (`four_dgs_slam_node`) that interfaces with the GS-ICP-SLAM backend.
1. **Input**: The node subscribes to RGB and Depth topics and synchronizes them (approximate sync).
2. **Processing**: Images are pushed to a multiprocessing queue.
3. **Tracking**: The `Tracker` process (part of GS-ICP-SLAM) consumes images, performs GICP registration (or uses external pose), and updates the 3D Gaussian Splatting map.
4. **Keyframing**: Keyframes are selected based on overlap ratios and optical flow magnitude to ensure sufficient baseline for mapping.
5. **Output**: The estimated camera pose is sent back to the ROS2 node via a queue and published as TF and Odometry.

### Launch
To start the SLAM node:
```bash
ros2 launch four_dgs_slam slam.launch.py camera_topic:=<your_rgb_topic> depth_topic:=<your_depth_topic> camera_info_topic:=<your_cam_info_topic>
```

### Parameters
| Parameter | Default | Description |
|-----------|---------|-------------|
| `camera_topic` | `/camera/rgb/image_raw` | RGB image topic |
| `depth_topic` | `/camera/depth/image_raw` | Depth image topic (alligned to RGB, float or mm) |
| `camera_info_topic` | `/camera/camera_info` | Camera intrinsics |
| `use_external_pose` | `False` | Use external pose instead of GICP |
| `pose_topic` | `/pose` | Topic for external pose (geometry_msgs/PoseStamped) |
| `rerun_viewer` | `True` | Enable Rerun viewer |
| `keyframe_th` | `0.7` | Keyframe selection threshold |
| `verbose` | `False` | Enable verbose logging |

### Topics
**Subscribed:**
- RGB Image: `sensor_msgs/Image` (rgb8)
- Depth Image: `sensor_msgs/Image` (16UC1 or 32FC1)
- Camera Info: `sensor_msgs/CameraInfo`
- Pose (Optional): `geometry_msgs/PoseStamped`

**Published:**
- Odometry: `/odometry` (`nav_msgs/Odometry`)
- TF: `odom` -> `camera_link`

## Performance & Limitations

### Framerate
- **Minimum**: 20 FPS is recommended for reliable GICP tracking. Lower framerates may result in tracking loss due to large inter-frame motion.
- **Optimization**: If running slowly, increase `downsample_rate` (default: 10) or reduce image resolution.

### External Pose
- If `use_external_pose` is `True`, the system bypasses the internal GICP registration for tracking.
- This is useful if you have a reliable VIO or robot odometry and want to use GS-ICP-SLAM for mapping only.
- Ensure the external pose is time-synchronized with the images. The node performs a simple timestamp check (< 0.05s diff).

### Rerun Viewer
- The `rerun` viewer provides real-time visualization of the camera trajectory, current point cloud, and keyframes.
- It is launched automatically if `rerun_viewer` is `True`.

## Troubleshooting
- **Tracking Lost**: If the system loses tracking, it currently does not have a relocalization module. You may need to restart the node.
- **High Latency**: Reduce image resolution or increase `downsample_rate`. Ensure your GPU is utilized (check `nvidia-smi`).
