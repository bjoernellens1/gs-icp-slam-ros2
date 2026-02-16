# GS-ICP-SLAM ROS2 Package - Tutorials

This document provides step-by-step tutorials for getting started with the 4D Gaussian Splatting SLAM system using ROS2.

## Table of Contents

1. [Quick Start Tutorial](#quick-start-tutorial)
2. [Basic Usage Tutorial](#basic-usage-tutorial)
3. [Advanced Configuration Tutorial](#advanced-configuration-tutorial)
4. [Real-time SLAM Tutorial](#real-time-slam-tutorial)
5. [Bag Playback Tutorial](#bag-playback-tutorial)
6. [Visualization Tutorial](#visualization-tutorial)
7. [Troubleshooting Tutorial](#troubleshooting-tutorial)

## Quick Start Tutorial

### Lesson 1: Environment Setup in 5 Minutes

**Objective:** Get the environment set up and running a basic SLAM node.

```bash
# 1. Navigate to the package
cd ~/dev/gs_icp_slam_ros2

# 2. Activate the conda environment
source $(conda info --base)/etc/profile.d/conda.sh
conda activate gs_icp_slam

# 3. Source ROS 2 environment
source /opt/ros/humble/setup.bash

# 4. Build the package (single time)
colcon build --packages-select gs_icp_slam --symlink-install

# 5. Source the package
source install/setup.bash

# 6. Check if everything is working
python3 -c "from gs_icp_slam_ros2 import SLAMNode; print('Success!')"
```

**Expected Output:**
```
Success!
```

### Lesson 2: Basic Node Operation

**Objective:** Understand how to run a basic SLAM node.

```bash
# Terminal 1: Start your camera node (example)
ros2 run your_camera_package camera_publisher

# Terminal 2: Start SLAM node
ros2 run gs_icp_slam gs_icp_slam_node
```

**Expected Behavior:**
- The SLAM node should connect to camera topics
- It will publish odometry and pose estimates
- Check topics: `ros2 topic list | grep slam`

## Basic Usage Tutorial

### Lesson 3: Using Launch Files

**Objective:** Use ROS2 launch files for easier management.

```bash
# Launch basic SLAM node
ros2 launch gs_icp_slam gs_icp_slam_basic.launch.py

# With custom camera topics
ros2 launch gs_icp_slam gs_icp_slam_basic.launch.py --ros-args \
    -r image:=/custom/image \
    -r depth:=/custom/depth
```

### Lesson 4: Real-time Camera Input

**Objective:** Process live camera input with SLAM.

```bash
# 1. Record some test data first (optional)
ros2 bag record /camera/image_raw /camera/depth --duration 30

# 2. Play the bag for testing
ros2 bag play test_bag.bag --clock

# 3. Run SLAM in separate terminal
ros2 run gs_icp_slam gs_icp_slam_node

# 4. Monitor output in another terminal
ros2 topic echo /slam/odometry
ros2 topic echo /slam/statistics
```

## Advanced Configuration Tutorial

### Lesson 5: Custom Configuration Files

**Objective:** Create and use custom configuration files.

```bash
# Create custom configuration
cat > my_config.yaml << 'EOF'
slam:
  camera_model: "pinhole"
  focal_length: 500.0
  num_gaussians: 150000
  
system:
  working_directory: "/home/user/my_slam_results"
  save_results: true

monitoring:
  publish_statistics: true
  publish_trajectory: true
EOF

# Run with custom config
ros2 launch gs_icp_slam gs_icp_slam_with_bag.launch.py \
    --config-file my_config.yaml \
    bag_file:=your_bag.bag
```

### Lesson 6: Performance Tuning

**Objective:** Optimize for real-time performance.

```bash
# Create performance-tuned config
cat > performance_config.yaml << 'EOF'
slam:
  num_gaussians: 30000
  tracking_max_iter: 15
  keyframe_distance: 1.2
  
gpu:
  cuda_device: 0
  batch_size: 4096
  
monitoring:
  publish_statistics: false
  publish_trajectory: false
EOF

# Use performance config
ros2 launch gs_icp_slam gs_icp_slam_with_bag.launch.py \
    --config-file performance_config.yaml
```

## Real-time SLAM Tutorial

### Lesson 7: Continuous Mapping

**Objective:** Run SLAM for extended periods to build large maps.

```bash
# For long-term mapping sessions
ros2 launch gs_icp_slam gs_icp_slam_visualization.launch.py

# In a separate terminal, control your robot
ros2 run teleop_twist_keyboard teleop_twist_keyboard

# Monitor performance
ros2 topic echo /slam/odometry
```

**Tips:**
- Increase keyframe distance for large scenes
- Save checkpoints regularly if needed
- Monitor memory usage

### Lesson 8: Dynamic Scene Handling

**Objective:** Handle dynamic objects in the scene.

```bash
# Default configuration already handles dynamics
# Check dynamic confidence threshold
cat config/slam_config.yaml | grep dynamic_conf

# For more aggressive dynamic filtering
cat > dynamic_config.yaml << 'EOF'
slam:
  dynamic_confidence_threshold: 0.85
EOF

# Use with SLAM
ros2 launch gs_icp_slam gs_icp_slam_with_bag.launch.py \
    --config-file dynamic_config.yaml
```

## Bag Playback Tutorial

### Lesson 9: Testing with Recorded Data

**Objective:** Test SLAM with pre-recorded data.

```bash
# 1. Record camera data
ros2 bag record \
    /camera/image_raw \
    /camera/depth \
    /camera/camera_info \
    -o test_recording

# 2. Play data
ros2 bag play test_recording.bag --clock

# 3. Run SLAM
ros2 run gs_icp_slam gs_icp_slam_node

# 4. Compare to standard SLAM
# Repeat steps 2-3 without SLAM to compare
```

### Lesson 10: Advanced Bag Controls

**Objective:** Use advanced bag playback features.

```bash
# Start bag playback at specific time
ros2 bag play test_recording.bag --start-time 10.0 --clock

# Play segment by time range
ros2 bag play test_recording.bag \
    --start-time 10.0 \
    --end-time 20.0 \
    --clock

# Skip frames for speed
ros2 bag play test_recording.bag --rate 2.0 --clock
```

## Visualization Tutorial

### Lesson 11: RViz2 Setup

**Objective:** Configure RViz2 for SLAM visualization.

```bash
# Launch RViz2
ros2 run rviz2 rviz2

# Then follow these steps in RViz2 interface:

# 1. Add TF display - Enable coordinate frame visualization
# 2. Add Map display - For point clouds
# 3. Add Odometry display - For trajectory tracking
# 4. Add Image display - For keyframes

# For automated setup, create rviz config:
# Copy from config/rviz2_default.rviz to your workspace
```

### Lesson 12: Real-time Monitoring

**Objective:** Monitor SLAM output in real-time.

```bash
# Terminal 1: Run SLAM
ros2 run gs_icp_slam gs_icp_slam_node

# Terminal 2: Monitor odometry
ros2 topic echo /slam/odometry

# Terminal 3: Monitor statistics
ros2 topic echo /slam/statistics

# Terminal 4: Monitor pose
ros2 topic echo /slam/pose

# Terminal 5: Check topic connections
ros2 topic hz /slam/odometry
ros2 topic info /slam/pose
```

## Troubleshooting Tutorial

### Lesson 13: Common Issues

**Issue 1: Node fails to connect to camera**

```bash
# Check if camera topics are published
ros2 topic list | grep camera

# Check if they have data
ros2 topic echo /camera/image_raw --once

# Remap if necessary
ros2 run gs_icp_slam gs_icp_slam_node -r image:=/remapped/topic
```

**Issue 2: Memory usage too high**

```yaml
# Reduce Gaussian count and disable features
slam:
  num_gaussians: 30000
  # ... other parameters

# Disable parallel processing
system:
  parallel_processing: false
```

**Issue 3: Poor SLAM accuracy**

```yaml
# Improve camera calibration
slam:
  focal_length: 527.0
  principal_point: [326.0, 247.0]

# Increase tracking iterations
slam:
  tracking_max_iter: 50

# Improve keyframe selection
slam:
  keyframe_distance: 0.3
  keyframe_angle: 0.3
```

### Lesson 14: Debug Mode

```bash
# Enable verbose logging
ros2 run gs_icp_slam gs_icp_slam_node --ros-args \
    --log-level debug

# Python debug mode (in your code)
system:
  verbose: true
  enable_debug: true
```

## Complete Workflow Example

### Comprehensive Example: From Setup to Results

```bash
# Step 1: Setup (one-time)
cd ~/dev/gs_icp_slam_ros2
conda activate gs_icp_slam
source /opt/ros/humble/setup.bash
colcon build --packages-select gs_icp_slam --symlink-install
source install/setup.bash

# Step 2: Record test data (optional)
ros2 bag record /camera/image_raw /camera/depth --duration 60

# Step 3: Prepare SLAM configuration
cat > my_experiment_config.yaml << 'EOF'
slam:
  num_gaussians: 100000
  tracking_max_iter: 30
  
system:
  working_directory: ~/dev/slam_results
  save_results: true

monitoring:
  publish_statistics: true
  publish_trajectory: true
EOF

# Step 4: Run experiment
ros2 bag play my_data.bag --clock
ros2 launch gs_icp_slam gs_icp_slam_with_bag.launch.py \
    --config-file my_experiment_config.yaml

# Step 5: Analyze results
ls ~/dev/slam_results/
```

## Next Steps

After completing these tutorials:

1. Explore the [API Documentation](./API.md)
2. Read the [Full README](../README.md)
3. Check [Contribution Guidelines](../CONTRIBUTING.md)
4. Review [Advanced Topics](./ADVANCED.md) for performance optimization

## Getting Help

If you encounter issues not covered in these tutorials:

1. Check the [Troubleshooting Section](../README.md#troubleshooting)
2. Review GitHub issues
3. Check ROS2 documentation
4. Consult SLAM-specific resources

## Practice Exercises

### Exercise 1
**Task:** Run SLAM with a recorded bag file and visualize the trajectory.

### Exercise 2
**Task:** Create a custom configuration for your specific camera hardware.

### Exercise 3
**Task:** Optimize SLAM parameters for real-time autonomous navigation.

### Exercise 4
**Task:** Process a longer dataset (5+ minutes) and save the reconstruction results.

### Exercise 5
**Task:** Compare dynamic segmention settings and evaluate scene reconstruction quality.