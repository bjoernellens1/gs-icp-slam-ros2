# 4D Gaussian Splatting SLAM - ROS2 Package

[![Python Version](https://img.shields.io/badge/python-3.8--3.11-blue.svg)](https://www.python.org/downloads/)
[![ROS2 Version](https://img.shields.io/badge/ROS2-humble_/_blue.svg)](https://index.ros.org/doc/ros2/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Conventional Commits](https://img.shields.io/badge/commits-conventional-ff0000.svg)](https://www.conventionalcommits.org/)

A complete ROS2 implementation of the **4D Gaussian Splatting SLAM** system for dynamic scene reconstruction and mapping. This package provides real-time RGB-D SLAM with advanced Gaussian Splatting techniques for high-quality scene generation.

## 🎯 Overview

The 4DGS-SLAM ROS2 package provides:

- **✅ Real-time SLAM**: Efficient RGB-D SLAM with high frame rate processing
- **✅ 4D Gaussian Splatting**: High-quality dynamic scene reconstruction
- **✅ ROS2 Integration**: Complete ROS2 node implementation with proper QoS settings
- **✅ Flexible Configuration**: Comprehensive YAML configuration system
- **✅ Visualization Support**: Integration with RViz2 for real-time monitoring
- **✅ Multi-platform Support**: Works with both real cameras and ROS bag files
- **✅ Production-ready**: Designed for real-world autonomous systems

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/4dgs_slam_ros2.git
cd 4dgs_slam_ros2

# Install the package
python3 scripts/install.py --pretrained
```

### Basic Usage

```bash
# Run with default configuration
ros2 run 4dgs_slam 4dgs_slam_node

# Use launch files
ros2 launch 4dgs_slam 4dgs_slam_basic.launch.py
ros2 launch 4dgs_slam 4dgs_slam_with_bag.launch.py bag_file:=your_bag.bag
ros2 launch 4dgs_slam 4dgs_slam_visualization.launch.py
```

## 📚 Documentation

- [📖 README](README.md) - Complete package overview and usage guide
- [📖 API Reference](docs/API.md) - Detailed API documentation
- [📖 Tutorials](docs/TUTORIALS.md) - Step-by-step usage guides
- [📖 Advanced Topics](docs/ADVANCED.md) - Optimization and advanced features
- [📖 Development Guide](docs/DEVELOPMENT.md) - For contributors and developers
- [📖 Contributing Guide](docs/CONTRIBUTING.md) - How to contribute

## 🎥 Features

### Core SLAM Features

- **RGB-D SLAM**: Feature tracking, pose estimation, Gaussian field updates
- **Dynamic Scene Handling**: Automatic dynamic object detection and filtering
- **4D Gaussian Reconstruction**: High-fidelity scene representation
- **Real-time Performance**: <100ms processing time on modern hardware
- **Robust Tracking**: Multi-threaded processing pipeline

### ROS2 Integration

- **Standard Topics**: Odometry, poses, trajectories, keyframes
- **Proper QoS**: Best-effort and reliable services
- **Launch Support**: Multiple ready-to-use launch files
- **RViz2 Integration**: Ready-to-use configuration
- **Bag Playback**: Test with pre-recorded data

### Advanced Features

- **Custom Configuration**: Extensive YAML config system
- **Performance Optimization**: Configurable for speed vs quality
- **Cross-platform**: Supports GPU acceleration
- **Monitoring**: Real-time statistics and diagnostics
- **Checkpoints**: Save and resume reconstruction

## 📋 Requirements

### System Requirements

- Ubuntu 20.04/22.04
- ROS 2 Humble or newer (rolling recommended)
- NVIDIA GPU with CUDA 11.x support
- Python 3.8 or newer
- 16GB+ RAM (recommended)

### Dependencies

```
# ROS 2 packages
- rclcpp
- sensor_msgs
- geometry_msgs
- nav_msgs
- cv_bridge

# Python packages
- numpy
- opencv-python
- scipy
- pyyaml
- torch (with CUDA 11.7)

# Development tools
- pytest
- colcon
- python3-dev (build utilities)
```
## 🐳 Docker Support
        
### Quick Start with Docker/Podman

We provide a Dockerfile for easy deployment. You can use Docker or Podman.

**Build the Image:**
```bash
# Using Podman (Recommended)
podman build -t 4dgs-slam-ros2 .

# Using Docker
docker build -t 4dgs-slam-ros2 .
```

**Run the Container:**
```bash
# Using Podman with GPU support
podman run --rm --gpus all --net=host -it 4dgs-slam-ros2

# Using Docker with GPU support
docker run --rm --gpus all --net=host -it 4dgs-slam-ros2
```

### CI/CD
This repository includes a GitHub Actions workflow that automatically builds and pushes the Docker image to the GitHub Container Registry (GHCR) on every push to the `main` branch.

### AMD GPU Support (ROCm / Strix Halo)

For users with AMD GPUs (including Strix Halo APUs), we provide a specific Dockerfile with ROCm 6.2 support.

**Build the Image:**
```bash
docker build -f Dockerfile.rocm -t 4dgs-slam-ros2:rocm-strix .
```

**Run the Container (Strix Halo):**
For Strix Halo (Ryzen AI Max 300 series), you may need to override the GFX version to mimic RDNA 3 (gfx1100) if native support isn't detected by PyTorch yet.

```bash
docker run --rm -it \
    --device=/dev/kfd \
    --device=/dev/dri \
    --group-add video \
    --group-add render \
    --ipc=host \
    --shm-size=8g \
    --security-opt seccomp=unconfined \
    -e HSA_OVERRIDE_GFX_VERSION=11.0.0 \
    4dgs-slam-ros2:rocm-strix
```
*Note: If you encounter issues with the `render` group (e.g. "Unable to find group"), try using the group ID directly (e.g. `--group-add 992` if your `/dev/kfd` belongs to group 992), or ensure the group exists.*
*Note: If Strix Halo detection fails or crashes, you can try `HSA_OVERRIDE_GFX_VERSION=11.0.0` (RDNA 3).*

## 🛠️ Installation
## 🛠️ Installation

### 1. Clone and Setup

```bash
git clone https://github.com/yourusername/4dgs_slam_ros2.git
cd 4dgs_slam_ros2

# Create conda environment (recommended)
conda create -n 4dgs_slam python=3.8
conda activate 4dgs_slam

# Install dependencies
pip install -r requirements.txt
source /opt/ros/humble/setup.bash

# Build the package
colcon build --packages-select 4dgs_slam --symlink-install

# Source the workspace
source install/setup.bash
```

### 2. Download Pretrained Models

```bash
# Download YOLOv9 for dynamic object detection
mkdir -p pretrained
cd pretrained
wget https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov9e-seg.pt
```

## 🚦 Usage Examples

### Basic SLAM with ROS Bag

```bash
# Terminal 1: Play your bag file
ros2 bag play your_sequence.bag --clock

# Terminal 2: Run SLAM
ros2 run 4dgs_slam 4dgs_slam_node
```

### Real-time Camera Input

```bash
# If you have a camera node running
ros2 run your_camera_package camera_publisher

# Run SLAM in another terminal
ros2 run 4dgs_slam 4dgs_slam_node
```

### Visualization

```bash
# Run complete system with RViz2
ros2 launch 4dgs_slam 4dgs_slam_visualization.launch.py
```

### Custom Configuration

```bash
# Use custom config file
cat > custom_config.yaml << 'EOF'
slam:
  num_gaussians: 150000
  tracking_max_iter: 50
EOF

ros2 launch 4dgs_slam 4dgs_slam_with_bag.launch.py --config-file custom_config.yaml
```

## 🔧 Configuration

### Main Configuration File

See `config/slam_config.yaml` for complete configuration options.

#### Key Parameters

```yaml
slam:
  # SLAM settings
  num_gaussians: 100000      # Number of Gaussians  
  focal_length: 527.0        # Camera focal length
  keyframe_distance: 0.5     # Keyframe selection distance
  tracking_max_iter: 30      # Pose estimation iterations

gpu:
  cuda_device: 0             # GPU index
  batch_size: 1024           # Processing batch size

monitoring:
  publish_odometry: true      # Publish odometry stream
  publish_pose: true         # Publish current pose
  publish_stats: true        # Publish statistics
```

## 📈 Performance

### Benchmarks

| Setting | Gaussians | Processing Time | Memory Usage | Quality |
|---------|-----------|-----------------|--------------|---------|
| Low | 30,000 | <20ms | ~250MB | Good for preview |
| Medium | 75,000 | ~50ms | ~600MB | Balanced |
| High | 150,000 | ~100ms | ~1.2GB | Best quality |

### Optimization Tips

```yaml
# For real-time performance
slam:
  num_gaussians: 30000
  tracking_max_iter: 15
  keyframe_distance: 1.2
  publish_statistics: false

# For high-quality scenes
slam:
  num_gaussians: 150000
  tracking_max_iter: 50
  keyframe_distance: 0.3
  print_memory_usage: true
```

## 📡 ROS2 Topics

### Published Topics

| Topic | Type | Description | QoS Profile |
|-------|------|-------------|-------------|
| `/slam/odometry` | `nav_msgs/Odometry` | SLAM odometry | Best-Effort |
| `/slam/pose` | `geometry_msgs/PoseStamped` | Current pose | Reliable |
| `/slam/trajectory` | `geometry_msgs/PoseStamped` | Trajectory points | Reliable |
| `/slam/keyframes` | `sensor_msgs/Image` | Keyframe images | Best-Effort |
| `/slam/statistics` | `std_msgs/String` | Performance stats | Reliable |
| `/slam/generation_progress` | `std_msgs/String` | Generation status | Reliable |

### Subscribed Topics

| Topic | Type | Description |
|-------|------|-------------|
| `/camera/image_raw` | `sensor_msgs/Image` | RGB camera input |
| `/camera/depth` | `sensor_msgs/Image` | Depth camera input |
| `/camera/camera_info` | `sensor_msgs/CameraInfo` | Camera calibration |

## 🐛 Troubleshooting

For comprehensive troubleshooting guides, see [Troubleshooting](docs/TUTORIALS.md#troubleshooting-tutorial).

### Common Issues

**Issue**: Node connects but doesn't process frames
```bash
# Check if camera topics are published
ros2 topic list | grep camera
ros2 topic echo /camera/image_raw --once
```

**Issue**: High memory usage
```yaml
# Reduce Gaussian count and disable features
slam:
  num_gaussians: 30000
monitoring:
  publish_statistics: false
```

**Issue**: Poor SLAM accuracy
```yaml
# Improve tracking parameters
slam:
  focal_length: 527.0
  tracking_max_iter: 50
  keyframe_distance: 0.3
```

## 🧪 Testing

```bash
# Run tests
colcon test --packages-select 4dgs_slam

# Run with logging
colcon test --packages-select 4dgs_slam --event-handlers console_direct+
```

## 📝 Citation

If you use this package in your research, please cite:

```bibtex
@article{li20254d,
  title={4{D} {G}aussian {S}platting {SLAM}},
  author={Li, Yanyan and Fang, Youxu and Zhu, Zunjie and Li, Kunyi and Ding, Yong and Tombari, Federico},
  journal={arXiv preprint arXiv:2503.16710},
  year={2025}
}
```

## 🤝 Contributing

We welcome contributions! Please see [Contributing Guide](docs/CONTRIBUTING.md) for details.

### Development Guidelines

- Follow PEP 8 style guide
- Maintain >80% code coverage
- Update documentation for changes
- Run tests before submitting
- Sign off contributions

## 🙏 Acknowledgments

- Original 4DGS-SLAM implementation: [Yanyan Li et al.](https://github.com/yanyan-li/4DGS-SLAM)
- Based on Gaussian Splatting, GeoGaussian, SC-GS, and MonoGS
- Used PyTorch and OpenCV for computer vision operations

## 📄 License

This package is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## 🗣️ Support

- **GitHub Issues**: [Report bugs or request features](https://github.com/yourusername/4dgs_slam_ros2/issues)
- **Documentation**: See [docs](docs) directory
- **Community**: Join discussions on GitHub

## 🔄 Changelog

See [CHANGELOG.md](CHANGELOG.md) for recent updates and version history.

## 📊 Project Statistics

- **Star**: [![GitHub Stars](https://img.shields.io/github/stars/yourusername/4dgs_slam_ros2)](https://github.com/yourusername/4dgs_slam_ros2)
- **Forks**: [![GitHub Forks](https://img.shields.io/github/forks/yourusername/4dgs_slam_ros2)](https://github.com/yourusername/4dgs_slam_ros2)

---

**Note**: This package provides ROS2 integration for the 4D Gaussian Splatting SLAM system. Full functionality requires the external 4DGS-SLAM dependencies and GPU acceleration.# 4dgs-slam-ros2
# 4dgs-slam-ros2
