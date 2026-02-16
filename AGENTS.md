# Agent Guidelines for gs-icp-slam-ros2

> [!IMPORTANT]
> **ALWAYS** use the provided Docker environment for development, testing, and running this project. The host system does NOT have the required ROS2 Humble or CUDA environment configured.
>
> **Project Name**: `gs_icp_slam` (formerly `four_dgs_slam`). Always use `gs_icp_slam` for package validation and launch commands.

## Architecture & Environment

- **Package Name**: `gs_icp_slam`
- **Workspace Path**: `/ws/src/gs_icp_slam` (in Docker)
- **ROS Version**: Humble
- **GPU Support**: NVIDIA CUDA (default) & AMD ROCm (optional)

## 🐳 Docker Workflow

### 1. Build the Image

**Option A: NVIDIA GPU (Standard)**
```bash
docker build -t gs-icp-slam-ros2 .
```

**Option B: AMD GPU (ROCm / Strix Halo)**
```bash
docker build -f Dockerfile.rocm -t gs-icp-slam-ros2:rocm .
```

### 2. Run the Container

**Interactive Development (NVIDIA)**
```bash
# Allow X11 forwarding
xhost +local:docker

docker run --gpus all -it --rm \
    --net=host \
    --ipc=host \
    --env="DISPLAY" \
    --env="QT_X11_NO_MITSHM=1" \
    --volume="/tmp/.X11-unix:/tmp/.X11-unix:rw" \
    --volume="$(pwd):/ws/src/gs_icp_slam" \
    gs-icp-slam-ros2 \
    bash
```

**Interactive Development (AMD ROCm)**
```bash
# Allow X11 forwarding
xhost +local:docker

# For Strix Halo, you might need HSA_OVERRIDE_GFX_VERSION=11.0.0
docker run --rm -it \
    --net=host \
    --ipc=host \
    --device=/dev/kfd \
    --device=/dev/dri \
    --group-add video \
    --group-add render \
    --env="DISPLAY" \
    --env="QT_X11_NO_MITSHM=1" \
    --volume="/tmp/.X11-unix:/tmp/.X11-unix:rw" \
    --volume="$(pwd):/workspace/src/gs_icp_slam" \
    -e HSA_OVERRIDE_GFX_VERSION=11.0.0 \
    gs-icp-slam-ros2:rocm \
    bash
```

### 3. Build & Source

Inside the container:
```bash
# Build the package
colcon build --packages-select gs_icp_slam --symlink-install

# Source the workspace
source install/setup.bash
```

### 4. Run Launch Files

**Standard SLAM**
```bash
ros2 launch gs_icp_slam slam.launch.py
```

**Bag File Playback**
```bash
ros2 launch gs_icp_slam gs_icp_slam_with_bag.launch.py bag_file:=/path/to/bag.mcap
```

**Visualizer**
```bash
ros2 launch gs_icp_slam gs_icp_slam_visualization.launch.py
```
