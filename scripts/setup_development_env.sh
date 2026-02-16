#!/usr/bin/env bash
#!/bin/bash
#
# Setup script for GS-ICP-SLAM ROS2 environment
#

# Check for required tools
echo "Checking for required tools..."
command -v python3 >/dev/null 2>&1 || { echo >&2 "Python3 is required but not installed."; exit 1; }
command -v conda >/dev/null 2>&1 || { echo >&2 "Anaconda/Miniconda is required but not installed."; exit 1; }

# Set up ROS 2 environment
if [ -f /opt/ros/humble/setup.bash ]; then
    echo "Activating ROS 2 Humble environment..."
    source /opt/ros/humble/setup.bash
elif [ -f /opt/ros/rolling/setup.bash ]; then
    echo "Activating ROS 2 Rolling environment..."
    source /opt/ros/rolling/setup.bash
else
    echo "WARNING: ROS 2 environment not found in /opt/ros/"
fi

# Create conda environment
echo "Setting up conda environment..."
conda create -n gs_icp_slam python=3.8 \
    -y || { echo >&2 "Failed to create conda environment"; exit 1; }

echo "Activating gs_icp_slam environment..."
source $(conda info --base)/etc/profile.d/conda.sh
conda activate gs_icp_slam

# Install Python dependencies
echo "Installing Python dependencies..."
pip install --upgrade pip
pip install pyyaml numpy opencv-python scipy
pip install torch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 \
    --extra-index-url https://download.pytorch.org/whl/cu117

# Install ROS 2 dependencies
echo "Installing ROS 2 dependencies..."
apt-get update
apt-get install -y \
    python3-dev \
    gcc \
    g++ \
    ros-humble-desktop \
    ros-humble-rviz2 \
    ros-humble-camera-info-manager \
    ros-humble-compressed-image-transport \
    ros-humble-image-transport

# Install package in development mode
echo "Installing gs_icp_slam package..."
echo "Installing gs_icp_slam package..."
pip install -e .

echo "Setup completed successfully!"
echo "To activate the environment: conda activate gs_icp_slam"
echo "To run the package: ros2 run gs_icp_slam gs_icp_slam_node"