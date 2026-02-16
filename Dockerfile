FROM osrf/ros:humble-desktop

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8

# Install basic system dependencies and tools
RUN apt-get -o APT::Sandbox::User=root -o Dir::Cache::archives="/tmp/" update && \
    apt-get -o APT::Sandbox::User=root -o Dir::Cache::archives="/tmp/" install -y --no-install-recommends \
    wget \
    git \
    python3-pip \
    ros-humble-ament-cmake-python \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
# Upgrade pip first
RUN python3 -m pip install --upgrade pip

# Install PyTorch with CUDA 11.7 support
RUN pip3 install torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu117

COPY requirements.txt /tmp/requirements.txt
RUN pip3 install -r /tmp/requirements.txt && rm /tmp/requirements.txt

# Source ROS 2 setup
RUN echo "source /opt/ros/humble/setup.bash" >> /root/.bashrc

# Create workspace
WORKDIR /workspace
RUN mkdir -p src/4dgs-slam-ros2

# Copy source code
COPY . /workspace/src/4dgs-slam-ros2/

# Build the workspace
# We need to setup CUDA environment for compilation if possible, but for now we try to build without explicit CUDA toolkit if not needed for core build
# (The user code might fail if it needs nvcc, but let's see)
RUN . /opt/ros/humble/setup.sh && \
    colcon build --symlink-install --cmake-args -DCMAKE_BUILD_TYPE=Release

# Download pretrained YOLOv9 model
RUN mkdir -p /workspace/src/4dgs-slam-ros2/pretrained && \
    wget https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov9e-seg.pt -P /workspace/src/4dgs-slam-ros2/pretrained

# Set entrypoint
COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh
ENTRYPOINT ["/entrypoint.sh"]
CMD ["bash"]
