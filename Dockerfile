
# GS ICP SLAM ROS2 Wrapper - Dockerfile

FROM nvidia/cuda:11.8.0-devel-ubuntu22.04

# Prevent interactive prompts
ENV DEBIAN_FRONTEND=noninteractive

# Install basics
RUN apt-get update && apt-get install -y \
    software-properties-common \
    curl \
    git \
    wget \
    build-essential \
    cmake \
    python3-pip \
    python3-dev \
    libpcl-dev \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Install ROS2 Humble
RUN add-apt-repository universe \
    && curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key -o /usr/share/keyrings/ros-archive-keyring.gpg \
    && echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu $(lsb_release -cs) main" | tee /etc/apt/sources.list.d/ros2.list > /dev/null \
    && apt-get update && apt-get install -y \
    ros-humble-desktop \
    ros-humble-cv-bridge \
    ros-humble-perception-pcl \
    && rm -rf /var/lib/apt/lists/*

# Install Python deps
# GS-ICP-SLAM asks for torch 2.0.0+cu118
RUN pip3 install --no-cache-dir \
    torch==2.0.0+cu118 \
    torchvision==0.15.1+cu118 \
    torchaudio==2.0.1 \
    --index-url https://download.pytorch.org/whl/cu118

RUN pip3 install --no-cache-dir \
    opencv-python \
    open3d \
    scipy \
    tqdm \
    torchmetrics \
    plyfile \
    rerun-sdk

# Install submodules
# We copy them into the image to build them
WORKDIR /app/submodules
ENV TORCH_CUDA_ARCH_LIST="6.0 6.1 7.0 7.5 8.0 8.6+PTX"
ENV TCNN_CUDA_ARCHITECTURES=86;80;75;70;61;60
COPY submodules/diff-gaussian-rasterization /app/submodules/diff-gaussian-rasterization
RUN pip3 install /app/submodules/diff-gaussian-rasterization

COPY submodules/simple-knn /app/submodules/simple-knn
RUN pip3 install /app/submodules/simple-knn

COPY submodules/fast_gicp /app/submodules/fast_gicp
WORKDIR /app/submodules/fast_gicp
RUN mkdir build && cd build && cmake .. && make -j$(nproc) && cd .. && python3 setup.py install

# Setup workspace
WORKDIR /ws/src/gs_icp_slam
# We will mount the source code here
# But for building potentially we need dependencies...

# Env setup
RUN echo "source /opt/ros/humble/setup.bash" >> ~/.bashrc
