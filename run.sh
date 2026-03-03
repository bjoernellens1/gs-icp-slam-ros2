#!/bin/bash
# run.sh — Run a GS-ICP-SLAM ROS2 container
# Usage:
#   ./run.sh            # auto-detect GPU and run appropriate image
#   ./run.sh cuda       # run CUDA (NVIDIA) image
#   ./run.sh rocm       # run ROCm (AMD) image
#   ./run.sh cpu        # run CPU-only image

set -e

# Prefer podman, fall back to docker
if command -v podman &>/dev/null; then
    DOCKER_CMD="podman"
else
    DOCKER_CMD="docker"
fi

echo "Using container engine: $DOCKER_CMD"

CONTAINER_NAME="gs_icp_slam_container"

run_cuda() {
    IMAGE_NAME="gs-icp-slam-ros2:cuda"
    echo "Running CUDA (NVIDIA) image: $IMAGE_NAME"
    $DOCKER_CMD run --rm -it \
        --gpus all \
        --ipc=host \
        --name "$CONTAINER_NAME" \
        "$IMAGE_NAME"
}

run_rocm() {
    IMAGE_NAME="gs-icp-slam-ros2:rocm"
    echo "Running ROCm (AMD) image: $IMAGE_NAME"

    DEVICE_FLAGS="--device=/dev/kfd --device=/dev/dri"
    GROUP_FLAGS="--group-add video"
    if grep -q "^render:" /etc/group; then
        RENDER_GID=$(grep "^render:" /etc/group | cut -d: -f3)
        GROUP_FLAGS="$GROUP_FLAGS --group-add $RENDER_GID"
    else
        echo "Warning: Group 'render' not found in /etc/group. Skipping '--group-add render'."
    fi

    $DOCKER_CMD run --rm -it \
        $DEVICE_FLAGS \
        $GROUP_FLAGS \
        --ipc=host \
        --security-opt seccomp=unconfined \
        -e HSA_OVERRIDE_GFX_VERSION=11.0.0 \
        --name "$CONTAINER_NAME" \
        "$IMAGE_NAME"
}

run_cpu() {
    IMAGE_NAME="gs-icp-slam-ros2:cpu"
    echo "Running CPU-only image: $IMAGE_NAME"
    $DOCKER_CMD run --rm -it \
        --ipc=host \
        --name "$CONTAINER_NAME" \
        "$IMAGE_NAME"
}

# Auto-detect if no target specified
auto_detect() {
    if command -v nvidia-smi &>/dev/null && nvidia-smi &>/dev/null 2>&1; then
        echo "Detected NVIDIA GPU — using CUDA image."
        run_cuda
    elif ls /dev/kfd &>/dev/null 2>&1; then
        echo "Detected AMD GPU (/dev/kfd) — using ROCm image."
        run_rocm
    else
        echo "No GPU detected — using CPU-only image."
        run_cpu
    fi
}

TARGET="${1:-auto}"

case "$TARGET" in
    cuda)  run_cuda ;;
    rocm)  run_rocm ;;
    cpu)   run_cpu ;;
    auto)  auto_detect ;;
    *)
        echo "Unknown target: $TARGET"
        echo "Usage: $0 [cuda|rocm|cpu|auto]"
        exit 1
        ;;
esac
