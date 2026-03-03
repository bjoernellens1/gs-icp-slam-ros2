#!/bin/bash
# build.sh — Build GS-ICP-SLAM ROS2 container images
# Usage:
#   ./build.sh            # build all variants
#   ./build.sh cuda       # build CUDA (NVIDIA) image only
#   ./build.sh rocm       # build ROCm (AMD) image only
#   ./build.sh cpu        # build CPU-only image only

set -e

# Prefer podman, fall back to docker
if command -v podman &>/dev/null; then
    DOCKER_CMD="podman"
else
    DOCKER_CMD="docker"
fi

echo "Using container engine: $DOCKER_CMD"

build_cuda() {
    echo ""
    echo "=== Building CUDA (NVIDIA) image ==="
    $DOCKER_CMD build -f Dockerfile.cuda -t gs-icp-slam-ros2:cuda .
    echo "=== CUDA image built: gs-icp-slam-ros2:cuda ==="
}

build_rocm() {
    echo ""
    echo "=== Building ROCm (AMD) image ==="
    $DOCKER_CMD build -f Dockerfile.rocm -t gs-icp-slam-ros2:rocm .
    echo "=== ROCm image built: gs-icp-slam-ros2:rocm ==="
}

build_cpu() {
    echo ""
    echo "=== Building CPU-only image ==="
    $DOCKER_CMD build -f Dockerfile.cpu -t gs-icp-slam-ros2:cpu .
    echo "=== CPU image built: gs-icp-slam-ros2:cpu ==="
}

TARGET="${1:-all}"

case "$TARGET" in
    cuda)   build_cuda ;;
    rocm)   build_rocm ;;
    cpu)    build_cpu ;;
    all)
        build_cuda
        build_rocm
        build_cpu
        echo ""
        echo "=== All images built successfully ==="
        $DOCKER_CMD images | grep gs-icp-slam-ros2
        ;;
    *)
        echo "Unknown target: $TARGET"
        echo "Usage: $0 [cuda|rocm|cpu|all]"
        exit 1
        ;;
esac
