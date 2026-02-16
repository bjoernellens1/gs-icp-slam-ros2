#!/bin/bash
set -e

# Source ROS 2
source /opt/ros/humble/setup.bash

# Source workspace
if [ -f "/workspace/install/setup.bash" ]; then
    source /workspace/install/setup.bash
fi

echo "Starting 4DGS-SLAM node..."
# Run in background
ros2 run four_dgs_slam four_dgs_slam_node --ros-args -p system.log_level:=DEBUG &
NODE_PID=$!

# Wait for 10 seconds
echo "Waiting for node to initialize..."
sleep 10

# Check if node is still running
if ps -p $NODE_PID > /dev/null; then
   echo "Node is running successfully!"
   # Kill the node
   kill $NODE_PID
   exit 0
else
   echo "Node crashed!"
   exit 1
fi
