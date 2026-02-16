# Agent Guidelines for 4dgs-slam-ros2

> [!IMPORTANT]
> **ALWAYS** use the provided Docker/Devcontainer environment for development, testing, and running this project. The host system does NOT have the required ROS2 Humble or CUDA environment configured.

## Running in Docker

1.  **Build the image**:
    ```bash
    docker build -t 4dgs-slam-ros2 .
    ```

2.  **Run with GUI support (Pre-requisite)**:
    Ensure you allow X11 connections on the host (if applicable):
    ```bash
    xhost +local:docker
    ```

3.  **Start the container**:
    ```bash
    docker run --gpus all -it --rm \
        --net=host \
        --env="DISPLAY" \
        --env="QT_X11_NO_MITSHM=1" \
        --volume="/tmp/.X11-unix:/tmp/.X11-unix:rw" \
        4dgs-slam-ros2 \
        ros2 run four_dgs_slam four_dgs_slam_node
    ```

## Running Graphical Apps (RViz2)

Inside the container, you can run RViz2:
```bash
ros2 run rviz2 rviz2
```
If configured correctly, the window should appear on your host screen.

## Development

The repository is mounted at `/workspace/src/4dgs-slam-ros2`.
Always source the setup script before running ROS commands:
```bash
source /opt/ros/humble/setup.bash
source install/setup.bash
```
