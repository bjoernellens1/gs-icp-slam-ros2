# 4DGS-SLAM ROS2 Package - API Reference

This document provides detailed information about the classes, methods, and interfaces provided by the 4DGS-SLAM ROS2 package.

## Table of Contents

- [Overview](#overview)
- [Classes](#classes)
  - [SLAMParameters](#slamparameters)
  - [SLAMNode](#slamnode)
  - [GaussianField](#gaussianfield)
  - [CameraParameters](#cameraparameters)
  - [KeyFrame](#keyframe)
- [Data Structures](#data-structures)
- [Utility Functions](#utility-functions)
- [Configuration](#configuration)

## Overview

The 4DGS-SLAM ROS2 package provides a comprehensive set of classes and functions for integrating the 4D Gaussian Splatting SLAM system with ROS2. The package is designed for real-time SLAM processing, dynamic scene reconstruction, and integration with ROS2 navigation and perception systems.

## Classes

### SLAMParameters

The primary configuration class that manages all SLAM system parameters.

#### Constructor

```python
SLAMParameters(config_file: Optional[str] = None)
```

**Parameters:**
- `config_file` (str, optional): Path to YAML configuration file. If provided, parameters are loaded from this file.

#### Methods

**get()**

Get nested configuration value by key path.

```python
get(*keys: str, default: Optional[Any] = None) -> Any
```

**Parameters:**
- `*keys`: Variable length argument of keys to navigate through the configuration dictionary.
- `default`: Default value to return if key path is not found.

**Returns:**
- Value at the specified configuration path.

**Example:**
```python
params = SLAMParameters()
focal_length = params.get('slam', 'focal_length')
num_gaussians = params.get('slam', 'num_gaussians', default=100000)
```

**set()**

Set nested configuration value at specified key path.

```python
set(*keys: str, value: Any) -> None
```

**Parameters:**
- `*keys`: Variable length argument of keys to set.
- `value`: Value to set.

**Example:**
```python
params = SLAMParameters()
params.set('slam', 'num_gaussians', 150000)
```

**save_to_file()**

Save current configuration to YAML file.

```python
save_to_file(file_path: str) -> None
```

**Parameters:**
- `file_path` (str): Path to save configuration file.

### SLAMNode

The main SLAM processing node that integrates with ROS2.

#### Constructor

```python
SLAMNode()
```

Initializes the SLAM system with default parameters.

#### Methods

**process_frame()**

Process a new RGB-D frame with SLAM system.

```python
process_frame(rgb_image: np.ndarray, depth_image: np.ndarray, timestamp: float) -> bool
```

**Parameters:**
- `rgb_image` (np.ndarray): RGB input image as numpy array.
- `depth_image` (np.ndarray): Depth input image as numpy array.
- `timestamp` (float): Timestamp of the frame.

**Returns:**
- bool: True if processing successful, False otherwise.

**get_reconstruction()**

Get current reconstruction state.

```python
get_reconstruction() -> Dict
```

**Returns:**
- Dictionary containing reconstruction state with camera trajectory, keyframes, and Gaussian fields.

**save_results()**

Save reconstruction results to file.

```python
save_results(output_path: str) -> bool
```

**Parameters:**
- `output_path` (str): Path to save reconstruction results.

**Returns:**
- bool: True if save successful.

**load_checkpoint()**

Load reconstruction from checkpoint file.

```python
load_checkpoint(checkpoint_path: str) -> bool
```

**Parameters:**
- `checkpoint_path` (str): Path to checkpoint file.

**Returns:**
- bool: True if loading successful.

**cleanup()**

Clean up resources and save final results.

```python
cleanup() -> None
```

### GaussianField

Represents the 4D Gaussian field for dynamic scene representation.

#### Constructor

```python
GaussianField(num_gaussians: int)
```

**Parameters:**
- `num_gaussians` (int): Number of Gaussians to manage.

#### Attributes

- `num_gaussians: int`: The number of Gaussians in the field.
- `positions: np.ndarray`: Position of each Gaussian (shape: [num_gaussians, 3]).
- `colors: np.ndarray`: Color of each Gaussian (shape: [num_gaussians, 3]).
- `scales: np.ndarray`: Scale of each Gaussian (shape: [num_gaussians, 3]).
- `rotations: np.ndarray`: Rotation of each Gaussian (shape: [num_gaussians, 4]).
- `opacities: np.ndarray`: Opacity of each Gaussian (shape: [num_gaussians]).
- `dynamics: Dict[int, float]`: Time-dependent parameters for each Gaussian.

#### Methods

**update()**

Update Gaussian parameters based on new frame information.

```python
update(frame_id: int, gaussian_params: np.ndarray) -> None
```

**Parameters:**
- `frame_id` (int): Current frame identifier.
- `gaussian_params` (np.ndarray): New Gaussian parameters from frame.

### CameraParameters

Data structure for camera parameters.

#### Attributes

- `camera_id: str`: Unique identifier for camera.
- `model_type: str`: Camera model type (e.g., 'pinhole', 'fisheye').
- `width: int`: Image width.
- `height: int`: Image height.
- `intrinsic_matrix: np.ndarray`: Camera intrinsic matrix (shape: [3, 3]).
- `distortion_coefficients: np.ndarray`: Distortion coefficients (shape: [5,] for simple models).
- `extrinsic_matrix: np.ndarray`: Extrinsic matrix (shape: [4, 4]).
- `projection_matrix: np.ndarray`: Projection matrix (shape: [4, 4]).

### KeyFrame

Represents a keyframe used in SLAM.

#### Attributes

- `frame_id: int`: Frame identifier.
- `timestamp: float`: Timestamp of the frame.
- `image: np.ndarray`: RGB image.
- `depth: np.ndarray`: Depth image.
- `camera_pose: np.ndarray`: Camera pose (shape: [4, 4]).
- `features: Optional[List]`: Extracted features (optional).

## Data Structures

### Configuration Dictionary Structure

The main configuration dictionary has the following hierarchical structure:

```python
{
    'slam': {  # Main SLAM parameters
        'camera_model': str,
        'focal_length': float,
        'principal_point': List[float],
        'num_gaussians': int,
        'tracking_max_iter': int,
        ...
    },
    'gpu': {  # GPU configuration
        'cuda_device': int,
        'batch_size': int,
        ...
    },
    'system': {  # System parameters
        'working_directory': str,
        'log_level': str,
        ...
    },
    'data': {  # Data input parameters
        'source_type': str,
        'image_topic': str,
        'depth_topic': str,
        ...
    },
    'monitoring': {  # Publishing and monitoring
        'publish_odometry': bool,
        'publish_poses': bool,
        ...
    }
}
```

## Utility Functions

**preprocess_image()**

Preprocess image for SLAM processing.

```python
def preprocess_image(image: np.ndarray) -> np.ndarray
```

**denoise_image()**

Denoise image using optimized algorithm.

```python
def denoise_image(image: np.ndarray, strength: float = 0.1) -> np.ndarray
```

**compute_depth_consistency()**

Compute depth consistency between two images.

```python
def compute_depth_consistency(img1: np.ndarray, img2: np.ndarray) -> np.ndarray
```

## Configuration

### Default Parameters

The default configuration provides balanced performance and quality settings:

```python
default_config = {
    'slam': {
        'num_gaussians': 100000,
        'focal_length': 527.0,
        'keyframe_distance': 0.5,
        'keyframe_angle': 0.4,
    },
    'gpu': {
        'cuda_device': 0,
        'batch_size': 1024,
    },
    'monitoring': {
        'publish_statistics': True,
    }
}
```

### Performance Tuning

For real-time applications, use performance-optimized configuration:

```python
# Reduced for better performance
'slam': {
    'num_gaussians': 30000,
    'tracking_max_iter': 15,
    'keyframe_distance': 1.0,
    'keyframe_angle': 0.6,
},
'system': {
    'working_directory': '/tmp/4dgs_slam_fast',
},
'monitoring': {
    'publish_statistics': False,
    'publish_trajectory': False,
}
```

### Quality Tuning

For maximum reconstruction quality:

```python
# Increased for better quality
'slam': {
    'num_gaussians': 200000,
    'tracking_max_iter': 50,
    'keyframe_distance': 0.5,
    'keyframe_angle': 0.4,
},
'monitoring': {
    'publish_statistics': True,
    'publish_trajectory': True,
}
```

## ROS2 Integration

### ROS2 Topics

The SLAM node publishes the following topics:

| Topic Name | Message Type | Description |
|------------|--------------|-------------|
| `/slam/odometry` | `nav_msgs/Odometry` | Odometry estimates |
| `/slam/pose` | `geometry_msgs/PoseStamped` | Current camera pose |
| `/slam/trajectory` | `geometry_msgs/PoseStamped` | Trajectory history |
| `/slam/keyframes` | `sensor_msgs/Image` | Extracted keyframes |
| `/slam/statistics` | `std_msgs/String` | Performance statistics |
| `/slam/generation_progress` | `std_msgs/String` | Generation process updates |

### ROS2 Services

The SLAM node provides the following services:

| Service Name | Type | Description |
|--------------|------|-------------|
| `/slam/get_reconstruction` | `4dgs_slam_ros2/srv/GetReconstruction` | Request reconstruction data |
| `/slam/save_checkpoint` | `4dgs_slam_ros2/srv/SaveCheckpoint` | Save current state to checkpoint |

## Error Handling

The SLAM system includes comprehensive error handling:

- **RuntimeErrors**: Thrown for processing failures
- **ValueErrors**: Raised for invalid parameter values
- **ImportErrors**: Handled for missing dependencies

All errors are logged with appropriate severity levels.

## Performance Notes

- Processing time per frame is typically <100ms on modern hardware
- Memory usage scales with Gaussian count (~8MB per 10k Gaussians)
- Recommended GPU: NVIDIA RTX 3060 or better
- Recommended RAM: 16GB or more for high Gaussian counts