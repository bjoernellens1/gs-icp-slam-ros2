# 4DGS-SLAM ROS2 Package - Development Guide

This document provides guidelines and best practices for developing, extending, and contributing to the 4D Gaussian Splatting SLAM ROS2 package.

## Table of Contents

1. [Development Environment](#development-environment)
2. [Code Structure](#code-structure)
3. [Coding Standards](#coding-standards)
4. [Adding New Features](#adding-new-features)
5. [Integration with External Libraries](#integration-with-external-libraries)
6. [Testing Guidelines](#testing-guidelines)
7. [Performance Optimization](#performance-optimization)
8. [Documentation Standards](#documentation-standards)
9. [Issue Reporting](#issue-reporting)
10. [Pull Request Guide](#pull-request-guide)

## Development Environment

### Prerequisites

```bash
# System requirements
- Ubuntu 20.04/22.04
- ROS 2 Humble or Rolling
- NVIDIA GPU with CUDA 11.x
- Python 3.8+
- Docker and docker-compose (optional)

# Install development tools
sudo apt install python3.8-dev gcc g++ cmake git
pip install pre-commit pytest pytest-cov black flake8 mypy
```

### Project Setup

```bash
# Clone repository
cd ~/dev
git clone https://github.com/yourusername/4dgs_slam_ros2.git
cd 4dgs_slam_ros2

# Install pre-commit hooks
pre-commit install

# Setup development environment
conda create -n 4dgs-slam-dev python=3.8 -y
conda activate 4dgs-slam-dev
source /opt/ros/humble/setup.bash
pip install -e .
```

## Code Structure

```
4dgs_slam_ros2/
├── CMakeLists.txt                      # CMake build configuration
├── package.xml                         # ROS2 package manifest
├── 4dgs_slam_node.py                  # Package entry point
├── docs/                              # Documentation directory
│   ├── API.md                         # Complete API reference
│   ├── TUTORIALS.md                   # Step-by-step tutorials
│   ├── ADVANCED.md                    # Advanced topics
│   └── DEVELOPMENT.md                 # This file
├── config/                            # Configuration files
│   ├── slam_config.yaml               # Main configuration
│   └── performance_config.yaml        # Performance settings
├── launch/                            # ROS2 launch files
│   ├── 4dgs_slam_basic.launch.py      # Basic launch
│   ├── 4dgs_slam_with_bag.launch.py   # Bag playback launch
│   └── 4dgs_slam_visualization.launch.py  # Visualization launch
├── scripts/                           # Utility scripts
│   ├── run_slam.py                    # SLAM runner script
│   └── setup_development_env.sh       # Setup script
├── src/4dgs_slam_ros2/
│   ├── __init__.py                    # Package initialization
│   ├── parameters.py                  # Configuration management
│   ├── node.py                        # Main SLAM node implementation
│   └── launch_helpers.py              # Launch utilities
├── tests/                             # Test files
│   └── test_parameters.py             # Test cases
└── examples/                          # Usage examples
    ├── example_basic.py               # Basic usage example
    ├── example_bag_playback.py        # Bag playback example
    ├── example_configuration.py       # Configuration example
    └── example_visualization.py       # Visualization example
```

## Coding Standards

### Python Code Style

```python
# Use type hints for better code clarity
import numpy as np
from typing import Optional, List, Dict

def process_frame(rgb_image: np.ndarray, 
                 depth_image: np.ndarray, 
                 timestamp: float) -> bool:
    """
    Process a new frame with SLAM system.
    
    Args:
        rgb_image: RGB input image as numpy array
        depth_image: Depth input image as numpy array
        timestamp: Timestamp of the frame
        
    Returns:
        True if processing successful
        
    Example:
        >>> process_rgb_frame(rgb_img, depth_img, time.time())
    """
    try:
        # Validate input
        if rgb_image is None or depth_image is None:
            raise ValueError("Input images cannot be None")
            
        # Process implementation
        result = _process_with_slam(rgb_image, depth_image, timestamp)
        
        return result
        
    except Exception as e:
        self.get_logger().error(f'Processing error: {e}')
        return False
```

**Style guidelines:**
- Follow PEP 8 style guide
- Use black for code formatting: `black --line-length 100 *.py`
- Use pylint/flake8 for code quality checks
- Use mypy for static type checking
- Add docstrings to all public functions and classes
- Use type hints for all parameters and return values

### ROS2 Specific Standards

```python
# ROS2 node implementation
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image

class MySLAMNode(Node):
    """
    ROS2 node for SLAM system.
    
    This node provides real-time SLAM processing and publishes
    to various topics for navigation and perception systems.
    """
    
    def __init__(self):
        """Initialize node with appropriate logging"""
        super().__init__('my_slam_node')
        self.get_logger().info('SLAM Node initialized')
        
        # Initialize subscribers with proper QoS
        self.sub_image = self.create_subscription(
            Image,
            '/camera/image',
            self.image_callback,
            qos_profile=DepthImageQualityOfService()
        )
    
    def image_callback(self, msg: Image) -> None:
        """Callback for image messages"""
        self.get_logger().debug('Received image')
        # Process image
```

## Adding New Features

### Extension Example: Custom Feature Extractor

```python
# Create new module: src/4dgs_slam_ros2/custom_features.py

class CustomFeatureExtractor:
    """Custom feature extraction implementation"""
    
    def __init__(self, parameters: Dict):
        """
        Initialize custom feature extractor.
        
        Args:
            parameters: Configuration parameters for feature extraction
        """
        self.parameters = parameters
        self.feature_cache = {}
    
    def extract_features(self, image: np.ndarray) -> Dict:
        """
        Extract features from image using custom method.
        
        Args:
            image: Input RGB image
            
        Returns:
            Dictionary containing extracted features
        """
        # Custom feature extraction logic
        features = {
            'keypoints': self._extract_custom_keypoints(image),
            'descriptors': self._compute_custom_descriptors(image),
            'metadata': {
                'algorithm': 'custom',
                'parameters': self.parameters,
                'timestamp': time.time()
            }
        }
        return features
    
    def _extract_custom_keypoints(self, image: np.ndarray) -> List:
        """Extract keypoints using custom method"""
        # Implementation of custom feature extraction
        pass
```

### Integration Points

1. **Configuration Management**: Add settings to `slam_config.yaml` and update `SLAMParameters` class
2. **Node Integration**: Extend existing methods in `SLAMNode` class
3. **Launch File**: Add new launch configurations in `launch/` directory
4. **Documentation**: Update relevant documentation files

## Integration with External Libraries

### Adding Python Dependencies

```python
# Add to setup.py (if Python package)
from setuptools import setup, find_packages
import os

def get_python_requirements():
    """Read Python requirements from requirements.txt"""
    with open('requirements.txt', 'r') as f:
        return [line.strip() for line in f if line.strip()]

setup(
    name='4dgs_slam_ros2',
    packages=find_packages(),
    install_requires=get_python_requirements(),
    extras_require={
        'dev': [
            'pytest>=7.0',
            'black>=20.0',
            'flake8>=3.8',
            'mypy>=0.9'
        ]
    }
)
```

### C++ Extension Development

```cpp
// Add C++ extension in src/4dgs_slam_ros2_cpp/
// Example: performance_optimization.cpp

#include "4dgs_slam_ros2_cpp/performance_optimization.hpp"

namespace 4dgs_slam_ros2::cpp {

void compute_raycasting_optimized(
    const GaussianField& field,
    const CameraPose& camera,
    float* output_buffer,
    int output_size
) {
    // Optimized CUDA implementation
    // ...
}

} // namespace
```

```cmake
# Update CMakeLists.txt for C++ extensions

add_library(4dgs_slam_ros2_cpp performance_optimization.cpp)
target_link_libraries(4dgs_slam_ros2_cpp PRIVATE 
    rclcpp
    sensor_msgs
    geometry_msgs
    cv_bridge
)

ament_target_dependencies(4dgs_slam_ros2_cpp
    rclcpp
    sensor_msgs
    geometry_msgs
)
```

## Testing Guidelines

### Unit Testing Example

```python
# tests/test_parameters.py

import unittest
from pathlib import Path
from 4dgs_slam_ros2.parameters import SLAMParameters

class TestSLAMParameters(unittest.TestCase):
    """Unit tests for SLAM parameters"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.test_config_path = Path(__file__).parent.parent / 'config' / 'slam_config.yaml'
    
    def test_parameter_initialization(self):
        """Test default parameter initialization"""
        params = SLAMParameters()
        self.assertEqual(params.get('slam', 'camera_model'), 'pinhole')
    
    def test_config_loading(self):
        """Test configuration file loading"""
        params = SLAMParameters(str(self.test_config_path))
        num_gaussians = params.get('slam', 'num_gaussians')
        self.assertIsInstance(num_gaussians, (int, float))
    
    def test_nested_parameter_get(self):
        """Test nested parameter access"""
        params = SLAMParameters()
        focal_length = params.get('slam', 'focal_length', default=500.0)
        self.assertEqual(focal_length, 527.0)
```

### Integration Testing

```bash
# Run tests
colcon test --packages-select 4dgs_slam
colcon test-result --verbose

# Run specific test
pytest tests/test_parameters.py -v

# Generate coverage report
pytest tests/ --cov=4dgs_slam --cov-report=html
```

### Testing Checklist

- [ ] All unit tests pass
- [ ] Integration tests complete
- [ ] Code coverage > 80%
- [ ] No linting errors
- [ ] Type checking passes
- [ ] Performance benchmarks maintained
- [ ] Documentation updated

## Performance Optimization

### Profile Code Performance

```python
import time
import cProfile
import pstats

def profile_function(func, *args, **kwargs):
    """Profile a function for performance analysis"""
    
    # Context manager for profiling
    profiler = cProfile.Profile()
    profiler.enable()
    
    result = func(*args, **kwargs)
    
    profiler.disable()
    
    # Print statistics
    stats = pstats.Stats(profiler)
    stats.strip_dirs()
    stats.sort_stats('cumulative')
    stats.print_stats(20)
    
    return result
```

### Performance Testing Framework

```python
# Create benchmark suite
class SLAMPerformanceBenchmark:
    """Performance benchmarking for SLAM operations"""
    
    def __init__(self):
        self.benchmark_results = []
    
    def benchmark_processing_time(self, num_frames=100):
        """Benchmark frame processing time"""
        
        results = []
        for i in range(num_frames):
            start = time.time()
            
            # Process frame
            self.slam_node.process_next_frame()
            
            end = time.time()
            results.append(end - start)
        
        metrics = {
            'avg_time': np.mean(results),
            'std_time': np.std(results),
            'min_time': np.min(results),
            'max_time': np.max(results),
            'fps': 1.0 / np.mean(results)
        }
        
        self.benchmark_results.append(metrics)
        return metrics
```

## Documentation Standards

### Docstring Format

```python
def function_name(param1: str, param2: int) -> bool:
    """
    Brief description of what the function does.
    
    This is a more detailed explanation of the function's purpose
    and behavior. It should cover:
    - Primary functionality
    - Key behaviors
    - Important edge cases
    
    Args:
        param1: Description of parameter 1
        param2: Description of parameter 2
        
    Returns:
        Description of return value
        
    Raises:
        ValueError: If validation fails
        RuntimeError: If processing fails
        
    Example:
        >>> result = function_name("test", 42)
        >>> print(result)
        True
    """
    pass
```

### Documentation Structure

1. **README.md**: High-level overview, installation, quick start
2. **API.md**: Complete API reference with all classes and methods
3. **TUTORIALS.md**: Step-by-step usage guides
4. **ADVANCED.md**: Advanced topics and optimization techniques
5. **CONTRIBUTING.md**: Development and contribution guidelines
6. **CHANGELOG.md**: Version history and changes

## Issue Reporting

### Issue Template

```markdown
## Bug Report
### Description
Describe the bug concisely.

### Expected Behavior
What should happen.

### Actual Behavior
What actually happens.

### Environment
- ROS 2 version: [e.g., Humble]
- Ubuntu version: [e.g., 22.04]
- GPU model: [e.g., RTX 3060]
- Python version: [e.g., 3.8]

### Steps to Reproduce
1. First step
2. Second step
3. ...

### Additional Context
Logs, screenshots, or any other relevant information.
```

### Submitting Issues

1. Check for existing issues
2. Create a detailed issue description
3. Include minimal reproduction case
4. Tag with appropriate category (bug, feature, question)
5. Attach relevant logs and environment details

## Pull Request Guide

### Prerequisites

- [ ] Code follows style guidelines (black, flake8)
- [ ] All tests pass
- [ ] Documentation updated
- [ ] No merge conflicts
- [ ] Commit messages follow conventional commits

### PR Template

```markdown
## Description
Brief description of the PR.

## Changes
- Feature X: Description
- Bug fix: Description
- Documentation: Description

## Testing
- [ ] Tests added/updated
- [ ] Manual testing completed
- [ ] Performance benchmarks maintained

## Checklist
- [ ] Code follows style guidelines
- [ ] Documentation updated
- [ ] No merge conflicts
- [ ] Commit messages clear
```

### Development Workflow

```bash
# 1. Create feature branch
git checkout -b feature/your-feature

# 2. Make changes and commit
git add .
git commit -m "feat: add your feature"

# 3. Push branch
git push origin feature/your-feature

# 4. Open Pull Request
# In GitHub: create pull request from branch to main

# 5. Address review feedback
# Make changes and update PR

# 6. Merge pull request
```

## Continuous Integration

### GitHub Actions Example

```yaml
# .github/workflows/test.yml
name: CI

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    
    steps:
    - uses: actions/checkout@v2
    
    - name: Set up ROS 2 environment
      run: |
        sudo apt update
        sudo apt install -y ros-humble-desktop
    
    - name: Set up Python
      uses: actions/setup-python@v2
      with:
        python-version: '3.8'
    
    - name: Install dependencies
      run: |
        pip install -r requirements.txt
        pip install pytest pytest-cov
    
    - name: Run tests
      run: colcon test --packages-select 4dgs_slam
    
    - name: Generate coverage report
      run: colcon test --packages-select 4dgs_slam --event-handlers+ console_direct+
```

## Code Review Checklist

- Code compiles and runs without errors
- All tests pass
- Code follows project style guidelines
- Complexity is within reasonable limits ($O(n)$ expected)
- No unnecessary imports or dependencies
- Error handling is appropriate
- Comments explain complex logic
- Functions are single-purpose
- Names are descriptive
- Documentation is complete

## Getting Help

- Check existing documentation and issues
- Review existing code for similar implementations
- Ask in the project's discussion forum
- Contact maintainers directly
- Consider seeking help from the ROS2 community

## Contributing Guidelines Summary

1. Always read the README and documentation first
2. Understand the codebase and existing implementations
3. Follow coding standards and best practices
4. Write tests for new functionality
5. Update documentation thoroughly
6. Submit well-structured, descriptive PRs
7. Be responsive to code reviews
8. Respect existing code style and architecture
9. Consider backwards compatibility
10. Test your changes thoroughly

For questions or contributions, please refer to the main project documentation or open an issue.