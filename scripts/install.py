#!/usr/bin/env python3
"""Complete installation and setup script for GS-ICP-SLAM ROS2 package"""

import sys
import os
import subprocess
import platform
import argparse
from pathlib import Path


class InstallationScript:
    """Comprehensive installation script for GS-ICP-SLAM ROS2"""
    
    def __init__(self, args=None):
        self.args = args or argparse.Namespace()
        self.errors = []
        self.successes = []
    
    def setup_environment(self):
        """Setup system environment"""
        print("=== Setting up System Environment ===")
        
        # Check Python version
        python_version = sys.version_info
        if python_version.major == 3 and python_version.minor >= 8:
            print(f"✓ Python {python_version.major}.{python_version.minor} detected")
            self.successes.append(f"Python version: {python_version.major}.{python_version.minor}")
        else:
            print(f"✗ Python version {python_version.major}.{python_version.minor} not supported")
            self.errors.append("Python 3.8+ required")
            return False
        
        # Check for conda
        if subprocess.run(["which", "conda"], capture_output=True).returncode == 0:
            print("✓ Anaconda/Miniconda detected")
        else:
            print("⚠ Anaconda/Miniconda not found (conda environment recommended)")
        
        return True
    
    def install_ros_dependencies(self):
        """Install ROS2 dependencies"""
        print("\n=== Installing ROS 2 Dependencies ===")
        
        # Check for ROS 2
        ros_setup = subprocess.run(
            ["test", "-f", "/opt/ros/humble/setup.bash"],
            capture_output=True,
            silent=True
        )
        
        if ros_setup.returncode == 0:
            print("✓ ROS 2 Humble environment found")
            self.successes.append("ROS 2 Humble detected")
            
            # Try to install ROS dependencies
            dependencies = [
                "ros-humble-desktop",
                "ros-humble-rviz2",
                "ros-humble-camera-info-manager",
                "ros-humble-compressed-image-transport",
                "ros-humble-image-transport",
                "ros-humble-cv-bridge",
                "python3-dev",
                "gcc",
                "g++",
                "cmake"
            ]
            
            for pkg in dependencies:
                print(f"Installing {pkg}...", end=" ")
                subprocess.run(
                    ["sudo", "apt-get", "install", "-y", pkg],
                    stderr=subprocess.DEVNULL
                )
                print("✓")
            
            self.successes.append("ROS 2 dependencies installed")
        else:
            print("⚠ ROS 2 environment not found in /opt/ros/")
            return False
        
        return True
    
    def setup_python_environment(self):
        """Setup Python environment"""
        print("\n=== Setting up Python Environment ===")
        
        try:
            # Install Python packages
            pip_packages = [
                "numpy>=1.19.0",
                "opencv-python>=4.5.0",
                "scipy>=1.8.0",
                "pyyaml>=5.4",
                "opencv-contrib-python>=4.5.0"
            ]
            
            for pkg in pip_packages:
                print(f"Installing {pkg}...", end=" ")
                subprocess.run(
                    ["pip", "install", "--upgrade", pkg],
                    stderr=subprocess.DEVNULL
                )
                print("✓")
            
            # Install PyTorch
            print("Installing PyTorch (CUDA 11.7)...")
            subprocess.run(
                ["pip", "install", "torch==1.13.1+cu117", "torchvision==0.14.1+cu117",
                 "torchaudio==0.13.1", "--extra-index-url", "https://download.pytorch.org/whl/cu117"],
                stderr=subprocess.DEVNULL
            )
            print("✓ PyTorch installed")
            
            self.successes.append("Python environment setup complete")
            return True
        except Exception as e:
            self.errors.append(f"Python environment setup failed: {e}")
            return False
    
    def install_package(self):
        """Install the GS-ICP-SLAM package"""
        print("\n=== Installing GS-ICP-SLAM ROS2 Package ===")
        
        try:
            # Activate ROS environment
            subprocess.run(
                ["source", "/opt/ros/humble/setup.bash"],
                shell=True,
                stderr=subprocess.DEVNULL
            )
            print("✓ ROS environment sourced")
            
            # Build the package
            print("Building package (this may take a few minutes)...")
            subprocess.run(
                ["colcon", "build", "--packages-select", "gs_icp_slam", "--symlink-install"],
                shell=True
            )
            print("✓ Package built")
            
            # Source the installation
            subprocess.run(
                ["source", "install/setup.bash"],
                shell=True,
                stderr=subprocess.DEVNULL
            )
            
            self.successes.append("GS-ICP-SLAM package installed")
            return True
        except Exception as e:
            self.errors.append(f"Package installation failed: {e}")
            return False
    
    def setup_pretrained_models(self):
        """Download and setup pretrained models"""
        print("\n=== Setting up Pretrained Models ===")
        
        pretrained_dir = Path("pretrained")
        pretrained_dir.mkdir(exist_ok=True)
        
        # Download YOLOv9 model
        yolo_path = pretrained_dir / "yolov9e-seg.pt"
        if not yolo_path.exists():
            print("Downloading YOLOv9 model...")
            subprocess.run(
                ["wget", "https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov9e-seg.pt",
                 "-P", str(pretrained_dir)]
            )
            print("✓ YOLOv9 model downloaded")
    
    def create_working_directories(self):
        """Create necessary working directories"""
        print("\n=== Creating Working Directories ===")
        
        directories = [
            "/tmp/gs_icp_slam_ros2/checkpoints",
            "/tmp/gs_icp_slam_ros2/results",
            "/tmp/gs_icp_slam_ros2/keyframes",
            "/tmp/gs_icp_slam_ros2/logs"
        ]
        
        for directory in directories:
            Path(directory).mkdir(parents=True, exist_ok=True)
            print(f"✓ {directory}/ created")
    
    def create_demo_script(self):
        """Create demonstration scripts"""
        print("\n=== Creating Demonstration Scripts ===")
        print("✓ Basic launch script available: gs_icp_slam_basic.launch.py")
        print("✓ Bag playback script available: gs_icp_slam_with_bag.launch.py")
        print("✓ Visualization script available: gs_icp_slam_visualization.launch.py")
    
    def run(self):
        """Run complete installation process"""
        print("=" * 60)
        print("GS-ICP-SLAM ROS2 Package Installation")
        print("=" * 60)
        
        # System check
        if not self.setup_environment():
            print("\n✗ Installation failed: System check failed")
            return False
        
        # Install ROS dependencies
        if not self.install_ros_dependencies():
            print("\n⚠ Installation continued with warnings")
        
        # Setup Python environment
        if not self.setup_python_environment():
            print("\n✗ Python setup failed")
            return False
        
        # Install package
        if not self.install_package():
            print("\n✗ Package installation failed")
            return False
        
        # Setup pretrained models
        if self.args.pretrained:
            self.setup_pretrained_models()
            self.create_working_directories()
        
        # Create demonstration scripts
        self.create_demo_script()
        
        # Print summary
        print("\n" + "=" * 60)
        print("Installation Summary")
        print("=" * 60)
        
        print("\n✓ Successes:")
        for success in self.successes:
            print(f"  • {success}")
        
        if self.errors:
            print("\n⚠ Warnings/Errors:")
            for error in self.errors:
                print(f"  • {error}")
        
        # Provide next steps
        print("\nNext Steps:")
        print("1. Source the environment:")
        print("   source install/setup.bash")
        print("\n2. Run SLAM with a bag file:")
        print("   ros2 run gs_icp_slam gs_icp_slam_node")
        print("\n3. Launch with visualization:")
        print("   ros2 launch gs_icp_slam gs_icp_slam_visualization.launch.py")
        print("\n4. See README.md for detailed usage instructions")
        
        return True


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description='Install and setup GS-ICP-SLAM ROS2 package')
    parser.add_argument('--pretrained', action='store_true', help='Download pretrained models')
    
    args = parser.parse_args()
    
    installer = InstallationScript(args)
    success = installer.run()
    
    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()