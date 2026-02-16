cmake_minimum_required(VERSION 3.8)
project(gs_icp_slam)

# Default to C++14
if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 14)
endif()

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(cv_bridge REQUIRED)

# Python source files configuration
ament_python_install_package(gs_icp_slam)

install(PROGRAMS
  gs_icp_slam/node.py
  DESTINATION lib/${PROJECT_NAME}
  RENAME gs_icp_slam_node
)

# Install launch files
install(DIRECTORY
  launch
  DESTINATION share/${PROJECT_NAME}
)

# Install configuration files
install(DIRECTORY
  config
  DESTINATION share/${PROJECT_NAME}
)

# Install scripts
install(DIRECTORY
  scripts
  DESTINATION share/${PROJECT_NAME}
)

# Install documentation
install(DIRECTORY
  docs
  DESTINATION share/${PROJECT_NAME}
)

# Install tests
install(DIRECTORY
  tests
  DESTINATION share/${PROJECT_NAME}
)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  ament_lint_auto_find_test_dependencies()
endif()

if(ANDROID)
  find_package(rosidl_default_generators REQUIRED)
  rosidl_generate_interfaces(${PROJECT_NAME}
    "msg/slam_parameters.msg"
  )
  ament_target_dependencies(4dgs_slam_ros2_cpp_rosidl_interface
    rclcpp
    sensor_msgs
    geometry_msgs
    std_msgs
  )
  ament_export_dependencies(rosidl_default_runtime)
  ament_export_dependencies(4dgs_slam_ros2_cpp_rosidl_interface)
endif()

# Export ROS2 configuration
ament_package()