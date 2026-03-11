cmake_minimum_required(VERSION 3.8)
project(carla-virtual-platoon)

# Default to C++14
if(NOT CMAKE_CXX_STANDARD)
set(CMAKE_CXX_STANDARD 14)
endif()

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
add_compile_options(-Wall -Wextra -Wl,-Bstatic -lrpc -lboost_filesystem -Wl,-Bdynamic -lpng -ltiff -ljpeg -lRecast -lDetour -lDetourCrowd)
endif()

if(NOT DEFINED ENV{CARLA_ROOT})
  message(FATAL_ERROR "Environment variable CARLA_ROOT is not set. Please set CARLA_ROOT to the root directory of your CARLA installation.")
endif()

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rclpy REQUIRED)
find_package(std_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(image_transport REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(Boost REQUIRED)
find_package(TIFF REQUIRED)

set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTOUIC ON)
set(CMAKE_AUTORCC ON)

set(ament_dependencies
  rclcpp
  std_msgs
  geometry_msgs
  sensor_msgs
  cv_bridge
  image_transport
  Boost
)

##############################################################################################
include(dependency.cmake)

add_executable(main
    src/main.cpp
    nodes/FrontCamera.cpp
    nodes/FrontRadar.cpp
    nodes/FrontLidar.cpp
    nodes/TruckControl.cpp
    nodes/TruckStatus.cpp
)

##############################################################################################
ament_target_dependencies(main ${ament_dependencies})

target_include_directories(main PUBLIC
  ${libCarla_INCLUDE_DIR}
  ${DEP_INCLUDE_DIR}
)
target_include_directories(main PUBLIC 
  ${CMAKE_SOURCE_DIR}/include
)

target_link_libraries(main 
  ${libCarla_LIBS} 
  ${Boost_LIBRARIES} 
  ${TIFF_LIBRARIES} 
  jpeg png
)
##############################################################################################

install(TARGETS
  main
  DESTINATION lib/${PROJECT_NAME}
)

install(PROGRAMS
  nodes/perception/perception_node.py
  nodes/perception/autonomy_compact_node.py
  nodes/perception/lane_detection_node.py
  nodes/perception/lane_follower_legacy.py
  nodes/perception/yolo_detection_node.py
  nodes/perception/lidar_obstacle_node.py
  nodes/perception/stop_line_detection_node.py
  nodes/planning/planning_node.py
  nodes/planning/behavior_fsm_node.py
  nodes/control/control_node.py
  nodes/control/controller_node.py
  nodes/fleet/fleet_command_node.py
  DESTINATION lib/${PROJECT_NAME}
)

install(FILES
  nodes/perception/lane_detector.py
  nodes/perception/stop_line_detector.py
  nodes/perception/front_vehicle_detector.py
  nodes/perception/debug_visualizer.py
  nodes/planning/behavior_fsm.py
  nodes/planning/local_planner.py
  nodes/planning/platoon_planner.py
  nodes/control/pid_controller.py
  nodes/control/lateral_controller.py
  nodes/control/longitudinal_controller.py
  DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY
  launch
  DESTINATION share/${PROJECT_NAME}
)

install(DIRECTORY
  config
  DESTINATION share/${PROJECT_NAME}
)

ament_package()
