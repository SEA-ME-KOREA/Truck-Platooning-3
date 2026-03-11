import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def _include(name: str, **kwargs):
    share = get_package_share_directory("carla-virtual-platoon")
    path = os.path.join(share, "launch", name)
    return IncludeLaunchDescription(
        PythonLaunchDescriptionSource(path),
        launch_arguments=kwargs.items(),
    )


def generate_launch_description():
    namespace = LaunchConfiguration("namespace")
    yolo_enabled = LaunchConfiguration("yolo_enabled")
    yolo_model = LaunchConfiguration("yolo_model")
    enable_control = LaunchConfiguration("enable_control")
    enable_fleet_command = LaunchConfiguration("enable_fleet_command")
    num_trucks = LaunchConfiguration("num_trucks")
    map_name = LaunchConfiguration("map_name")
    debug_visualization = LaunchConfiguration("debug_visualization")
    publish_debug_topics = LaunchConfiguration("publish_debug_topics")
    show_opencv_windows = LaunchConfiguration("show_opencv_windows")
    sliding_window_visualization = LaunchConfiguration("sliding_window_visualization")

    return LaunchDescription(
        [
            DeclareLaunchArgument("namespace", default_value="truck0"),
            DeclareLaunchArgument("yolo_enabled", default_value="false"),
            DeclareLaunchArgument("yolo_model", default_value="yolov8n.pt"),
            DeclareLaunchArgument("debug_visualization", default_value="false"),
            DeclareLaunchArgument("publish_debug_topics", default_value="true"),
            DeclareLaunchArgument("show_opencv_windows", default_value="false"),
            DeclareLaunchArgument("sliding_window_visualization", default_value="true"),
            DeclareLaunchArgument("enable_control", default_value="true"),
            DeclareLaunchArgument("enable_fleet_command", default_value="false"),
            DeclareLaunchArgument("num_trucks", default_value="3"),
            DeclareLaunchArgument("map_name", default_value="Town04_Opt"),
            _include(
                "perception_stack.launch.py",
                namespace=namespace,
                yolo_enabled=yolo_enabled,
                yolo_model=yolo_model,
                debug_visualization=debug_visualization,
                publish_debug_topics=publish_debug_topics,
                show_opencv_windows=show_opencv_windows,
                sliding_window_visualization=sliding_window_visualization,
            ),
            _include("planning_stack.launch.py", namespace=namespace),
            _include(
                "control_stack.launch.py",
                namespace=namespace,
                enable_control=enable_control,
            ),
            Node(
                package="carla-virtual-platoon",
                executable="fleet_command_node.py",
                name="fleet_command",
                output="screen",
                condition=IfCondition(enable_fleet_command),
                parameters=[
                    {
                        "num_trucks": num_trucks,
                        "map_name": map_name,
                        "command_topic": "/platoon/command",
                    }
                ],
            ),
        ]
    )
