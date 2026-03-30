import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, OpaqueFunction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def _generate_autonomy_launches(context):
    share_dir = get_package_share_directory("carla-virtual-platoon")
    autonomy_launch = os.path.join(share_dir, "launch", "autonomy_structured.launch.py")

    num_trucks = int(LaunchConfiguration("num_trucks").perform(context))
    enable_fleet_command = (
        LaunchConfiguration("enable_fleet_command").perform(context).lower() == "true"
    )

    common_arguments = {
        "debug_visualization": LaunchConfiguration("debug_visualization"),
        "publish_debug_topics": LaunchConfiguration("publish_debug_topics"),
        "show_opencv_windows": LaunchConfiguration("show_opencv_windows"),
        "sliding_window_visualization": LaunchConfiguration("sliding_window_visualization"),
        "enable_control": LaunchConfiguration("enable_control"),
        # Fleet command must be created only once below.
        "enable_fleet_command": "false",
        "num_trucks": LaunchConfiguration("num_trucks"),
        "map_name": LaunchConfiguration("map_name"),
    }

    launches = [
        Node(
            executable="/home/webebe/ros2_ws/src/carla-virtual-platoon/nodes/fleet/carla_pose_publisher.py",
            name="carla_pose_publisher",
            output="screen",
        )
    ]
    for truck_idx in range(num_trucks):
        launch_arguments = dict(common_arguments)
        launch_arguments["namespace"] = f"truck{truck_idx}"
        launches.append(
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource(autonomy_launch),
                launch_arguments=launch_arguments.items(),
            )
        )

    if enable_fleet_command:
        launches.append(
            Node(
                package="carla-virtual-platoon",
                executable="fleet_command_node.py",
                name="fleet_command",
                output="screen",
                parameters=[
                    {
                        "num_trucks": LaunchConfiguration("num_trucks"),
                        "map_name": LaunchConfiguration("map_name"),
                        "command_topic": "/platoon/command",
                    }
                ],
            )
        )

    return launches


def generate_launch_description():
    return LaunchDescription(
        [
            DeclareLaunchArgument("num_trucks", default_value="3"),
            DeclareLaunchArgument("map_name", default_value="Town04_Opt"),
            DeclareLaunchArgument("enable_control", default_value="true"),
            DeclareLaunchArgument("enable_fleet_command", default_value="true"),
            DeclareLaunchArgument("debug_visualization", default_value="false"),
            DeclareLaunchArgument("publish_debug_topics", default_value="true"),
            DeclareLaunchArgument("show_opencv_windows", default_value="false"),
            DeclareLaunchArgument("sliding_window_visualization", default_value="true"),
            OpaqueFunction(function=_generate_autonomy_launches),
        ]
    )
