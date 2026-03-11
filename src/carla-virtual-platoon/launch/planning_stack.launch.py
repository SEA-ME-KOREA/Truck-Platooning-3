from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    namespace = LaunchConfiguration("namespace")

    return LaunchDescription(
        [
            DeclareLaunchArgument("namespace", default_value="truck0"),
            Node(
                package="carla-virtual-platoon",
                executable="planning_node.py",
                name="planning",
                namespace=namespace,
                output="screen",
            ),
        ]
    )
