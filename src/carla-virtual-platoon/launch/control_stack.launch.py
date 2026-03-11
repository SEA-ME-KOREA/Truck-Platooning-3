from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    namespace = LaunchConfiguration("namespace")
    enable_control = LaunchConfiguration("enable_control")

    return LaunchDescription(
        [
            DeclareLaunchArgument("namespace", default_value="truck0"),
            DeclareLaunchArgument("enable_control", default_value="true"),
            Node(
                package="carla-virtual-platoon",
                executable="control_node.py",
                name="control",
                namespace=namespace,
                output="screen",
                condition=IfCondition(enable_control),
            ),
        ]
    )
