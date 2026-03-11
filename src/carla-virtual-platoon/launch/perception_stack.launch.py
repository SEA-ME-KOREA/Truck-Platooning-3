from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    namespace = LaunchConfiguration("namespace")
    yolo_enabled = LaunchConfiguration("yolo_enabled")
    yolo_model = LaunchConfiguration("yolo_model")
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
            Node(
                package="carla-virtual-platoon",
                executable="perception_node.py",
                name="perception",
                namespace=namespace,
                output="screen",
                parameters=[
                    {
                        "use_debug_visualization": debug_visualization,
                        "publish_debug_topics": publish_debug_topics,
                        "show_opencv_windows": show_opencv_windows,
                        "use_sliding_window_visualization": sliding_window_visualization,
                    }
                ],
            ),
            Node(
                package="carla-virtual-platoon",
                executable="yolo_detection_node.py",
                name="yolo_detection",
                namespace=namespace,
                output="screen",
                condition=IfCondition(yolo_enabled),
                parameters=[{"enabled": True, "model_path": yolo_model}],
            ),
        ]
    )
