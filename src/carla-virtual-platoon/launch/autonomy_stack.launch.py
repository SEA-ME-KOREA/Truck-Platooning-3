import launch
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.conditions import IfCondition, UnlessCondition
from launch.substitutions import LaunchConfiguration, PythonExpression
from launch_ros.actions import Node


def _generate_nodes(context):
    namespace = LaunchConfiguration("namespace").perform(context)
    yolo_enabled = LaunchConfiguration("yolo_enabled").perform(context).lower() == "true"
    model_path = LaunchConfiguration("yolo_model").perform(context)
    enable_fleet_command = (
        LaunchConfiguration("enable_fleet_command").perform(context).lower() == "true"
    )
    num_trucks = int(LaunchConfiguration("num_trucks").perform(context))
    map_name = LaunchConfiguration("map_name").perform(context)
    debug_visualization = (
        LaunchConfiguration("debug_visualization").perform(context).lower() == "true"
    )
    publish_debug_topics = (
        LaunchConfiguration("publish_debug_topics").perform(context).lower() == "true"
    )
    show_opencv_windows = (
        LaunchConfiguration("show_opencv_windows").perform(context).lower() == "true"
    )
    sliding_window_visualization = (
        LaunchConfiguration("sliding_window_visualization").perform(context).lower() == "true"
    )

    common_ns = namespace.strip("/")

    nodes = [
        Node(
            package="carla-virtual-platoon",
            executable="perception_node.py",
            name="perception",
            namespace=common_ns,
            output="screen",
            parameters=[
                {
                    "use_debug_visualization": debug_visualization,
                    "publish_debug_topics": publish_debug_topics,
                    "show_opencv_windows": show_opencv_windows,
                    "use_sliding_window_visualization": sliding_window_visualization,
                }
            ],
            condition=UnlessCondition(LaunchConfiguration("compact_mode")),
        ),
        Node(
            package="carla-virtual-platoon",
            executable="planning_node.py",
            name="planning",
            namespace=common_ns,
            output="screen",
            condition=UnlessCondition(LaunchConfiguration("compact_mode")),
        ),
        Node(
            package="carla-virtual-platoon",
            executable="control_node.py",
            name="control",
            namespace=common_ns,
            output="screen",
            condition=IfCondition(
                PythonExpression(
                    [
                        "not ",
                        LaunchConfiguration("compact_mode"),
                        " and ",
                        LaunchConfiguration("enable_control"),
                    ]
                )
            ),
        ),
        Node(
            package="carla-virtual-platoon",
            executable="autonomy_compact_node.py",
            name="autonomy_compact",
            namespace=common_ns,
            output="screen",
            condition=IfCondition(LaunchConfiguration("compact_mode")),
        ),
        Node(
            package="carla-virtual-platoon",
            executable="yolo_detection_node.py",
            name="yolo_detection",
            namespace=common_ns,
            output="screen",
            parameters=[{"enabled": yolo_enabled, "model_path": model_path}],
        ),
        Node(
            package="carla-virtual-platoon",
            executable="fleet_command_node.py",
            name="fleet_command",
            output="screen",
            condition=IfCondition(LaunchConfiguration("enable_fleet_command")),
            parameters=[
                {"num_trucks": num_trucks, "map_name": map_name, "command_topic": "/platoon/command"}
            ],
        ),
    ]

    return nodes


def generate_launch_description():
    return LaunchDescription(
        [
            DeclareLaunchArgument("namespace", default_value="truck0"),
            DeclareLaunchArgument("num_trucks", default_value="3"),
            DeclareLaunchArgument("map_name", default_value="Town04_Opt"),
            DeclareLaunchArgument("compact_mode", default_value="true"),
            DeclareLaunchArgument("yolo_enabled", default_value="false"),
            DeclareLaunchArgument("enable_control", default_value="false"),
            DeclareLaunchArgument("enable_fleet_command", default_value="false"),
            DeclareLaunchArgument("debug_visualization", default_value="false"),
            DeclareLaunchArgument("publish_debug_topics", default_value="true"),
            DeclareLaunchArgument("show_opencv_windows", default_value="false"),
            DeclareLaunchArgument("sliding_window_visualization", default_value="true"),
            DeclareLaunchArgument("yolo_model", default_value="yolov8n.pt"),
            OpaqueFunction(function=_generate_nodes),
        ]
    )
