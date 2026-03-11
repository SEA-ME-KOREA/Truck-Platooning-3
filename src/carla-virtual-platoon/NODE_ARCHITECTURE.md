# Node Architecture (Perception / Planning / Control)

## Source Tree
- `nodes/perception/`
  - `autonomy_compact_node.py`
  - `lane_detection_node.py`
  - `lane_follower_legacy.py`
  - `lidar_obstacle_node.py`
  - `stop_line_detection_node.py`
  - `yolo_detection_node.py`
- `nodes/planning/`
  - `behavior_fsm_node.py`
- `nodes/control/`
  - `controller_node.py`
- `nodes/fleet/`
  - `fleet_command_node.py`

## 1) Perception
- `lane_detection_node.py`
  - Subscribe: `front_camera`
  - Publish: `perception/lane_center_offset`, `perception/lane_confidence`
- `lidar_obstacle_node.py`
  - Subscribe: `front_lidar`
  - Publish: `perception/front_obstacle_distance`
- `stop_line_detection_node.py`
  - Subscribe: `front_camera`
  - Publish: `perception/stop_line_detected`
- `yolo_detection_node.py` (optional)
  - Subscribe: `front_camera`
  - Publish: `perception/detections_json`, `perception/traffic_light_state`

Launch:
```bash
ros2 launch carla-virtual-platoon perception_stack.launch.py namespace:=truck0 yolo_enabled:=false
```

## 2) Planning
- `behavior_fsm_node.py`
  - Subscribe:
    - `perception/lane_confidence`
    - `perception/front_obstacle_distance`
    - `perception/traffic_light_state`
    - `perception/stop_line_detected`
  - Publish:
    - `control/target_speed`
    - `control/behavior_mode`

Launch:
```bash
ros2 launch carla-virtual-platoon planning_stack.launch.py namespace:=truck0
```

## 3) Control
- `controller_node.py`
  - Subscribe:
    - `perception/lane_center_offset`
    - `perception/lane_confidence`
    - `perception/front_obstacle_distance`
    - `control/target_speed`
    - `control/behavior_mode`
    - `control/enabled`
    - `velocity`
  - Publish:
    - `steer_control`
    - `throttle_control`

Launch:
```bash
ros2 launch carla-virtual-platoon control_stack.launch.py namespace:=truck0 enable_control:=true
```

## Integrated launch (structured)

Single command to run Perception + Planning + Control:
```bash
ros2 launch carla-virtual-platoon autonomy_structured.launch.py namespace:=truck0 enable_control:=true
```

Optional fleet command node:
```bash
ros2 launch carla-virtual-platoon autonomy_structured.launch.py namespace:=truck0 enable_control:=true enable_fleet_command:=true num_trucks:=3
```
