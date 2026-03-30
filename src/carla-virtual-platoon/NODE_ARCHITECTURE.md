# Node Architecture

## Current Runtime Path

The supported autonomy runtime is the structured stack:

- `perception`: `nodes/perception/perception_node.py`
- `planning`: `nodes/planning/planning_node.py`
- `control`: `nodes/control/control_node.py`
- `fleet`: `nodes/fleet/fleet_command_node.py`

The supported map/runtime target is:

- map: `Town04_Opt`
- bridge launch: `launch/carla-virtual-platoon.launch.py`
- autonomy launch: `launch/autonomy_fleet.launch.py`

## Topic Contract

Per truck namespace:

- inputs from bridge
  - `front_camera`
  - `front_lidar`
  - `velocity`

- perception outputs
  - `perception/lane_error`
  - `perception/lane_curvature`
  - `perception/lane_confidence`
  - `perception/stop_line_detected`
  - `perception/front_vehicle_distance`

- planning outputs
  - `planning/state`
  - `planning/target_speed`
  - `planning/target_steer_error`

- control inputs
  - `control/enabled`

- control outputs
  - `steer_control`
  - `throttle_control`
  - `brake_control`
  - `vehicle_cmd/steer`
  - `vehicle_cmd/throttle`
  - `vehicle_cmd/brake`

## Debug Topics

Per truck namespace:

- `perception/debug/raw_image`
- `perception/debug/threshold_image`
- `perception/debug/warp_image`
- `perception/debug/histogram_image`
- `perception/debug/sliding_window_image`
- `perception/debug/lane_overlay_image`
- `perception/debug/stop_line_image`

Bird's-eye style lane debug is primarily:

- `perception/debug/warp_image`
- `perception/debug/sliding_window_image`

## Launches

Bridge and spawn:

```bash
ros2 launch carla-virtual-platoon carla-virtual-platoon.launch.py NumTrucks:=3 Map:=Town04_Opt
```

Structured autonomy fleet:

```bash
ros2 launch carla-virtual-platoon autonomy_fleet.launch.py num_trucks:=3 map_name:=Town04_Opt enable_control:=true enable_fleet_command:=true
```

Fleet commands:

- `START`
- `STOP`

Command topic:

```text
/platoon/command
```

## Notes

- `RESET` is not currently part of the stable runtime path because the Python `carla` module is not available in this environment.
- Legacy nodes and launch paths remain in the repository for reference, but the current supported path is the structured stack above.
