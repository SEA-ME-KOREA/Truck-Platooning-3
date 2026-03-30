# CARLA ROS2 플래투닝 다음 세션 체크리스트

기준 날짜: 2026-03-30

## 핵심 결론

- 현재 문제는 `truck2`만의 추종 문제가 아니라, 전체 트럭에 공통인 차선 인식/추적 불안정성일 가능성이 큼
- 속도를 올리면 불안정성이 더 심해졌음
- 다음 세션에서는 숫자 튜닝보다 `truck0` 차선 디버그 영상을 먼저 확인해야 함

## 현재 유지할 기준값

### Planning

- `cruise_speed = 3.0`
- `slow_speed = 2.2`
- `platoon_max_speed = 3.5`
- `steer_target_gain = 1.08`
- `steer_target_limit = 0.36`
- `steer_error_alpha = 0.90`
- `lane_conf_threshold = 0.1`
- `lane_conf_stop_threshold = 0.08`
- `steer_enable_conf_threshold = 0.12`

### Perception

- `lane_white_close_kernel_height = 17`
- `warped_gap_close_kernel_height = 55`
- `warped_gap_close_kernel_width = 7`
- `lane_preview_y_ratio = 0.35`
- `lane_far_preview_y_ratio = 0.18`
- `lane_far_preview_weight = 0.4`
- `warp_top_left_x_ratio = 0.28`
- `warp_top_right_x_ratio = 0.72`

### Control

- `steer_kp = 12.0`
- `steer_kd = 0.18`
- `steer_slowdown_start_deg = 1.2`
- `steer_slowdown_factor = 0.30`

## 다음 세션 시작 절차

1. CARLA를 완전히 새로 실행
2. 셸 A

```bash
exec bash
source ~/.bashrc
tx
ts
```

3. 셸 B

```bash
exec bash
source ~/.bashrc
pad
```

4. 셸 C

```bash
exec bash
source ~/.bashrc
pls
```

5. 바로 `truck0` 디버그 뷰어 실행

```bash
ros2f python3 /home/webebe/ros2_ws/tools/lane_debug_viewer.py --namespace truck0
```

## 가장 먼저 볼 화면

- `raw_image`
- `warp_image`
- `sliding_window_image`
- `lane_overlay_image`

## 확인 질문

`truck0`가 실제 차선을 잡고 있는가, 아니면 아래 중 하나를 차선으로 잘못 인식하는가?

- 도로 경계
- 그림자
- 잘못된 좌우 차선 쌍

## 관련 코드 위치

- Planning 기본값: `src/carla-virtual-platoon/nodes/planning/planning_node.py`
- Perception 기본값과 디버그 토픽: `src/carla-virtual-platoon/nodes/perception/perception_node.py`
- Control 기본값과 steer 기반 감속: `src/carla-virtual-platoon/nodes/control/control_node.py`
- 디버그 모자이크 뷰어: `tools/lane_debug_viewer.py`
