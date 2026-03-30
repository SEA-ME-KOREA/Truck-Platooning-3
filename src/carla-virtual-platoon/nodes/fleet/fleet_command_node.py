#!/usr/bin/env python3
from typing import Dict, Tuple

import rclpy
from rclpy.node import Node
from std_msgs.msg import Bool, String

try:
    import carla  # type: ignore
except Exception:  # pragma: no cover
    carla = None


TRUCK_LOCATIONS = {
    "Town04_Opt": {
        0: (-270.990, 30.0, 2.0, 0.0),
        1: (-300.990, 30.0, 2.0, 0.22),
        2: (-330.990, 30.0, 2.0, 0.22),
    },
    "IHP": {
        0: (-1139.0, 4.5, 1.0, 0.0),
        1: (-1169.0, 4.5, 1.0, 0.0),
        2: (-1199.0, 4.5, 1.0, 0.0),
    },
    "k-track": {
        0: (-75.6, 50.75, 2.0, 0.0),
        1: (-95.6, 50.75, 2.0, 0.0),
        2: (-115.6, 50.75, 2.0, 0.0),
    },
}


class FleetCommandNode(Node):
    def __init__(self) -> None:
        super().__init__("fleet_command_node")
        self.declare_parameter("num_trucks", 3)
        self.declare_parameter("map_name", "Town04_Opt")
        self.declare_parameter("carla_host", "localhost")
        self.declare_parameter("carla_port", 2000)
        self.declare_parameter("command_topic", "/platoon/command")

        self._enable_pubs = {}
        for i in range(int(self.get_parameter("num_trucks").value)):
            topic = f"/truck{i}/control/enabled"
            self._enable_pubs[i] = self.create_publisher(Bool, topic, 10)
        self._fleet_enabled = False

        self.create_subscription(
            String, self.get_parameter("command_topic").value, self._on_command, 10
        )
        self.create_timer(0.5, self._publish_current_enable_state)
        self.get_logger().info(
            f"fleet_command_node listening on {self.get_parameter('command_topic').value}"
        )

    def _publish_enable_all(self, enabled: bool) -> None:
        self._fleet_enabled = enabled
        msg = Bool()
        msg.data = enabled
        for pub in self._enable_pubs.values():
            pub.publish(msg)

    def _publish_current_enable_state(self) -> None:
        self._publish_enable_all(self._fleet_enabled)

    def _on_command(self, msg: String) -> None:
        cmd = (msg.data or "").strip().upper()
        if cmd in ("START", "RUN", "GO"):
            self._publish_enable_all(True)
            self.get_logger().info("Fleet control enabled")
            return
        if cmd in ("STOP", "PAUSE", "HALT"):
            self._publish_enable_all(False)
            self.get_logger().info("Fleet control disabled")
            return
        if cmd == "RESET":
            self._publish_enable_all(False)
            ok = self._reset_positions()
            self.get_logger().info(f"Fleet reset {'succeeded' if ok else 'failed'}")
            return

        self.get_logger().warn("Unknown command. Use START / STOP / RESET")

    def _reset_positions(self) -> bool:
        if carla is None:
            self.get_logger().warn("Python carla module not available; cannot reset positions.")
            return False

        map_name = str(self.get_parameter("map_name").value)
        positions = TRUCK_LOCATIONS.get(map_name)
        if not positions:
            self.get_logger().warn(f"No reset positions configured for map '{map_name}'")
            return False

        host = str(self.get_parameter("carla_host").value)
        port = int(self.get_parameter("carla_port").value)

        try:
            client = carla.Client(host, port)
            client.set_timeout(5.0)
            world = client.get_world()
            actors = world.get_actors()

            moved_any = False
            for idx, (x, y, z, yaw) in positions.items():
                if idx not in self._enable_pubs:
                    continue
                truck_role = f"truck{idx}"
                trailer_role = f"trailer{idx}"

                truck = self._find_actor_by_role(actors, truck_role)
                trailer = self._find_actor_by_role(actors, trailer_role)
                if truck is None:
                    self.get_logger().warn(f"Actor '{truck_role}' not found")
                    continue

                truck_tf = carla.Transform(
                    carla.Location(x=x + 5.2, y=y, z=z),
                    carla.Rotation(pitch=0.0, yaw=yaw, roll=0.0),
                )
                truck.set_target_velocity(carla.Vector3D(0.0, 0.0, 0.0))
                truck.set_target_angular_velocity(carla.Vector3D(0.0, 0.0, 0.0))
                truck.set_transform(truck_tf)
                moved_any = True

                if trailer is not None:
                    trailer_tf = carla.Transform(
                        carla.Location(x=x, y=y, z=z),
                        carla.Rotation(pitch=0.0, yaw=yaw, roll=0.0),
                    )
                    trailer.set_target_velocity(carla.Vector3D(0.0, 0.0, 0.0))
                    trailer.set_target_angular_velocity(carla.Vector3D(0.0, 0.0, 0.0))
                    trailer.set_transform(trailer_tf)

            return moved_any
        except Exception as exc:
            self.get_logger().warn(f"Reset failed: {exc}")
            return False

    @staticmethod
    def _find_actor_by_role(actors, role_name: str):
        for actor in actors:
            try:
                if actor.attributes.get("role_name") == role_name:
                    return actor
            except Exception:
                continue
        return None


def main() -> None:
    rclpy.init()
    node = FleetCommandNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
