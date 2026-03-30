#!/usr/bin/env python3

import tkinter as tk

import rclpy
from rclpy.node import Node
from std_msgs.msg import Bool, Float32, String


def get_stipple(ratio: float) -> str:
    if ratio >= 0.75:
        return "gray75"
    if ratio >= 0.5:
        return "gray50"
    if ratio >= 0.25:
        return "gray25"
    return "gray12"


class PlatoonDashboard(Node):
    def __init__(self) -> None:
        super().__init__("platoon_dashboard")
        self._state = {
            i: {
                "velocity": 0.0,
                "distance": 0.0,
                "distance_valid": False,
                "target_speed": 0.0,
                "enabled": False,
                "planning_state": "UNKNOWN",
            }
            for i in range(3)
        }

        for i in range(3):
            ns = f"/truck{i}"
            self.create_subscription(Float32, f"{ns}/velocity", lambda m, i=i: self._set(i, "velocity", m.data), 10)
            self.create_subscription(
                Float32,
                f"{ns}/perception/front_vehicle_distance",
                lambda m, i=i: self._set_distance(i, m.data),
                10,
            )
            self.create_subscription(
                Float32, f"{ns}/planning/target_speed", lambda m, i=i: self._set(i, "target_speed", m.data), 10
            )
            self.create_subscription(
                Bool, f"{ns}/control/enabled", lambda m, i=i: self._set(i, "enabled", bool(m.data)), 10
            )
            self.create_subscription(
                String, f"{ns}/planning/state", lambda m, i=i: self._set(i, "planning_state", m.data or "UNKNOWN"), 10
            )

    def _set(self, truck_id: int, key: str, value) -> None:
        self._state[truck_id][key] = value

    def _set_distance(self, truck_id: int, value: float) -> None:
        self._state[truck_id]["distance"] = float(value)
        self._state[truck_id]["distance_valid"] = float(value) < 999.0


class TkinterDashboard:
    def __init__(self, ros_node: PlatoonDashboard):
        self.ros_node = ros_node
        self.root = tk.Tk()
        self.root.title("Platoon Dashboard")
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

        self.speed_bars = {}
        self.distance_bars = {}
        self.speed_labels = {}
        self.distance_labels = {}
        self.target_speed_labels = {}
        self.state_labels = {}
        self.enabled_labels = {}

        self.max_speed_kmh = 60.0
        self.max_distance = 50.0

        for i in range(3):
            frame = tk.Frame(self.root, padx=10, pady=8)
            frame.pack()

            tk.Label(frame, text=f"Truck {i}", font=("Arial", 16)).grid(
                row=0, column=0, columnspan=2, sticky="w"
            )
            self.state_labels[i] = tk.Label(frame, text="State: UNKNOWN", font=("Arial", 12))
            self.state_labels[i].grid(row=1, column=0, sticky="w")
            self.enabled_labels[i] = tk.Label(frame, text="Enabled: False", font=("Arial", 12))
            self.enabled_labels[i].grid(row=1, column=1, sticky="w")

            self.speed_labels[i] = tk.Label(frame, text="Speed: N/A", font=("Arial", 12))
            self.speed_labels[i].grid(row=2, column=0, sticky="w")
            self.speed_bars[i] = tk.Canvas(frame, width=140, height=20, bg="white")
            self.speed_bars[i].grid(row=2, column=1, padx=10)

            self.target_speed_labels[i] = tk.Label(frame, text="Target: N/A", font=("Arial", 12))
            self.target_speed_labels[i].grid(row=3, column=0, sticky="w")

            self.distance_labels[i] = tk.Label(frame, text="Distance: N/A", font=("Arial", 12))
            self.distance_labels[i].grid(row=4, column=0, sticky="w")
            self.distance_bars[i] = tk.Canvas(frame, width=140, height=20, bg="white")
            self.distance_bars[i].grid(row=4, column=1, padx=10)

    def update_ui(self):
        for i in range(3):
            data = self.ros_node._state[i]
            speed_kmh = float(data["velocity"]) * 3.6
            target_kmh = float(data["target_speed"]) * 3.6
            dist = float(data["distance"])
            dist_valid = bool(data["distance_valid"])

            self.state_labels[i].config(text=f"State: {data['planning_state']}")
            self.enabled_labels[i].config(text=f"Enabled: {data['enabled']}")
            self.speed_labels[i].config(text=f"Speed: {speed_kmh:.1f} km/h")
            self.target_speed_labels[i].config(text=f"Target: {target_kmh:.1f} km/h")
            self.distance_labels[i].config(
                text=f"Distance: {dist:.1f} m" if dist_valid else "Distance: N/A"
            )

            speed_ratio = min(max(speed_kmh / self.max_speed_kmh, 0.0), 1.0)
            speed_canvas = self.speed_bars[i]
            speed_canvas.delete("all")
            speed_canvas.create_rectangle(
                0, 0, 140 * speed_ratio, 20, fill="red", stipple=get_stipple(speed_ratio)
            )

            distance_ratio = min(max(dist / self.max_distance, 0.0), 1.0) if dist_valid else 0.0
            distance_canvas = self.distance_bars[i]
            distance_canvas.delete("all")
            distance_canvas.create_rectangle(
                0, 0, 140 * distance_ratio, 20, fill="blue", stipple=get_stipple(distance_ratio)
            )

        self.root.after(100, self.update_ui)

    def on_closing(self):
        self.root.destroy()

    def run(self):
        self.update_ui()
        self.root.mainloop()


def main() -> None:
    rclpy.init()
    node = PlatoonDashboard()

    def spin_once():
        if rclpy.ok():
            rclpy.spin_once(node, timeout_sec=0.01)
            dashboard.root.after(10, spin_once)

    dashboard = TkinterDashboard(node)
    dashboard.root.after(10, spin_once)
    try:
        dashboard.run()
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
