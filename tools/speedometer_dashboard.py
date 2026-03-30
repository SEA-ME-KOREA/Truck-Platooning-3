#!/usr/bin/env python3

import signal
import tkinter as tk

import matplotlib

matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import numpy as np
import rclpy
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from rclpy.node import Node
from std_msgs.msg import Float32


class TruckSpeedSubscriber(Node):
    def __init__(self, ui):
        super().__init__("truck_speed_subscriber")
        self.ui = ui
        self.truck_speeds = {0: 0.0, 1: 0.0, 2: 0.0}

        self.create_subscription(Float32, "/truck0/velocity", lambda m: self._set_speed(0, m), 10)
        self.create_subscription(Float32, "/truck1/velocity", lambda m: self._set_speed(1, m), 10)
        self.create_subscription(Float32, "/truck2/velocity", lambda m: self._set_speed(2, m), 10)
        self.create_timer(0.1, self._update_ui)

    def _set_speed(self, truck_id: int, msg: Float32) -> None:
        self.truck_speeds[truck_id] = float(msg.data) * 3.6

    def _update_ui(self) -> None:
        self.ui.update_speed_gauges(self.truck_speeds)


class SpeedometerUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Truck Speedometer")
        self.max_speed = 80.0
        self.needle_length = 50.0

        self.fig, self.axs = plt.subplots(
            1, 3, subplot_kw={"projection": "polar"}, figsize=(10, 5)
        )
        self.fig.subplots_adjust(wspace=0.4)

        self.speed_lines = []
        for idx, ax in enumerate(self.axs):
            ax.set_ylim(0, self.needle_length)
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            ax.set_title(f"Truck {idx}", fontsize=17, fontweight="bold", color="navy")
            ax.set_rticks([])
            ax.set_theta_direction(-1)
            ax.set_theta_offset(-3 * np.pi / 4)
            ax.grid(False)

            speed_line, = ax.plot([0, 0], [0, self.needle_length], color="r", linewidth=4)
            self.speed_lines.append(speed_line)

            tick_length_small = 5
            tick_length_big = 10
            small_ticks = np.arange(0, self.max_speed + 1, 5)
            big_ticks = np.arange(0, self.max_speed + 1, 10)
            small_angles = np.deg2rad((small_ticks / self.max_speed) * 270)

            for tick, angle in zip(small_ticks, small_angles):
                inner_radius = (
                    self.needle_length - tick_length_big
                    if tick in big_ticks
                    else self.needle_length - tick_length_small
                )
                ax.plot([angle, angle], [inner_radius, self.needle_length], color="darkblue", lw=2)

        self.canvas = FigureCanvasTkAgg(self.fig, master=self.root)
        self.canvas.get_tk_widget().pack()

        self.label_frame = tk.Frame(self.root)
        self.label_frame.pack(pady=10)
        self.speed_labels = []
        for i in range(3):
            label = tk.Label(
                self.label_frame, text=f"Truck {i}: 0.0 km/h", font=("Helvetica", 10)
            )
            label.pack(side=tk.LEFT, padx=20)
            self.speed_labels.append(label)

    def update_speed_gauges(self, truck_speeds):
        for i in range(3):
            speed = float(truck_speeds.get(i, 0.0))
            angle = np.deg2rad((speed / self.max_speed) * 180)
            self.speed_lines[i].set_xdata([0, angle])
            self.speed_lines[i].set_ydata([0, self.needle_length])
            self.speed_labels[i].config(text=f"Truck {i}: {speed:.1f} km/h")
        self.canvas.draw()


def main():
    rclpy.init()
    root = tk.Tk()
    ui = SpeedometerUI(root)
    node = TruckSpeedSubscriber(ui)

    def on_closing():
        try:
            node.destroy_node()
            rclpy.shutdown()
        finally:
            root.destroy()

    def signal_handler(sig, frame):
        on_closing()

    root.protocol("WM_DELETE_WINDOW", on_closing)
    signal.signal(signal.SIGINT, signal_handler)

    def ros_spin():
        if rclpy.ok():
            rclpy.spin_once(node, timeout_sec=0.1)
            root.after(1, ros_spin)

    root.after(1, ros_spin)
    root.mainloop()


if __name__ == "__main__":
    main()
