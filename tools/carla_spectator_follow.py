#!/usr/bin/env python3

import math

import carla
import rclpy


class CarlaSpectatorFollower:
    def __init__(self):
        """CARLA 클라이언트 연결"""
        self.client = carla.Client("localhost", 2000)
        self.client.set_timeout(10.0)
        self.world = self.client.get_world()
        self.target_vehicle = None
        self.previous_location = None
        self.previous_rotation = None

    def find_truck1_vehicle(self):
        """같은 type_id 차량 중 role_name이 'truck1'인 차량 찾기"""
        target_role = "truck1"
        actors = self.world.get_actors()
        vehicles = [actor for actor in actors if actor.type_id.startswith("vehicle.daf")]
        truck1_vehicles = [v for v in vehicles if v.attributes.get("role_name") == target_role]

        if truck1_vehicles:
            self.target_vehicle = truck1_vehicles[0]
            print(
                f"\n '{target_role}' 찾음: ID: {self.target_vehicle.id}, 모델: {self.target_vehicle.type_id}"
            )
        else:
            print(f"'{target_role}' 역할 차량을 찾지 못했습니다.")

    def lerp(self, start, end, alpha):
        """선형 보간 (Lerp)"""
        return start + (end - start) * alpha

    def lerp_angle(self, start, end, alpha):
        """회전 각도(Yaw) 선형 보간 (360도 불연속성 문제 해결)"""
        diff = ((end - start + 180) % 360) - 180
        return (start + diff * alpha) % 360

    def get_relative_location(self, vehicle_transform, dx, dy, dz):
        """상대좌표를 월드 좌표로 변환"""
        yaw = math.radians(vehicle_transform.rotation.yaw)
        x = vehicle_transform.location.x + dx * math.cos(yaw) - dy * math.sin(yaw)
        y = vehicle_transform.location.y + dx * math.sin(yaw) + dy * math.cos(yaw)
        z = vehicle_transform.location.z + dz
        return carla.Location(x=x, y=y, z=z)

    def follow_vehicle(self):
        """Spectator 시점을 truck1 차량에 부드럽게 고정 (동기화)"""
        spectator = self.world.get_spectator()

        try:
            print("Spectator 시점을 'truck1' 차량에 부드럽게 고정 중...")

            alpha_position = 0.2
            alpha_rotation = 0.03

            while True:
                self.world.wait_for_tick()
                if not self.target_vehicle:
                    self.find_truck1_vehicle()
                    if not self.target_vehicle:
                        continue
                    self.previous_location = self.target_vehicle.get_transform().location
                    self.previous_rotation = self.target_vehicle.get_transform().rotation

                vehicle_transform = self.target_vehicle.get_transform()

                target_location = self.get_relative_location(
                    vehicle_transform, dx=0, dy=0, dz=80
                )

                smooth_x = self.lerp(
                    self.previous_location.x, target_location.x, alpha_position
                )
                smooth_y = self.lerp(
                    self.previous_location.y, target_location.y, alpha_position
                )
                smooth_z = self.lerp(
                    self.previous_location.z, target_location.z, alpha_position
                )

                target_yaw = vehicle_transform.rotation.yaw + 90
                smooth_yaw = self.lerp_angle(
                    self.previous_rotation.yaw, target_yaw, alpha_rotation
                )

                spectator_transform = carla.Transform(
                    carla.Location(x=smooth_x, y=smooth_y, z=smooth_z),
                    carla.Rotation(pitch=-85, yaw=smooth_yaw, roll=0),
                )
                spectator.set_transform(spectator_transform)

                self.previous_location = carla.Location(
                    x=smooth_x, y=smooth_y, z=smooth_z
                )
                self.previous_rotation.yaw = smooth_yaw

        except KeyboardInterrupt:
            print("\n시뮬레이션 중지.")
        finally:
            print("프로그램 종료.")


def main(args=None):
    rclpy.init(args=args)
    follower = CarlaSpectatorFollower()
    follower.follow_vehicle()


if __name__ == "__main__":
    main()
