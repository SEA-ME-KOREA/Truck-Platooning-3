#!/usr/bin/env python3

import carla


def main() -> None:
    client = carla.Client("localhost", 2000)
    client.set_timeout(5.0)
    world = client.get_world()
    actors = world.get_actors()

    target_roles = {f"truck{i}" for i in range(3)} | {f"trailer{i}" for i in range(3)}
    destroyed = 0

    for actor in actors:
        try:
            role_name = actor.attributes.get("role_name", "")
        except Exception:
            continue
        if role_name in target_roles:
            try:
                actor.destroy()
                destroyed += 1
            except Exception:
                pass

    print(f"destroyed {destroyed} actors")


if __name__ == "__main__":
    main()
