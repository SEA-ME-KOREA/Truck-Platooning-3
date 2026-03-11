#!/usr/bin/env python3

from typing import Dict

from sensor_msgs.msg import Image


class DebugImagePublisher:
    """Small helper that keeps debug image publishers grouped by topic name."""

    def __init__(self, node, topic_map: Dict[str, str]) -> None:
        self._node = node
        self._publishers = {
            key: node.create_publisher(Image, topic, 10) for key, topic in topic_map.items()
        }

    def publish(self, key: str, msg: Image) -> None:
        publisher = self._publishers.get(key)
        if publisher is not None:
            publisher.publish(msg)
