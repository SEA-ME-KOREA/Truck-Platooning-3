#!/usr/bin/env python3

from dataclasses import dataclass, field
from enum import Enum
from typing import List


class ManeuverState(str, Enum):
    IDLE = "IDLE"
    LEADER_EXITS_LANE = "LEADER_EXITS_LANE"
    LEADER_CREATES_GAP = "LEADER_CREATES_GAP"
    SUCCESSOR_ENTERS_GAP = "SUCCESSOR_ENTERS_GAP"
    FOLLOWER_ENTERS_GAP = "FOLLOWER_ENTERS_GAP"
    LEADER_REENTERS_LANE = "LEADER_REENTERS_LANE"
    PROMOTE_TARGET_EXITS = "PROMOTE_TARGET_EXITS"
    PROMOTE_PLATOON_CREATES_GAP = "PROMOTE_PLATOON_CREATES_GAP"
    PROMOTE_TARGET_REENTERS = "PROMOTE_TARGET_REENTERS"
    REORDER_COMPLETE = "REORDER_COMPLETE"
    COOLDOWN = "COOLDOWN"


@dataclass
class PlatoonState:
    truck_order: List[int] = field(default_factory=lambda: [0, 1, 2])
    maneuver_state: ManeuverState = ManeuverState.IDLE
    reorder_direction: str = "left"
    exiting_leader_id: int = -1
    successor_id: int = -1
    follower_id: int = -1
    promote_target_id: int = -1
    promote_original_leader_id: int = -1

    def reset_targets(self) -> None:
        self.exiting_leader_id = -1
        self.successor_id = -1
        self.follower_id = -1
        self.promote_target_id = -1
        self.promote_original_leader_id = -1
