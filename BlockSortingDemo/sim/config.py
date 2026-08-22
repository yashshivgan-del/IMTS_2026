"""Config loading for the block sorting cell."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import yaml

from .kinematics import Joints, Kinematics

DEFAULT_PATH = Path(__file__).resolve().parent.parent / "config" / "cell.yaml"


@dataclass(frozen=True)
class Block:
    id: str
    color: str
    label: str
    start_x: float
    start_y: float


@dataclass(frozen=True)
class Slot:
    id: str
    label: str
    x: float
    y: float


@dataclass(frozen=True)
class StagingSpot:
    id: str
    label: str
    x: float
    y: float


class CellConfig:
    def __init__(self, raw: dict):
        self.raw = raw
        self.site = raw["site"]
        self.units = raw["units"]
        self.arm = raw["arm"]
        self.connection = raw.get("connection", {})
        self.limits = raw["limits"]
        self.envelope = raw["envelope"]
        self.gripper = raw["gripper"]
        self.heights = raw["heights"]
        self.vision = raw["vision"]

        self.kin = Kinematics(raw["arm"], raw["envelope"])

        # Parse blocks
        block_cfg = raw["blocks"]
        self.block_size = float(block_cfg["size"])
        self.blocks: dict[str, Block] = {}
        for color_def in block_cfg["colors"]:
            bid = color_def["id"]
            start = block_cfg["start_positions"][bid]
            self.blocks[bid] = Block(
                id=bid,
                color=color_def["color"],
                label=color_def["label"],
                start_x=float(start["x"]),
                start_y=float(start["y"]),
            )

        # Parse slots
        self.slots: dict[str, Slot] = {}
        for s in raw["slots"]:
            self.slots[s["id"]] = Slot(
                id=s["id"],
                label=s["label"],
                x=float(s["x"]),
                y=float(s["y"]),
            )

        # Parse staging positions (temporary safe locations for re-sorting)
        self.staging: dict[str, StagingSpot] = {}
        for s in raw.get("staging", []):
            self.staging[s["id"]] = StagingSpot(
                id=s["id"],
                label=s["label"],
                x=float(s["x"]),
                y=float(s["y"]),
            )

        # Home joints
        self.home_joints = Joints.from_dict(raw["home"]["joints"])

    @property
    def backend_name(self) -> str:
        return os.environ.get("CELL_BACKEND", self.raw.get("backend", "sim"))

    def sanity_check(self) -> list[str]:
        """Verify home position and all slots are reachable."""
        problems: list[str] = []

        # Check home
        for msg in self.kin.joint_violations(self.home_joints):
            problems.append(f"home: {msg}")
        for msg in self.kin.envelope_violations(self.home_joints):
            problems.append(f"home: {msg}")

        # Check all slots are reachable at grip height
        grip_z = float(self.heights["grip_z"])
        for slot in self.slots.values():
            joints = self.kin.inverse(slot.x, slot.y, grip_z)
            if joints is None:
                problems.append(f"slot '{slot.id}' at ({slot.x}, {slot.y}) unreachable")

        # Check all starting block positions are reachable
        for block in self.blocks.values():
            joints = self.kin.inverse(block.start_x, block.start_y, grip_z)
            if joints is None:
                problems.append(
                    f"block '{block.id}' start ({block.start_x}, {block.start_y}) unreachable"
                )

        # Check all staging positions are reachable
        for spot in self.staging.values():
            joints = self.kin.inverse(spot.x, spot.y, grip_z)
            if joints is None:
                problems.append(
                    f"staging '{spot.id}' at ({spot.x}, {spot.y}) unreachable"
                )

        return problems

    def to_scene(self) -> dict:
        """Everything the Three.js scene needs."""
        return {
            "site": self.site,
            "measured": bool(self.arm.get("measured", False)),
            "arm": {
                "base_height": self.kin.base_height,
                "upper_arm": self.kin.upper_arm,
                "forearm": self.kin.forearm,
                "eoat": self.kin.eoat,
            },
            "envelope": self.envelope,
            "gripper": self.gripper,
            "blocks": [
                {
                    "id": b.id,
                    "color": b.color,
                    "label": b.label,
                    "size": self.block_size,
                    "start_x": b.start_x,
                    "start_y": b.start_y,
                }
                for b in self.blocks.values()
            ],
            "slots": [
                {"id": s.id, "label": s.label, "x": s.x, "y": s.y}
                for s in self.slots.values()
            ],
            "heights": self.heights,
            "home_joints": self.home_joints.as_dict(),
        }


def load_config(path: str | os.PathLike | None = None) -> CellConfig:
    p = Path(path or os.environ.get("CELL_CONFIG") or DEFAULT_PATH)
    with open(p) as fh:
        return CellConfig(yaml.safe_load(fh))
