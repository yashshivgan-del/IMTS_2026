"""Simulated arm with gripper and block state tracking.

Moves in real time, animates gripper open/close, tracks which block is held
and where each block currently sits. The overhead camera detection just reads
the simulated state.
"""

from __future__ import annotations

import logging
import os
import random
import threading
import time

from ..config import CellConfig
from ..kinematics import Joints

log = logging.getLogger(__name__)


class SimulatedBackend:
    name = "sim"

    def __init__(self) -> None:
        self._joints = Joints()
        self._lock = threading.Lock()
        self._connected = False
        self._homed = False
        self._stop = threading.Event()
        self._gripper = "open"  # open | closed | gripping
        self._held_block: str | None = None

        # Block positions: {block_id: {x, y}} - updated during pick/place
        self._block_positions: dict[str, dict[str, float]] = {}
        # Which slot each block is in (if any): {block_id: slot_id}
        self._block_slots: dict[str, str | None] = {}

    # -- lifecycle --------------------------------------------------------- #

    def connect(self, cfg: CellConfig) -> None:
        if self._connected:
            return
        # Initialize block positions from config
        for block in cfg.blocks.values():
            self._block_positions[block.id] = {"x": block.start_x, "y": block.start_y}
            self._block_slots[block.id] = None
        self._connected = True
        log.info("sim backend connected (site=%s)", cfg.site)

    def is_connected(self) -> bool:
        return self._connected

    def release(self) -> None:
        self._connected = False

    def estop(self) -> None:
        self._stop.set()

    def clear_estop(self) -> None:
        self._stop.clear()

    # -- motion ------------------------------------------------------------ #

    def home(self, cfg: CellConfig) -> None:
        self.move_to(cfg.home_joints, cfg)
        self._homed = True

    @property
    def homed(self) -> bool:
        return self._homed

    def read_joints(self) -> Joints:
        with self._lock:
            return self._joints

    def move_to(self, target: Joints, cfg: CellConfig) -> None:
        scale = float(os.environ.get("SIM_TIME_SCALE", "1.0"))
        start = self.read_joints()
        duration = cfg.kin.travel_seconds(start, target) * scale

        if duration <= 0:
            with self._lock:
                self._joints = target
            return

        t0 = time.monotonic()
        while True:
            if self._stop.is_set():
                raise RuntimeError("motion stopped by estop")
            frac = (time.monotonic() - t0) / duration
            if frac >= 1.0:
                break
            with self._lock:
                self._joints = start.lerp(target, _smoothstep(frac))
            time.sleep(1 / 60)

        with self._lock:
            self._joints = target

    # -- gripper ----------------------------------------------------------- #

    def gripper_open(self, cfg: CellConfig) -> None:
        scale = float(os.environ.get("SIM_TIME_SCALE", "1.0"))
        time.sleep(float(cfg.gripper["actuate_time_s"]) * scale)
        with self._lock:
            if self._held_block:
                # Releasing a block - place it at current gripper position
                pose = cfg.kin.forward(self._joints)
                self._block_positions[self._held_block] = {
                    "x": round(pose.x, 1),
                    "y": round(pose.y, 1),
                }
                self._held_block = None
            self._gripper = "open"

    def gripper_close(self, cfg: CellConfig) -> None:
        scale = float(os.environ.get("SIM_TIME_SCALE", "1.0"))
        time.sleep(float(cfg.gripper["actuate_time_s"]) * scale)
        with self._lock:
            # Check if there's a block at the current position
            pose = cfg.kin.forward(self._joints)
            grabbed = self._find_block_at(pose.x, pose.y, cfg.block_size)
            if grabbed:
                self._held_block = grabbed
                self._gripper = "gripping"
            else:
                self._gripper = "closed"

    def gripper_state(self) -> str:
        with self._lock:
            return self._gripper

    def _find_block_at(self, x: float, y: float, block_size: float) -> str | None:
        """Find a block within grab distance of (x, y)."""
        grab_radius = block_size * 0.8  # some tolerance
        for bid, pos in self._block_positions.items():
            if bid == self._held_block:
                continue
            dx = pos["x"] - x
            dy = pos["y"] - y
            if (dx * dx + dy * dy) < grab_radius * grab_radius:
                return bid
        return None

    # -- vision (simulated overhead camera) -------------------------------- #

    def detect_blocks(self, cfg: CellConfig) -> list[dict]:
        """Return positions of all blocks not currently held."""
        scale = float(os.environ.get("SIM_TIME_SCALE", "1.0"))
        time.sleep(float(cfg.vision.get("capture_dwell_s", 0.3)) * scale)

        with self._lock:
            results = []
            for block in cfg.blocks.values():
                if block.id == self._held_block:
                    continue  # Can't see a block that's being held
                pos = self._block_positions.get(block.id)
                if pos:
                    results.append({
                        "id": block.id,
                        "color": block.color,
                        "label": block.label,
                        "x": pos["x"],
                        "y": pos["y"],
                        "confidence": cfg.vision.get("detection_confidence", 0.95),
                    })
            return results

    # -- state query ------------------------------------------------------- #

    def get_block_positions(self) -> dict[str, dict[str, float]]:
        with self._lock:
            return dict(self._block_positions)

    def get_held_block(self) -> str | None:
        with self._lock:
            return self._held_block

    def get_block_slots(self) -> dict[str, str | None]:
        with self._lock:
            return dict(self._block_slots)

    def set_block_slot(self, block_id: str, slot_id: str | None) -> None:
        with self._lock:
            self._block_slots[block_id] = slot_id

    def reset_blocks(self, cfg: CellConfig) -> None:
        """Put blocks back to starting positions."""
        with self._lock:
            self._held_block = None
            self._gripper = "open"
            for block in cfg.blocks.values():
                self._block_positions[block.id] = {
                    "x": block.start_x,
                    "y": block.start_y,
                }
                self._block_slots[block.id] = None


def _smoothstep(t: float) -> float:
    return t * t * (3.0 - 2.0 * t)
