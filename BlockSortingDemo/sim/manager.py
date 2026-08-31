"""Cell manager for block sorting: safety, planning, execution.

The AI decides WHAT order the blocks should be in.
This module decides HOW to physically do it safely.

The structural guarantee: execute_sort() only accepts a plan_id minted by
plan_sort(). Single use, TTL-bounded. A forged or stale id is refused.
"""

from __future__ import annotations

import logging
import math
import secrets
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from enum import Enum

from .config import CellConfig
from .kinematics import Joints

log = logging.getLogger(__name__)


class CellState(str, Enum):
    DISCONNECTED = "disconnected"
    IDLE = "idle"
    MOVING = "moving"
    PICKING = "picking"
    PLACING = "placing"
    COOLDOWN = "cooldown"
    FAULT = "fault"
    ESTOP = "estop"


class JobState(str, Enum):
    QUEUED = "queued"
    RUNNING = "running"
    DONE = "done"
    CANCELLED = "cancelled"
    FAILED = "failed"


@dataclass
class Violation:
    code: str
    message: str


@dataclass
class _PickPlaceOp:
    block_id: str
    block_label: str
    slot_id: str
    slot_label: str
    pick_x: float
    pick_y: float
    place_x: float
    place_y: float


@dataclass
class _Plan:
    plan_id: str
    operations: list[_PickPlaceOp]
    sequence: list[str]
    est_seconds: float
    created: float


@dataclass
class _Job:
    job_id: str
    plan: _Plan
    state: JobState = JobState.QUEUED
    step_index: int = 0
    current_action: str = ""
    error: str = ""
    completed_ops: list[dict] = field(default_factory=list)
    cancel: threading.Event = field(default_factory=threading.Event)


PLAN_TTL_SECONDS = 120


class CellManager:
    def __init__(self, cfg: CellConfig, backend=None):
        self.cfg = cfg
        self.backend = backend or _make_backend(cfg)
        self._lock = threading.Lock()
        self._plans: dict[str, _Plan] = {}
        self._jobs: dict[str, _Job] = {}
        self._active: _Job | None = None
        self._fault: str | None = None
        self._estop = False
        self._cooldown_until = 0.0
        self._sort_count = 0

        problems = cfg.sanity_check()
        if problems:
            log.warning("config sanity check issues: %s", problems)

        self.backend.connect(cfg)

    # -- status ------------------------------------------------------------ #

    def status(self) -> dict:
        with self._lock:
            active = self._active
            fault = self._fault
            estop = self._estop
            cooldown = max(0.0, self._cooldown_until - time.time())

        if estop:
            state = CellState.ESTOP
        elif fault:
            state = CellState.FAULT
        elif not self.backend.is_connected():
            state = CellState.DISCONNECTED
        elif active is not None:
            if active.current_action.startswith("pick"):
                state = CellState.PICKING
            elif active.current_action.startswith("place"):
                state = CellState.PLACING
            else:
                state = CellState.MOVING
        elif cooldown > 0:
            state = CellState.COOLDOWN
        else:
            state = CellState.IDLE

        joints = self.backend.read_joints()
        pose = self.cfg.kin.forward(joints)

        return {
            "state": state.value,
            "backend": self.backend.name,
            "site": self.cfg.site,
            "joints": {k: round(v, 2) for k, v in joints.as_dict().items()},
            "pose": {
                "x": round(pose.x, 1),
                "y": round(pose.y, 1),
                "z": round(pose.z, 1),
            },
            "gripper": self.backend.gripper_state(),
            "held_block": self.backend.get_held_block(),
            "current_job": active.job_id if active else None,
            "current_action": active.current_action if active else "",
            "sorts_completed": self._sort_count,
            "seconds_until_ready": math.ceil(cooldown * 100) / 100,
            "fault_detail": fault,
            "block_positions": self.backend.get_block_positions(),
        }

    # -- detection --------------------------------------------------------- #

    def detect_blocks(self) -> dict:
        """Use overhead camera to find blocks."""
        blocks = self.backend.detect_blocks(self.cfg)
        block_slots = self.backend.get_block_slots()

        # Build slot occupancy info
        slot_occupancy: dict[str, str | None] = {}
        for slot in self.cfg.slots.values():
            slot_occupancy[slot.id] = None
        for bid, sid in block_slots.items():
            if sid is not None and sid in slot_occupancy:
                slot_occupancy[sid] = bid

        return {
            "blocks": blocks,
            "count": len(blocks),
            "slots": [
                {
                    "id": s.id,
                    "label": s.label,
                    "x": s.x,
                    "y": s.y,
                    "occupied_by": slot_occupancy.get(s.id),
                }
                for s in self.cfg.slots.values()
            ],
            "blocks_in_slots": any(v is not None for v in slot_occupancy.values()),
        }

    # -- planning ---------------------------------------------------------- #

    def plan_sort(self, sequence: list[str]) -> dict:
        """Create a pick-and-place plan from a desired color sequence.

        sequence: e.g. ["green", "red", "yellow"]
        Maps each color to the corresponding slot in order.
        """
        violations: list[Violation] = []

        if not sequence:
            violations.append(Violation("EMPTY_SEQUENCE", "Need at least one block."))
            return {"ok": False, "plan_id": None, "violations": [v.__dict__ for v in violations]}

        if len(sequence) > len(self.cfg.slots):
            violations.append(Violation(
                "TOO_MANY_BLOCKS",
                f"{len(sequence)} blocks requested but only {len(self.cfg.slots)} slots."
            ))

        # Validate all colors exist
        for color in sequence:
            if color not in self.cfg.blocks:
                violations.append(Violation(
                    "UNKNOWN_BLOCK",
                    f"No block with id '{color}'. Available: {list(self.cfg.blocks.keys())}"
                ))

        # Check for duplicates
        if len(set(sequence)) != len(sequence):
            violations.append(Violation("DUPLICATE", "Each block can only appear once."))

        if violations:
            return {"ok": False, "plan_id": None, "violations": [v.__dict__ for v in violations]}

        # Build operations — slot-conflict-aware planning
        # If a destination slot is occupied by a different block, relocate it first.
        block_positions = self.backend.get_block_positions()
        block_slots = self.backend.get_block_slots()  # {block_id: slot_id or None}
        slot_list = list(self.cfg.slots.values())
        operations: list[_PickPlaceOp] = []
        est = 0.0

        grip_z = float(self.cfg.heights["grip_z"])
        prev_joints = self.backend.read_joints()

        # Build reverse map: which block is in which slot right now
        # slot_occupants: {slot_id: block_id}
        slot_occupants: dict[str, str] = {}
        for bid, sid in block_slots.items():
            if sid is not None:
                slot_occupants[sid] = bid

        # Track which staging spots are available
        staging_list = list(self.cfg.staging.values())
        used_staging: set[str] = set()

        # Simulated positions: track where blocks are during planning
        # (so relocation ops update positions for subsequent steps)
        sim_positions: dict[str, dict[str, float]] = dict(block_positions)
        sim_slot_occupants: dict[str, str] = dict(slot_occupants)

        for i, color in enumerate(sequence):
            block = self.cfg.blocks[color]
            slot = slot_list[i]

            # Check if the destination slot is occupied by a DIFFERENT block
            occupant = sim_slot_occupants.get(slot.id)
            if occupant is not None and occupant != color:
                # Need to relocate the occupant to a staging spot first
                # Find an available staging spot that isn't occupied by another block
                staging_spot = None
                for spot in staging_list:
                    if spot.id in used_staging:
                        continue
                    # Check no block is currently sitting at this staging position
                    spot_free = True
                    for bid, bpos in sim_positions.items():
                        if bid == occupant:
                            continue  # the block we're about to move doesn't count
                        dx = bpos["x"] - spot.x
                        dy = bpos["y"] - spot.y
                        if abs(dx) < 30 and abs(dy) < 30:  # within block-size proximity
                            spot_free = False
                            break
                    if spot_free:
                        staging_spot = spot
                        used_staging.add(spot.id)
                        break

                if staging_spot is None:
                    violations.append(Violation(
                        "NO_STAGING",
                        f"No staging spot available to relocate block '{occupant}' from {slot.id}."
                    ))
                    continue

                # Validate occupant's current position (should be at the slot)
                occ_pos = sim_positions.get(occupant)
                if occ_pos is None:
                    violations.append(Violation(
                        "BLOCK_NOT_FOUND",
                        f"Occupant block '{occupant}' position unknown."
                    ))
                    continue

                # Check occupant pick reachable
                occ_pick_joints = self.cfg.kin.inverse(occ_pos["x"], occ_pos["y"], grip_z)
                if occ_pick_joints is None:
                    violations.append(Violation(
                        "UNREACHABLE",
                        f"Cannot reach occupant '{occupant}' at ({occ_pos['x']}, {occ_pos['y']})."
                    ))
                    continue

                # Check staging place reachable
                stg_place_joints = self.cfg.kin.inverse(staging_spot.x, staging_spot.y, grip_z)
                if stg_place_joints is None:
                    violations.append(Violation(
                        "UNREACHABLE",
                        f"Cannot reach staging '{staging_spot.id}' at ({staging_spot.x}, {staging_spot.y})."
                    ))
                    continue

                occ_block = self.cfg.blocks[occupant]
                operations.append(_PickPlaceOp(
                    block_id=occupant,
                    block_label=occ_block.label,
                    slot_id=staging_spot.id,
                    slot_label=staging_spot.label,
                    pick_x=occ_pos["x"],
                    pick_y=occ_pos["y"],
                    place_x=staging_spot.x,
                    place_y=staging_spot.y,
                ))

                # Update simulated state
                sim_positions[occupant] = {"x": staging_spot.x, "y": staging_spot.y}
                del sim_slot_occupants[slot.id]

                # Estimate time for relocation
                est += self.cfg.kin.travel_seconds(prev_joints, occ_pick_joints)
                est += 3.0  # pick
                est += self.cfg.kin.travel_seconds(occ_pick_joints, stg_place_joints)
                est += 3.0  # place
                prev_joints = stg_place_joints

            # Now handle the main block placement
            # If the block is already in its target slot, skip it
            current_slot_of_block = None
            for sid, bid in sim_slot_occupants.items():
                if bid == color:
                    current_slot_of_block = sid
                    break

            if current_slot_of_block == slot.id:
                # Block is already in the correct slot, no move needed
                continue

            pos = sim_positions.get(color)
            if pos is None:
                violations.append(Violation("BLOCK_NOT_FOUND", f"Block '{color}' position unknown."))
                continue

            # Check pick position reachable
            pick_joints = self.cfg.kin.inverse(pos["x"], pos["y"], grip_z)
            if pick_joints is None:
                violations.append(Violation("UNREACHABLE", f"Cannot reach block '{color}' at ({pos['x']}, {pos['y']})."))
                continue

            # Check place position reachable
            place_joints = self.cfg.kin.inverse(slot.x, slot.y, grip_z)
            if place_joints is None:
                violations.append(Violation("UNREACHABLE", f"Cannot reach slot '{slot.id}' at ({slot.x}, {slot.y})."))
                continue

            operations.append(_PickPlaceOp(
                block_id=color,
                block_label=block.label,
                slot_id=slot.id,
                slot_label=slot.label,
                pick_x=pos["x"],
                pick_y=pos["y"],
                place_x=slot.x,
                place_y=slot.y,
            ))

            # Update simulated state
            sim_positions[color] = {"x": slot.x, "y": slot.y}
            # Remove block from its previous slot (if any)
            if current_slot_of_block is not None:
                del sim_slot_occupants[current_slot_of_block]
            sim_slot_occupants[slot.id] = color

            # Estimate time
            est += self.cfg.kin.travel_seconds(prev_joints, pick_joints)
            est += 3.0  # pick sequence (lower, grab, lift)
            if place_joints:
                est += self.cfg.kin.travel_seconds(pick_joints, place_joints)
            est += 3.0  # place sequence (lower, release, lift)
            prev_joints = place_joints or prev_joints

        if violations:
            return {"ok": False, "plan_id": None, "violations": [v.__dict__ for v in violations]}

        # Check time budget
        budget = float(self.cfg.limits["max_plan_seconds"])
        if est > budget:
            violations.append(Violation(
                "TOO_SLOW", f"Plan needs {est:.0f}s; budget is {budget:.0f}s."
            ))
            return {"ok": False, "plan_id": None, "violations": [v.__dict__ for v in violations]}

        # Check cell readiness
        st = self.status()
        if st["state"] in ("fault", "estop", "disconnected"):
            violations.append(Violation("CELL_NOT_READY", f"Cell is {st['state']}."))
            return {"ok": False, "plan_id": None, "violations": [v.__dict__ for v in violations]}

        plan = _Plan(
            plan_id="plan_" + secrets.token_hex(5),
            operations=operations,
            sequence=list(sequence),
            est_seconds=est,
            created=time.time(),
        )
        with self._lock:
            self._plans[plan.plan_id] = plan

        return {
            "ok": True,
            "plan_id": plan.plan_id,
            "violations": [],
            "operations": [
                {
                    "block": op.block_label,
                    "block_id": op.block_id,
                    "from": {"x": op.pick_x, "y": op.pick_y},
                    "to": {"x": op.place_x, "y": op.place_y},
                    "slot": op.slot_label,
                }
                for op in operations
            ],
            "est_seconds": round(est, 1),
        }

    # -- state-based planning ---------------------------------------------- #

    def plan_arrangement(self, target_state: dict[str, str]) -> dict:
        """Plan moves to reach a desired board state with minimal changes.

        target_state: dict mapping slot_id → block_id for slots that should change.
            e.g. {"slot_2": "green"} means "put green in slot 2, leave others as-is"
            e.g. {"slot_1": "red", "slot_2": "green", "slot_3": "yellow"} full arrangement

        Slots not mentioned in target_state remain unchanged.
        The planner computes the minimal set of moves, handling conflicts
        (relocating blocks from occupied destination slots) automatically.
        """
        violations: list[Violation] = []

        if not target_state:
            violations.append(Violation("EMPTY_TARGET", "Target state must specify at least one slot."))
            return {"ok": False, "plan_id": None, "violations": [v.__dict__ for v in violations]}

        # Validate slot IDs
        valid_slot_ids = set(self.cfg.slots.keys())
        for slot_id in target_state:
            if slot_id not in valid_slot_ids:
                violations.append(Violation(
                    "UNKNOWN_SLOT",
                    f"No slot '{slot_id}'. Available: {list(valid_slot_ids)}"
                ))

        # Validate block IDs
        for slot_id, block_id in target_state.items():
            if block_id not in self.cfg.blocks:
                violations.append(Violation(
                    "UNKNOWN_BLOCK",
                    f"No block '{block_id}'. Available: {list(self.cfg.blocks.keys())}"
                ))

        # Check for duplicate blocks in target
        target_blocks = list(target_state.values())
        if len(set(target_blocks)) != len(target_blocks):
            violations.append(Violation("DUPLICATE", "A block cannot be placed in multiple slots."))

        if violations:
            return {"ok": False, "plan_id": None, "violations": [v.__dict__ for v in violations]}

        # Get current state
        block_positions = self.backend.get_block_positions()
        block_slots = self.backend.get_block_slots()  # {block_id: slot_id or None}

        # Build current slot→block map
        current_slot_to_block: dict[str, str | None] = {}
        for slot_id in valid_slot_ids:
            current_slot_to_block[slot_id] = None
        for bid, sid in block_slots.items():
            if sid is not None and sid in current_slot_to_block:
                current_slot_to_block[sid] = bid

        # Build the full desired state: merge target with current for unmentioned slots
        desired_slot_to_block: dict[str, str | None] = dict(current_slot_to_block)
        for slot_id, block_id in target_state.items():
            desired_slot_to_block[slot_id] = block_id

        # Check if a block appears both in an unchanged slot AND a target slot
        # (i.e., block is currently in slot X and target says put it in slot Y)
        # In that case we need to move it, which means slot X becomes empty — that's fine.

        # Determine which slots actually need to change
        changes: list[tuple[str, str]] = []  # (slot_id, desired_block_id)
        for slot_id, desired_block in desired_slot_to_block.items():
            if desired_block is None:
                continue
            current_block = current_slot_to_block.get(slot_id)
            if current_block != desired_block:
                changes.append((slot_id, desired_block))

        if not changes:
            return {
                "ok": True,
                "plan_id": None,
                "message": "Board already matches the requested state. No moves needed.",
                "violations": [],
                "operations": [],
                "est_seconds": 0,
            }

        # Build operations using the same conflict-aware logic as plan_sort
        slot_list_map = self.cfg.slots  # dict[str, Slot]
        operations: list[_PickPlaceOp] = []
        est = 0.0
        grip_z = float(self.cfg.heights["grip_z"])
        prev_joints = self.backend.read_joints()

        # Track simulated state during planning
        sim_positions: dict[str, dict[str, float]] = dict(block_positions)
        sim_slot_occupants: dict[str, str] = {}
        for bid, sid in block_slots.items():
            if sid is not None:
                sim_slot_occupants[sid] = bid

        staging_list = list(self.cfg.staging.values())
        used_staging: set[str] = set()

        for slot_id, desired_block in changes:
            slot = slot_list_map[slot_id]
            block = self.cfg.blocks[desired_block]

            # Check if destination slot is occupied by a different block
            occupant = sim_slot_occupants.get(slot_id)
            if occupant is not None and occupant != desired_block:
                # Check if this occupant has a place in the desired state elsewhere
                # If so, it will be moved later. If not, move to staging.
                occupant_target_slot = None
                for s_id, b_id in changes:
                    if b_id == occupant:
                        occupant_target_slot = s_id
                        break

                if occupant_target_slot is None:
                    # Occupant is not needed elsewhere in the plan — move to staging
                    staging_spot = None
                    for spot in staging_list:
                        if spot.id in used_staging:
                            continue
                        spot_free = True
                        for bid, bpos in sim_positions.items():
                            if bid == occupant:
                                continue
                            dx = bpos["x"] - spot.x
                            dy = bpos["y"] - spot.y
                            if abs(dx) < 30 and abs(dy) < 30:
                                spot_free = False
                                break
                        if spot_free:
                            staging_spot = spot
                            used_staging.add(spot.id)
                            break

                    if staging_spot is None:
                        violations.append(Violation(
                            "NO_STAGING",
                            f"No staging spot to relocate '{occupant}' from {slot_id}."
                        ))
                        continue

                    occ_pos = sim_positions.get(occupant)
                    if occ_pos is None:
                        violations.append(Violation("BLOCK_NOT_FOUND", f"Occupant '{occupant}' position unknown."))
                        continue

                    occ_pick = self.cfg.kin.inverse(occ_pos["x"], occ_pos["y"], grip_z)
                    stg_place = self.cfg.kin.inverse(staging_spot.x, staging_spot.y, grip_z)
                    if occ_pick is None or stg_place is None:
                        violations.append(Violation("UNREACHABLE", f"Cannot relocate '{occupant}'."))
                        continue

                    occ_block = self.cfg.blocks[occupant]
                    operations.append(_PickPlaceOp(
                        block_id=occupant,
                        block_label=occ_block.label,
                        slot_id=staging_spot.id,
                        slot_label=staging_spot.label,
                        pick_x=occ_pos["x"],
                        pick_y=occ_pos["y"],
                        place_x=staging_spot.x,
                        place_y=staging_spot.y,
                    ))
                    sim_positions[occupant] = {"x": staging_spot.x, "y": staging_spot.y}
                    del sim_slot_occupants[slot_id]
                    est += self.cfg.kin.travel_seconds(prev_joints, occ_pick)
                    est += 3.0
                    est += self.cfg.kin.travel_seconds(occ_pick, stg_place)
                    est += 3.0
                    prev_joints = stg_place

                else:
                    # Occupant will be moved later — still need to get it out of the way now
                    staging_spot = None
                    for spot in staging_list:
                        if spot.id in used_staging:
                            continue
                        spot_free = True
                        for bid, bpos in sim_positions.items():
                            if bid == occupant:
                                continue
                            dx = bpos["x"] - spot.x
                            dy = bpos["y"] - spot.y
                            if abs(dx) < 30 and abs(dy) < 30:
                                spot_free = False
                                break
                        if spot_free:
                            staging_spot = spot
                            used_staging.add(spot.id)
                            break

                    if staging_spot is None:
                        violations.append(Violation(
                            "NO_STAGING",
                            f"No staging spot to relocate '{occupant}' from {slot_id}."
                        ))
                        continue

                    occ_pos = sim_positions.get(occupant)
                    if occ_pos is None:
                        violations.append(Violation("BLOCK_NOT_FOUND", f"Occupant '{occupant}' position unknown."))
                        continue

                    occ_pick = self.cfg.kin.inverse(occ_pos["x"], occ_pos["y"], grip_z)
                    stg_place = self.cfg.kin.inverse(staging_spot.x, staging_spot.y, grip_z)
                    if occ_pick is None or stg_place is None:
                        violations.append(Violation("UNREACHABLE", f"Cannot relocate '{occupant}'."))
                        continue

                    occ_block = self.cfg.blocks[occupant]
                    operations.append(_PickPlaceOp(
                        block_id=occupant,
                        block_label=occ_block.label,
                        slot_id=staging_spot.id,
                        slot_label=staging_spot.label,
                        pick_x=occ_pos["x"],
                        pick_y=occ_pos["y"],
                        place_x=staging_spot.x,
                        place_y=staging_spot.y,
                    ))
                    sim_positions[occupant] = {"x": staging_spot.x, "y": staging_spot.y}
                    del sim_slot_occupants[slot_id]
                    est += self.cfg.kin.travel_seconds(prev_joints, occ_pick)
                    est += 3.0
                    est += self.cfg.kin.travel_seconds(occ_pick, stg_place)
                    est += 3.0
                    prev_joints = stg_place

            # Now place the desired block into the slot
            # First check if it's already there
            current_slot_of_block = None
            for sid, bid in sim_slot_occupants.items():
                if bid == desired_block:
                    current_slot_of_block = sid
                    break

            if current_slot_of_block == slot_id:
                continue  # Already in correct slot

            pos = sim_positions.get(desired_block)
            if pos is None:
                violations.append(Violation("BLOCK_NOT_FOUND", f"Block '{desired_block}' position unknown."))
                continue

            pick_joints = self.cfg.kin.inverse(pos["x"], pos["y"], grip_z)
            place_joints = self.cfg.kin.inverse(slot.x, slot.y, grip_z)
            if pick_joints is None:
                violations.append(Violation("UNREACHABLE", f"Cannot reach block '{desired_block}'."))
                continue
            if place_joints is None:
                violations.append(Violation("UNREACHABLE", f"Cannot reach slot '{slot_id}'."))
                continue

            operations.append(_PickPlaceOp(
                block_id=desired_block,
                block_label=block.label,
                slot_id=slot_id,
                slot_label=slot.label,
                pick_x=pos["x"],
                pick_y=pos["y"],
                place_x=slot.x,
                place_y=slot.y,
            ))

            # Update simulated state
            sim_positions[desired_block] = {"x": slot.x, "y": slot.y}
            if current_slot_of_block is not None:
                del sim_slot_occupants[current_slot_of_block]
            sim_slot_occupants[slot_id] = desired_block

            est += self.cfg.kin.travel_seconds(prev_joints, pick_joints)
            est += 3.0
            est += self.cfg.kin.travel_seconds(pick_joints, place_joints)
            est += 3.0
            prev_joints = place_joints

        if violations:
            return {"ok": False, "plan_id": None, "violations": [v.__dict__ for v in violations]}

        if not operations:
            return {
                "ok": True,
                "plan_id": None,
                "message": "No moves required — board already matches target.",
                "violations": [],
                "operations": [],
                "est_seconds": 0,
            }

        # Check time budget
        budget = float(self.cfg.limits["max_plan_seconds"])
        if est > budget:
            violations.append(Violation("TOO_SLOW", f"Plan needs {est:.0f}s; budget is {budget:.0f}s."))
            return {"ok": False, "plan_id": None, "violations": [v.__dict__ for v in violations]}

        # Check cell readiness
        st = self.status()
        if st["state"] in ("fault", "estop", "disconnected"):
            violations.append(Violation("CELL_NOT_READY", f"Cell is {st['state']}."))
            return {"ok": False, "plan_id": None, "violations": [v.__dict__ for v in violations]}

        plan = _Plan(
            plan_id="plan_arr_" + secrets.token_hex(5),
            operations=operations,
            sequence=list(target_state.values()),
            est_seconds=est,
            created=time.time(),
        )
        with self._lock:
            self._plans[plan.plan_id] = plan

        return {
            "ok": True,
            "plan_id": plan.plan_id,
            "violations": [],
            "operations": [
                {
                    "block": op.block_label,
                    "block_id": op.block_id,
                    "from": {"x": op.pick_x, "y": op.pick_y},
                    "to": {"x": op.place_x, "y": op.place_y},
                    "slot": op.slot_label,
                }
                for op in operations
            ],
            "est_seconds": round(est, 1),
            "changes_made": len(changes),
            "message": f"Plan requires {len(operations)} moves to reach target state.",
        }

    # -- execution --------------------------------------------------------- #

    def execute_sort(self, plan_id: str) -> tuple[str | None, list[Violation]]:
        """Execute an approved sort plan."""
        with self._lock:
            plan = self._plans.get(plan_id)
            if plan is None:
                return None, [Violation("UNKNOWN_PLAN", "No such plan. Call plan_sort again.")]
            if time.time() - plan.created > PLAN_TTL_SECONDS:
                self._plans.pop(plan_id, None)
                return None, [Violation("PLAN_EXPIRED", "Plan is stale; re-plan.")]
            if self._active is not None:
                return None, [Violation("BUSY", "Cell is already running a job.")]
            if self._estop:
                return None, [Violation("ESTOP", "Cell is in emergency stop.")]
            if self._fault:
                return None, [Violation("FAULT", self._fault)]
            if time.time() < self._cooldown_until:
                left = self._cooldown_until - time.time()
                return None, [Violation("COOLDOWN", f"Cooldown, {left:.1f}s remaining.")]

            self._plans.pop(plan_id, None)
            job = _Job(job_id="job_" + secrets.token_hex(5), plan=plan)
            self._jobs[job.job_id] = job
            self._active = job

        threading.Thread(target=self._run, args=(job,), daemon=True).start()
        return job.job_id, []

    def _run(self, job: _Job) -> None:
        """Execute pick-and-place operations sequentially."""
        try:
            job.state = JobState.RUNNING
            cfg = self.cfg
            lift_z = float(cfg.heights["lift_z"])
            grip_z = float(cfg.heights["grip_z"])
            approach_z = float(cfg.heights["approach_z"])

            # Home first if not already
            if not self.backend.homed:
                job.current_action = "homing"
                self.backend.home(cfg)

            for i, op in enumerate(job.plan.operations):
                if job.cancel.is_set():
                    job.state = JobState.CANCELLED
                    return

                job.step_index = i

                # --- PICK SEQUENCE ---
                # 1. Move to above the block (approach height)
                job.current_action = f"pick:approach:{op.block_id}"
                approach_joints = cfg.kin.inverse(op.pick_x, op.pick_y, approach_z)
                if approach_joints is None:
                    raise RuntimeError(f"Cannot reach approach for {op.block_id}")
                self.backend.move_to(approach_joints, cfg)

                # 2. Open gripper
                job.current_action = f"pick:open:{op.block_id}"
                self.backend.gripper_open(cfg)

                # 3. Lower to grip height
                job.current_action = f"pick:lower:{op.block_id}"
                grip_joints = cfg.kin.inverse(op.pick_x, op.pick_y, grip_z)
                if grip_joints is None:
                    raise RuntimeError(f"Cannot reach grip for {op.block_id}")
                self.backend.move_to(grip_joints, cfg)

                # 4. Close gripper (grab block)
                job.current_action = f"pick:grab:{op.block_id}"
                self.backend.gripper_close(cfg)

                # 5. Lift
                job.current_action = f"pick:lift:{op.block_id}"
                lift_joints = cfg.kin.inverse(op.pick_x, op.pick_y, lift_z)
                if lift_joints is None:
                    raise RuntimeError(f"Cannot reach lift for {op.block_id}")
                self.backend.move_to(lift_joints, cfg)

                # --- PLACE SEQUENCE ---
                # 6. Move to above the slot (lift height)
                job.current_action = f"place:move:{op.slot_id}"
                place_lift_joints = cfg.kin.inverse(op.place_x, op.place_y, lift_z)
                if place_lift_joints is None:
                    raise RuntimeError(f"Cannot reach slot lift for {op.slot_id}")
                self.backend.move_to(place_lift_joints, cfg)

                # 7. Lower to grip height at slot
                job.current_action = f"place:lower:{op.slot_id}"
                place_joints = cfg.kin.inverse(op.place_x, op.place_y, grip_z)
                if place_joints is None:
                    raise RuntimeError(f"Cannot reach slot {op.slot_id}")
                self.backend.move_to(place_joints, cfg)

                # 8. Open gripper (release block)
                job.current_action = f"place:release:{op.slot_id}"
                self.backend.gripper_open(cfg)

                # 9. Lift away
                job.current_action = f"place:lift:{op.slot_id}"
                self.backend.move_to(place_lift_joints, cfg)

                # Track slot assignment:
                # First clear any previous slot this block occupied
                old_slots = self.backend.get_block_slots()
                old_slot = old_slots.get(op.block_id)
                if old_slot is not None and old_slot != op.slot_id:
                    # Block moved from a real slot — clear the old assignment
                    pass  # set_block_slot below handles the new assignment

                # Determine if this is a staging relocation or a final placement
                is_staging = op.slot_id.startswith("stage_")
                if is_staging:
                    # Moving to staging: clear slot assignment (block is no longer in a slot)
                    self.backend.set_block_slot(op.block_id, None)
                else:
                    self.backend.set_block_slot(op.block_id, op.slot_id)

                job.completed_ops.append({
                    "block": op.block_label,
                    "block_id": op.block_id,
                    "slot": op.slot_label,
                    "slot_id": op.slot_id,
                })

            # Return home
            job.current_action = "homing"
            self.backend.home(cfg)

            job.state = JobState.DONE
            job.current_action = "complete"
            self._sort_count += 1

            # Start cooldown
            cooldown_after = int(self.cfg.limits.get("cooldown_after_sorts", 999))
            if self._sort_count % cooldown_after == 0:
                self._cooldown_until = time.time() + float(
                    self.cfg.limits.get("cooldown_seconds", 45)
                )

        except Exception as e:
            log.exception("sort job failed")
            job.state = JobState.FAILED
            job.error = str(e)
            with self._lock:
                self._fault = str(e)
        finally:
            with self._lock:
                self._active = None

    # -- job status -------------------------------------------------------- #

    def job_status(self, job_id: str) -> dict:
        job = self._jobs.get(job_id)
        if job is None:
            return {"error": "unknown job_id"}
        total = len(job.plan.operations)
        return {
            "job_id": job.job_id,
            "state": job.state.value,
            "step_index": job.step_index,
            "step_count": total,
            "pct": round(100 * len(job.completed_ops) / max(1, total), 1),
            "current_action": job.current_action,
            "completed_ops": job.completed_ops,
            "sequence": job.plan.sequence,
            "error": job.error or None,
        }

    # -- cancel / recover / estop ------------------------------------------ #

    def cancel(self, job_id: str) -> bool:
        job = self._jobs.get(job_id)
        if job and job.state == JobState.RUNNING:
            job.cancel.set()
            return True
        return False

    def recover(self) -> dict:
        with self._lock:
            self._fault = None
            self._estop = False
            self._plans.clear()
        self.backend.clear_estop()
        self.backend.home(self.cfg)
        return {"recovered": True, "state": self.status()["state"]}

    def estop(self) -> dict:
        with self._lock:
            self._estop = True
        self.backend.estop()
        return {"estopped": True}

    def reset_blocks(self) -> dict:
        """Reset all blocks to starting positions using the robot arm.

        Generates pick-and-place operations to physically move each block
        from its current position back to its starting position.
        Returns a job_id to poll with get_sort_status.
        """
        with self._lock:
            if self._active is not None:
                return {"ok": False, "reason": "Cannot reset while a job is running."}
            if self._estop:
                return {"ok": False, "reason": "Cell is in emergency stop."}
            if self._fault:
                return {"ok": False, "reason": f"Cell fault: {self._fault}"}

        # Build operations to move each block back to start
        block_positions = self.backend.get_block_positions()
        grip_z = float(self.cfg.heights["grip_z"])
        operations: list[_PickPlaceOp] = []

        for block in self.cfg.blocks.values():
            pos = block_positions.get(block.id)
            if pos is None:
                continue

            # Skip if block is already at its start position (within tolerance)
            dx = pos["x"] - block.start_x
            dy = pos["y"] - block.start_y
            if abs(dx) < 5 and abs(dy) < 5:
                continue

            # Verify pick and place are reachable
            pick_joints = self.cfg.kin.inverse(pos["x"], pos["y"], grip_z)
            if pick_joints is None:
                continue
            place_joints = self.cfg.kin.inverse(block.start_x, block.start_y, grip_z)
            if place_joints is None:
                continue

            operations.append(_PickPlaceOp(
                block_id=block.id,
                block_label=block.label,
                slot_id="start_" + block.id,
                slot_label=f"Start ({block.label})",
                pick_x=pos["x"],
                pick_y=pos["y"],
                place_x=block.start_x,
                place_y=block.start_y,
            ))

        if not operations:
            return {"ok": True, "message": "All blocks already at starting positions.",
                    "block_positions": self.backend.get_block_positions()}

        # Create a plan and job, execute it
        plan = _Plan(
            plan_id="plan_reset_" + secrets.token_hex(5),
            operations=operations,
            sequence=["reset"],
            est_seconds=len(operations) * 8.0,
            created=time.time(),
        )

        with self._lock:
            job = _Job(job_id="job_reset_" + secrets.token_hex(5), plan=plan)
            self._jobs[job.job_id] = job
            self._active = job

        threading.Thread(target=self._run_reset, args=(job,), daemon=True).start()
        return {"ok": True, "job_id": job.job_id, "moves": len(operations),
                "block_positions": self.backend.get_block_positions()}

    def _run_reset(self, job: _Job) -> None:
        """Execute reset operations — move blocks to start positions."""
        try:
            job.state = JobState.RUNNING
            cfg = self.cfg
            lift_z = float(cfg.heights["lift_z"])
            grip_z = float(cfg.heights["grip_z"])
            approach_z = float(cfg.heights["approach_z"])

            if not self.backend.homed:
                job.current_action = "homing"
                self.backend.home(cfg)

            for i, op in enumerate(job.plan.operations):
                if job.cancel.is_set():
                    job.state = JobState.CANCELLED
                    return

                job.step_index = i

                # PICK from current position
                job.current_action = f"pick:approach:{op.block_id}"
                approach_joints = cfg.kin.inverse(op.pick_x, op.pick_y, approach_z)
                if approach_joints is None:
                    raise RuntimeError(f"Cannot reach approach for {op.block_id}")
                self.backend.move_to(approach_joints, cfg)

                job.current_action = f"pick:open:{op.block_id}"
                self.backend.gripper_open(cfg)

                job.current_action = f"pick:lower:{op.block_id}"
                grip_joints = cfg.kin.inverse(op.pick_x, op.pick_y, grip_z)
                if grip_joints is None:
                    raise RuntimeError(f"Cannot reach grip for {op.block_id}")
                self.backend.move_to(grip_joints, cfg)

                job.current_action = f"pick:grab:{op.block_id}"
                self.backend.gripper_close(cfg)

                job.current_action = f"pick:lift:{op.block_id}"
                lift_joints = cfg.kin.inverse(op.pick_x, op.pick_y, lift_z)
                if lift_joints is None:
                    raise RuntimeError(f"Cannot reach lift for {op.block_id}")
                self.backend.move_to(lift_joints, cfg)

                # PLACE at start position
                job.current_action = f"place:move:{op.block_id}"
                place_lift_joints = cfg.kin.inverse(op.place_x, op.place_y, lift_z)
                if place_lift_joints is None:
                    raise RuntimeError(f"Cannot reach start lift for {op.block_id}")
                self.backend.move_to(place_lift_joints, cfg)

                job.current_action = f"place:lower:{op.block_id}"
                place_joints = cfg.kin.inverse(op.place_x, op.place_y, grip_z)
                if place_joints is None:
                    raise RuntimeError(f"Cannot reach start for {op.block_id}")
                self.backend.move_to(place_joints, cfg)

                job.current_action = f"place:release:{op.block_id}"
                self.backend.gripper_open(cfg)

                job.current_action = f"place:lift:{op.block_id}"
                self.backend.move_to(place_lift_joints, cfg)

                # Clear slot assignment
                self.backend.set_block_slot(op.block_id, None)

                job.completed_ops.append({
                    "block": op.block_label,
                    "block_id": op.block_id,
                    "slot": op.slot_label,
                    "slot_id": op.slot_id,
                })

            # Return home
            job.current_action = "homing"
            self.backend.home(cfg)

            job.state = JobState.DONE
            job.current_action = "complete"

        except Exception as e:
            log.exception("reset job failed")
            job.state = JobState.FAILED
            job.error = str(e)
            with self._lock:
                self._fault = str(e)
        finally:
            with self._lock:
                self._active = None


def _make_backend(cfg: CellConfig):
    name = cfg.backend_name
    if name == "roarm":
        from .backends.roarm import RoArmBackend
        return RoArmBackend()
    if name == "sim":
        from .backends.simulated import SimulatedBackend
        return SimulatedBackend()
    raise ValueError(f"Unknown backend: {name}")
