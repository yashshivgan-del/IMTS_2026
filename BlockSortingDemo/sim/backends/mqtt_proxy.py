"""MQTT proxy backend (runs on EC2, inside the MCP server).

This backend implements the SAME interface as SimulatedBackend and
RoArmBackend, but it does not touch hardware. Instead, each motion
primitive is published as an MQTT command to a broker, and the backend
blocks until the booth laptop agent replies with a result.

    Manager (EC2)  ->  MqttProxyBackend.move_to()
                          │ publish robo/cmd {id, op:"move_to", joints}
                          ▼
                       MQTT Broker (EC2)
                          ▼
                       Laptop agent  -> runs roarm.py locally -> arm
                          │ publish robo/result {id, ok, joints}
                          ▼
                       MqttProxyBackend  <- unblocks move_to()

Why this stays fast: only ONE round trip per primitive crosses the
internet. The tight feedback-poll loop lives on the laptop's LAN.

Live state: the laptop publishes robo/state ~10 Hz. This backend caches the
latest state so read_joints() (called at 20 Hz by the 3D mirror) returns
instantly without any network round trip.

Block tracking (positions, slots, held block) lives here on the EC2 side,
identical to the sim/roarm backends, because it's pure bookkeeping the
manager relies on for planning.
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time

from ..config import CellConfig
from ..kinematics import Joints
from .. import mqtt_protocol as proto

log = logging.getLogger(__name__)


class MqttProxyBackend:
    name = "mqtt_proxy"

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._connected = False
        self._homed = False
        self._estopped = False

        # MQTT client (paho) — imported lazily in connect()
        self._client = None
        self._broker_host = "localhost"
        self._broker_port = 1883
        self._cmd_timeout = proto.DEFAULT_CMD_TIMEOUT_S

        # Correlation: pending command id -> Event + result slot
        self._pending: dict[str, dict] = {}
        self._pending_lock = threading.Lock()

        # Latest live state received from the laptop (robo/state)
        self._last_joints = Joints()
        self._last_gripper = "open"
        self._last_state_ts = 0.0
        self._agent_online = False

        # Block state tracking (EC2-side bookkeeping, same as sim/roarm)
        self._block_positions: dict[str, dict[str, float]] = {}
        self._block_slots: dict[str, str | None] = {}
        self._held_block: str | None = None

    # -- lifecycle --------------------------------------------------------- #

    def connect(self, cfg: CellConfig) -> None:
        if self._connected:
            return

        conn = cfg.connection or {}
        mqtt_cfg = conn.get("mqtt", {}) if isinstance(conn.get("mqtt"), dict) else {}
        self._broker_host = (
            mqtt_cfg.get("host")
            or os.environ.get("MQTT_HOST")
            or "localhost"
        )
        use_tls = bool(mqtt_cfg.get("use_tls", False))
        default_port = 8883 if use_tls else 1883
        self._broker_port = int(mqtt_cfg.get("port") or os.environ.get("MQTT_PORT") or default_port)
        self._cmd_timeout = float(mqtt_cfg.get("cmd_timeout_s") or proto.DEFAULT_CMD_TIMEOUT_S)
        username = mqtt_cfg.get("username") or os.environ.get("MQTT_USERNAME")
        password = mqtt_cfg.get("password") or os.environ.get("MQTT_PASSWORD")
        client_id = mqtt_cfg.get("client_id") or f"ec2-proxy-{os.getpid()}"

        try:
            import paho.mqtt.client as mqtt
        except ImportError as e:
            raise RuntimeError(
                "paho-mqtt is required for the mqtt_proxy backend. "
                "Install with: pip install paho-mqtt"
            ) from e

        # See laptop_agent.py's connect() for why VERSION1 is pinned
        # explicitly: paho-mqtt >=2.0 defaults to VERSION2's 5-arg callback
        # signatures, which silently mismatch this file's 4-arg
        # on_connect/on_message and never fire (no crash, just nothing
        # happens -- easy to mistake for a hung connection).
        client = mqtt.Client(
            callback_api_version=mqtt.CallbackAPIVersion.VERSION1,
            client_id=client_id, clean_session=True,
        )

        if use_tls:
            # AWS IoT Core (and most managed MQTT brokers) authenticate via
            # mutual TLS certs, NOT username/password -- username_pw_set()
            # is skipped in this path since IoT Core ignores/rejects it.
            ca_cert = mqtt_cfg.get("ca_cert")
            cert_file = mqtt_cfg.get("cert_file")
            key_file = mqtt_cfg.get("key_file")
            if not (ca_cert and cert_file and key_file):
                raise RuntimeError(
                    "mqtt.use_tls is true but ca_cert/cert_file/key_file are "
                    "not all set in cell.yaml connection.mqtt -- required for "
                    "AWS IoT Core's mutual-TLS auth."
                )
            client.tls_set(ca_certs=ca_cert, certfile=cert_file, keyfile=key_file)
        elif username:
            client.username_pw_set(username, password)

        client.on_connect = self._on_connect
        client.on_message = self._on_message

        client.connect(self._broker_host, self._broker_port, keepalive=30)
        client.loop_start()
        self._client = client

        # Seed block state from config (no camera)
        for block in cfg.blocks.values():
            self._block_positions[block.id] = {"x": block.start_x, "y": block.start_y}
            self._block_slots[block.id] = None

        if mqtt_cfg.get("skip_connect_handshake"):
            # No live Pi-side agent subscribed yet -- publish/subscribe still
            # works (visible on robo/cmd in the MQTT test client), just skip
            # blocking on a reply that will never arrive.
            log.warning(
                "mqtt.skip_connect_handshake is set -- NOT waiting for a "
                "laptop agent to confirm arm connection. Commands will be "
                "published but any that need a reply (e.g. detect_blocks) "
                "will time out and fall back to cached data until a real "
                "agent is subscribed and replying on %s.", proto.TOPIC_RESULT,
            )
        else:
            # Ask the laptop agent to connect to the arm and confirm it's alive.
            result = self._request(proto.OP_CONNECT, {}, timeout=self._cmd_timeout)
            if not result.get("ok"):
                raise RuntimeError(
                    f"Laptop agent did not confirm arm connection: {result.get('error')}. "
                    f"Is the laptop agent running and connected to broker {self._broker_host}?"
                )

        self._connected = True
        self._estopped = False
        log.info("mqtt_proxy backend connected (broker=%s:%d)", self._broker_host, self._broker_port)

    def is_connected(self) -> bool:
        return self._connected

    def detect_kit_plan(self, placements: list[dict]) -> list[dict]:
        """Detect all blocks and return coordinates for each placement."""
        result = self._request(
            proto.OP_DETECT_KIT_PLAN,
            {"placements": placements},
            timeout=max(self._cmd_timeout, 45.0),  # detection can take 15-20s on Pi
        )
        if not result.get("ok"):
            raise RuntimeError(f"detect_kit_plan failed: {result.get('error')}")
        return result.get("placements", [])

    def capture_image(self) -> str:
        """Ask Pi to capture a frame and return it as base64 JPEG."""
        result = self._request(proto.OP_CAPTURE_IMAGE, {}, timeout=15.0)
        if not result.get("ok"):
            raise RuntimeError(f"capture_image failed: {result.get('error')}")
        return result.get("image_b64", "")

    def execute_single_placement(self, placement: dict) -> dict:
        """Execute a single placement. Must have color, cell, seq, pick_x, pick_y, place_x, place_y.
        Returns the completed placement with pick/place settle info.
        """
        result = self._request(
            proto.OP_EXECUTE_SINGLE,
            {"placement": placement},
            timeout=max(self._cmd_timeout, 45.0),  # one block ~25-30s
        )
        if not result.get("ok"):
            raise RuntimeError(f"execute_single_placement failed: {result.get('error')}")
        return result.get("completed", {})
        """Execute a pre-resolved plan. Each placement must have:
        color, cell, seq, pick_x, pick_y, place_x, place_y.
        Placements are sorted by 'seq' on the laptop side.
        Returns list of completed placements.
        """
        result = self._request(
            proto.OP_EXECUTE_PLAN,
            {"placements": placements},
            timeout=max(self._cmd_timeout * len(placements), 90.0 * len(placements)),
        )
        if not result.get("ok"):
            raise RuntimeError(f"execute_plan failed: {result.get('error')}")
        return result.get("completed", [])

    def release(self) -> None:
        self._connected = False
        if self._client is not None:
            try:
                self._client.loop_stop()
                self._client.disconnect()
            except Exception:
                pass

    # -- MQTT callbacks ---------------------------------------------------- #

    def _on_connect(self, client, userdata, flags, rc):
        client.subscribe([(proto.TOPIC_RESULT, proto.QOS_RESULT),
                          (proto.TOPIC_STATE, proto.QOS_STATE)])
        log.info("mqtt_proxy subscribed to %s and %s", proto.TOPIC_RESULT, proto.TOPIC_STATE)

    def _on_message(self, client, userdata, msg):
        try:
            payload = json.loads(msg.payload.decode("utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError):
            return

        if msg.topic == proto.TOPIC_RESULT:
            self._handle_result(payload)
        elif msg.topic == proto.TOPIC_STATE:
            self._handle_state(payload)

    def _handle_result(self, payload: dict):
        cmd_id = payload.get("id")
        if not cmd_id:
            return
        with self._pending_lock:
            slot = self._pending.get(cmd_id)
            if slot is not None:
                slot["result"] = payload
                slot["event"].set()

    def _handle_state(self, payload: dict):
        j = payload.get("joints") or {}
        with self._lock:
            self._last_joints = Joints(
                base=float(j.get("base", self._last_joints.base)),
                shoulder=float(j.get("shoulder", self._last_joints.shoulder)),
                elbow=float(j.get("elbow", self._last_joints.elbow)),
                wrist=float(j.get("wrist", self._last_joints.wrist)),
            )
            self._last_gripper = payload.get("gripper", self._last_gripper)
            self._last_state_ts = time.time()
            self._agent_online = bool(payload.get("connected", True))

    # -- request/response over MQTT ---------------------------------------- #

    def _request(self, op: str, extra: dict, timeout: float | None = None) -> dict:
        """Publish a command and block until the matching result arrives."""
        if self._client is None:
            raise RuntimeError("mqtt_proxy not connected")

        timeout = timeout if timeout is not None else self._cmd_timeout
        cmd_id = proto.new_cmd_id()
        message = {"id": cmd_id, "op": op, **extra}

        event = threading.Event()
        with self._pending_lock:
            self._pending[cmd_id] = {"event": event, "result": None}

        self._client.publish(proto.TOPIC_CMD, json.dumps(message), qos=proto.QOS_CMD)

        got = event.wait(timeout)
        with self._pending_lock:
            slot = self._pending.pop(cmd_id, None)

        if not got or slot is None or slot["result"] is None:
            raise RuntimeError(f"Timeout waiting for '{op}' result (id={cmd_id}, {timeout}s)")

        result = slot["result"]

        # Update cached joints/gripper from the result if present
        rj = result.get("joints")
        if rj:
            with self._lock:
                self._last_joints = Joints(
                    base=float(rj.get("base", self._last_joints.base)),
                    shoulder=float(rj.get("shoulder", self._last_joints.shoulder)),
                    elbow=float(rj.get("elbow", self._last_joints.elbow)),
                    wrist=float(rj.get("wrist", self._last_joints.wrist)),
                )
        if "gripper" in result:
            with self._lock:
                self._last_gripper = result["gripper"]

        return result

    # -- motion ------------------------------------------------------------ #

    def home(self, cfg: CellConfig) -> None:
        result = self._request(proto.OP_HOME, {})
        if not result.get("ok"):
            raise RuntimeError(f"home failed: {result.get('error')}")
        self._homed = True

    @property
    def homed(self) -> bool:
        return self._homed

    def read_joints(self) -> Joints:
        # Return cached live state — no network round trip. The laptop pushes
        # robo/state at ~10 Hz, plenty fresh for the 20 Hz mirror.
        with self._lock:
            return self._last_joints

    def move_to(self, target: Joints, cfg: CellConfig) -> None:
        if self._estopped:
            raise RuntimeError("motion refused: estop active")
        result = self._request(proto.OP_MOVE_TO, {"joints": target.as_dict()})
        if not result.get("ok"):
            raise RuntimeError(f"move_to failed: {result.get('error')}")

    # -- gripper ----------------------------------------------------------- #

    def gripper_open(self, cfg: CellConfig) -> None:
        result = self._request(proto.OP_GRIPPER, {"action": "open"})
        if not result.get("ok"):
            raise RuntimeError(f"gripper_open failed: {result.get('error')}")
        with self._lock:
            if self._held_block:
                pose = cfg.kin.forward(self._last_joints)
                self._block_positions[self._held_block] = {
                    "x": round(pose.x, 1),
                    "y": round(pose.y, 1),
                }
                self._held_block = None
            self._last_gripper = "open"

    def gripper_close(self, cfg: CellConfig) -> None:
        result = self._request(proto.OP_GRIPPER, {"action": "close"})
        if not result.get("ok"):
            raise RuntimeError(f"gripper_close failed: {result.get('error')}")
        with self._lock:
            pose = cfg.kin.forward(self._last_joints)
            grabbed = self._find_block_at(pose.x, pose.y, cfg.block_size)
            if grabbed:
                self._held_block = grabbed
                self._last_gripper = "gripping"
            else:
                self._last_gripper = "closed"

    def gripper_state(self) -> str:
        with self._lock:
            return self._last_gripper

    def _find_block_at(self, x: float, y: float, block_size: float) -> str | None:
        grab_radius = block_size * 0.8
        for bid, pos in self._block_positions.items():
            if bid == self._held_block:
                continue
            dx = pos["x"] - x
            dy = pos["y"] - y
            if (dx * dx + dy * dy) < grab_radius * grab_radius:
                return bid
        return None

    # -- vision -------------------------------------------------------------- #

    def detect_blocks(self, cfg: CellConfig) -> list[dict]:
        """Publish a "detect" command and relay the laptop's camera result.

        Falls back to the last-known cached positions (seeded from
        cell.yaml) if the laptop agent doesn't reply -- keeps this backend
        usable for planning even before the Pi-side camera relay exists.
        """
        try:
            result = self._request(proto.OP_DETECT, {})
        except Exception:
            log.exception("detect_blocks: MQTT relay failed, falling back to cache")
            result = None

        if result is not None and result.get("ok") and "blocks" in result:
            blocks = result["blocks"]
            with self._lock:
                for b in blocks:
                    self._block_positions[b["id"]] = {"x": b["x"], "y": b["y"]}
            return blocks

        with self._lock:
            results = []
            for block in cfg.blocks.values():
                if block.id == self._held_block:
                    continue
                pos = self._block_positions.get(block.id)
                if pos:
                    results.append({
                        "id": block.id,
                        "color": block.color,
                        "label": block.label,
                        "x": pos["x"],
                        "y": pos["y"],
                        "confidence": 1.0,
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
        with self._lock:
            self._held_block = None
            self._last_gripper = "open"
            for block in cfg.blocks.values():
                self._block_positions[block.id] = {
                    "x": block.start_x,
                    "y": block.start_y,
                }
                self._block_slots[block.id] = None

    # -- safety ------------------------------------------------------------ #

    def estop(self) -> None:
        self._estopped = True
        try:
            self._request(proto.OP_ESTOP, {}, timeout=5.0)
        except Exception:
            log.exception("estop relay failed")

    def clear_estop(self) -> None:
        self._estopped = False
        try:
            self._request(proto.OP_CLEAR_ESTOP, {}, timeout=5.0)
        except Exception:
            log.exception("clear_estop relay failed")
