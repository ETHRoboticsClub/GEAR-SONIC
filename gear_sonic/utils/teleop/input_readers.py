"""The operator's tracking, as the robot's teleop module reports it.

The module (ETHRC's ``ethrc_teleop``) reads the devices; this server never does.
The robot-side bridge relays each of the module's tracking frames on the
feedback socket as the topic ``tracking`` followed by msgpack::

    {"sequence": int, "sample_time_ns": int,
     "head": [x, y, z, qx, qy, qz, qw] | None,
     "left_controller" / "right_controller": {"grip": [7], "aim": [7] | None,
         "primary", "secondary", "menu", "stick_click": bool,
         "trigger", "squeeze", "stick_x", "stick_y": float} | None,
     "left_hand" / "right_hand": {"positions": [78], "rotations": [104]} | None,
     "body": {"positions": [72], "rotations": [96]} | None}

A device is present only while the module judges it tracked (finite, unit
quaternions, still changing). Poses are in the module's operator space, REP-103
(x forward, y left, z up); ``TrackingReader`` converts them to the OpenXR stage
convention (x right, y up, z backward) the retargeting below was written in.
"""

from dataclasses import dataclass
import logging
import threading
import time
from typing import Any

import msgpack
import numpy as np
from scipy.spatial.transform import Rotation
import zmq

logger = logging.getLogger(__name__)

TRACKING_TOPIC = b"tracking"

# Frames arrive at the module's poll rate (60-90 Hz); without one for this long the
# module or the bridge stopped, and nothing counts as tracked. 12 frames at 60 Hz.
INPUT_STALE_AFTER_S = 0.2

# Runtimes normalize quaternions in float32 (error ~1e-7); a larger deviation is a
# corrupt pose, which the module never sends as tracked.
QUAT_NORM_TOLERANCE = 1e-3

# Rows: OpenXR stage axes in operator space; operator = C @ openxr, openxr = C.T @ operator.
OPERATOR_FROM_OPENXR = np.array([[0.0, 0.0, -1.0], [-1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
_OPENXR_FROM_OPERATOR = Rotation.from_matrix(OPERATOR_FROM_OPENXR.T)

# XR_BD_body_tracking joints (core.BodyJoint).
NUM_BODY_JOINTS = 24
BODY_PELVIS, BODY_NECK = 0, 12
BODY_LEFT_WRIST, BODY_RIGHT_WRIST = 20, 21
BODY_LEFT_HAND, BODY_RIGHT_HAND = 22, 23

# XR_EXT_hand_tracking joints; the wrist's index (core.HandJoint.WRIST).
NUM_HAND_JOINTS = 26
_HAND_JOINT_WRIST = 1

# Headset three-point synthesis (no body tracker). Orientations of device frames
# are mapped onto the XR_BD joint frame of the same hand by right-multiplication,
# q_bd = q_device * R. XR_BD joint frames coincide with the reference-space axes
# when the operator stands in T-pose facing -Z; that is the frame OFFSETS in
# pico_manager_thread_server.py turn into the G1 zero-pose wrist frames.
_DEVICE_TO_BD_WRIST = {
    # OpenXR grip pose: +X out of the left palm / into the right palm, -Z through the closed fist toward the thumb.
    ("grip", "left"): Rotation.from_euler("z", -90, degrees=True),
    ("grip", "right"): Rotation.from_euler("z", 90, degrees=True),
    # XR_EXT_hand_tracking wrist joint: -Z toward the fingertips, +Y out of the back of the hand.
    ("hand", "left"): Rotation.from_euler("y", -90, degrees=True),
    ("hand", "right"): Rotation.from_euler("y", 90, degrees=True),
}
# The robot's third point is torso_link + 0.35 m, not the head. The pelvis is
# placed this far below the head; ThreePointPose calibrates wrist positions
# against G1 FK, so the value only has to be a plausible standing eye-to-hip height.
HEAD_TO_PELVIS_HEIGHT_M = 0.7
_OPENXR_UP = np.array([0.0, 1.0, 0.0])


@dataclass(frozen=True)
class Tracking:
    """Which input classes currently deliver fresh, tracked data."""

    body: bool = False
    head: bool = False
    controllers: bool = False
    hands: bool = False

    def as_array(self) -> np.ndarray:
        """``uint8[4]`` in manager_state order BODY, HEAD, CONTROLLERS, HANDS."""
        return np.array([self.body, self.head, self.controllers, self.hands], dtype=np.uint8)


def _build_controller_dict(sides: dict[str, dict[str, float]]) -> dict[str, Any] | None:
    """Merge per-side inputs into the controller dict schema that
    ``pico_manager_thread_server`` consumes (left/right trigger, squeeze,
    thumbstick, click, primary/secondary and menu click)."""
    if not sides:
        return None
    out: dict[str, Any] = {}
    for side, inp in sides.items():
        out[f"{side}_trigger_value"] = inp["trigger_value"]
        out[f"{side}_squeeze_value"] = inp["squeeze_value"]
        out[f"{side}_thumbstick"] = [inp["thumbstick_x"], inp["thumbstick_y"]]
        out[f"{side}_thumbstick_click"] = inp["thumbstick_click"]
        out[f"{side}_primary_click"] = inp["primary_click"]
        out[f"{side}_secondary_click"] = inp["secondary_click"]
        out[f"{side}_menu_click"] = inp["menu_click"]
    return out


def synthesize_three_point_body(
    head: np.ndarray, left_wrist: np.ndarray, right_wrist: np.ndarray, wrist_source: str
) -> np.ndarray:
    """Build a (24, 7) XR_BD body array from a headset and two wrist poses.

    Only PELVIS, NECK, LEFT_HAND and RIGHT_HAND are filled (other joints are
    identity at the origin), which is all ``_process_3pt_pose`` reads. Inputs
    are ``[x, y, z, qx, qy, qz, qw]`` in the OpenXR reference space;
    ``wrist_source`` is ``"grip"`` (controllers) or ``"hand"`` (hand tracking).
    The pelvis keeps a world-fixed heading; the neck takes only the head's yaw,
    so looking down does not bend the robot's torso.
    """
    body = np.zeros((NUM_BODY_JOINTS, 7), dtype=np.float32)
    body[:, 6] = 1.0

    body[BODY_PELVIS, :3] = head[:3] - HEAD_TO_PELVIS_HEIGHT_M * _OPENXR_UP
    # Heading of the view direction: yaw of the intrinsic yaw-pitch-roll (Y-X-Z) angles.
    yaw = Rotation.from_quat(head[3:]).as_euler("YXZ")[0]
    body[BODY_NECK, :3] = head[:3]
    body[BODY_NECK, 3:] = Rotation.from_euler("y", yaw).as_quat()

    for joint, side, wrist in (
        (BODY_LEFT_HAND, "left", left_wrist),
        (BODY_RIGHT_HAND, "right", right_wrist),
    ):
        body[joint, :3] = wrist[:3]
        rot = Rotation.from_quat(wrist[3:]) * _DEVICE_TO_BD_WRIST[(wrist_source, side)]
        body[joint, 3:] = rot.as_quat()
    return body


def to_openxr(positions: np.ndarray, rotations: np.ndarray) -> np.ndarray:
    """``(N, 7)`` OpenXR stage poses ``[x, y, z, qx, qy, qz, qw]`` from ``(N, 3)``
    operator-space positions and ``(N, 4)`` xyzw rotations. Device-local axes are
    unchanged; only the reference space converts. Raises ValueError for a
    non-finite value or a quaternion that is not unit length."""
    if not (np.isfinite(positions).all() and np.isfinite(rotations).all()):
        raise ValueError("non-finite pose")
    if (np.abs(np.linalg.norm(rotations, axis=1) - 1.0) > QUAT_NORM_TOLERANCE).any():
        raise ValueError("quaternion is not unit length")
    xyz = positions @ OPERATOR_FROM_OPENXR  # row vectors: C.T @ p
    quat = (_OPENXR_FROM_OPERATOR * Rotation.from_quat(rotations)).as_quat()
    return np.concatenate([xyz, quat], axis=1)


def _pose(value: Any) -> np.ndarray:
    return to_openxr(np.asarray(value[:3], float)[None], np.asarray(value[3:7], float)[None])[0]


def _joints(value: dict, count: int) -> np.ndarray:
    positions = np.asarray(value["positions"], float).reshape(count, 3)
    rotations = np.asarray(value["rotations"], float).reshape(count, 4)
    return to_openxr(positions, rotations)


def _controller_inputs(value: dict) -> dict[str, float]:
    return {
        "trigger_value": float(value["trigger"]),
        "squeeze_value": float(value["squeeze"]),
        "thumbstick_x": float(value["stick_x"]),
        "thumbstick_y": float(value["stick_y"]),
        "thumbstick_click": float(value["stick_click"]),
        "primary_click": float(value["primary"]),
        "secondary_click": float(value["secondary"]),
        "menu_click": float(value["menu"]),
    }


class TrackingReader:
    """The newest tracking frame the bridge relayed, behind the reader interface
    the streamers use: ``get_latest()`` (body samples), ``get_controller_data()``,
    ``get_three_point_body()``, ``get_tracking()``, ``get_input_timestamp_ns()``.

    ``_update`` takes one decoded frame; ``start`` runs a thread that feeds it from
    a SUB socket on the feedback port. A frame that does not decode or has the
    wrong shape is dropped with a warning.
    """

    def __init__(
        self, host: str | None = None, port: int | None = None, context=None, clock=time.monotonic_ns
    ):
        self._address = None if host is None else f"tcp://{host}:{port}"
        self._context = context
        self._clock = clock
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()
        self._received_ns: int | None = None
        self._latest: dict[str, Any] | None = None
        self._controllers: dict[str, Any] | None = None
        self._tracking = Tracking()
        self._three_point: tuple[str, Any] | None = None
        self._input_timestamp_ns = 0
        self._last_body_ns: int | None = None
        self._fps_ema = 0.0

    def start(self) -> None:
        if self._address is None or self._thread is not None:
            return
        self._thread = threading.Thread(target=self._run, name="tracking", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)

    def _run(self) -> None:
        socket = (self._context or zmq.Context.instance()).socket(zmq.SUB)
        socket.setsockopt(zmq.SUBSCRIBE, TRACKING_TOPIC)
        socket.connect(self._address)
        try:
            while not self._stop.is_set():
                if not socket.poll(timeout=100):
                    continue
                message = socket.recv()
                try:
                    frame = msgpack.unpackb(message[len(TRACKING_TOPIC) :], raw=False)
                except Exception as e:
                    logger.warning("[TrackingReader] undecodable frame dropped (%s)", e)
                    continue
                self._update(frame, self._clock())
        finally:
            socket.close(linger=0)

    def _live(self) -> bool:
        return (
            self._received_ns is not None
            and self._clock() - self._received_ns <= INPUT_STALE_AFTER_S * 1e9
        )

    def get_latest(self) -> dict[str, Any] | None:
        """Latest tracked body sample (body-only consumers gate on its timestamp)."""
        with self._lock:
            return self._latest

    def get_controller_data(self) -> dict[str, Any] | None:
        with self._lock:
            return self._controllers if self._live() else None

    def get_tracking(self) -> Tracking:
        with self._lock:
            return self._tracking if self._live() else Tracking()

    def get_timestamp_ns(self) -> int:
        with self._lock:
            sample = self._latest
        return int(sample["timestamp_ns"]) if sample else 0

    def get_input_timestamp_ns(self) -> int:
        """Advances with every frame that tracks anything."""
        with self._lock:
            return self._input_timestamp_ns

    def get_three_point_body(self) -> tuple[str, np.ndarray] | None:
        """``(source, (24, 7) XR_BD array)`` for the 3-point pose, or None.

        ``source`` is ``"body"`` while a body is tracked, else ``"headset"``
        (synthesized from head + controllers or hands).
        """
        with self._lock:
            three_point = self._three_point if self._live() else None
        if three_point is None:
            return None
        source, data = three_point
        if source == "body":
            return source, data
        return source, synthesize_three_point_body(*data)

    def _update(self, frame: Any, now_ns: int) -> None:
        try:
            sample_ns = int(frame["sample_time_ns"])
            head = None if frame.get("head") is None else _pose(frame["head"])
            grips, inputs = {}, {}
            for side in ("left", "right"):
                controller = frame.get(f"{side}_controller")
                if controller is not None:
                    grips[side] = _pose(controller["grip"])
                    inputs[side] = _controller_inputs(controller)
            wrists = {
                side: _joints(frame[f"{side}_hand"], NUM_HAND_JOINTS)[_HAND_JOINT_WRIST]
                for side in ("left", "right")
                if frame.get(f"{side}_hand") is not None
            }
            body = None
            if frame.get("body") is not None:
                body = _joints(frame["body"], NUM_BODY_JOINTS).astype(np.float32)
        except (KeyError, TypeError, ValueError, IndexError) as e:
            logger.warning("[TrackingReader] malformed frame dropped (%s)", e)
            return

        controllers, hands = len(grips) == 2, len(wrists) == 2
        tracking = Tracking(
            body=body is not None, head=head is not None, controllers=controllers, hands=hands
        )
        if body is not None:
            three_point = ("body", body)
        elif head is not None and controllers:
            three_point = ("headset", (head, grips["left"], grips["right"], "grip"))
        elif head is not None and hands:
            three_point = ("headset", (head, wrists["left"], wrists["right"], "hand"))
        else:
            three_point = None

        sample = None
        if body is not None:
            dt = 0.0 if self._last_body_ns is None else (sample_ns - self._last_body_ns) * 1e-9
            if dt > 0.0:
                self._fps_ema = 1.0 / dt if self._fps_ema == 0.0 else 0.9 * self._fps_ema + 0.1 / dt
            self._last_body_ns = sample_ns
            sample = {
                "body_poses_np": body,
                "timestamp_realtime": time.time(),
                "timestamp_monotonic": now_ns * 1e-9,
                "timestamp_ns": sample_ns,
                "dt": dt,
                "fps": self._fps_ema,
            }

        with self._lock:
            if tracking != self._tracking:
                logger.info("[TrackingReader] Tracking: %s", tracking)
            self._received_ns = now_ns
            self._tracking = tracking
            self._controllers = _build_controller_dict(inputs)
            self._three_point = three_point
            if tracking != Tracking():
                self._input_timestamp_ns = sample_ns
            if sample is not None:
                self._latest = sample
