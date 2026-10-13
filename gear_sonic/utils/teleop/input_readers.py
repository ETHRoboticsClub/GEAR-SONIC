"""Input source readers for body tracking data.

PicoReader         -- pulls data from XRoboToolkit SDK (Pico headset).
IsaacTeleopReader  -- in-process IsaacTeleop / CloudXR DeviceIO session.
"""

from dataclasses import dataclass
import logging
import threading
import time
from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation

logger = logging.getLogger(__name__)

try:
    import xrobotoolkit_sdk as xrt
except ImportError:
    xrt = None


class PicoReader:
    """Background reader that pulls Pico/XRT data and computes dt/FPS."""

    STALE_TIMEOUT = 5.0

    def __init__(self, max_queue_size: int = 15):
        del max_queue_size
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._fps_ema = 0.0
        self._last_stamp_ns = None
        self._latest = None
        self._lock = threading.Lock()
        self._last_new_data_time = time.monotonic()
        self._disconnected = threading.Event()

    def start(self):
        if not self._thread.is_alive():
            self._thread.start()

    def stop(self):
        self._stop.set()
        if self._thread.is_alive():
            self._thread.join(timeout=1.0)

    def get_latest(self):
        with self._lock:
            return self._latest

    @property
    def disconnected(self) -> bool:
        return self._disconnected.is_set()

    def clear_disconnect(self):
        self._disconnected.clear()
        self._last_new_data_time = time.monotonic()
        self._last_stamp_ns = None
        self._fps_ema = 0.0

    def get_timestamp_ns(self) -> int:
        if xrt is None:
            return 0
        return int(xrt.get_time_stamp_ns())

    def _run(self):
        last_report = time.time()
        while not self._stop.is_set():
            if xrt is None or not xrt.is_body_data_available():
                if (
                    time.monotonic() - self._last_new_data_time > self.STALE_TIMEOUT
                    and not self._disconnected.is_set()
                ):
                    logger.warning(
                        "[PicoReader] No new data for %.1fs, flagging disconnect",
                        self.STALE_TIMEOUT,
                    )
                    self._disconnected.set()
                time.sleep(0.001)
                continue

            stamp_ns = xrt.get_time_stamp_ns()
            prev_stamp_ns = self._last_stamp_ns
            if prev_stamp_ns is not None and stamp_ns == prev_stamp_ns:
                if (
                    time.monotonic() - self._last_new_data_time > self.STALE_TIMEOUT
                    and not self._disconnected.is_set()
                ):
                    logger.warning(
                        "[PicoReader] Timestamps stale for %.1fs, flagging disconnect",
                        self.STALE_TIMEOUT,
                    )
                    self._disconnected.set()
                time.sleep(0.000001)
                continue

            self._last_new_data_time = time.monotonic()
            if self._disconnected.is_set():
                logger.info("[PicoReader] Fresh data received, connection restored")
                self._disconnected.clear()

            device_dt = ((stamp_ns - prev_stamp_ns) * 1e-9) if prev_stamp_ns is not None else 0.0
            if device_dt > 0.0:
                inst = 1.0 / device_dt
                self._fps_ema = inst if self._fps_ema == 0.0 else (0.9 * self._fps_ema + 0.1 * inst)
            self._last_stamp_ns = stamp_ns

            try:
                body_poses = xrt.get_body_joints_pose()
                sample = {
                    "body_poses_np": np.array(body_poses),
                    "timestamp_realtime": time.time(),
                    "timestamp_monotonic": time.monotonic(),
                    "timestamp_ns": stamp_ns,
                    "dt": device_dt,
                    "fps": self._fps_ema,
                }
                with self._lock:
                    self._latest = sample

                now = time.time()
                if now - last_report >= 5.0:
                    logger.info(
                        "[PicoReader] dt_ts: %.2f ms, fps: %.2f",
                        device_dt * 1000.0,
                        self._fps_ema,
                    )
                    last_report = now
            except Exception:
                logger.exception("[PicoReader] read error")


# Full-body sources selectable for the Isaac Teleop input (TrackerVendor "body.<name>").
BODY_VENDORS = ("pico-xr", "xsens")
XSENS_DEFAULT_PORT = 9764  # MVN Studio's "Isaac Teleop" network streamer preset

# How long a device's data may stay bit-identical before it counts as no longer
# tracked. CloudXR and the body.xsens vendor keep returning the last sample after
# their source stops (MVN stops streaming, headset disconnects), while live
# tracked human motion never repeats bit-exactly between frames. 0.2 s is 12
# frames at 60 Hz (MVN's stream rate, and no headset runs slower), so a few
# dropped datagrams or polls landing between two device frames do not trip it.
INPUT_STALE_AFTER_S = 0.2

# XR_BD_body_tracking joints (core.BodyJoint).
NUM_BODY_JOINTS = 24
BODY_PELVIS, BODY_NECK = 0, 12
BODY_LEFT_WRIST, BODY_RIGHT_WRIST = 20, 21
BODY_LEFT_HAND, BODY_RIGHT_HAND = 22, 23
# GEAR-SONIC takes its wrists from LEFT_HAND/RIGHT_HAND. Xsens MVN tracks no
# fingers and sends those two as invalid copies of LEFT_WRIST/RIGHT_WRIST.
_HAND_JOINT_FROM_WRIST = ((BODY_LEFT_HAND, BODY_LEFT_WRIST), (BODY_RIGHT_HAND, BODY_RIGHT_WRIST))

# XR_EXT_hand_tracking joint index of the wrist (core.HandJoint.WRIST).
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


class _StaleGuard:
    """Tracks whether one device's data is still changing between polls."""

    def __init__(self):
        self._last: np.ndarray | None = None
        self._changed_ns = 0

    def update(self, data: np.ndarray | None, now_ns: int) -> tuple[bool, bool]:
        """Return ``(fresh, live)``: changed since the previous poll, changed recently."""
        if data is None:
            self._last = None
            return False, False
        fresh = self._last is None or not np.array_equal(data, self._last)
        if fresh:
            self._last = data.copy()
            self._changed_ns = now_ns
        return fresh, now_ns - self._changed_ns <= INPUT_STALE_AFTER_S * 1e9


def _pose7(pose: Any) -> np.ndarray:
    """``[x, y, z, qx, qy, qz, qw]`` from a schema ``Pose``."""
    p, o = pose.position, pose.orientation
    return np.array([p.x, p.y, p.z, o.x, o.y, o.z, o.w], dtype=np.float64)


def _body_joints(full_body: Any) -> tuple[np.ndarray, np.ndarray] | None:
    """``(poses (24, 7) float32 xyz+xyzw, valid (24,) bool)`` from a ``FullBodyPose`` view."""
    if full_body is None:
        return None
    joints = full_body.joints
    poses = np.concatenate(
        [np.asarray(joints.positions), np.asarray(joints.orientations)], axis=1
    ).astype(np.float32)
    valid = np.asarray(joints.is_valid).astype(bool) & np.isfinite(poses).all(axis=1)
    return poses, valid


def _fill_hand_joints_from_wrists(poses: np.ndarray, valid: np.ndarray) -> None:
    """Use a valid wrist joint for its invalid hand joint, in place."""
    for hand, wrist in _HAND_JOINT_FROM_WRIST:
        if not valid[hand] and valid[wrist]:
            poses[hand] = poses[wrist]
            valid[hand] = True


def _controller_grip(snapshot: Any) -> np.ndarray | None:
    grip = None if snapshot is None else snapshot.grip_pose
    if grip is None or not grip.is_valid:
        return None
    pose = _pose7(grip.pose)
    return pose if np.isfinite(pose).all() else None


def _hand_wrist(hand: Any) -> np.ndarray | None:
    if hand is None:
        return None
    joints = hand.joints
    if not joints.is_valid[_HAND_JOINT_WRIST]:
        return None
    pose = np.concatenate(
        [joints.positions[_HAND_JOINT_WRIST], joints.orientations[_HAND_JOINT_WRIST]]
    ).astype(np.float64)
    return pose if np.isfinite(pose).all() else None


def _controller_inputs_to_dict_side(snapshot: Any) -> dict[str, float]:
    """Project one ControllerSnapshot.inputs into the dict shape consumed by helpers."""
    inputs = snapshot.inputs
    return {
        "trigger_value": float(inputs.trigger_value),
        "squeeze_value": float(inputs.squeeze_value),
        "thumbstick_x": float(inputs.thumbstick_x),
        "thumbstick_y": float(inputs.thumbstick_y),
        "thumbstick_click": float(inputs.thumbstick_click),
        "primary_click": float(inputs.primary_click),
        "secondary_click": float(inputs.secondary_click),
    }


def _build_controller_dict(sides: dict[str, dict[str, float]]) -> dict[str, Any] | None:
    """Merge per-side inputs into the controller dict schema that
    ``pico_manager_thread_server`` consumes (left/right trigger, squeeze,
    thumbstick, click, primary/secondary click)."""
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


class IsaacTeleopReader:
    """Background reader using the in-process IsaacTeleop / CloudXR DeviceIO session.

    Drop-in alternative to ``PicoReader`` — same ``get_latest()`` /
    ``get_controller_data()`` contract, plus ``get_tracking()``,
    ``get_input_timestamp_ns()`` and ``get_three_point_body()``. Any subset of
    body tracker, headset, controllers and hand tracking may be present;
    nothing waits for a particular device.
    """

    def __init__(
        self,
        max_queue_size: int = 15,
        poll_hz: float = 90.0,
        client: Any = None,
        **client_kwargs: Any,
    ):
        del max_queue_size

        if client is None:
            try:
                from gear_sonic.utils.teleop.isaac_teleop_client import IsaacTeleopClient
            except ImportError as e:
                raise RuntimeError(
                    "isaaccapture is required for --input-source isaac-teleop but was not "
                    "found. Install via install_scripts/install_pico.sh, which runs:\n"
                    "  uv pip install 'isaaccapture[cloudxr]==1.7.6rc1' --prerelease=allow "
                    "--extra-index-url https://pypi.nvidia.com"
                ) from e
            client = IsaacTeleopClient(**client_kwargs)
        self._client = client
        self._period = 1.0 / max(1.0, float(poll_hz))

        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._lock = threading.Lock()
        self._latest: dict[str, Any] | None = None
        self._latest_controller: dict[str, Any] | None = None
        self._tracking = Tracking()
        self._input_timestamp_ns = 0
        self._three_point: tuple[str, Any] | None = None
        self._fps_ema = 0.0
        self._last_stamp_ns: int | None = None
        self._last_report = time.monotonic()

        self._body_guard = _StaleGuard()
        self._head_guard = _StaleGuard()
        self._controller_guards = {"left": _StaleGuard(), "right": _StaleGuard()}
        self._hand_guards = {"left": _StaleGuard(), "right": _StaleGuard()}

    def start(self) -> None:
        self._client.start_runtime()
        if not self._thread.is_alive():
            self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread.is_alive():
            self._thread.join(timeout=1.0)
        try:
            self._client.close()
        except Exception:
            logger.exception("Failed to close IsaacTeleopClient cleanly")

    def get_latest(self) -> dict[str, Any] | None:
        """Latest fresh, tracked body sample (body-only consumers gate on its timestamp)."""
        with self._lock:
            return self._latest

    def get_controller_data(self) -> dict[str, Any] | None:
        """Inputs of the controllers that are currently tracked and live."""
        with self._lock:
            return self._latest_controller

    def get_tracking(self) -> Tracking:
        with self._lock:
            return self._tracking

    def get_timestamp_ns(self) -> int:
        with self._lock:
            sample = self._latest
        return int(sample["timestamp_ns"]) if sample else 0

    def get_input_timestamp_ns(self) -> int:
        """Advances whenever any fresh, tracked input arrived."""
        with self._lock:
            return self._input_timestamp_ns

    def get_three_point_body(self) -> tuple[str, np.ndarray] | None:
        """``(source, (24, 7) XR_BD array)`` for the 3-point pose, or None.

        ``source`` is ``"body"`` while a body tracker is tracked, else
        ``"headset"`` (synthesized from head + controllers or hands).
        """
        with self._lock:
            three_point = self._three_point
        if three_point is None:
            return None
        source, data = three_point
        if source == "body":
            return source, data
        return source, synthesize_three_point_body(*data)

    def _run(self) -> None:
        last_error = None
        while not self._stop.is_set():
            if not self._client.has_session:
                try:
                    self._client.open_session(self._stop)
                except Exception as e:
                    if str(e) != last_error:
                        logger.exception("[IsaacTeleopReader] Opening the session failed; retrying")
                        last_error = str(e)
                    self._stop.wait(1.0)
                    continue
                last_error = None
            try:
                raw = self._client._get_tracker_data()  # noqa: SLF001 — internal API by design
            except Exception:
                logger.exception("[IsaacTeleopReader] DeviceIO update failed")
                raw = None
            self._update(raw, time.monotonic_ns())
            self._stop.wait(self._period)

    def _update(self, raw: dict[str, Any] | None, now_ns: int) -> None:
        """Recompute tracking, inputs and samples from one DeviceIO poll."""
        raw = raw or {}
        fresh_inputs = False

        body = _body_joints(raw.get("full_body"))
        body_fresh, body_live = self._body_guard.update(None if body is None else body[0], now_ns)
        body_tracked = False
        if body is not None:
            body_poses, valid = body
            _fill_hand_joints_from_wrists(body_poses, valid)
            # POSE mode retargets every joint, so every joint must be valid.
            body_tracked = body_live and bool(valid.all())
            fresh_inputs |= body_tracked and body_fresh

        head = raw.get("head")
        head_pose = _pose7(head.pose) if head is not None and head.is_tracked else None
        head_fresh, head_live = self._head_guard.update(head_pose, now_ns)
        head_tracked = head_pose is not None and head_live

        grips, controller_inputs, controllers_fresh = {}, {}, False
        for side, guard in self._controller_guards.items():
            snapshot = raw.get(f"{side}_controller")
            grip = _controller_grip(snapshot)
            inputs = None if grip is None else _controller_inputs_to_dict_side(snapshot)
            data = None if grip is None else np.concatenate([grip, list(inputs.values())])
            fresh, live = guard.update(data, now_ns)
            if grip is not None and live:
                grips[side] = grip
                controller_inputs[side] = inputs
                controllers_fresh |= fresh
        controllers = len(grips) == 2

        wrists, hands_fresh = {}, False
        for side, guard in self._hand_guards.items():
            wrist = _hand_wrist(raw.get(f"{side}_hand"))
            fresh, live = guard.update(wrist, now_ns)
            if wrist is not None and live:
                wrists[side] = wrist
                hands_fresh |= fresh
        hands = len(wrists) == 2

        fresh_inputs |= head_tracked and head_fresh
        fresh_inputs |= controllers and controllers_fresh
        fresh_inputs |= hands and hands_fresh

        if body_tracked:
            three_point = ("body", body_poses)
        elif head_tracked and controllers:
            three_point = ("headset", (head_pose, grips["left"], grips["right"], "grip"))
        elif head_tracked and hands:
            three_point = ("headset", (head_pose, wrists["left"], wrists["right"], "hand"))
        else:
            three_point = None

        sample = None
        if body_tracked and body_fresh:
            prev_stamp_ns = self._last_stamp_ns
            device_dt = ((now_ns - prev_stamp_ns) * 1e-9) if prev_stamp_ns is not None else 0.0
            if device_dt > 0.0:
                inst = 1.0 / device_dt
                self._fps_ema = inst if self._fps_ema == 0.0 else (0.9 * self._fps_ema + 0.1 * inst)
            self._last_stamp_ns = now_ns
            sample = {
                "body_poses_np": body_poses,
                "timestamp_realtime": time.time(),
                "timestamp_monotonic": now_ns * 1e-9,
                "timestamp_ns": now_ns,
                "dt": device_dt,
                "fps": self._fps_ema,
            }

        tracking = Tracking(
            body=body_tracked, head=head_tracked, controllers=controllers, hands=hands
        )
        with self._lock:
            if tracking != self._tracking:
                logger.info("[IsaacTeleopReader] Tracking: %s", tracking)
            self._tracking = tracking
            self._latest_controller = _build_controller_dict(controller_inputs)
            self._three_point = three_point
            if fresh_inputs:
                self._input_timestamp_ns = now_ns
            if sample is not None:
                self._latest = sample

        if sample is not None and now_ns * 1e-9 - self._last_report >= 5.0:
            logger.info(
                "[IsaacTeleopReader] dt: %.2f ms, fps: %.2f", sample["dt"] * 1000.0, self._fps_ema
            )
            self._last_report = now_ns * 1e-9
