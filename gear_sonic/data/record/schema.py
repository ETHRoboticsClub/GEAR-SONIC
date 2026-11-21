# ruff: noqa: E501  -- the CALLS block is an exact contract and stays one line per signature.
"""Frozen schema for G1 teleop episode recording.

Other workstreams import this module and do not edit it. Types below are the
on-disk and in-process contract. Functions named in CALLS are implemented by
later workstreams; their signatures live here so the contract has one source.

CONSTANTS
    FPS
        Dataset and builder grid rate, in hertz. Row times are
        ``t_k = t0 + (k * 1_000_000_000 + fps // 2) // fps``.
    TOKEN_DIM
        Motion-token width. ``action.motion_token`` and ``StateSample.token``.
    PICO_DIM
        Width of ``TrackingSample.vector`` and ``observation.pico``.
    BODY_DIM
        MuJoCo body-joint width (legs, waist, arms). Not ``state_dim``.
    HAND_DIM
        Dex3 motors on one hand.
    IMAGE_HEIGHT, IMAGE_WIDTH
        Height and width stored in the LeRobot video feature. These are the
        build defaults (360x640). Raw ZED capture stays 1280x720 per eye
        (the left half of a 2560x720 YUYV frame) and is scaled at encode time.
    REPO_ID
        LeRobot repository id for this dataset family.
    ROBOT_TYPE
        LeRobot robot type string.
    STATE_INTERP_MAX_NS
        Largest gap, in nanoseconds, between bracketing robot-state samples
        for which joint positions, joint velocities, and base angular velocity
        may be linearly interpolated and ``base_quat`` may be slerped.
        Absolute window (60 ms). It does not rescale if ``FPS`` changes.
        The row stays valid and sets ``STATE_INTERP``. A longer gap holds the
        last finite state and clears ``quality.valid``.
    CAMERA_STALE_NS
        Camera age above which the row sets ``CAMERA_STALE`` and
        ``quality.valid = 0``. Absolute window (100 ms).
    PICO_INTERP_MAX_NS
        Largest gap, in nanoseconds, between bracketing tracking samples for
        which poses are interpolated (lerp xyz, slerp quat). Buttons and
        tracked flags come from the latest sample at or before ``t_k``.
        Absolute window (150 ms). Sets ``PICO_INTERP``. A longer gap holds the
        last vector and sets ``PICO_STALE``. Pico flags do not clear
        ``quality.valid``.
    ACTION_MAX_AGE_NS
        One grid period at ``FPS``, ``1_000_000_000 // FPS``, same value as
        ``period_ns(FPS)``. Action, hand actions, and the motion token are
        copied from the latest ``g1_debug`` sample at or before ``t_k`` only
        when its age is within this bound and ``token`` is not None. They are
        never interpolated.
    STATE_INTERP, ACTION_MISSING, CAMERA_STALE, PICO_INTERP, PICO_STALE
        Bits of ``quality.flags`` / ``Row.flags``. Distinct powers of two:
        ``1 << 0`` through ``1 << 4`` in that order.
    RECORD_MAGIC, RECORD_VERSION
        Raw-log framing. Little-endian layout is
        ``magic | uint32 version | uint64 stamp_ns | uint64 recv_ns |
        uint32 payload_len | payload``.
    PICO_NAMES
        Length ``PICO_DIM`` (209). ``TrackingSample.vector`` and
        ``observation.pico`` use this order, which flattens
        ``pico_sim_bridge.sample``:

        * head: ``head_tracked``, then xyz, then xyzw (8).
        * ``left_controller``, then ``right_controller`` (16 each):
          ``tracked``, grip xyz, grip xyzw, ``trigger``, ``squeeze``,
          ``stick_x``, ``stick_y``, ``primary``, ``secondary``, ``menu``,
          ``stick_click``.
        * ``body_tracked``, then ``body_pos_{i}_{x,y,z}`` for ``i`` in 0..23,
          then ``body_rot_{i}_{qx,qy,qz,qw}`` for ``i`` in 0..23.

        Quaternion component order is xyzw, matching
        ``scipy.spatial.transform.Rotation.as_quat`` in ``pico_sim_bridge``.
        Tracked flags and buttons are 0 or 1 stored as floats. A device whose
        sample value is None is still valid data: its tracked flag is 0 and
        its pose components are NaN. Controller axes and buttons are 0 when
        that controller is None. A missing body sets ``body_tracked`` to 0 and
        every body pose component to NaN. NaN is allowed in the vector.

    ``quality.valid`` and ``quality.flags`` are LeRobot ``int64`` columns of
    shape ``(1,)``. The recording plan's int8 / int32 widths are the logical
    widths; LeRobot datasets reliably store int64, so the feature dict uses
    int64. ``Row.valid`` is 1 only when ``ACTION_MISSING`` and ``CAMERA_STALE``
    are unset and the robot state is usable (not a long state gap). Pico flags
    alone do not clear it. ``Row.token`` is 64 zeros only when
    ``ACTION_MISSING`` is set, so the column keeps a shape; ``quality.valid``
    is 0 in that case.

CALLS
    - zmq_sources.subscribe_state(host="127.0.0.1", port=5557) -> iterator of (recv_ns: int, payload: bytes). SUB connect only. Filter g1_debug. Ignore other topics. Never bind.
    - zmq_sources.subscribe_tracking(host="127.0.0.1", port=5558) -> iterator of (recv_ns: int, payload: bytes). Filter tracking only. Ignore forwarded g1_debug.
    - zmq_sources.decode_g1_debug(payload: bytes) -> StateSample | None. payload is msgpack map WITHOUT the topic prefix. Stamp is NOT inside this function; caller sets stamp_ns to recv time. Return None if body arrays are wrong length or non-finite. token None if length != 64. Do not zero-fill inside the decoder.
    - zmq_sources.decode_tracking(payload: bytes) -> TrackingSample | None. payload is msgpack of pico_sim_bridge.sample dict. stamp_ns = sample_time_ns. None device -> tracked 0 and NaN poses, still valid. Wrong body lengths -> valid False / return None.
    - zed_v4l2.left_eye_rgb(yuyv: bytes | memoryview, width=2560, height=720) -> numpy uint8 (height, width//2, 3) RGB. YUYV is Y0 U Y1 V. Left eye is the first width//2 pixels of each row. Use only Y and the chroma that belongs to those pixels (standard YUYV: pixel 2*i uses Y at byte 4*i? Wait: YUYV packs two pixels in 4 bytes: Y0,U,Y1,V for pixels 0 and 1. Left eye is pixels 0..1279, which is the first 1280*2 = 2560 bytes of each 2560*2-byte row. Yes: each row is 2560 pixels * 2 bytes = 5120 bytes. Left eye is the first 2560 bytes of each row (1280 pixels). Crop rows at byte 2560 of the row, not byte 2560 of the frame.
    - zed_v4l2.iter_frames(device: str) -> iterator of objects with stamp_ns, sequence, yuyv bytes, error: bool. Negotiates 2560x720 YUYV 30fps 4 buffers. Drops nothing itself; sets error True when bytesused is short or V4L2 error flag. Requires TIMESTAMP_MONOTONIC else raises. Tests must not need a device; this function is only called live.
    - session_log.SessionWriter(root: Path). append_state/append_tracking/append_jpeg(stamp, sequence, jpeg_bytes, error=False)/mark_event/close. Bounded queues. append from capture threads must not block on disk: if queue full, increment drop count and return. health() dict. Reader SessionReader tolerates truncated tail.
    - build.build_rows(state, tracking, images, *, fps, t0_ns, layout) -> list[Row]
      state: list[StateSample] sorted by stamp
      tracking: list[TrackingSample]
      images: list[ImageRef]  (valid frames only)
      Rules:
      * t_k = t0 + (k*1_000_000_000 + fps//2)//fps
      * Robot state: latest sample with stamp<=t_k. If its age <= one period, use it. Else if bracketing samples are <= STATE_INTERP_MAX_NS apart, linearly interpolate joints/vel/ang vel and slerp base_quat, set STATE_INTERP, row stays valid. Longer gap: row invalid (hold last finite state so shapes exist).
      * Action, hand actions, token: NEVER interpolate. Use the latest g1_debug sample with stamp<=t_k only if age <= action_max_age_ns(fps) (1.5 periods) AND token is not None. Otherwise hold last value, set ACTION_MISSING, valid=0.
      * Camera: latest ImageRef with stamp<=t_k (never a future frame). Record age. Age > CAMERA_STALE_NS => CAMERA_STALE and valid=0. No image at all => CAMERA_STALE valid=0 image_index=-1.
      * PICO: interpolate poses between bracketing samples if gap <= PICO_INTERP_MAX_NS (lerp xyz, slerp quat, buttons/tracked flags from the latest sample at or before t_k). Set PICO_INTERP. If gap larger, hold last and set PICO_STALE. PICO flags alone do NOT clear valid.
      * A None-device tracking sample is valid data (tracked 0, NaNs) and can be a bracket endpoint; do not treat it as missing.
      * quality.valid is 1 only when ACTION_MISSING and CAMERA_STALE are unset and state is usable (not a long state gap).
    - writer.write_episode(rows, frames_rgb, root, *, fps, task, t0_ns, metadata, image_hw) -> path. LeRobot v3 LeRobotDataset.create vcodec h264. Also writes meta/grid_clock.json and meta/quality/episode_NNNNNN.json.
    - convert.segment_invalid(rows_valid_mask, *, min_segment_frames) -> list of (start, end) half-open index ranges of contiguous valid rows, dropping ranges shorter than min_segment_frames.

The g1_debug msgpack map (topic prefix already stripped) carries ``body_q``,
``body_dq``, and ``last_action`` in ``body_joint_names`` order; hand vectors
in the supplemental hand-actuator order recorded in the layout json; ``token_state``
(length 64, or missing / empty / wrong length -> ``StateSample.token is None``);
``base_quat`` wxyz; ``base_ang_vel``; and ``index``. ``decode_g1_debug`` does
not read a timestamp. The caller sets ``StateSample.stamp_ns`` to receive time.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import struct

# Dataset grid. Row timestamps use this rate unless build_rows is passed another fps.
FPS = 50

# Motion token published as g1_debug token_state. Wrong length stays None in StateSample.
TOKEN_DIM = 64

# Flattened pico_sim_bridge.sample vector. Asserted against PICO_NAMES below.
PICO_DIM = 209

# MuJoCo body joints: two 6-dof legs, 3-dof waist, two 7-dof arms.
BODY_DIM = 29

# Unitree Dex3 motors on one hand.
HAND_DIM = 7

# Encoded ego video. Raw capture remains 1280x720 per eye.
IMAGE_HEIGHT = 360
IMAGE_WIDTH = 640

# LeRobot dataset identity for episodes written by writer.write_episode.
REPO_ID = "unitree/g1_pico_sonic"
ROBOT_TYPE = "g1"

# Absolute interp / stale windows, in nanoseconds. Not rescaled with FPS.
STATE_INTERP_MAX_NS = 60_000_000
CAMERA_STALE_NS = 100_000_000
PICO_INTERP_MAX_NS = 150_000_000

# quality.flags bits. Logical flags; the column dtype is int64 (see features).
STATE_INTERP = 1 << 0
ACTION_MISSING = 1 << 1
CAMERA_STALE = 1 << 2
PICO_INTERP = 1 << 3
PICO_STALE = 1 << 4

# Raw session-log framing. Little-endian. payload follows the 28-byte header.
RECORD_MAGIC = b"G1RS"
RECORD_VERSION = 1
_RECORD_HEADER = struct.Struct("<4sIQQI")

_LAYOUT_PATH = Path(__file__).resolve().parent / "g1_sonic_layout.json"


def period_ns(fps: int = FPS) -> int:
    """Return one control period in nanoseconds, ``1_000_000_000 // fps``."""
    if isinstance(fps, bool) or not isinstance(fps, int) or fps <= 0:
        raise ValueError(f"fps must be a positive int, got {fps!r}")
    return 1_000_000_000 // fps


def action_max_age_ns(fps: int = FPS) -> int:
    """Return how old the latest g1_debug sample may be and still supply action and token.

    The controller publishes once per period with about 1 ms of jitter, so an
    exact one-period limit marks about one grid row in ten missing. One and a
    half periods accepts that jitter and still flags a dropped message.
    """
    return period_ns(fps) * 3 // 2


ACTION_MAX_AGE_NS = action_max_age_ns(FPS)


def _pico_names() -> tuple[str, ...]:
    names: list[str] = [
        "head_tracked",
        "head_x",
        "head_y",
        "head_z",
        "head_qx",
        "head_qy",
        "head_qz",
        "head_qw",
    ]
    for side in ("left_controller", "right_controller"):
        names.append(f"{side}_tracked")
        names.extend(
            [
                f"{side}_grip_x",
                f"{side}_grip_y",
                f"{side}_grip_z",
                f"{side}_grip_qx",
                f"{side}_grip_qy",
                f"{side}_grip_qz",
                f"{side}_grip_qw",
            ]
        )
        names.extend(
            [
                f"{side}_trigger",
                f"{side}_squeeze",
                f"{side}_stick_x",
                f"{side}_stick_y",
                f"{side}_primary",
                f"{side}_secondary",
                f"{side}_menu",
                f"{side}_stick_click",
            ]
        )
    names.append("body_tracked")
    for index in range(24):
        names.extend([f"body_pos_{index}_x", f"body_pos_{index}_y", f"body_pos_{index}_z"])
    for index in range(24):
        names.extend(
            [
                f"body_rot_{index}_qx",
                f"body_rot_{index}_qy",
                f"body_rot_{index}_qz",
                f"body_rot_{index}_qw",
            ]
        )
    if len(names) != PICO_DIM:
        raise AssertionError(f"PICO_NAMES length {len(names)} != {PICO_DIM}")
    return tuple(names)


PICO_NAMES: tuple[str, ...] = _pico_names()


def _as_int(value, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{field} must be an int, got {type(value).__name__}")
    return value


def _as_bool(value, field: str) -> bool:
    if not isinstance(value, bool):
        raise TypeError(f"{field} must be a bool, got {type(value).__name__}")
    return value


def _as_floats(values, length: int, field: str) -> tuple[float, ...]:
    if isinstance(values, (str, bytes, bytearray)):
        raise TypeError(f"{field} must be a sequence of numbers")
    try:
        converted = tuple(float(item) for item in values)
    except TypeError as exc:
        raise TypeError(f"{field} must be a sequence of numbers") from exc
    if len(converted) != length:
        raise ValueError(f"{field} length {len(converted)} != {length}")
    return converted


def _as_token(values) -> tuple[float, ...] | None:
    if values is None:
        return None
    return _as_floats(values, TOKEN_DIM, "token")


def pack_record(stamp_ns: int, recv_ns: int, payload: bytes) -> bytes:
    """Frame one raw log record.

    Layout, little-endian: ``b"G1RS"``, uint32 version 1, uint64 ``stamp_ns``,
    uint64 ``recv_ns``, uint32 payload length, then ``payload``.
    """
    stamp_ns = _as_int(stamp_ns, "stamp_ns")
    recv_ns = _as_int(recv_ns, "recv_ns")
    if isinstance(payload, memoryview):
        payload = payload.tobytes()
    elif isinstance(payload, bytearray):
        payload = bytes(payload)
    elif not isinstance(payload, bytes):
        raise TypeError("payload must be bytes")
    header = _RECORD_HEADER.pack(RECORD_MAGIC, RECORD_VERSION, stamp_ns, recv_ns, len(payload))
    return header + payload


def iter_records(path):
    """Yield ``(stamp_ns, recv_ns, payload)`` from a session log.

    A truncated final record, whether the header or the payload was cut, stops
    the iterator without raising. A complete header with the wrong magic or
    version is corruption and raises ``ValueError``.
    """
    with open(path, "rb") as handle:
        header_size = _RECORD_HEADER.size
        while True:
            prefix = handle.read(header_size)
            if len(prefix) < header_size:
                return
            magic, version, stamp_ns, recv_ns, payload_len = _RECORD_HEADER.unpack(prefix)
            if magic != RECORD_MAGIC or version != RECORD_VERSION:
                raise ValueError(
                    f"bad record header at offset {handle.tell() - header_size}: "
                    f"magic={magic!r} version={version}"
                )
            payload = handle.read(payload_len)
            if len(payload) < payload_len:
                return
            yield stamp_ns, recv_ns, payload


def load_layout() -> dict:
    """Load ``g1_sonic_layout.json`` from the same directory as this module.

    Keys used by the builder and by ``features``:

    * ``joint_names``: observation.state names, robot-model order.
    * ``state_dim``: ``len(joint_names)``.
    * ``groups``: ``left_leg``, ``right_leg``, ``waist``, ``left_arm``,
      ``right_arm``, ``left_hand``, ``right_hand``. Each value is
      ``{"start", "end", "names"}`` with ``end`` exclusive.
    * ``body_joint_names``: the 29 MuJoCo body joints, g1_debug body order.
    * ``unassigned``: joint names in ``joint_names`` outside those seven groups.
      Empty when every configuration joint is assigned. See ``unassigned_note``.
    """
    with _LAYOUT_PATH.open(encoding="utf-8") as handle:
        layout = json.load(handle)
    if not isinstance(layout, dict):
        raise ValueError(f"{_LAYOUT_PATH} must contain a JSON object")
    return layout


def _feature(dtype: str, shape: tuple[int, ...], names) -> dict:
    return {"dtype": dtype, "shape": shape, "names": list(names)}


def features(image_height: int = IMAGE_HEIGHT, image_width: int = IMAGE_WIDTH) -> dict:
    """Return the LeRobot feature dict for one G1 pico/sonic episode.

    ``image_height`` and ``image_width`` override the encoded ego-view size.
    They do not change raw capture (1280x720 per eye).

    ``quality.valid`` and ``quality.flags`` are ``int64`` of shape ``(1,)``.
    The plan's int8 and int32 are logical widths; LeRobot stores int64 reliably,
    so those are the dtypes written into the dataset.
    """
    if isinstance(image_height, bool) or not isinstance(image_height, int) or image_height <= 0:
        raise ValueError(f"image_height must be a positive int, got {image_height!r}")
    if isinstance(image_width, bool) or not isinstance(image_width, int) or image_width <= 0:
        raise ValueError(f"image_width must be a positive int, got {image_width!r}")

    layout = load_layout()
    joint_names = list(layout["joint_names"])
    state_dim = int(layout["state_dim"])
    body_names = list(layout["body_joint_names"])
    if state_dim != len(joint_names):
        raise ValueError(f"layout state_dim {state_dim} != len(joint_names) {len(joint_names)}")
    if len(body_names) != BODY_DIM:
        raise ValueError(f"layout body_joint_names length {len(body_names)} != {BODY_DIM}")

    motor = [f"motor_{index}" for index in range(HAND_DIM)]
    left_motor = [f"left_hand_motor_{index}" for index in range(HAND_DIM)]
    right_motor = [f"right_hand_motor_{index}" for index in range(HAND_DIM)]
    token_names = [f"token_{index}" for index in range(TOKEN_DIM)]
    return {
        "observation.state": _feature("float32", (state_dim,), joint_names),
        "observation.projected_gravity": _feature(
            "float32", (3,), ["gravity_x", "gravity_y", "gravity_z"]
        ),
        "observation.base_quat": _feature("float32", (4,), ["w", "x", "y", "z"]),
        "observation.base_ang_vel": _feature("float32", (3,), ["x", "y", "z"]),
        "observation.joint_velocity": _feature("float32", (BODY_DIM,), body_names),
        "observation.left_hand_dq": _feature("float32", (HAND_DIM,), motor),
        "observation.right_hand_dq": _feature("float32", (HAND_DIM,), motor),
        "observation.g1_debug_index": _feature("int64", (1,), ["g1_debug_index"]),
        "observation.pico": _feature("float32", (PICO_DIM,), PICO_NAMES),
        "observation.pico_sequence": _feature("int64", (1,), ["pico_sequence"]),
        "observation.sample_age_s": _feature("float32", (3,), ["robot", "pico", "camera"]),
        "quality.valid": _feature("int64", (1,), ["valid"]),
        "quality.flags": _feature("int64", (1,), ["flags"]),
        "action": _feature("float32", (BODY_DIM,), body_names),
        "action.left_hand": _feature("float32", (HAND_DIM,), left_motor),
        "action.right_hand": _feature("float32", (HAND_DIM,), right_motor),
        "action.motion_token": _feature("float32", (TOKEN_DIM,), token_names),
        "observation.images.ego_view": {
            "dtype": "video",
            "shape": (image_height, image_width, 3),
            "names": ["height", "width", "channel"],
        },
    }


@dataclass(frozen=True)
class StateSample:
    """One decoded ``g1_debug`` sample, without the topic prefix.

    ``stamp_ns`` is receive time, assigned by the caller. ``decode_g1_debug``
    does not invent it. Vectors are tuples of float. ``token`` is None when
    ``token_state`` is missing or its length is not ``TOKEN_DIM``; this type
    does not zero-fill a short token (pass None). Body arrays that are the
    wrong length or non-finite never become a ``StateSample``; the decoder
    returns None instead.

    ``body_q``, ``body_dq``, and ``last_action`` follow ``body_joint_names``.
    Hand vectors follow the layout's ``hand_actuator_joint_names``, not the
    contiguous ``groups.left_hand`` / ``groups.right_hand`` state slices.
    ``base_quat`` is wxyz.
    """

    stamp_ns: int
    index: int
    body_q: tuple[float, ...]
    body_dq: tuple[float, ...]
    left_hand_q: tuple[float, ...]
    left_hand_dq: tuple[float, ...]
    right_hand_q: tuple[float, ...]
    right_hand_dq: tuple[float, ...]
    last_action: tuple[float, ...]
    last_left_hand_action: tuple[float, ...]
    last_right_hand_action: tuple[float, ...]
    token: tuple[float, ...] | None
    base_quat: tuple[float, ...]
    base_ang_vel: tuple[float, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "stamp_ns", _as_int(self.stamp_ns, "stamp_ns"))
        object.__setattr__(self, "index", _as_int(self.index, "index"))
        object.__setattr__(self, "body_q", _as_floats(self.body_q, BODY_DIM, "body_q"))
        object.__setattr__(self, "body_dq", _as_floats(self.body_dq, BODY_DIM, "body_dq"))
        object.__setattr__(
            self, "left_hand_q", _as_floats(self.left_hand_q, HAND_DIM, "left_hand_q")
        )
        object.__setattr__(
            self, "left_hand_dq", _as_floats(self.left_hand_dq, HAND_DIM, "left_hand_dq")
        )
        object.__setattr__(
            self, "right_hand_q", _as_floats(self.right_hand_q, HAND_DIM, "right_hand_q")
        )
        object.__setattr__(
            self,
            "right_hand_dq",
            _as_floats(self.right_hand_dq, HAND_DIM, "right_hand_dq"),
        )
        object.__setattr__(
            self, "last_action", _as_floats(self.last_action, BODY_DIM, "last_action")
        )
        object.__setattr__(
            self,
            "last_left_hand_action",
            _as_floats(self.last_left_hand_action, HAND_DIM, "last_left_hand_action"),
        )
        object.__setattr__(
            self,
            "last_right_hand_action",
            _as_floats(self.last_right_hand_action, HAND_DIM, "last_right_hand_action"),
        )
        object.__setattr__(self, "token", _as_token(self.token))
        object.__setattr__(self, "base_quat", _as_floats(self.base_quat, 4, "base_quat"))
        object.__setattr__(self, "base_ang_vel", _as_floats(self.base_ang_vel, 3, "base_ang_vel"))


@dataclass(frozen=True)
class TrackingSample:
    """One decoded ``pico_sim_bridge.sample`` dict.

    ``stamp_ns`` is ``sample_time_ns``. ``vector`` has length ``PICO_DIM`` in
    ``PICO_NAMES`` order and may contain NaN. ``valid`` stays true for a
    None device (tracked 0, NaN poses). The decoder returns None, or a sample
    with ``valid`` false, when body position or rotation lengths are wrong.
    """

    stamp_ns: int
    sequence: int
    body_update_sequence: int
    vector: tuple[float, ...]
    valid: bool

    def __post_init__(self) -> None:
        object.__setattr__(self, "stamp_ns", _as_int(self.stamp_ns, "stamp_ns"))
        object.__setattr__(self, "sequence", _as_int(self.sequence, "sequence"))
        object.__setattr__(
            self,
            "body_update_sequence",
            _as_int(self.body_update_sequence, "body_update_sequence"),
        )
        object.__setattr__(self, "vector", _as_floats(self.vector, PICO_DIM, "vector"))
        object.__setattr__(self, "valid", _as_bool(self.valid, "valid"))


@dataclass(frozen=True)
class ImageRef:
    """A kept camera frame with no pixels.

    The log stores JPEG bytes. ``build_rows`` takes these refs plus an index
    into the episode's kept frames so the row builder stays pure. ``ImageSample``
    in the capture path is this identity plus pixels; pixels are not a field here.
    """

    stamp_ns: int
    sequence: int
    valid: bool

    def __post_init__(self) -> None:
        object.__setattr__(self, "stamp_ns", _as_int(self.stamp_ns, "stamp_ns"))
        object.__setattr__(self, "sequence", _as_int(self.sequence, "sequence"))
        object.__setattr__(self, "valid", _as_bool(self.valid, "valid"))


@dataclass(frozen=True)
class Row:
    """One grid row at ``t_ns``.

    ``state`` has length ``state_dim`` from the layout. ``joint_velocity``,
    ``action`` follow ``body_joint_names``. Hand velocities and hand actions
    are the 7-motor g1_debug vectors (``motor_0`` .. ``motor_6``), not slices
    of ``state``. ``token`` is length ``TOKEN_DIM``; it is zeros only when
    ``ACTION_MISSING`` is set, and then ``valid`` is 0. ``pico`` is length
    ``PICO_DIM``. ``ages`` is ``(robot_s, pico_s, camera_s)``. ``valid`` is
    1 or 0. ``image_index`` indexes the episode's kept frames, or -1 when the
    row has no frame.

    ``flags`` ORs ``STATE_INTERP``, ``ACTION_MISSING``, ``CAMERA_STALE``,
    ``PICO_INTERP``, and ``PICO_STALE``.
    """

    t_ns: int
    state: tuple[float, ...]
    projected_gravity: tuple[float, ...]
    base_quat: tuple[float, ...]
    base_ang_vel: tuple[float, ...]
    joint_velocity: tuple[float, ...]
    left_hand_dq: tuple[float, ...]
    right_hand_dq: tuple[float, ...]
    action: tuple[float, ...]
    left_hand_action: tuple[float, ...]
    right_hand_action: tuple[float, ...]
    token: tuple[float, ...]
    pico: tuple[float, ...]
    pico_sequence: int
    g1_index: int
    ages: tuple[float, ...]
    valid: int
    flags: int
    image_index: int

    def __post_init__(self) -> None:
        state_dim = int(load_layout()["state_dim"])
        object.__setattr__(self, "t_ns", _as_int(self.t_ns, "t_ns"))
        object.__setattr__(self, "state", _as_floats(self.state, state_dim, "state"))
        object.__setattr__(
            self,
            "projected_gravity",
            _as_floats(self.projected_gravity, 3, "projected_gravity"),
        )
        object.__setattr__(self, "base_quat", _as_floats(self.base_quat, 4, "base_quat"))
        object.__setattr__(self, "base_ang_vel", _as_floats(self.base_ang_vel, 3, "base_ang_vel"))
        object.__setattr__(
            self,
            "joint_velocity",
            _as_floats(self.joint_velocity, BODY_DIM, "joint_velocity"),
        )
        object.__setattr__(
            self, "left_hand_dq", _as_floats(self.left_hand_dq, HAND_DIM, "left_hand_dq")
        )
        object.__setattr__(
            self,
            "right_hand_dq",
            _as_floats(self.right_hand_dq, HAND_DIM, "right_hand_dq"),
        )
        object.__setattr__(self, "action", _as_floats(self.action, BODY_DIM, "action"))
        object.__setattr__(
            self,
            "left_hand_action",
            _as_floats(self.left_hand_action, HAND_DIM, "left_hand_action"),
        )
        object.__setattr__(
            self,
            "right_hand_action",
            _as_floats(self.right_hand_action, HAND_DIM, "right_hand_action"),
        )
        object.__setattr__(self, "token", _as_floats(self.token, TOKEN_DIM, "token"))
        object.__setattr__(self, "pico", _as_floats(self.pico, PICO_DIM, "pico"))
        object.__setattr__(self, "pico_sequence", _as_int(self.pico_sequence, "pico_sequence"))
        object.__setattr__(self, "g1_index", _as_int(self.g1_index, "g1_index"))
        object.__setattr__(self, "ages", _as_floats(self.ages, 3, "ages"))
        valid = _as_int(self.valid, "valid")
        if valid not in (0, 1):
            raise ValueError(f"valid must be 0 or 1, got {valid}")
        object.__setattr__(self, "valid", valid)
        object.__setattr__(self, "flags", _as_int(self.flags, "flags"))
        image_index = _as_int(self.image_index, "image_index")
        if image_index < -1:
            raise ValueError(f"image_index must be >= -1, got {image_index}")
        object.__setattr__(self, "image_index", image_index)
