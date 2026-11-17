"""Offline 50 Hz grid for one G1 teleop episode.

``build_rows`` places robot state, the applied action, the motion token, PICO
tracking, and camera frames on

    t_k = t0_ns + (k * 1_000_000_000 + fps // 2) // fps

and stops at the later of the last state stamp and the last image stamp.
Joint state may be interpolated across a short gap. Action, hand actions, and
the motion token are never blended: they are taken from the latest
``g1_debug`` sample at or before ``t_k`` only when that sample is still inside
one grid period and carries a token. Otherwise the numeric action is held and
the token column is 64 zeros.
"""

from __future__ import annotations

import bisect
import math

import numpy as np

from gear_sonic.data.record.schema import (
    ACTION_MISSING,
    BODY_DIM,
    CAMERA_STALE,
    CAMERA_STALE_NS,
    HAND_DIM,
    PICO_DIM,
    PICO_INTERP,
    PICO_INTERP_MAX_NS,
    PICO_NAMES,
    PICO_STALE,
    STATE_INTERP,
    STATE_INTERP_MAX_NS,
    TOKEN_DIM,
    ImageRef,
    Row,
    StateSample,
    TrackingSample,
    period_ns,
)

_ZERO_TOKEN = (0.0,) * TOKEN_DIM
_ZERO_BODY = (0.0,) * BODY_DIM
_ZERO_HAND = (0.0,) * HAND_DIM
_ZERO_ANG = (0.0, 0.0, 0.0)
_IDENTITY_QUAT = (1.0, 0.0, 0.0, 0.0)
_PICO_HELD_TOKENS = ("trigger", "squeeze", "stick", "primary", "secondary", "menu")


def _held_pico_name(name: str) -> bool:
    """Buttons, sticks, and tracked flags stay on the latest sample."""
    if name.endswith("_tracked"):
        return True
    return any(token in name for token in _PICO_HELD_TOKENS)


def _pico_axes() -> tuple[np.ndarray, tuple[tuple[int, int, int, int], ...]]:
    """Position indexes and xyzw quaternion groups, taken from ``PICO_NAMES``."""
    positions: list[int] = []
    quaternions: dict[str, dict[str, int]] = {}
    for index, name in enumerate(PICO_NAMES):
        if name.endswith("_qx"):
            quaternions.setdefault(name[:-3], {})["qx"] = index
        elif name.endswith("_qy"):
            quaternions.setdefault(name[:-3], {})["qy"] = index
        elif name.endswith("_qz"):
            quaternions.setdefault(name[:-3], {})["qz"] = index
        elif name.endswith("_qw"):
            quaternions.setdefault(name[:-3], {})["qw"] = index
        elif _held_pico_name(name):
            continue
        elif name.endswith(("_x", "_y", "_z")):
            positions.append(index)
    groups: list[tuple[int, int, int, int]] = []
    for parts in quaternions.values():
        groups.append((parts["qx"], parts["qy"], parts["qz"], parts["qw"]))
    return np.asarray(positions, dtype=np.int64), tuple(groups)


_POS_INDEX, _QUAT_GROUPS = _pico_axes()


def _floats(values) -> tuple[float, ...]:
    return tuple(float(value) for value in values)


def _lerp(left, right, alpha: float) -> np.ndarray:
    start = np.asarray(left, dtype=np.float64)
    end = np.asarray(right, dtype=np.float64)
    return (1.0 - alpha) * start + alpha * end


def _slerp(q0, q1, alpha: float) -> np.ndarray:
    """Spherical interpolation of one quaternion. A non-finite endpoint stays NaN."""
    start = np.asarray(q0, dtype=np.float64)
    end = np.asarray(q1, dtype=np.float64)
    if not np.isfinite(start).all() or not np.isfinite(end).all():
        return np.full(4, np.nan)
    norm0 = float(np.linalg.norm(start))
    norm1 = float(np.linalg.norm(end))
    if norm0 == 0.0 or norm1 == 0.0:
        return np.full(4, np.nan)
    start = start / norm0
    end = end / norm1
    dot = float(np.dot(start, end))
    if dot < 0.0:
        end = -end
        dot = -dot
    dot = min(1.0, dot)
    if dot > 0.9995:
        blended = start + alpha * (end - start)
        norm = float(np.linalg.norm(blended))
        return start if norm == 0.0 else blended / norm
    theta = math.acos(dot)
    sin_theta = math.sin(theta)
    weight0 = math.sin((1.0 - alpha) * theta) / sin_theta
    weight1 = math.sin(alpha * theta) / sin_theta
    return weight0 * start + weight1 * end


def _projected_gravity(quat) -> tuple[float, float, float]:
    """Rotate world gravity ``(0, 0, -1)`` by the inverse of a wxyz quaternion."""
    raw = np.asarray(quat, dtype=np.float64)
    norm = float(np.linalg.norm(raw))
    if norm == 0.0:
        w, x, y, z = _IDENTITY_QUAT
    else:
        w, x, y, z = (float(part) for part in raw / norm)
    gravity = (0.0, 0.0, -1.0)
    scale = 2.0 * w * w - 1.0
    ax, ay, az = (component * scale for component in gravity)
    cx = y * gravity[2] - z * gravity[1]
    cy = z * gravity[0] - x * gravity[2]
    cz = x * gravity[1] - y * gravity[0]
    cross_scale = 2.0 * w
    bx, by, bz = cx * cross_scale, cy * cross_scale, cz * cross_scale
    dot = x * gravity[0] + y * gravity[1] + z * gravity[2]
    along = 2.0 * dot
    return (ax - bx + x * along, ay - by + y * along, az - bz + z * along)


class _Scatter:
    """Map g1_debug body order and hand-actuator order onto ``joint_names``."""

    def __init__(self, layout: dict) -> None:
        joint_names = list(layout["joint_names"])
        self.state_dim = int(layout["state_dim"])
        if self.state_dim != len(joint_names):
            raise ValueError(
                f"layout state_dim {self.state_dim} != len(joint_names) {len(joint_names)}"
            )
        index = {name: position for position, name in enumerate(joint_names)}
        self._body = [index[name] for name in layout["body_joint_names"]]
        hands = layout["hand_actuator_joint_names"]
        self._left = [index[name] for name in hands["left"]]
        self._right = [index[name] for name in hands["right"]]

    def state(self, body_q, left_q, right_q) -> tuple[float, ...]:
        values = [0.0] * self.state_dim
        for position, value in zip(self._body, body_q, strict=True):
            values[position] = float(value)
        for position, value in zip(self._left, left_q, strict=True):
            values[position] = float(value)
        for position, value in zip(self._right, right_q, strict=True):
            values[position] = float(value)
        return tuple(values)


def _by_stamp(samples):
    return [
        sample
        for _, sample in sorted(enumerate(samples), key=lambda item: (item[1].stamp_ns, item[0]))
    ]


def _frames_by_stamp(images: list[ImageRef]) -> list[tuple[int, ImageRef]]:
    kept = [(index, image) for index, image in enumerate(images) if image.valid]
    kept.sort(key=lambda item: (item[1].stamp_ns, item[0]))
    return kept


def _bracket(samples, stamps: list[int], t_ns: int):
    """Latest sample at or before ``t_ns``, and the following sample if any."""
    index = bisect.bisect_right(stamps, t_ns) - 1
    if index < 0:
        return None, None
    previous = samples[index]
    following = samples[index + 1] if index + 1 < len(samples) else None
    return previous, following


def _grid(t0_ns: int, fps: int, t_end_ns: int):
    previous = None
    k = 0
    while True:
        t_k = t0_ns + (k * 1_000_000_000 + fps // 2) // fps
        if t_k > t_end_ns:
            return
        if previous is not None and t_k <= previous:
            raise ValueError(f"grid step did not advance at k={k} for fps={fps}")
        yield t_k
        previous = t_k
        k += 1


def _episode_end_ns(state, images) -> int | None:
    stamps = []
    if state:
        stamps.append(max(sample.stamp_ns for sample in state))
    if images:
        stamps.append(max(image.stamp_ns for image in images))
    if not stamps:
        return None
    return max(stamps)


def _interpolate_tracking(
    previous: TrackingSample, following: TrackingSample, t_ns: int
) -> np.ndarray:
    alpha = (t_ns - previous.stamp_ns) / (following.stamp_ns - previous.stamp_ns)
    base = np.asarray(previous.vector, dtype=np.float64)
    other = np.asarray(following.vector, dtype=np.float64)
    mixed = base.copy()
    if _POS_INDEX.size:
        mixed[_POS_INDEX] = (1.0 - alpha) * base[_POS_INDEX] + alpha * other[_POS_INDEX]
    for group in _QUAT_GROUPS:
        columns = list(group)
        mixed[columns] = _slerp(base[columns], other[columns], alpha)
    return mixed


def _tracking_at(previous: TrackingSample | None, following: TrackingSample | None, t_ns: int):
    if previous is None:
        return _floats(np.zeros(PICO_DIM)), 0, 0, 0.0
    age_ns = t_ns - previous.stamp_ns
    gap_ns = None if following is None else following.stamp_ns - previous.stamp_ns
    if following is not None and t_ns > previous.stamp_ns and gap_ns <= PICO_INTERP_MAX_NS:
        vector = _floats(_interpolate_tracking(previous, following, t_ns))
        return vector, previous.sequence, PICO_INTERP, age_ns / 1_000_000_000
    flags = 0
    # A bracket wider than the window cannot be interpolated. The held sample
    # is also stale when its own age exceeds that same window.
    if age_ns > PICO_INTERP_MAX_NS or (gap_ns is not None and gap_ns > PICO_INTERP_MAX_NS):
        flags |= PICO_STALE
    return previous.vector, previous.sequence, flags, age_ns / 1_000_000_000


def _kinematics(
    previous: StateSample, following: StateSample | None, t_ns: int, period: int, scatter: _Scatter
):
    """Return state fields, whether they are usable, and any ``STATE_INTERP`` bit."""
    age_ns = t_ns - previous.stamp_ns
    if age_ns <= period:
        return (
            scatter.state(previous.body_q, previous.left_hand_q, previous.right_hand_q),
            previous.base_quat,
            previous.base_ang_vel,
            previous.body_dq,
            previous.left_hand_dq,
            previous.right_hand_dq,
            True,
            0,
        )
    gap_ns = None if following is None else following.stamp_ns - previous.stamp_ns
    if following is not None and gap_ns <= STATE_INTERP_MAX_NS:
        alpha = (t_ns - previous.stamp_ns) / gap_ns
        quat = _floats(_slerp(previous.base_quat, following.base_quat, alpha))
        return (
            scatter.state(
                _lerp(previous.body_q, following.body_q, alpha),
                _lerp(previous.left_hand_q, following.left_hand_q, alpha),
                _lerp(previous.right_hand_q, following.right_hand_q, alpha),
            ),
            quat,
            _floats(_lerp(previous.base_ang_vel, following.base_ang_vel, alpha)),
            _floats(_lerp(previous.body_dq, following.body_dq, alpha)),
            _floats(_lerp(previous.left_hand_dq, following.left_hand_dq, alpha)),
            _floats(_lerp(previous.right_hand_dq, following.right_hand_dq, alpha)),
            True,
            STATE_INTERP,
        )
    return (
        scatter.state(previous.body_q, previous.left_hand_q, previous.right_hand_q),
        previous.base_quat,
        previous.base_ang_vel,
        previous.body_dq,
        previous.left_hand_dq,
        previous.right_hand_dq,
        False,
        0,
    )


def _action_at(previous: StateSample | None, t_ns: int, period: int):
    """Hold the latest sample's action numbers. Accept the token only while it is fresh."""
    if previous is None:
        return _ZERO_BODY, _ZERO_HAND, _ZERO_HAND, _ZERO_TOKEN, ACTION_MISSING
    age_ns = t_ns - previous.stamp_ns
    if age_ns <= period and previous.token is not None:
        return (
            previous.last_action,
            previous.last_left_hand_action,
            previous.last_right_hand_action,
            previous.token,
            0,
        )
    return (
        previous.last_action,
        previous.last_left_hand_action,
        previous.last_right_hand_action,
        _ZERO_TOKEN,
        ACTION_MISSING,
    )


def _camera_at(frames: list[tuple[int, ImageRef]], stamps: list[int], t_ns: int):
    chosen, _following = _bracket(frames, stamps, t_ns)
    if chosen is None:
        return -1, 0.0, CAMERA_STALE
    index, image = chosen
    age_ns = t_ns - image.stamp_ns
    flags = CAMERA_STALE if age_ns > CAMERA_STALE_NS else 0
    return index, age_ns / 1_000_000_000, flags


def _row_at(
    t_ns: int,
    *,
    period: int,
    scatter: _Scatter,
    state_samples,
    state_stamps: list[int],
    tracks,
    track_stamps: list[int],
    frames,
    frame_stamps: list[int],
) -> Row:
    previous, following = _bracket(state_samples, state_stamps, t_ns)
    if previous is None:
        state = (0.0,) * scatter.state_dim
        quat = _IDENTITY_QUAT
        ang_vel = _ZERO_ANG
        joint_velocity = _ZERO_BODY
        left_dq = _ZERO_HAND
        right_dq = _ZERO_HAND
        state_usable = False
        state_flags = 0
        g1_index = 0
        robot_age = 0.0
    else:
        state, quat, ang_vel, joint_velocity, left_dq, right_dq, state_usable, state_flags = (
            _kinematics(previous, following, t_ns, period, scatter)
        )
        g1_index = previous.index
        robot_age = (t_ns - previous.stamp_ns) / 1_000_000_000

    action, left_action, right_action, token, action_flags = _action_at(previous, t_ns, period)
    pico_prev, pico_next = _bracket(tracks, track_stamps, t_ns)
    pico, pico_sequence, pico_flags, pico_age = _tracking_at(pico_prev, pico_next, t_ns)
    image_index, camera_age, camera_flags = _camera_at(frames, frame_stamps, t_ns)

    flags = state_flags | action_flags | pico_flags | camera_flags
    usable = state_usable and (flags & ACTION_MISSING) == 0 and (flags & CAMERA_STALE) == 0
    return Row(
        t_ns=t_ns,
        state=state,
        projected_gravity=_projected_gravity(quat),
        base_quat=quat,
        base_ang_vel=ang_vel,
        joint_velocity=joint_velocity,
        left_hand_dq=left_dq,
        right_hand_dq=right_dq,
        action=action,
        left_hand_action=left_action,
        right_hand_action=right_action,
        token=token,
        pico=pico,
        pico_sequence=pico_sequence,
        g1_index=g1_index,
        ages=(robot_age, pico_age, camera_age),
        valid=1 if usable else 0,
        flags=flags,
        image_index=image_index,
    )


def build_rows(
    state: list[StateSample],
    tracking: list[TrackingSample],
    images: list[ImageRef],
    *,
    fps: int,
    t0_ns: int,
    layout: dict,
) -> list[Row]:
    """Sample ``state``, ``tracking``, and ``images`` onto the ``fps`` grid."""
    period = period_ns(fps)
    t_end = _episode_end_ns(state, images)
    if t_end is None:
        return []

    state_samples = _by_stamp(state)
    tracks = _by_stamp(sample for sample in tracking if sample.valid)
    frames = _frames_by_stamp(images)
    scatter = _Scatter(layout)
    return [
        _row_at(
            t_ns,
            period=period,
            scatter=scatter,
            state_samples=state_samples,
            state_stamps=[sample.stamp_ns for sample in state_samples],
            tracks=tracks,
            track_stamps=[sample.stamp_ns for sample in tracks],
            frames=frames,
            frame_stamps=[image.stamp_ns for _, image in frames],
        )
        for t_ns in _grid(t0_ns, fps, t_end)
    ]
