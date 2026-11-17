"""ZMQ subscribers and offline decoders for ``g1_debug`` and ``tracking``.

``ZMQOutputHandler`` and ``pico_sim_bridge`` concatenate the topic bytes onto
the msgpack payload with no separator. Subscribers connect a ``SUB`` socket
(they never bind), keep a topic filter, and yield the payload with that prefix
removed. ``recv_ns`` is ``time.monotonic_ns()`` taken immediately after
``recv``. The receive high-water mark is 1000 and the queue is not collapsed,
so a slow recorder still observes every frame that fits.

``decode_g1_debug`` does not read a timestamp. ``stamp_ns`` on the returned
sample is 0; callers ``dataclasses.replace`` it with receive time.
"""

from __future__ import annotations

from collections.abc import Iterator
import math
import time

import msgpack
import zmq

from gear_sonic.data.record.schema import (
    BODY_DIM,
    HAND_DIM,
    PICO_DIM,
    TOKEN_DIM,
    StateSample,
    TrackingSample,
)

_STATE_TOPIC = b"g1_debug"
_TRACKING_TOPIC = b"tracking"

# pico_sim_bridge.sample body: 24 joints, xyz then xyzw, stored flat.
_BODY_POSITIONS = 72
_BODY_ROTATIONS = 96
_POSE = 7  # xyz + xyzw
_CONTROLLER_SCALARS = (
    "trigger",
    "squeeze",
    "stick_x",
    "stick_y",
    "primary",
    "secondary",
    "menu",
    "stick_click",
)

# Msgpack keys whose lengths and finiteness gate a StateSample. Token is separate.
_FINITE_VECTORS = (
    ("body_q", BODY_DIM),
    ("body_dq", BODY_DIM),
    ("last_action", BODY_DIM),
    ("left_hand_q", HAND_DIM),
    ("left_hand_dq", HAND_DIM),
    ("right_hand_q", HAND_DIM),
    ("right_hand_dq", HAND_DIM),
    ("last_left_hand_action", HAND_DIM),
    ("last_right_hand_action", HAND_DIM),
)


def subscribe_state(
    host: str = "127.0.0.1", port: int = 5557
) -> Iterator[tuple[int, bytes]]:
    """Yield ``(recv_ns, payload)`` for ``g1_debug``. Connects; never binds."""
    yield from _subscribe(host, port, _STATE_TOPIC)


def subscribe_tracking(
    host: str = "127.0.0.1", port: int = 5558
) -> Iterator[tuple[int, bytes]]:
    """Yield ``(recv_ns, payload)`` for ``tracking``. Ignores forwarded ``g1_debug``."""
    yield from _subscribe(host, port, _TRACKING_TOPIC)


def decode_g1_debug(payload: bytes) -> StateSample | None:
    """Decode a topic-stripped ``g1_debug`` map.

    Returns None when ``body_q``, ``body_dq``, or ``last_action`` is not 29
    finite floats, or any hand vector is not 7 finite floats. ``token`` is
    None unless ``token_state`` is exactly 64 finite floats; a wrong token
    does not reject the sample and is not zero-filled. ``base_quat`` is wxyz.
    """
    data = _unpack_map(payload)
    if data is None:
        return None
    index = _required_int(data, "index")
    if index is None:
        return None

    fields: dict[str, tuple[float, ...]] = {}
    for name, length in _FINITE_VECTORS:
        parsed = _numbers(data.get(name), length, finite=True)
        if parsed is None:
            return None
        fields[name] = parsed

    base_quat = _numbers(data.get("base_quat"), 4, finite=False)
    base_ang_vel = _numbers(data.get("base_ang_vel"), 3, finite=False)
    if base_quat is None or base_ang_vel is None:
        return None

    token = None
    if "token_state" in data and data["token_state"] is not None:
        token = _numbers(data["token_state"], TOKEN_DIM, finite=True)

    return StateSample(
        stamp_ns=0,
        index=index,
        body_q=fields["body_q"],
        body_dq=fields["body_dq"],
        left_hand_q=fields["left_hand_q"],
        left_hand_dq=fields["left_hand_dq"],
        right_hand_q=fields["right_hand_q"],
        right_hand_dq=fields["right_hand_dq"],
        last_action=fields["last_action"],
        last_left_hand_action=fields["last_left_hand_action"],
        last_right_hand_action=fields["last_right_hand_action"],
        token=token,
        base_quat=base_quat,
        base_ang_vel=base_ang_vel,
    )


def decode_tracking(payload: bytes) -> TrackingSample | None:
    """Decode a ``pico_sim_bridge.sample`` dict.

    ``stamp_ns`` is ``sample_time_ns``. The vector follows ``PICO_NAMES``.
    A None device stays valid: tracked flag 0, pose NaN, and controller
    scalars 0. Body position length other than 72 or rotation length other
    than 96 returns None.
    """
    data = _unpack_map(payload)
    if data is None:
        return None
    stamp_ns = _required_int(data, "sample_time_ns")
    sequence = _required_int(data, "sequence")
    body_update_sequence = _required_int(data, "body_update_sequence")
    if stamp_ns is None or sequence is None or body_update_sequence is None:
        return None

    vector: list[float] = []
    head = _pose_components(data.get("head"))
    if head is None:
        return None
    vector.extend(head)
    for side in ("left_controller", "right_controller"):
        controller = _controller_components(data.get(side))
        if controller is None:
            return None
        vector.extend(controller)
    body = _body_components(data.get("body"))
    if body is None:
        return None
    vector.extend(body)
    if len(vector) != PICO_DIM:
        return None
    return TrackingSample(
        stamp_ns=stamp_ns,
        sequence=sequence,
        body_update_sequence=body_update_sequence,
        vector=tuple(vector),
        valid=True,
    )


def _subscribe(host: str, port: int, topic: bytes) -> Iterator[tuple[int, bytes]]:
    context = zmq.Context()
    socket = context.socket(zmq.SUB)
    try:
        socket.setsockopt(zmq.RCVHWM, 1000)
        socket.setsockopt(zmq.SUBSCRIBE, topic)
        socket.connect(f"tcp://{host}:{port}")
        while True:
            raw = socket.recv()
            recv_ns = time.monotonic_ns()
            if not raw.startswith(topic):
                continue
            yield recv_ns, raw[len(topic) :]
    finally:
        socket.close(linger=0)
        context.term()


def _unpack_map(payload: bytes) -> dict | None:
    if isinstance(payload, memoryview):
        payload = payload.tobytes()
    elif isinstance(payload, bytearray):
        payload = bytes(payload)
    elif not isinstance(payload, bytes):
        return None
    try:
        data = msgpack.unpackb(payload, raw=False)
    except (msgpack.UnpackException, msgpack.ExtraData, ValueError, TypeError):
        return None
    if not isinstance(data, dict):
        return None
    return data


def _required_int(data: dict, key: str) -> int | None:
    value = data.get(key)
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    return value


def _numbers(value: object, length: int, *, finite: bool) -> tuple[float, ...] | None:
    if value is None or isinstance(value, (str, bytes, bytearray, dict)):
        return None
    try:
        items = tuple(value)  # type: ignore[arg-type]
    except TypeError:
        return None
    if len(items) != length:
        return None
    numbers: list[float] = []
    for item in items:
        if isinstance(item, bool) or not isinstance(item, (int, float)):
            return None
        number = float(item)
        if finite and not math.isfinite(number):
            return None
        numbers.append(number)
    return tuple(numbers)


def _scalar(value: object) -> float | None:
    if isinstance(value, bool):
        return 1.0 if value else 0.0
    if isinstance(value, (int, float)):
        number = float(value)
        if math.isfinite(number):
            return number
    return None


def _pose_components(pose: object) -> list[float] | None:
    """Tracked flag plus xyz xyzw. None pose -> tracked 0 and NaN components."""
    if pose is None:
        return [0.0, *([math.nan] * _POSE)]
    numbers = _numbers(pose, _POSE, finite=False)
    if numbers is None:
        return None
    return [1.0, *numbers]


def _controller_components(controller: object) -> list[float] | None:
    """16 floats: tracked, grip xyzw, then the eight controller scalars."""
    if controller is None:
        return [0.0, *([math.nan] * _POSE), *([0.0] * len(_CONTROLLER_SCALARS))]
    if not isinstance(controller, dict):
        return None
    grip = _numbers(controller.get("grip"), _POSE, finite=False)
    if grip is None:
        return None
    scalars: list[float] = []
    for key in _CONTROLLER_SCALARS:
        if key not in controller:
            return None
        number = _scalar(controller[key])
        if number is None:
            return None
        scalars.append(number)
    return [1.0, *grip, *scalars]


def _body_components(body: object) -> list[float] | None:
    """Tracked flag, then 72 positions, then 96 rotations. None body stays valid."""
    width = _BODY_POSITIONS + _BODY_ROTATIONS
    if body is None:
        return [0.0, *([math.nan] * width)]
    if not isinstance(body, dict):
        return None
    positions = body.get("positions")
    rotations = body.get("rotations")
    if _sequence_length(positions) != _BODY_POSITIONS or _sequence_length(rotations) != _BODY_ROTATIONS:
        return None
    decoded_positions = _numbers(positions, _BODY_POSITIONS, finite=False)
    decoded_rotations = _numbers(rotations, _BODY_ROTATIONS, finite=False)
    if decoded_positions is None or decoded_rotations is None:
        return None
    return [1.0, *decoded_positions, *decoded_rotations]


def _sequence_length(value: object) -> int | None:
    if value is None or isinstance(value, (str, bytes, bytearray, dict)):
        return None
    try:
        return len(value)  # type: ignore[arg-type]
    except TypeError:
        return None
