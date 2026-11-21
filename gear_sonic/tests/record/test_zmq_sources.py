"""Offline decoders and loopback subscribers for g1_debug and tracking. No hardware."""

from dataclasses import replace
import math
import threading
import time

import msgpack
import zmq

from gear_sonic.data.record.schema import PICO_DIM, PICO_NAMES
from gear_sonic.data.record.zmq_sources import (
    decode_g1_debug,
    decode_tracking,
    subscribe_state,
    subscribe_tracking,
)

# Ports the live recorder and the deploy stack already use.
_RESERVED_PORTS = (5556, 5557, 5558, 60061)


def _g1_dict(**overrides) -> dict:
    data = {
        "control_loop_type": "cpp",
        "index": 17,
        "ros_timestamp": 12345.6,
        "body_q": [float(i) for i in range(29)],
        "body_dq": [float(i) + 0.5 for i in range(29)],
        "last_action": [float(100 + i) for i in range(29)],
        "left_hand_q": [float(i) for i in range(7)],
        "left_hand_dq": [float(10 + i) for i in range(7)],
        "right_hand_q": [float(20 + i) for i in range(7)],
        "right_hand_dq": [float(30 + i) for i in range(7)],
        "last_left_hand_action": [float(40 + i) for i in range(7)],
        "last_right_hand_action": [float(50 + i) for i in range(7)],
        "base_quat": [0.1, 0.2, 0.3, 0.4],
        "base_ang_vel": [1.5, -2.5, 3.5],
        "token_state": [float(i) for i in range(64)],
    }
    data.update(overrides)
    return data


def _pack_g1(**overrides) -> bytes:
    return msgpack.packb(_g1_dict(**overrides), use_bin_type=True)


def _controller(grip: list[float], **scalars) -> dict:
    fields = {
        "grip": grip,
        "aim": None,
        "trigger": 0.25,
        "squeeze": 0.75,
        "stick_x": -1.0,
        "stick_y": 1.0,
        "primary": True,
        "secondary": False,
        "menu": True,
        "stick_click": False,
    }
    fields.update(scalars)
    return fields


def _tracking_frame(*, sequence: int = 11, right=None, body=True) -> dict:
    positions: list[float] = []
    rotations: list[float] = []
    for joint in range(24):
        positions.extend((100.0 + joint, 0.25, -0.5))
        rotations.extend((0.01 * joint, 0.02, 0.03, 0.9))
    return {
        "sequence": sequence,
        "sample_time_ns": 987_654_321,
        "body_update_sequence": 42,
        "head": [1.5, 2.5, 3.5, 0.0, 0.1, 0.2, 0.3],
        "left_controller": _controller([4.0, 5.0, 6.0, 0.0, 0.0, 0.0, 1.0]),
        "right_controller": right,
        "body": None if not body else {"positions": positions, "rotations": rotations},
    }


def _pack_tracking(frame: dict) -> bytes:
    return msgpack.packb(frame, use_bin_type=True)


def _by_name(vector) -> dict[str, float]:
    assert len(vector) == len(PICO_NAMES) == PICO_DIM == 209
    return dict(zip(PICO_NAMES, vector))


def test_decode_g1_debug_keeps_64_float_token_and_leaves_stamp_at_zero():
    token = tuple(float(i) for i in range(64))
    sample = decode_g1_debug(_pack_g1(index=99, token_state=list(token)))
    assert sample is not None
    assert sample.stamp_ns == 0
    assert sample.index == 99
    assert sample.token == token
    assert sample.body_q == tuple(float(i) for i in range(29))
    assert sample.body_dq == tuple(float(i) + 0.5 for i in range(29))
    assert sample.last_action == tuple(float(100 + i) for i in range(29))
    assert sample.left_hand_q == tuple(float(i) for i in range(7))
    assert sample.last_right_hand_action == tuple(float(50 + i) for i in range(7))
    assert sample.base_quat == (0.1, 0.2, 0.3, 0.4)
    assert sample.base_ang_vel == (1.5, -2.5, 3.5)
    stamped = replace(sample, stamp_ns=123)
    assert stamped.stamp_ns == 123
    assert sample.stamp_ns == 0

    bare = _g1_dict()
    del bare["token_state"]
    missing = decode_g1_debug(msgpack.packb(bare, use_bin_type=True))
    assert missing is not None and missing.token is None

    non_finite = [0.0] * 64
    non_finite[2] = math.nan
    kept = decode_g1_debug(_pack_g1(token_state=non_finite))
    assert kept is not None and kept.token is None
    assert kept.body_q[0] == 0.0


def test_wrong_token_length_sets_token_none_and_keeps_the_sample():
    for length in (0, 128):
        sample = decode_g1_debug(_pack_g1(token_state=[0.5] * length))
        assert sample is not None
        assert sample.token is None
        assert sample.token != tuple(0.0 for _ in range(64))
        assert sample.body_q == tuple(float(i) for i in range(29))
        assert sample.stamp_ns == 0


def test_nonfinite_or_wrong_length_body_or_hand_is_none():
    body = [float(i) for i in range(29)]
    body[4] = math.nan
    assert decode_g1_debug(_pack_g1(body_q=body)) is None
    velocities = [float(i) + 0.5 for i in range(29)]
    velocities[0] = math.inf
    assert decode_g1_debug(_pack_g1(body_dq=velocities)) is None
    assert decode_g1_debug(_pack_g1(last_action=[0.3] * 28)) is None
    assert decode_g1_debug(_pack_g1(last_action=[0.3] * 30)) is None
    assert decode_g1_debug(_pack_g1(left_hand_q=[math.nan] * 7)) is None
    assert decode_g1_debug(_pack_g1(right_hand_dq=[0.1] * 6)) is None
    assert decode_g1_debug(_pack_g1(last_left_hand_action=[0.1] * 8)) is None
    assert decode_g1_debug(_pack_g1(last_right_hand_action=[math.inf] * 7)) is None
    assert decode_g1_debug(_pack_g1(base_quat=[1.0, 0.0, 0.0])) is None
    assert decode_g1_debug(_pack_g1(base_ang_vel=[1.0, 2.0])) is None
    assert decode_g1_debug(b"") is None
    assert decode_g1_debug(b"g1_debug" + _pack_g1()) is None


def test_decode_tracking_frame_with_one_none_device_and_no_hands():
    frame = _tracking_frame(right=None, body=True)
    assert frame["right_controller"] is None
    assert "left_hand" not in frame and "right_hand" not in frame
    assert len(frame["body"]["positions"]) == 72
    assert len(frame["body"]["rotations"]) == 96

    sample = decode_tracking(_pack_tracking(frame))
    assert sample is not None
    assert sample.valid is True
    assert sample.stamp_ns == 987_654_321
    assert sample.sequence == 11
    assert sample.body_update_sequence == 42
    names = _by_name(sample.vector)

    assert names["head_tracked"] == 1.0
    assert names["head_x"] == 1.5
    assert names["head_z"] == 3.5
    assert names["head_qx"] == 0.0
    assert names["head_qw"] == 0.3
    assert names["left_controller_tracked"] == 1.0
    assert names["left_controller_grip_x"] == 4.0
    assert names["left_controller_grip_z"] == 6.0
    assert names["left_controller_grip_qw"] == 1.0
    assert names["left_controller_trigger"] == 0.25
    assert names["left_controller_squeeze"] == 0.75
    assert names["left_controller_stick_x"] == -1.0
    assert names["left_controller_stick_y"] == 1.0
    assert names["left_controller_primary"] == 1.0
    assert names["left_controller_secondary"] == 0.0
    assert names["left_controller_menu"] == 1.0
    assert names["left_controller_stick_click"] == 0.0
    assert names["right_controller_tracked"] == 0.0
    for name in (
        "right_controller_grip_x",
        "right_controller_grip_y",
        "right_controller_grip_z",
        "right_controller_grip_qx",
        "right_controller_grip_qy",
        "right_controller_grip_qz",
        "right_controller_grip_qw",
    ):
        assert math.isnan(names[name]), name
    for name in (
        "right_controller_trigger",
        "right_controller_squeeze",
        "right_controller_stick_x",
        "right_controller_stick_y",
        "right_controller_primary",
        "right_controller_secondary",
        "right_controller_menu",
        "right_controller_stick_click",
    ):
        assert names[name] == 0.0
    assert names["body_tracked"] == 1.0
    assert names["body_pos_0_x"] == 100.0
    assert names["body_pos_0_y"] == 0.25
    assert names["body_pos_0_z"] == -0.5
    assert names["body_pos_1_x"] == 101.0
    assert names["body_pos_23_x"] == 123.0
    assert names["body_pos_23_z"] == -0.5
    assert names["body_rot_0_qx"] == 0.0
    assert names["body_rot_0_qy"] == 0.02
    assert names["body_rot_0_qw"] == 0.9
    assert names["body_rot_1_qx"] == 0.01
    assert names["body_rot_23_qx"] == 0.23
    assert names["body_rot_23_qw"] == 0.9

    missing_body = _tracking_frame(body=False)
    missing = decode_tracking(_pack_tracking(missing_body))
    assert missing is not None and missing.valid is True
    missing_names = _by_name(missing.vector)
    assert missing_names["body_tracked"] == 0.0
    assert missing_names["head_tracked"] == 1.0
    start = PICO_NAMES.index("body_pos_0_x")
    assert all(math.isnan(value) for value in missing.vector[start:])


def test_decode_tracking_wrong_body_length_is_none():
    short_positions = _tracking_frame()
    short_positions["body"]["positions"] = short_positions["body"]["positions"][:-1]
    assert decode_tracking(_pack_tracking(short_positions)) is None

    short_rotations = _tracking_frame()
    short_rotations["body"]["rotations"] = short_rotations["body"]["rotations"][:-1]
    assert decode_tracking(_pack_tracking(short_rotations)) is None

    long_positions = _tracking_frame()
    long_positions["body"]["positions"] = long_positions["body"]["positions"] + [0.0]
    assert decode_tracking(_pack_tracking(long_positions)) is None

    no_rotations = _tracking_frame()
    del no_rotations["body"]["rotations"]
    assert decode_tracking(_pack_tracking(no_rotations)) is None
    assert decode_tracking(b"") is None
    assert decode_tracking(b"tracking" + _pack_tracking(_tracking_frame())) is None


def _bind_ephemeral():
    context = zmq.Context()
    publisher = None
    try:
        for _ in range(8):
            publisher = context.socket(zmq.PUB)
            publisher.setsockopt(zmq.LINGER, 0)
            publisher.setsockopt(zmq.SNDHWM, 1000)
            publisher.bind("tcp://127.0.0.1:0")
            endpoint = publisher.getsockopt(zmq.LAST_ENDPOINT)
            if isinstance(endpoint, bytes):
                endpoint = endpoint.decode()
            port = int(str(endpoint).rsplit(":", 1)[-1])
            if port not in _RESERVED_PORTS:
                return context, publisher, port
            publisher.close(linger=0)
            publisher = None
        raise AssertionError("ephemeral bind landed on a reserved recorder port")
    except Exception:
        if publisher is not None:
            publisher.close(linger=0)
        context.term()
        raise


def _send(publisher, frame: bytes) -> None:
    try:
        publisher.send(frame, flags=zmq.NOBLOCK)
    except zmq.Again:
        return


def test_subscribe_tracking_strips_prefix_and_ignores_g1_debug():
    context, publisher, port = _bind_ephemeral()
    assert port not in _RESERVED_PORTS
    frame_a = msgpack.packb({"part": "A"}, use_bin_type=True)
    frame_b = msgpack.packb({"part": "B"}, use_bin_type=True)
    frame_c = msgpack.packb({"part": "C"}, use_bin_type=True)
    g1 = msgpack.packb({"index": 1, "marker": "g1"}, use_bin_type=True)
    assert frame_a != g1

    got: list[tuple[int, bytes]] = []
    lock = threading.Lock()
    pause = threading.Event()
    release = threading.Event()

    def collect() -> None:
        iterator = subscribe_tracking("127.0.0.1", port)
        try:
            first = True
            for recv_ns, payload in iterator:
                with lock:
                    got.append((recv_ns, payload))
                    payloads = [item for _, item in got]
                if first:
                    first = False
                    pause.set()
                    if not release.wait(timeout=3):
                        return
                if frame_b in payloads and frame_c in payloads:
                    return
        finally:
            iterator.close()

    thread = threading.Thread(target=collect, name="tracking-sub", daemon=True)
    thread.start()
    try:
        deadline = time.monotonic() + 5
        while not pause.is_set() and time.monotonic() < deadline:
            _send(publisher, b"g1_debug" + g1)
            _send(publisher, b"tracking" + frame_a)
            time.sleep(0.01)
        assert pause.is_set(), "tracking subscriber never connected"
        _send(publisher, b"g1_debug" + g1)
        _send(publisher, b"tracking" + frame_b)
        _send(publisher, b"tracking" + frame_c)
        release.set()
        thread.join(timeout=3)
        assert not thread.is_alive()
        with lock:
            payloads = [item for _, item in got]
            stamps = [stamp for stamp, _ in got]
        assert frame_a in payloads
        assert payloads.index(frame_b) < payloads.index(frame_c)
        assert g1 not in payloads
        assert all(not payload.startswith(b"tracking") for payload in payloads)
        assert all(not payload.startswith(b"g1_debug") for payload in payloads)
        assert stamps == sorted(stamps)
        assert all(isinstance(stamp, int) and stamp > 0 for stamp in stamps)
    finally:
        release.set()
        publisher.close(linger=0)
        context.term()


def test_subscribe_state_strips_g1_debug_prefix():
    context, publisher, port = _bind_ephemeral()
    state = _pack_g1()
    tracking = _pack_tracking(_tracking_frame(sequence=3))
    got: list[tuple[int, bytes]] = []

    def collect() -> None:
        iterator = subscribe_state(port=port)
        try:
            recv_ns, payload = next(iterator)
            got.append((recv_ns, payload))
        finally:
            iterator.close()

    thread = threading.Thread(target=collect, name="state-sub", daemon=True)
    thread.start()
    try:
        deadline = time.monotonic() + 5
        while not got and time.monotonic() < deadline:
            _send(publisher, b"tracking" + tracking)
            _send(publisher, b"robot_config" + msgpack.packb({"nope": True}))
            _send(publisher, b"g1_debug" + state)
            time.sleep(0.01)
        thread.join(timeout=2)
        assert got, "subscribe_state received nothing"
        recv_ns, payload = got[0]
        assert payload == state
        assert not payload.startswith(b"g1_debug")
        assert payload != tracking
        decoded = decode_g1_debug(payload)
        assert decoded is not None and decoded.index == 17 and decoded.stamp_ns == 0
        assert isinstance(recv_ns, int) and 0 < recv_ns <= time.monotonic_ns()
    finally:
        publisher.close(linger=0)
        context.term()
