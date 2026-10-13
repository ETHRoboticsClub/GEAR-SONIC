import time

import msgpack
import numpy as np
import pytest
import zmq

from gear_sonic.scripts import pico_manager_thread_server as server
from gear_sonic.scripts.pico_manager_thread_server import StreamMode
from gear_sonic.tests.teleop import fakes
from gear_sonic.utils.teleop.input_readers import IsaacTeleopReader, Tracking

MS = 1_000_000
T0 = 10_000 * MS

ALL = [StreamMode.POSE, StreamMode.PLANNER, StreamMode.PLANNER_VR_3PT]


@pytest.mark.parametrize(
    "tracking, expected",
    [
        (Tracking(), []),
        (Tracking(body=True), ALL),
        (Tracking(head=True), [StreamMode.PLANNER]),
        (Tracking(controllers=True), [StreamMode.PLANNER]),
        (Tracking(hands=True), [StreamMode.PLANNER]),
        (Tracking(head=True, controllers=True), [StreamMode.PLANNER, StreamMode.PLANNER_VR_3PT]),
        (Tracking(head=True, hands=True), [StreamMode.PLANNER, StreamMode.PLANNER_VR_3PT]),
        (Tracking(controllers=True, hands=True), [StreamMode.PLANNER]),
    ],
)
def test_available_modes(tracking, expected):
    assert server.available_modes(tracking) == expected


class FakePoseStreamer:
    def __init__(self, calls):
        self.calls = calls

    def on_mode_exit(self):
        self.calls.append("pose.on_mode_exit")

    def reset_yaw(self):
        self.calls.append("pose.reset_yaw")


class FakePlannerStreamer:
    def __init__(self, calls):
        self.calls = calls

    def calibrate_now(self):
        self.calls.append("planner.calibrate_now")

    def reset_yaw(self):
        self.calls.append("planner.reset_yaw")

    def save_upper_body_position_target(self):
        self.calls.append("planner.save_upper_body_position_target")

    def recalibrate_for_vr3pt(self):
        self.calls.append("planner.recalibrate_for_vr3pt")


def _manager():
    calls = []
    socket = fakes.RecordingSocket()
    modes = server.StreamModeManager(
        socket,
        reader=None,
        pose_streamer=FakePoseStreamer(calls),
        planner_streamer=FakePlannerStreamer(calls),
    )
    return modes, socket, calls


def _iterate(modes, available, request=None):
    """The request part of one run_pico_manager iteration (no buttons pressed)."""
    modes.receive_mode_request(request)
    new_mode = modes.mode_request_hop(available)
    if new_mode != modes.current_mode:
        modes.begin_switch(new_mode)
        modes.finish_switch(new_mode)


def _commands(socket):
    out = []
    for message in socket.sent:
        _, fields = fakes.decode(message, b"command")
        out.append((int(fields["start"][0]), int(fields["stop"][0]), int(fields["planner"][0])))
    return out


def test_request_from_off_to_pose_steps_through_planner():
    modes, socket, calls = _manager()

    _iterate(modes, ALL, {"sequence": 1, "mode": StreamMode.POSE.value})
    assert modes.current_mode == StreamMode.PLANNER
    assert calls == ["planner.calibrate_now", "planner.reset_yaw"]
    assert modes.applied_mode_request == 0

    _iterate(modes, ALL)
    assert modes.current_mode == StreamMode.POSE
    assert calls[2:] == ["pose.reset_yaw"]

    _iterate(modes, ALL)
    assert modes.applied_mode_request == 1
    assert _commands(socket) == [(1, 0, 1), (1, 0, 0)]


def test_request_for_an_unavailable_mode_is_refused():
    modes, socket, calls = _manager()

    _iterate(modes, [StreamMode.PLANNER], {"sequence": 4, "mode": StreamMode.POSE.value})

    assert modes.current_mode == StreamMode.OFF
    assert modes.applied_mode_request == 4
    assert calls == [] and socket.sent == []


@pytest.mark.parametrize(
    "mode",
    [
        StreamMode.OFF.value,
        StreamMode.PLANNER_FROZEN_UPPER_BODY.value,
        StreamMode.POSE_PAUSE.value,
        99,
    ],
)
def test_modes_outside_the_request_contract_are_refused(mode):
    modes, socket, _ = _manager()
    _iterate(modes, ALL, {"sequence": 1, "mode": StreamMode.PLANNER.value})
    _iterate(modes, ALL)
    sent = len(socket.sent)

    _iterate(modes, ALL, {"sequence": 2, "mode": mode})

    assert modes.current_mode == StreamMode.PLANNER
    assert modes.applied_mode_request == 2
    assert len(socket.sent) == sent


def test_target_lost_midway_is_refused_where_the_steps_stopped():
    modes, _, _ = _manager()
    _iterate(modes, ALL, {"sequence": 1, "mode": StreamMode.POSE.value})
    assert modes.current_mode == StreamMode.PLANNER

    _iterate(modes, [StreamMode.PLANNER])  # body tracking dropped

    assert modes.current_mode == StreamMode.PLANNER
    assert modes.applied_mode_request == 1


def test_resent_request_is_applied_once():
    modes, socket, _ = _manager()
    request = {"sequence": 7, "mode": StreamMode.PLANNER.value}
    _iterate(modes, ALL, request)
    _iterate(modes, ALL, request)
    assert modes.applied_mode_request == 7
    # The operator moves on with the buttons; the robot keeps resending sequence 7.
    modes.begin_switch(StreamMode.POSE)
    modes.finish_switch(StreamMode.POSE)
    sent = len(socket.sent)

    for _ in range(3):
        _iterate(modes, ALL, request)

    assert modes.current_mode == StreamMode.POSE
    assert len(socket.sent) == sent

    _iterate(modes, ALL, {"sequence": 8, "mode": StreamMode.PLANNER.value})
    assert modes.current_mode == StreamMode.PLANNER


def test_planner_vr3pt_round_trip_uses_the_button_side_effects():
    modes, _, calls = _manager()
    _iterate(modes, ALL, {"sequence": 1, "mode": StreamMode.PLANNER.value})
    _iterate(modes, ALL)
    calls.clear()

    _iterate(modes, ALL, {"sequence": 2, "mode": StreamMode.PLANNER_VR_3PT.value})
    assert modes.current_mode == StreamMode.PLANNER_VR_3PT
    assert modes.vr3pt_parent_mode == StreamMode.PLANNER
    assert calls == ["planner.recalibrate_for_vr3pt"]

    _iterate(modes, ALL, {"sequence": 3, "mode": StreamMode.PLANNER.value})
    assert modes.current_mode == StreamMode.PLANNER
    # Returning from the VR 3PT sub-mode keeps the planner's yaw.
    assert calls == ["planner.recalibrate_for_vr3pt"]
    _iterate(modes, ALL)
    assert modes.applied_mode_request == 3


def test_request_out_of_frozen_vr3pt_goes_through_pose():
    modes, _, _ = _manager()
    modes.current_mode = StreamMode.PLANNER_VR_3PT
    modes.vr3pt_parent_mode = StreamMode.PLANNER_FROZEN_UPPER_BODY

    _iterate(modes, ALL, {"sequence": 1, "mode": StreamMode.PLANNER.value})
    assert modes.current_mode == StreamMode.POSE
    _iterate(modes, ALL)
    assert modes.current_mode == StreamMode.PLANNER


def test_manager_state_reports_tracking_available_modes_and_applied_request():
    modes, _, _ = _manager()
    _iterate(modes, ALL, {"sequence": 5, "mode": StreamMode.PLANNER.value})
    _iterate(modes, ALL)
    tracking = Tracking(head=True, controllers=True)

    message = modes.manager_state_message(True, False, tracking, server.available_modes(tracking))

    header, fields = fakes.decode(message, b"manager_state")
    assert header["v"] == 3
    dtypes = {f["name"]: f["dtype"] for f in header["fields"]}
    assert dtypes == {
        "stream_mode": "i32",
        "toggle_data_collection": "bool",
        "toggle_data_abort": "bool",
        "tracking": "u8",
        "available_modes": "i32",
        "applied_mode_request": "i64",
    }
    assert fields["stream_mode"].tolist() == [2]
    assert fields["toggle_data_collection"].tolist() == [True]
    assert fields["tracking"].tolist() == [0, 1, 1, 0]
    assert fields["available_modes"].tolist() == [2, 5]
    assert fields["applied_mode_request"].tolist() == [5]


def test_manager_state_before_any_request_and_without_tracking():
    modes, _, _ = _manager()
    _, fields = fakes.decode(
        modes.manager_state_message(False, False, Tracking(), []), b"manager_state"
    )
    assert fields["stream_mode"].tolist() == [0]
    assert fields["tracking"].tolist() == [0, 0, 0, 0]
    assert fields["available_modes"].shape == (0,)
    assert fields["applied_mode_request"].tolist() == [0]


def test_mode_request_reader_takes_the_latest_well_formed_request():
    context = zmq.Context.instance()
    publisher = context.socket(zmq.PUB)
    port = publisher.bind_to_random_port("tcp://127.0.0.1")
    reader = server.ModeRequestReader("127.0.0.1", port, context)
    try:
        assert reader.poll() is None  # nothing sent: returns immediately

        def publish(payload):
            publisher.send(server.MODE_REQUEST_TOPIC + payload)

        deadline = time.monotonic() + 5.0
        request = None
        while request is None and time.monotonic() < deadline:  # wait out the PUB/SUB join
            publish(msgpack.packb({"sequence": 1, "mode": 2}))
            time.sleep(0.02)
            request = reader.poll()
        assert request == {"sequence": 1, "mode": 2}

        publisher.send(b"g1_debug" + msgpack.packb({"sequence": 9, "mode": 1}))
        publish(msgpack.packb({"sequence": 2, "mode": 5}))
        publish(b"\xc1")  # not msgpack
        publish(msgpack.packb({"sequence": 0, "mode": 1}))
        publish(msgpack.packb({"sequence": True, "mode": 1}))
        publish(msgpack.packb([3, 1]))
        time.sleep(0.1)
        assert reader.poll() == {"sequence": 2, "mode": 5}
        assert reader.poll() is None
    finally:
        reader.close()
        publisher.close(linger=0)


def test_planner_sends_on_controller_input_alone():
    reader = IsaacTeleopReader(client=fakes.FakeClient())
    socket = fakes.RecordingSocket()
    planner = server.PlannerStreamer(socket=socket, reader=reader, three_point=None, poll_hz=1000)

    def controllers(y):
        return fakes.raw(
            left_controller=fakes.controller([-0.2, y, -0.3], thumbstick=(0.0, 1.0)),
            right_controller=fakes.controller([0.2, y, -0.3]),
        )

    reader._update(controllers(1.0), T0)
    planner.run_once(StreamMode.PLANNER)
    assert len(socket.sent) == 1
    _, fields = fakes.decode(socket.sent[0], b"planner")
    assert np.linalg.norm(fields["movement"]) > 0.5  # left stick pushed: walking

    planner.run_once(StreamMode.PLANNER)
    assert len(socket.sent) == 1  # no fresh input, nothing resent

    reader._update(controllers(1.01), T0 + 10 * MS)
    planner.run_once(StreamMode.PLANNER)
    assert len(socket.sent) == 2

    reader._update(controllers(1.01), T0 + 300 * MS)  # frozen controllers
    planner.run_once(StreamMode.PLANNER)
    assert len(socket.sent) == 2
