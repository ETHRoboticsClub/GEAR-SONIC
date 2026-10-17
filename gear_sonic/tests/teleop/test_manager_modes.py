"""Manager loop: mode requests, controller buttons and manager_state.

Every test drives ``StreamModeManager.step``, the body of run_pico_manager's
loop. Readers are doubles at the reader boundary (``ScriptedReader``) or the
real TrackingReader fed with tracking frames.
"""

import time

import msgpack
import numpy as np
import pytest
from scipy.spatial.transform import Rotation
import zmq

from gear_sonic.scripts import pico_manager_thread_server as server
from gear_sonic.scripts.pico_manager_thread_server import StreamMode
from gear_sonic.tests.teleop import fakes
from gear_sonic.utils.teleop.input_readers import (
    Tracking,
    TrackingReader,
    synthesize_three_point_body,
)

MS = 1_000_000
T0 = 10_000 * MS

ALL = [StreamMode.POSE, StreamMode.PLANNER, StreamMode.PLANNER_VR_3PT]
BODY = Tracking(body=True)
HEADSET = Tracking(head=True, controllers=True)

BUTTONS = {
    "A": "right_primary_click",
    "B": "right_secondary_click",
    "X": "left_primary_click",
    "Y": "left_secondary_click",
    "click": "left_thumbstick_click",
}


class ScriptedReader(TrackingReader):
    """TrackingReader whose tracking, buttons and 3-point input are set by the test."""

    def __init__(self, tracking=Tracking()):
        super().__init__()
        self.tracking = tracking
        self.pressed = set()
        self.left_stick = (0.0, 0.0)
        self.three_point = None
        self._input_ns = 0

    def get_tracking(self):
        return self.tracking

    def get_controller_data(self):
        data = {key: float(name in self.pressed) for name, key in BUTTONS.items()}
        data["left_thumbstick"] = list(self.left_stick)
        return data

    def get_three_point_body(self):
        return self.three_point

    def get_input_timestamp_ns(self):
        self._input_ns += 1  # every iteration brings fresh input
        return self._input_ns


class Requests:
    """``ModeRequestReader`` double: each poll returns the next queued message."""

    def __init__(self):
        self.queue = []

    def send(self, sequence, mode):
        self.queue.append({"sequence": sequence, "mode": getattr(mode, "value", mode)})

    def poll(self):
        return self.queue.pop(0) if self.queue else None


class FakePoseStreamer:
    def __init__(self, calls):
        self.calls = calls

    def on_mode_exit(self):
        self.calls.append("pose.on_mode_exit")

    def reset_yaw(self):
        self.calls.append("pose.reset_yaw")

    def run_once(self):
        self.calls.append("pose.run_once")


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

    def run_once(self, mode):
        self.calls.append(f"planner.run_once({mode.name})")


class Loop:
    """A StreamModeManager wired to doubles, stepped like run_pico_manager does."""

    def __init__(self, reader=None, planner=None, tracking=BODY):
        self.calls = []
        self.socket = fakes.RecordingSocket()
        self.reader = reader if reader is not None else ScriptedReader(tracking)
        self.requests = Requests()
        if planner is None:
            planner = FakePlannerStreamer(self.calls)
        else:
            planner.socket = self.socket
        self.modes = server.StreamModeManager(
            self.socket, self.reader, FakePoseStreamer(self.calls), planner
        )

    def step(self, times=1):
        for _ in range(times):
            self.modes.step(self.requests)

    def press(self, *buttons):
        """One iteration with ``buttons`` held, then one with them released."""
        self.reader.pressed = set(buttons)
        self.step()
        self.reader.pressed = set()
        self.step()

    def start(self):
        """A+B+X+Y: the only way control starts."""
        self.press("A", "B", "X", "Y")
        assert self.mode == StreamMode.PLANNER

    @property
    def mode(self):
        return self.modes.current_mode

    def messages(self, topic):
        return [fakes.decode(m, topic)[1] for m in self.socket.sent if m.startswith(topic)]

    def commands(self):
        return [
            (int(f["start"][0]), int(f["stop"][0]), int(f["planner"][0]))
            for f in self.messages(b"command")
        ]

    def manager_state(self):
        return self.messages(b"manager_state")[-1]

    def switch_calls(self):
        return [c for c in self.calls if "run_once" not in c]


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


def test_a_request_never_starts_control(capsys):
    loop = Loop()
    loop.step()
    assert loop.manager_state()["available_modes"].shape == (0,)  # nothing in OFF

    loop.requests.send(1, StreamMode.PLANNER)
    loop.step(3)

    assert loop.mode == StreamMode.OFF
    assert loop.calls == [] and loop.commands() == []
    assert loop.manager_state()["applied_mode_request"].tolist() == [1]
    assert loop.manager_state()["available_modes"].shape == (0,)
    assert "refused: control starts on the controllers (A+B+X+Y)" in capsys.readouterr().out

    loop.start()
    assert loop.switch_calls() == ["planner.calibrate_now", "planner.reset_yaw"]
    assert loop.manager_state()["available_modes"].tolist() == [1, 2, 5]
    loop.requests.send(2, StreamMode.POSE)  # the same kind of request now works
    loop.step(2)
    assert loop.mode == StreamMode.POSE
    assert loop.modes.applied_mode_request == 2
    assert loop.commands() == [(1, 0, 1), (1, 0, 0)]


def test_request_from_pose_to_vr3pt_steps_through_planner():
    loop = Loop()
    loop.start()
    loop.press("A", "X")
    assert loop.mode == StreamMode.POSE
    loop.calls.clear()
    loop.requests.send(1, StreamMode.PLANNER_VR_3PT)

    loop.step()
    assert loop.mode == StreamMode.PLANNER
    assert loop.manager_state()["applied_mode_request"].tolist() == [0]
    loop.step()
    assert loop.mode == StreamMode.PLANNER_VR_3PT
    loop.step()
    assert loop.manager_state()["applied_mode_request"].tolist() == [1]
    assert loop.switch_calls() == [
        "pose.on_mode_exit",
        "planner.reset_yaw",
        "planner.recalibrate_for_vr3pt",
    ]
    assert loop.commands()[-2:] == [(1, 0, 1), (1, 0, 1)]


def test_request_for_an_unavailable_mode_is_refused():
    loop = Loop(tracking=Tracking(controllers=True))
    loop.start()
    commands, calls = loop.commands(), list(loop.calls)
    loop.requests.send(4, StreamMode.POSE)

    loop.step(3)

    assert loop.mode == StreamMode.PLANNER
    assert loop.manager_state()["applied_mode_request"].tolist() == [4]
    assert loop.switch_calls() == [c for c in calls if "run_once" not in c]
    assert loop.commands() == commands


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
    loop = Loop()
    loop.start()
    loop.requests.send(2, mode)

    loop.step(2)

    assert loop.mode == StreamMode.PLANNER
    assert loop.modes.applied_mode_request == 2
    assert loop.commands() == [(1, 0, 1)]


def test_target_lost_midway_is_refused_where_the_steps_stopped():
    loop = Loop()
    loop.start()
    loop.press("A", "X")
    loop.requests.send(1, StreamMode.PLANNER_VR_3PT)
    loop.step()
    assert loop.mode == StreamMode.PLANNER

    loop.reader.tracking = Tracking(controllers=True)  # VR 3PT no longer available
    loop.step(2)

    assert loop.mode == StreamMode.PLANNER
    assert loop.modes.applied_mode_request == 1


def test_resent_request_is_applied_once():
    loop = Loop()
    loop.start()
    loop.requests.send(7, StreamMode.PLANNER_VR_3PT)
    loop.step(2)
    assert loop.modes.applied_mode_request == 7
    loop.press("A", "X")  # the operator moves on to POSE
    assert loop.mode == StreamMode.POSE
    commands = loop.commands()

    for _ in range(3):  # the robot keeps resending sequence 7
        loop.requests.send(7, StreamMode.PLANNER_VR_3PT)
        loop.step()

    assert loop.mode == StreamMode.POSE
    assert loop.commands() == commands

    loop.requests.send(8, StreamMode.PLANNER)
    loop.step()
    assert loop.mode == StreamMode.PLANNER


def test_planner_vr3pt_round_trip_uses_the_button_side_effects():
    loop = Loop()
    loop.start()
    loop.calls.clear()

    loop.requests.send(2, StreamMode.PLANNER_VR_3PT)
    loop.step()
    assert loop.mode == StreamMode.PLANNER_VR_3PT
    assert loop.modes.vr3pt_parent_mode == StreamMode.PLANNER
    assert loop.switch_calls() == ["planner.recalibrate_for_vr3pt"]

    loop.requests.send(3, StreamMode.PLANNER)
    loop.step(2)
    assert loop.mode == StreamMode.PLANNER
    # Returning from the VR 3PT sub-mode keeps the planner's yaw.
    assert loop.switch_calls() == ["planner.recalibrate_for_vr3pt"]
    assert loop.modes.applied_mode_request == 3


def _enter_frozen(loop):
    loop.press("A", "B", "X", "Y")  # OFF -> PLANNER
    loop.press("A", "X")  # PLANNER -> POSE
    loop.press("B", "Y")  # POSE -> FROZEN
    assert loop.mode == StreamMode.PLANNER_FROZEN_UPPER_BODY


@pytest.mark.parametrize("tracking", [BODY, HEADSET], ids=["pose available", "no pose"])
@pytest.mark.parametrize("in_vr3pt", [False, True], ids=["frozen", "vr3pt from frozen"])
def test_requests_never_pass_through_pose(tracking, in_vr3pt):
    loop = Loop()
    _enter_frozen(loop)
    if in_vr3pt:
        loop.press("click")
        assert loop.mode == StreamMode.PLANNER_VR_3PT
    loop.reader.tracking = tracking
    mode, commands = loop.mode, loop.commands()

    loop.requests.send(1, StreamMode.PLANNER)
    loop.step(3)

    # Every path to PLANNER from here runs through POSE: refused, nothing sent.
    assert loop.mode == mode
    assert loop.commands() == commands
    assert loop.modes.applied_mode_request == 1


def test_no_request_produces_back_to_back_opposite_commands():
    for tracking in (BODY, HEADSET, Tracking(head=True, hands=True), Tracking(controllers=True)):
        available = server.available_modes(tracking)
        # Post-start modes only; POSE_PAUSE is held by a button.
        for start in set(StreamMode) - {StreamMode.OFF, StreamMode.POSE_PAUSE}:
            for target in available:
                loop = Loop(tracking=tracking)
                loop.modes.current_mode = start
                loop.requests.send(1, target)
                loop.step(4)
                flags = [c[2] for c in loop.commands()]  # planner=1, pose=0
                flips = [a == c != b for a, b, c in zip(flags, flags[1:], flags[2:])]
                assert not any(flips), (tracking, start, target, loop.commands())
                assert loop.mode in (start, target)


def test_buttons_cancel_a_queued_request(capsys):
    loop = Loop()
    loop.start()
    loop.press("A", "X")
    loop.requests.send(3, StreamMode.PLANNER_VR_3PT)
    loop.step()
    assert loop.mode == StreamMode.PLANNER  # first hop of POSE -> VR 3PT

    loop.press("A", "X")  # the operator goes back to POSE before the request finished

    assert loop.mode == StreamMode.POSE
    assert loop.modes.applied_mode_request == 3
    assert "buttons took over" in capsys.readouterr().out
    loop.step(3)
    assert loop.mode == StreamMode.POSE


def test_manager_state_reports_tracking_available_modes_and_applied_request():
    loop = Loop(tracking=HEADSET)
    loop.start()
    loop.requests.send(5, StreamMode.PLANNER)
    loop.step()

    header, _ = fakes.decode(loop.socket.sent[-1], b"manager_state")
    fields = loop.manager_state()
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
    assert fields["tracking"].tolist() == [0, 1, 1, 0]
    assert fields["available_modes"].tolist() == [2, 5]
    assert fields["applied_mode_request"].tolist() == [5]


def test_manager_state_is_published_every_iteration_without_tracking():
    loop = Loop(tracking=Tracking())
    loop.step(3)
    states = loop.messages(b"manager_state")
    assert len(states) == 3
    assert states[-1]["stream_mode"].tolist() == [0]
    assert states[-1]["tracking"].tolist() == [0, 0, 0, 0]
    assert states[-1]["available_modes"].shape == (0,)
    assert states[-1]["applied_mode_request"].tolist() == [0]


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


# --- Loops with the real PlannerStreamer and ThreePointPose -------------------


@pytest.fixture
def planner(monkeypatch):
    monkeypatch.setattr(server, "get_g1_key_frame_poses", fakes.g1_key_frame_poses)

    def make(reader):
        three_point = server.ThreePointPose(robot_model=fakes.FakeRobotModel())
        return server.PlannerStreamer(socket=None, reader=reader, three_point=three_point)

    return make


def test_planner_sends_on_controller_input_alone(planner):
    reader = fakes.reader()
    socket = fakes.RecordingSocket()
    streamer = planner(reader)
    streamer.socket = socket

    def controllers(y):
        return fakes.raw(
            left_controller=fakes.controller([-0.2, y, -0.3], thumbstick=(0.0, 1.0)),
            right_controller=fakes.controller([0.2, y, -0.3]),
        )

    fakes.feed(reader, controllers(1.0), T0)
    streamer.run_once(StreamMode.PLANNER)
    assert len(socket.sent) == 1
    _, fields = fakes.decode(socket.sent[0], b"planner")
    assert np.linalg.norm(fields["movement"]) > 0.5  # left stick pushed: walking

    streamer.run_once(StreamMode.PLANNER)
    assert len(socket.sent) == 1  # no fresh input, nothing resent

    fakes.feed(reader, controllers(1.01), T0 + 10 * MS)
    streamer.run_once(StreamMode.PLANNER)
    assert len(socket.sent) == 2

    fakes.feed(reader, fakes.raw(), T0 + 300 * MS)  # the module lost the controllers
    streamer.run_once(StreamMode.PLANNER)
    assert len(socket.sent) == 2


def _turning(right_x, y):
    return fakes.raw(
        left_controller=fakes.controller([-0.2, y, -0.3]),
        right_controller=fakes.controller([0.2, y, -0.3], thumbstick=(right_x, 0.0)),
    )


def _facing_after(planner, frame_times_ms, right_x=1.0):
    """Planner heading (rad) after one frame at each of ``frame_times_ms``."""
    reader = fakes.reader()
    streamer = planner(reader)
    streamer.socket = fakes.RecordingSocket()
    for i, t_ms in enumerate(frame_times_ms):
        fakes.feed(reader, _turning(right_x, 1.0 + 1e-3 * i), T0 + t_ms * MS)
        streamer.run_once(StreamMode.PLANNER)
    _, fields = fakes.decode(streamer.socket.sent[-1], b"planner")
    facing = fields["facing"].reshape(-1)
    return np.arctan2(facing[1], facing[0])


def test_planner_sends_every_fresh_frame_without_waiting(planner, monkeypatch):
    def no_sleep(_):
        raise AssertionError("the planner paced itself")

    monkeypatch.setattr(server.time, "sleep", no_sleep)
    reader = fakes.reader()
    streamer = planner(reader)
    streamer.socket = fakes.RecordingSocket()
    for i in range(9):  # 90 Hz tracking
        fakes.feed(reader, _turning(0.0, 1.0 + 1e-3 * i), T0 + round(i * 1000 / 90) * MS)
        streamer.run_once(StreamMode.PLANNER)
    assert len(streamer.socket.sent) == 9


def test_turn_rate_is_independent_of_the_frame_rate(planner):
    # Right stick fully right for 0.6 s turns right by yaw_gain (1.5 rad/s) * 0.6 s.
    expected = -1.5 * 0.6
    at_100_hz = _facing_after(planner, range(0, 601, 10))
    at_20_hz = _facing_after(planner, range(0, 601, 50))
    assert at_100_hz == pytest.approx(expected, abs=1e-6)
    assert at_20_hz == pytest.approx(expected, abs=1e-6)


def test_a_tracking_gap_turns_no_further_than_the_stale_limit(planner):
    # 0.1 s of frames, a 1 s gap, then one frame: the gap counts as 0.2 s.
    assert _facing_after(planner, [0, 50, 100, 1100]) == pytest.approx(-1.5 * 0.3, abs=1e-6)


def test_only_the_buttons_calibrate(planner):
    reader = ScriptedReader(HEADSET)
    reader.three_point = ("headset", _headset_body(heading_deg=0.0))
    streamer = planner(reader)
    loop = Loop(reader=reader, planner=streamer)

    loop.requests.send(1, StreamMode.PLANNER)
    loop.step(2)
    assert not streamer.three_point.is_calibrated
    assert loop.commands() == []

    loop.start()
    assert streamer.three_point.is_calibrated
    assert streamer.three_point_source == "headset"


def test_vr3pt_without_three_point_input_still_sends_locomotion(planner):
    reader = ScriptedReader(HEADSET)
    loop = Loop(reader=reader, planner=planner(reader))
    loop.start()
    loop.requests.send(1, StreamMode.PLANNER_VR_3PT)
    loop.step(2)
    assert loop.mode == StreamMode.PLANNER_VR_3PT

    reader.left_stick = (0.0, 1.0)
    loop.step()

    planner_messages = loop.messages(b"planner")
    last = planner_messages[-1]
    assert np.linalg.norm(last["movement"]) > 0.5
    assert "vr_position" not in last and "vr_orientation" not in last


def _headset_body(heading_deg, head_yaw_deg=0.0):
    """Synthesized XR_BD body of an operator facing ``heading_deg`` about +Y."""
    heading = Rotation.from_euler("y", heading_deg, degrees=True)
    head = np.concatenate(
        [
            heading.apply([0.0, 1.65, 0.0]),
            (heading * Rotation.from_euler("y", head_yaw_deg, degrees=True)).as_quat(),
        ]
    )
    grip = heading * Rotation.from_matrix(np.column_stack([[-1, 0, 0], [0, 0, -1], [0, -1, 0]]))
    wrists = [
        np.concatenate([heading.apply([x, 1.05, -0.15]), grip.as_quat()]) for x in (-0.15, 0.25)
    ]
    return synthesize_three_point_body(head, *wrists, "grip")


def test_switching_three_point_source_recalibrates_the_neck(planner):
    reader = ScriptedReader(BODY)
    # The body tracker's yaw origin: the operator faced -Z at calibration.
    reader.three_point = ("body", _headset_body(heading_deg=0.0))
    streamer = planner(reader)
    loop = Loop(reader=reader, planner=streamer)
    loop.press("A", "B", "X", "Y")  # OFF -> PLANNER, calibrates from the body
    loop.press("click")  # VR 3PT
    assert loop.mode == StreamMode.PLANNER_VR_3PT
    loop.press("click")  # back to PLANNER

    # Body tracking is lost; the headset's yaw origin differs by 90 degrees.
    headset = _headset_body(heading_deg=-90.0)
    reader.tracking = HEADSET
    reader.three_point = ("headset", headset)
    loop.step()
    loop.press("click")  # VR 3PT again: recalibrates against the measured q
    assert loop.mode == StreamMode.PLANNER_VR_3PT
    loop.step()
    last = loop.messages(b"planner")[-1]
    position, orientation = last["vr_position"].reshape(3, 3), last["vr_orientation"].reshape(3, 4)

    fresh = server.ThreePointPose(robot_model=fakes.FakeRobotModel())
    fresh.reset_with_measured_q(np.zeros(29))
    expected = fresh.process_smpl_pose(headset)
    np.testing.assert_allclose(position, expected[:, :3], atol=1e-5)
    np.testing.assert_allclose(orientation, expected[:, 3:], atol=1e-5)
    neck = Rotation.from_quat(orientation[2], scalar_first=True)
    assert neck.magnitude() == pytest.approx(0.0, abs=1e-5)  # no constant torso offset


def test_three_point_input_from_another_source_is_withheld_until_reentry(planner):
    reader = ScriptedReader(BODY)
    reader.three_point = ("body", _headset_body(heading_deg=0.0))
    loop = Loop(reader=reader, planner=planner(reader))
    loop.press("A", "B", "X", "Y")
    loop.press("click")
    assert loop.mode == StreamMode.PLANNER_VR_3PT

    reader.tracking = HEADSET
    reader.three_point = ("headset", _headset_body(heading_deg=-90.0))
    loop.step(2)
    reader.three_point = ("body", _headset_body(heading_deg=0.0))  # and back again
    loop.step(2)

    for message in loop.messages(b"planner")[-4:]:
        assert "vr_position" not in message  # the dropped calibration is not reused
