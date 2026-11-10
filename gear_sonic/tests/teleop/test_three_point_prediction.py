"""ThreePointPredictor: oracles are closed-form motions (constant linear and
angular velocity) sampled at a headset rate, not the predictor's own math."""

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from gear_sonic.scripts import pico_manager_thread_server as server
from gear_sonic.tests.teleop import fakes
from gear_sonic.utils.teleop.prediction import ThreePointPredictor

MS = 1_000_000
PERIOD_NS = round(1e9 / 90)
BASE = np.array([[0.3, 0.2, 0.1], [0.3, -0.2, 0.1], [0.0, 0.0, 0.4]])


def _poses(t_s, velocity=np.zeros(3), omega=np.zeros(3)):
    pose = np.zeros((3, 7))
    pose[:, :3] = BASE + velocity * t_s
    pose[:, 3:] = Rotation.from_rotvec(omega * t_s).as_quat(scalar_first=True)
    return pose


def _feed(predictor, n, **motion):
    out = None
    for i in range(n):
        out = predictor(10_000 * MS + i * PERIOD_NS, _poses(i * PERIOD_NS * 1e-9, **motion))
    return out, (n - 1) * PERIOD_NS * 1e-9


def test_constant_velocity_is_extrapolated_by_the_lead():
    velocity = np.array([0.4, -0.2, 0.1])  # m/s, a brisk reach
    predictor = ThreePointPredictor(0.05)
    out, t_s = _feed(predictor, 10, velocity=velocity)
    np.testing.assert_allclose(out[:, :3], BASE + velocity * (t_s + 0.05), atol=1e-9)


def test_constant_angular_velocity_is_extrapolated_by_the_lead():
    omega = np.array([0.0, 0.0, 2.0])  # rad/s about +z
    predictor = ThreePointPredictor(0.05)
    out, t_s = _feed(predictor, 10, omega=omega)
    expected = Rotation.from_rotvec(omega * (t_s + 0.05))
    got = Rotation.from_quat(out[:, 3:], scalar_first=True)
    assert np.all((got * expected.inv()).magnitude() < 1e-9)


def test_still_input_is_unchanged():
    out, _ = _feed(ThreePointPredictor(0.08), 10)
    np.testing.assert_allclose(out, _poses(0.0), atol=1e-12)


def test_the_lead_is_clamped():
    predictor = ThreePointPredictor(0.1, max_shift_m=0.05, max_turn_rad=0.2)
    out, t_s = _feed(predictor, 10, velocity=np.array([3.0, 0, 0]), omega=np.array([0, 10.0, 0]))
    shift = out[:, :3] - (BASE + np.array([3.0, 0, 0]) * t_s)
    np.testing.assert_allclose(np.linalg.norm(shift, axis=1), 0.05, atol=1e-9)
    newest = Rotation.from_rotvec(np.array([0, 10.0, 0]) * t_s)
    got = Rotation.from_quat(out[:, 3:], scalar_first=True)
    np.testing.assert_allclose((got * newest.inv()).magnitude(), 0.2, atol=1e-9)


@pytest.mark.parametrize("seed", range(3))
def test_arrival_jitter_does_not_disturb_a_steady_motion(seed):
    # The headset samples at 90 Hz; the stamps are arrival times, which Wi-Fi
    # and TCP delay by a few milliseconds each, in order.
    velocity = np.array([0.4, -0.2, 0.1])
    predictor = ThreePointPredictor(0.05)
    rng = np.random.default_rng(seed)
    arrival_ns, errors = 0, []
    for i in range(90):
        sample_ns = 10_000 * MS + i * PERIOD_NS
        arrival_ns = max(arrival_ns + 1, sample_ns + int(rng.exponential(4 * MS)))
        t_s = i * PERIOD_NS * 1e-9
        out = predictor(arrival_ns, _poses(t_s, velocity=velocity))
        if i >= 10:
            errors.append(np.abs(out[:, :3] - (BASE + velocity * (t_s + 0.05))).max())
    assert max(errors) < 0.003  # m; slopes against the stamps erred by 7-41 mm here


def test_a_gap_restarts_the_estimate():
    predictor = ThreePointPredictor(0.05)
    _feed(predictor, 10, velocity=np.array([0.5, 0, 0]))
    resumed = _poses(2.0)
    out = predictor(10_000 * MS + 2_000 * MS, resumed)  # 2 s later: no velocity yet
    np.testing.assert_array_equal(out, resumed)


def test_input_is_not_modified():
    predictor = ThreePointPredictor(0.05)
    for i in range(5):
        pose = _poses(i * PERIOD_NS * 1e-9, velocity=np.array([0.5, 0, 0]))
        before = pose.copy()
        predictor(10_000 * MS + i * PERIOD_NS, pose)
        np.testing.assert_array_equal(pose, before)


@pytest.mark.parametrize("lead_s", [0.0, -0.01])
def test_a_non_positive_lead_is_refused(lead_s):
    with pytest.raises(ValueError):
        ThreePointPredictor(lead_s)


def _planner(lead_s):
    reader = fakes.reader()
    socket = fakes.RecordingSocket()
    three_point = server.ThreePointPose(robot_model=fakes.FakeRobotModel())
    planner = server.PlannerStreamer(
        socket, reader, three_point, zmq_feedback_port=59999, three_point_lead_s=lead_s
    )
    planner.feedback_reader.poll_feedback = lambda: None
    return reader, socket, planner


def _headset(x):
    return fakes.raw(
        head=fakes.head([0.0, 1.6, 0.0]),
        left_controller=fakes.controller([-0.2 + x, 1.0, -0.3]),
        right_controller=fakes.controller([0.2 + x, 1.0, -0.3]),
    )


def _sent_wrists(socket):
    _, fields = fakes.decode(socket.sent[-1], b"planner")
    return fields["vr_position"].reshape(3, 3)[:2]


def test_the_planner_sends_led_targets_only_when_asked(monkeypatch):
    monkeypatch.setattr(server, "get_g1_key_frame_poses", fakes.g1_key_frame_poses)
    speed = 0.5  # m/s along OpenXR +X (the operator's right)
    sent = {}
    for lead_s in (0.0, 0.05):
        reader, socket, planner = _planner(lead_s)
        fakes.feed(reader, _headset(0.0), 10_000 * MS)
        planner.calibrate_now()
        for i in range(1, 10):
            fakes.feed(reader, _headset(speed * i * PERIOD_NS * 1e-9), 10_000 * MS + i * PERIOD_NS)
            planner.run_once(server.StreamMode.PLANNER_VR_3PT)
        sent[lead_s] = _sent_wrists(socket)
    # The operator's right is the robot's -y: the lead moves both wrists 2.5 cm that way.
    np.testing.assert_allclose(sent[0.05] - sent[0.0], [[0, -0.025, 0]] * 2, atol=1e-6)
