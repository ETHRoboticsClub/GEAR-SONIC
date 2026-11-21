import threading
import time

import msgpack
import numpy as np
import pytest
from scipy.spatial.transform import Rotation
import zmq

from gear_sonic.tests.teleop import fakes
from gear_sonic.utils.teleop.input_readers import (
    TRACKING_TOPIC,
    Tracking,
    TrackingReader,
    to_openxr,
)

MS = 1_000_000  # ns
T0 = 10_000 * MS


def _body_positions(offset=0.0):
    rng = np.random.default_rng(0)
    return rng.normal(size=(24, 3)) + offset


def _quest(offset=0.0, head_tracked=True, thumbstick=(0.0, 0.0), **buttons):
    return fakes.raw(
        head=fakes.head([0.0, 1.6 + offset, 0.0], tracked=head_tracked),
        left_controller=fakes.controller([-0.2, 1.0 + offset, -0.3]),
        right_controller=fakes.controller(
            [0.2, 1.0 + offset, -0.3], thumbstick=thumbstick, **buttons
        ),
    )


def test_operator_space_converts_to_openxr_by_physical_direction():
    # Operator space: x forward, y left, z up. OpenXR: forward -Z, left -X, up +Y.
    positions = np.array([[1.0, 0, 0], [0, 1.0, 0], [0, 0, 1.0], [2.0, -3.0, 0.5]])
    identity = np.tile([0.0, 0, 0, 1], (4, 1))
    poses = to_openxr(positions, identity)
    np.testing.assert_allclose(
        poses[:, :3], [[0, 0, -1.0], [-1.0, 0, 0], [0, 1.0, 0], [3.0, 0.5, -2.0]], atol=1e-12
    )
    # A device with OpenXR's own axes (right, up, back) as operator-space columns,
    # turned 90 degrees to the operator's left about up: its forward (-Z) points left.
    openxr_axes = Rotation.from_matrix(np.column_stack([[0, -1.0, 0], [0, 0, 1.0], [-1.0, 0, 0]]))
    turned = (Rotation.from_euler("z", 90, degrees=True) * openxr_axes).as_quat()
    pose = to_openxr(np.zeros((1, 3)), turned[None])[0]
    device_forward = Rotation.from_quat(pose[3:]).apply([0, 0, -1.0])
    np.testing.assert_allclose(device_forward, [-1.0, 0, 0], atol=1e-12)
    np.testing.assert_allclose(
        Rotation.from_quat(pose[3:]).as_matrix(),
        Rotation.from_euler("y", 90, degrees=True).as_matrix(),
        atol=1e-12,
    )


def test_body_is_body_tracking_in_openxr_space():
    positions = _body_positions()
    reader = fakes.reader()
    fakes.feed(reader, fakes.raw(body=fakes.body(positions)), T0)

    assert reader.get_tracking() == Tracking(body=True)
    sample = reader.get_latest()
    assert sample["body_poses_np"].shape == (24, 7)
    np.testing.assert_allclose(sample["body_poses_np"][:, :3], positions, atol=1e-5)
    assert reader.get_timestamp_ns() == T0
    assert reader.get_input_timestamp_ns() == T0
    source, three_point = reader.get_three_point_body()
    assert source == "body"
    np.testing.assert_array_equal(three_point, sample["body_poses_np"])


def test_quest_controllers_and_head():
    reader = fakes.reader()
    fakes.feed(reader, _quest(thumbstick=(0.25, -0.5)), T0)

    assert reader.get_tracking() == Tracking(head=True, controllers=True)
    assert reader.get_latest() is None
    assert reader.get_controller_data()["right_thumbstick"] == [0.25, -0.5]
    source, three_point = reader.get_three_point_body()
    assert source == "headset"
    assert three_point.shape == (24, 7)


def test_controller_buttons_reach_the_controller_dict():
    reader = fakes.reader()
    fakes.feed(reader, _quest(primary=True, menu=True, trigger=0.75), T0)

    data = reader.get_controller_data()
    assert data["right_primary_click"] == 1.0 and data["left_primary_click"] == 0.0
    assert data["right_menu_click"] == 1.0
    assert data["right_trigger_value"] == 0.75


def test_hands_and_head():
    reader = fakes.reader()
    frame = fakes.raw(
        head=fakes.head([0.0, 1.6, 0.0]),
        left_hand=fakes.hand([-0.2, 1.0, -0.3]),
        right_hand=fakes.hand([0.2, 1.0, -0.3]),
    )
    fakes.feed(reader, frame, T0)

    assert reader.get_tracking() == Tracking(head=True, hands=True)
    assert reader.get_controller_data() is None
    assert reader.get_three_point_body()[0] == "headset"


def test_one_controller_or_one_hand_is_not_enough():
    reader = fakes.reader()
    frame = fakes.raw(
        left_controller=fakes.controller([-0.2, 1.0, -0.3]),
        right_hand=fakes.hand([0.2, 1.0, -0.3]),
    )
    fakes.feed(reader, frame, T0)

    assert reader.get_tracking() == Tracking()
    assert reader.get_input_timestamp_ns() == 0
    assert reader.get_three_point_body() is None


def test_nothing_received():
    reader = fakes.reader()

    assert reader.get_tracking() == Tracking()
    assert reader.get_latest() is None
    assert reader.get_controller_data() is None
    assert reader.get_input_timestamp_ns() == 0
    assert reader.get_three_point_body() is None


def test_cached_body_does_not_trigger_another_mapping_but_buttons_still_update():
    reader = fakes.reader()
    frame = {**_quest(), "body": fakes.body(_body_positions()), "body_update_sequence": 1}
    fakes.feed(reader, frame, T0)
    original = reader.get_latest()
    # A new controller packet carries the same body counter and cached pose.
    frame["right_controller"]["primary"] = True
    fakes.feed(reader, frame, T0 + 10 * MS)
    assert reader.get_input_timestamp_ns() == T0 + 10 * MS
    assert reader.get_controller_data()["right_primary_click"] == 1.0
    assert reader.get_latest() is original
    # Stationary body packets still count as fresh when their SDK counter advances.
    frame["body_update_sequence"] = 2
    fakes.feed(reader, frame, T0 + 14 * MS)
    assert reader.get_timestamp_ns() == T0 + 14 * MS
    assert reader.get_latest()["dt"] == pytest.approx(0.014)


def test_head_alone_tracks_the_head_without_three_points():
    reader = fakes.reader()
    fakes.feed(reader, _quest(head_tracked=False), T0)
    assert reader.get_tracking() == Tracking(controllers=True)
    assert reader.get_three_point_body() is None


def test_a_silent_module_tracks_nothing():
    reader = fakes.reader()
    fakes.feed(reader, _quest(), T0)

    reader._clock.now_ns = T0 + 200 * MS
    assert reader.get_tracking() == Tracking(head=True, controllers=True)
    reader._clock.now_ns = T0 + 201 * MS
    assert reader.get_tracking() == Tracking()
    assert reader.get_controller_data() is None
    assert reader.get_three_point_body() is None

    fakes.feed(reader, _quest(offset=0.01), T0 + 250 * MS)
    assert reader.get_tracking() == Tracking(head=True, controllers=True)


def test_a_frame_without_tracking_does_not_advance_the_input_timestamp():
    reader = fakes.reader()
    fakes.feed(reader, _quest(), T0)
    fakes.feed(reader, fakes.raw(), T0 + 10 * MS)

    assert reader.get_input_timestamp_ns() == T0
    assert reader.get_tracking() == Tracking()


def test_wait_for_input_returns_at_once_for_input_not_yet_seen():
    reader = fakes.reader()
    fakes.feed(reader, _quest(), T0)

    start = time.monotonic()
    assert reader.wait_for_input(0, timeout_s=5.0) == T0
    assert time.monotonic() - start < 0.5


def test_wait_for_input_wakes_on_the_next_tracked_frame():
    reader = fakes.reader()
    fakes.feed(reader, _quest(), T0)
    threading.Timer(0.05, fakes.feed, (reader, _quest(offset=0.01), T0 + 10 * MS)).start()

    start = time.monotonic()
    assert reader.wait_for_input(T0, timeout_s=5.0) == T0 + 10 * MS
    assert time.monotonic() - start < 2.0


def test_wait_for_input_times_out_on_frames_that_track_nothing():
    reader = fakes.reader()
    fakes.feed(reader, _quest(), T0)
    threading.Timer(0.01, fakes.feed, (reader, fakes.raw(), T0 + 10 * MS)).start()

    start = time.monotonic()
    assert reader.wait_for_input(T0, timeout_s=0.1) == T0
    assert time.monotonic() - start >= 0.1


def test_losing_the_body_keeps_its_last_sample_but_not_its_tracking():
    reader = fakes.reader()
    fakes.feed(reader, fakes.raw(body=fakes.body(_body_positions())), T0)
    fakes.feed(reader, _quest(), T0 + 10 * MS)

    assert reader.get_tracking() == Tracking(head=True, controllers=True)
    assert reader.get_timestamp_ns() == T0  # body-only consumers see no new sample
    assert reader.get_three_point_body()[0] == "headset"


def test_malformed_frames_are_dropped():
    reader = fakes.reader()
    fakes.feed(reader, _quest(), T0)
    short_body = {"positions": [0.0] * 71, "rotations": [0.0, 0.0, 0.0, 1.0] * 24}
    for frame in (
        {"sample_time_ns": T0 + MS, "body": short_body},
        {"sample_time_ns": T0 + MS, "head": [0.0, 1.0]},
        {"sample_time_ns": T0 + MS, "left_controller": {"grip": [0.0] * 7}},
        {"head": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]},
        None,
    ):
        reader._update(frame, T0 + MS)
        assert reader.get_tracking() == Tracking(head=True, controllers=True)


@pytest.mark.parametrize(
    "position, quat",
    [
        ([np.nan, 1.0, 0.0], [0, 0, 0, 1.0]),
        ([0.0, 1.0, 0.0], [0, np.inf, 0, 1.0]),
        ([0.0, 1.0, 0.0], [0, 0, 0, 0.0]),
        ([0.0, 1.0, 0.0], [0, 0, 0, 1.01]),
    ],
    ids=["nan position", "inf quaternion", "zero quaternion", "non-unit quaternion"],
)
def test_a_frame_with_a_corrupt_pose_is_dropped(position, quat):
    reader = fakes.reader()
    fakes.feed(reader, _quest(), T0)
    corrupt = _quest(offset=0.01)
    corrupt["head"] = [*position, *quat]
    fakes.feed(reader, corrupt, T0 + MS)
    assert reader.get_input_timestamp_ns() == T0, "the corrupt frame was not taken"
    body = fakes.raw(body=fakes.body(_body_positions()))
    body["body"]["positions"][3:6] = position
    body["body"]["rotations"][4:8] = quat
    fakes.feed(reader, body, T0 + 2 * MS)
    assert reader.get_latest() is None


def test_frames_arrive_on_the_feedback_socket():
    context = zmq.Context()
    publisher = context.socket(zmq.PUB)
    port = publisher.bind_to_random_port("tcp://127.0.0.1")
    reader = TrackingReader("127.0.0.1", port, context)
    reader.start()
    try:
        frame = {**_quest(), "sequence": 1, "sample_time_ns": T0}
        for _ in range(200):  # a SUB drops what is published before it connects
            publisher.send(TRACKING_TOPIC + msgpack.packb(frame))
            if reader.get_tracking() != Tracking():
                break
            time.sleep(0.01)
        assert reader.get_tracking() == Tracking(head=True, controllers=True)
        assert reader.get_input_timestamp_ns() == T0
    finally:
        reader.stop()
        publisher.close(linger=0)
        context.term()
