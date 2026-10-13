import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from gear_sonic.tests.teleop import fakes
from gear_sonic.utils.teleop.input_readers import IsaacTeleopReader, Tracking

MS = 1_000_000  # ns
T0 = 10_000 * MS


def _reader():
    return IsaacTeleopReader(client=fakes.FakeClient())


def _body_positions(offset=0.0):
    rng = np.random.default_rng(0)
    return rng.normal(size=(24, 3)).astype(np.float32) + np.float32(offset)


def _quest(offset=0.0, head_tracked=True, thumbstick=(0.0, 0.0)):
    return fakes.raw(
        head=fakes.head([0.0, 1.6 + offset, 0.0], tracked=head_tracked),
        left_controller=fakes.controller([-0.2, 1.0 + offset, -0.3]),
        right_controller=fakes.controller([0.2, 1.0 + offset, -0.3], thumbstick=thumbstick),
    )


def test_pico_body_alone_is_body_tracking():
    reader = _reader()
    reader._update(fakes.raw(full_body=fakes.body(_body_positions())), T0)

    assert reader.get_tracking() == Tracking(body=True)
    sample = reader.get_latest()
    assert sample["body_poses_np"].shape == (24, 7)
    assert reader.get_timestamp_ns() == T0
    assert reader.get_input_timestamp_ns() == T0
    source, three_point = reader.get_three_point_body()
    assert source == "body"
    np.testing.assert_array_equal(three_point, sample["body_poses_np"])


def test_xsens_hand_joints_take_the_wrist_poses():
    positions = _body_positions()
    orientations = np.tile(np.array([0, 0, 0, 1], dtype=np.float32), (24, 1))
    orientations[20] = Rotation.from_euler("x", 30, degrees=True).as_quat()
    orientations[21] = Rotation.from_euler("y", -40, degrees=True).as_quat()
    valid = np.ones(24, dtype=np.uint8)
    valid[22] = valid[23] = 0  # MVN: LEFT_HAND/RIGHT_HAND are invalid copies
    positions[22] = positions[23] = 99.0  # whatever an invalid joint carries must not leak

    reader = _reader()
    reader._update(fakes.raw(full_body=fakes.body(positions, orientations, valid)), T0)

    assert reader.get_tracking().body
    poses = reader.get_latest()["body_poses_np"]
    np.testing.assert_array_equal(poses[22, :3], positions[20])
    np.testing.assert_array_equal(poses[23, :3], positions[21])
    np.testing.assert_array_equal(poses[22, 3:], orientations[20])
    np.testing.assert_array_equal(poses[23, 3:], orientations[21])


@pytest.mark.parametrize("invalid_joint", [0, 7, 12, 20])
def test_body_with_an_untracked_joint_is_not_body_tracking(invalid_joint):
    valid = np.ones(24, dtype=np.uint8)
    valid[invalid_joint] = 0
    if invalid_joint == 20:
        valid[22] = 0  # the hand joint cannot borrow an invalid wrist either
    reader = _reader()
    reader._update(fakes.raw(full_body=fakes.body(_body_positions(), valid=valid)), T0)

    assert reader.get_tracking() == Tracking()
    assert reader.get_latest() is None
    assert reader.get_three_point_body() is None


def test_quest_controllers_and_head():
    reader = _reader()
    reader._update(_quest(), T0)

    assert reader.get_tracking() == Tracking(head=True, controllers=True)
    assert reader.get_latest() is None
    assert reader.get_controller_data()["right_thumbstick"] == [0.0, 0.0]
    source, three_point = reader.get_three_point_body()
    assert source == "headset"
    assert three_point.shape == (24, 7)


def test_avp_hands_and_head():
    reader = _reader()
    reader._update(
        fakes.raw(
            head=fakes.head([0.0, 1.6, 0.0]),
            left_hand=fakes.hand([-0.2, 1.0, -0.3]),
            right_hand=fakes.hand([0.2, 1.0, -0.3]),
        ),
        T0,
    )

    assert reader.get_tracking() == Tracking(head=True, hands=True)
    assert reader.get_controller_data() is None
    assert reader.get_three_point_body()[0] == "headset"


def test_one_controller_or_one_hand_is_not_enough():
    reader = _reader()
    reader._update(
        fakes.raw(
            left_controller=fakes.controller([-0.2, 1.0, -0.3]),
            right_controller=fakes.controller([0.2, 1.0, -0.3], valid=False),
            left_hand=fakes.hand([-0.2, 1.0, -0.3], valid=False),
            right_hand=fakes.hand([0.2, 1.0, -0.3]),
        ),
        T0,
    )

    assert reader.get_tracking() == Tracking()
    assert reader.get_input_timestamp_ns() == 0


def test_nothing_connected():
    reader = _reader()
    reader._update(None, T0)

    assert reader.get_tracking() == Tracking()
    assert reader.get_latest() is None
    assert reader.get_controller_data() is None
    assert reader.get_input_timestamp_ns() == 0
    assert reader.get_three_point_body() is None
    np.testing.assert_array_equal(reader.get_tracking().as_array(), [0, 0, 0, 0])


def test_head_that_is_valid_but_not_tracked_is_not_head_tracking():
    reader = _reader()
    reader._update(_quest(head_tracked=False), T0)

    assert reader.get_tracking() == Tracking(controllers=True)
    assert reader.get_three_point_body() is None


def test_frozen_body_stops_being_tracked():
    reader = _reader()
    frozen = fakes.raw(full_body=fakes.body(_body_positions()))
    reader._update(frozen, T0)
    first_sample = reader.get_latest()

    reader._update(frozen, T0 + 100 * MS)
    assert reader.get_tracking().body
    assert reader.get_latest() is first_sample  # nothing new to stream
    assert reader.get_input_timestamp_ns() == T0

    reader._update(frozen, T0 + 201 * MS)
    assert reader.get_tracking() == Tracking()
    assert reader.get_three_point_body() is None

    moved = _body_positions()
    moved[5, 0] = np.nextafter(moved[5, 0], np.float32(np.inf))  # one ulp is a new sample
    reader._update(fakes.raw(full_body=fakes.body(moved)), T0 + 210 * MS)
    assert reader.get_tracking().body
    assert reader.get_timestamp_ns() == T0 + 210 * MS
    assert reader.get_input_timestamp_ns() == T0 + 210 * MS


def test_frozen_controller_inputs_are_dropped():
    reader = _reader()
    pushing_forward = _quest(thumbstick=(0.0, 1.0))
    reader._update(pushing_forward, T0)
    assert reader.get_controller_data()["right_thumbstick"] == [0.0, 1.0]

    reader._update(pushing_forward, T0 + 250 * MS)
    tracking = reader.get_tracking()
    assert not tracking.controllers and not tracking.head
    assert reader.get_controller_data() is None


def test_controllers_alone_advance_the_input_timestamp():
    reader = _reader()
    controllers = fakes.raw(
        left_controller=fakes.controller([-0.2, 1.0, -0.3]),
        right_controller=fakes.controller([0.2, 1.0, -0.3]),
    )
    reader._update(controllers, T0)
    assert reader.get_tracking() == Tracking(controllers=True)
    assert reader.get_input_timestamp_ns() == T0

    reader._update(controllers, T0 + 10 * MS)
    assert reader.get_input_timestamp_ns() == T0

    pressed = fakes.raw(
        left_controller=fakes.controller([-0.2, 1.0, -0.3]),
        right_controller=fakes.controller([0.2, 1.0, -0.3], primary_click=True),
    )
    reader._update(pressed, T0 + 20 * MS)
    assert reader.get_input_timestamp_ns() == T0 + 20 * MS
