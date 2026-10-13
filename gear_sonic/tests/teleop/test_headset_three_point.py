"""Headset three-point pose (head + controllers or hand tracking, no body tracker).

Oracles come from physical descriptions and the G1 URDF, not from the
synthesis code: the operator stands facing -Z in the OpenXR reference space
(+X right, +Y up) and calibrates in the G1 zero pose (upper arms down,
forearms forward, thumbs up, palms facing each other). Controller grip frames
are built from the OpenXR grip definition and hand-tracking wrist frames from
the XR_EXT_hand_tracking definition.
"""

from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from gear_sonic.scripts import pico_manager_thread_server as server
from gear_sonic.tests.teleop import fakes
from gear_sonic.utils.teleop.input_readers import IsaacTeleopReader

URDF = Path(server.__file__).parents[1] / "data/robots/g1/g1_29dof_with_hand.urdf"
MS = 1_000_000
T0 = 10_000 * MS

# Physical directions of the operator facing -Z, as OpenXR vectors.
FORWARD, LEFT, UP = np.array([0, 0, -1.0]), np.array([-1.0, 0, 0]), np.array([0, 1.0, 0])


def _robot_to_openxr(v):
    """Robot (x forward, y left, z up) direction as an OpenXR vector."""
    return v[0] * FORWARD + v[1] * LEFT + v[2] * UP


def _frame(x, y, z):
    """Rotation whose local axes are the given OpenXR directions."""
    matrix = np.column_stack([x, y, z])
    assert np.allclose(matrix @ matrix.T, np.eye(3)) and np.isclose(np.linalg.det(matrix), 1.0)
    return Rotation.from_matrix(matrix)


def _grip(side):
    """OpenXR grip of a hand in the calibration pose: +X out of the left palm and
    into the right palm, -Z through the closed fist toward the thumb (up)."""
    # Both palms face medially, so for both hands +X points to the operator's left.
    x = LEFT
    z = -UP
    return _frame(x, np.cross(z, x), z)


def _hand_wrist(side):
    """XR_EXT_hand_tracking wrist in the calibration pose: -Z toward the fingertips
    (forward), +Y out of the back of the hand (lateral)."""
    z = -FORWARD
    y = LEFT if side == "left" else -LEFT
    return _frame(np.cross(y, z), y, z)


# --- G1 model from the URDF -------------------------------------------------


def _urdf_joints():
    joints = {}
    for joint in ET.parse(URDF).getroot().findall("joint"):
        origin = joint.find("origin")
        if origin is None:  # URDF default: identity
            origin = ET.Element("origin")
        axis = joint.find("axis")
        joints[joint.get("name")] = dict(
            parent=joint.find("parent").get("link"),
            child=joint.find("child").get("link"),
            xyz=np.array([float(v) for v in origin.get("xyz", "0 0 0").split()]),
            rpy=np.array([float(v) for v in origin.get("rpy", "0 0 0").split()]),
            axis=None if axis is None else np.array([float(v) for v in axis.get("xyz").split()]),
        )
    return joints


JOINTS = _urdf_joints()


def _link_rotation_at_zero(link):
    """Orientation of ``link`` in the pelvis frame with every joint at zero."""
    by_child = {j["child"]: j for j in JOINTS.values()}
    rotation = Rotation.identity()
    while link in by_child:
        joint = by_child[link]
        rotation = Rotation.from_euler("xyz", joint["rpy"]) * rotation
        link = joint["parent"]
    assert link == "pelvis"
    return rotation


def _fake_key_frame_poses(robot_model, q=None):
    del robot_model, q
    poses = {}
    for side, y in (("left", 0.2), ("right", -0.2)):
        rotation = _link_rotation_at_zero(f"{side}_wrist_yaw_link")
        poses[f"{side}_wrist"] = {
            "position": np.array([0.3, y, 0.1]),
            "orientation_wxyz": rotation.as_quat(scalar_first=True),
        }
    return poses


class _FakeRobotModel:
    def get_configuration_from_actuated_joints(self, body_actuated_joint_values):
        return body_actuated_joint_values


@pytest.fixture
def three_point(monkeypatch):
    monkeypatch.setattr(server, "get_g1_key_frame_poses", _fake_key_frame_poses)
    return server.ThreePointPose(robot_model=_FakeRobotModel())


# --- Operator input ----------------------------------------------------------

HEAD = np.array([0.1, 1.65, 0.2])
HANDS = {"left": np.array([-0.15, 1.05, -0.15]), "right": np.array([0.25, 1.05, -0.15])}


def _three_point_body(
    head_rotation=Rotation.identity(),
    hand_rotation=None,
    lift=None,
    source="grip",
    heading=Rotation.identity(),
):
    """Synthesized XR_BD body for the operator turned by ``heading`` about +Y.

    ``hand_rotation`` is a world-frame rotation of both hands about their wrists
    (in the operator's own frame), ``lift`` a per-side OpenXR offset.
    """
    lift = lift or {}
    hand_rotation = hand_rotation or Rotation.identity()
    devices = {}
    for side, rest in HANDS.items():
        position = heading.apply(rest + lift.get(side, 0.0))
        frame = _grip(side) if source == "grip" else _hand_wrist(side)
        rotation = heading * hand_rotation * frame
        if source == "grip":
            devices[f"{side}_controller"] = fakes.controller(position, rotation)
        else:
            devices[f"{side}_hand"] = fakes.hand(position, rotation)
    reader = IsaacTeleopReader(client=fakes.FakeClient())
    reader._update(
        fakes.raw(head=fakes.head(heading.apply(HEAD), heading * head_rotation), **devices), T0
    )
    kind, body = reader.get_three_point_body()
    assert kind == "headset"
    return body


def _neck(pose):
    return Rotation.from_quat(pose[2, 3:], scalar_first=True)


def _wrist(pose, side):
    return Rotation.from_quat(pose[0 if side == "left" else 1, 3:], scalar_first=True)


@pytest.mark.parametrize("heading_deg", [0.0, -90.0, 135.0])
def test_head_yaw_turns_the_neck_target_about_z_only(three_point, heading_deg):
    heading = Rotation.from_euler("y", heading_deg, degrees=True)
    assert three_point.calibrate_now(_three_point_body(heading=heading))

    # Turn the head 30 deg to the left, then look 40 deg down and tilt it 15 deg.
    head = Rotation.from_euler("YXZ", [30, -40, 15], degrees=True)
    pose = three_point.process_smpl_pose(_three_point_body(head_rotation=head, heading=heading))

    expected = Rotation.from_euler("z", 30, degrees=True)
    assert (_neck(pose) * expected.inv()).magnitude() == pytest.approx(0.0, abs=1e-5)


@pytest.mark.parametrize("source", ["grip", "hand"])
@pytest.mark.parametrize("heading_deg", [0.0, -90.0])
def test_raising_the_right_hand_raises_only_the_right_wrist_target(
    three_point, source, heading_deg
):
    heading = Rotation.from_euler("y", heading_deg, degrees=True)
    three_point.calibrate_now(_three_point_body(source=source, heading=heading))
    rest = three_point.process_smpl_pose(_three_point_body(source=source, heading=heading))

    raised = three_point.process_smpl_pose(
        _three_point_body(source=source, heading=heading, lift={"right": 0.10 * UP})
    )

    np.testing.assert_allclose(raised[1, :3] - rest[1, :3], [0.0, 0.0, 0.10], atol=1e-5)
    np.testing.assert_allclose(raised[0], rest[0], atol=1e-6)


def _wrist_joint_axes(side):
    """Each wrist joint's axis in the wrist_yaw_link frame and in the robot frame at zero."""
    wrist = _link_rotation_at_zero(f"{side}_wrist_yaw_link")
    axes = {}
    for name in ("roll", "pitch", "yaw"):
        joint = JOINTS[f"{side}_wrist_{name}_joint"]
        child = _link_rotation_at_zero(joint["child"])
        physical = child.apply(joint["axis"])
        axes[name] = (wrist.inv().apply(physical), physical)
    return axes


def test_the_g1_wrist_roll_axis_is_the_forearm():
    for side in ("left", "right"):
        roll = JOINTS[f"{side}_wrist_roll_joint"]
        pitch = JOINTS[f"{side}_wrist_pitch_joint"]
        # The next joint sits along the roll axis: it is the forearm direction,
        # pointing forward in the zero pose.
        assert np.allclose(np.cross(roll["axis"], pitch["xyz"]), 0.0)
        assert np.dot(roll["axis"], pitch["xyz"]) > 0
        _, physical = _wrist_joint_axes(side)["roll"]
        np.testing.assert_allclose(physical, [1.0, 0.0, 0.0], atol=1e-3)


@pytest.mark.parametrize("source", ["grip", "hand"])
@pytest.mark.parametrize("side", ["left", "right"])
@pytest.mark.parametrize("joint", ["roll", "pitch", "yaw"])
def test_turning_the_hand_turns_the_wrist_target_about_the_matching_g1_axis(
    three_point, source, side, joint
):
    three_point.calibrate_now(_three_point_body(source=source))
    local_axis, physical_axis = _wrist_joint_axes(side)[joint]
    angle = np.deg2rad(25.0)
    turn = Rotation.from_rotvec(angle * _robot_to_openxr(physical_axis))

    pose = three_point.process_smpl_pose(_three_point_body(source=source, hand_rotation=turn))

    calibrated = Rotation.from_quat(
        _fake_key_frame_poses(None)[f"{side}_wrist"]["orientation_wxyz"], scalar_first=True
    )
    delta = (calibrated.inv() * _wrist(pose, side)).as_rotvec()
    np.testing.assert_allclose(delta, angle * local_axis, atol=2e-3)
