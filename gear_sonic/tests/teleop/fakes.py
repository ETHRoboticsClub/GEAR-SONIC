"""Tracking frames as the robot's teleop module sends them, and a packed-message decoder.

Tests describe devices in OpenXR stage space (+X right, +Y up, operator facing
-Z), as the retargeting code is written; these builders convert to the module's
operator space (REP-103: x forward, y left, z up) with their own axis mapping,
independent of ``input_readers.to_openxr``.
"""

import json

import numpy as np
from scipy.spatial.transform import Rotation

HEADER_SIZE = 1280
_DTYPES = {"f32": "<f4", "f64": "<f8", "i32": "<i4", "i64": "<i8", "u8": "u1", "bool": "?"}

# Operator-space axes written as OpenXR vectors: forward is -Z, left is -X, up is +Y.
_OPERATOR_AXES_IN_OPENXR = np.column_stack([[0, 0, -1.0], [-1.0, 0, 0], [0, 1.0, 0]])
_OPERATOR_FROM_OPENXR = Rotation.from_matrix(_OPERATOR_AXES_IN_OPENXR.T)


def operator_position(openxr_position):
    return _OPERATOR_AXES_IN_OPENXR.T @ np.asarray(openxr_position, dtype=float)


def operator_rotation(openxr_rotation: Rotation) -> Rotation:
    return _OPERATOR_FROM_OPENXR * openxr_rotation


def pose(position, rotation=Rotation.identity()):
    """``[x, y, z, qx, qy, qz, qw]`` in operator space for an OpenXR pose."""
    return [*operator_position(position), *operator_rotation(rotation).as_quat()]


def head(position, rotation=Rotation.identity(), tracked=True):
    return pose(position, rotation) if tracked else None


def controller(
    position, rotation=Rotation.identity(), valid=True, thumbstick=(0.0, 0.0), **buttons
):
    if not valid:
        return None
    inputs = dict(
        primary=False,
        secondary=False,
        menu=False,
        stick_click=False,
        trigger=0.0,
        squeeze=0.0,
        stick_x=float(thumbstick[0]),
        stick_y=float(thumbstick[1]),
    )
    inputs.update(buttons)
    grip = pose(position, rotation)
    return {"grip": grip, "aim": grip, **inputs}


def _joint_set(positions, orientations):
    """Packed operator-space joints from (N, 3) OpenXR positions and (N, 4) xyzw rotations."""
    return {
        "positions": [v for p in positions for v in operator_position(p)],
        "rotations": [
            v for q in orientations for v in operator_rotation(Rotation.from_quat(q)).as_quat()
        ],
    }


def hand(wrist_position, wrist_rotation=Rotation.identity(), valid=True):
    """26 XR_EXT_hand_tracking joints, only the wrist (1) placed."""
    if not valid:
        return None
    positions = np.zeros((26, 3))
    orientations = np.tile([0.0, 0.0, 0.0, 1.0], (26, 1))
    positions[1] = wrist_position
    orientations[1] = wrist_rotation.as_quat()
    return _joint_set(positions, orientations)


def body(positions, orientations=None):
    """24 XR_BD body joints at OpenXR ``positions``."""
    if orientations is None:
        orientations = np.tile([0.0, 0.0, 0.0, 1.0], (24, 1))
    return _joint_set(np.asarray(positions, float), np.asarray(orientations, float))


def raw(
    body=None,
    head=None,
    left_controller=None,
    right_controller=None,
    left_hand=None,
    right_hand=None,
):
    """One tracking frame without its clock; ``feed`` stamps it."""
    return dict(
        body=body,
        head=head,
        left_controller=left_controller,
        right_controller=right_controller,
        left_hand=left_hand,
        right_hand=right_hand,
    )


class Clock:
    def __init__(self, now_ns=0):
        self.now_ns = now_ns

    def __call__(self):
        return self.now_ns


def reader():
    """A TrackingReader on a test clock, fed by ``feed``."""
    from gear_sonic.utils.teleop.input_readers import TrackingReader

    return TrackingReader(clock=Clock())


def feed(reader, frame, t_ns):
    """Deliver ``frame`` sampled and received at ``t_ns``."""
    reader._clock.now_ns = t_ns
    reader._update({**frame, "sample_time_ns": t_ns}, t_ns)


class RecordingSocket:
    def __init__(self):
        self.sent = []

    def send(self, message):
        self.sent.append(message)


def decode(message: bytes, topic: bytes):
    """Decode a ``[topic][header][fields]`` message into (header, {name: array})."""
    assert message.startswith(topic)
    start = len(topic)
    header = json.loads(message[start : start + HEADER_SIZE].rstrip(b"\x00"))
    offset = start + HEADER_SIZE
    fields = {}
    for field in header["fields"]:
        dtype = np.dtype(_DTYPES[field["dtype"]])
        count = int(np.prod(field["shape"]))
        fields[field["name"]] = np.frombuffer(
            message, dtype=dtype, count=count, offset=offset
        ).reshape(field["shape"])
        offset += count * dtype.itemsize
    assert offset == len(message)
    return header, fields


class FakeRobotModel:
    """Stands in for the Pinocchio G1 model; FK comes from ``g1_key_frame_poses``."""

    def get_configuration_from_actuated_joints(self, body_actuated_joint_values):
        return body_actuated_joint_values


def g1_key_frame_poses(robot_model, q=None):
    """Fixed G1 wrist key frames (identity orientation, pelvis frame)."""
    del robot_model, q
    return {
        f"{side}_wrist": {
            "position": np.array([0.3, y, 0.1]),
            "orientation_wxyz": np.array([1.0, 0, 0, 0]),
        }
        for side, y in (("left", 0.2), ("right", -0.2))
    }
