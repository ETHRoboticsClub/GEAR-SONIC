"""Stand-ins for isaaccapture DeviceIO schema views and a packed-message decoder.

The views mirror the attributes the readers use (see isaaccapture's
full_body/head/controller/hand pybind bindings), so these tests run without
isaaccapture, a CloudXR runtime or any device.
"""

import json
from types import SimpleNamespace

import numpy as np
from scipy.spatial.transform import Rotation

HEADER_SIZE = 1280
_DTYPES = {"f32": "<f4", "f64": "<f8", "i32": "<i4", "i64": "<i8", "u8": "u1", "bool": "?"}


def pose(position, quat_xyzw=(0.0, 0.0, 0.0, 1.0)):
    x, y, z = (float(v) for v in position)
    qx, qy, qz, qw = (float(v) for v in quat_xyzw)
    return SimpleNamespace(
        position=SimpleNamespace(x=x, y=y, z=z),
        orientation=SimpleNamespace(x=qx, y=qy, z=qz, w=qw),
    )


def body(positions, orientations=None, valid=None):
    """``FullBodyPose`` view: 24 joints with bulk ``positions``/``orientations``/``is_valid``."""
    positions = np.asarray(positions, dtype=np.float32)
    if orientations is None:
        orientations = np.tile(np.array([0, 0, 0, 1], dtype=np.float32), (24, 1))
    if valid is None:
        valid = np.ones(24, dtype=np.uint8)
    return SimpleNamespace(
        joints=SimpleNamespace(
            positions=positions,
            orientations=np.asarray(orientations, dtype=np.float32),
            is_valid=np.asarray(valid, dtype=np.uint8),
        ),
        all_joint_poses_tracked=bool(np.all(valid)),
    )


def head(position, rotation=Rotation.identity(), tracked=True):
    return SimpleNamespace(
        pose=pose(position, rotation.as_quat()), is_valid=True, is_tracked=tracked
    )


def controller(
    position, rotation=Rotation.identity(), valid=True, thumbstick=(0.0, 0.0), **buttons
):
    inputs = dict(
        primary_click=False,
        secondary_click=False,
        thumbstick_click=False,
        menu_click=False,
        thumbstick_x=float(thumbstick[0]),
        thumbstick_y=float(thumbstick[1]),
        squeeze_value=0.0,
        trigger_value=0.0,
    )
    inputs.update(buttons)
    grip = SimpleNamespace(pose=pose(position, rotation.as_quat()), is_valid=valid)
    return SimpleNamespace(grip_pose=grip, aim_pose=grip, inputs=SimpleNamespace(**inputs))


def hand(wrist_position, wrist_rotation=Rotation.identity(), valid=True):
    """``HandPose`` view: 26 XR_EXT_hand_tracking joints, only the wrist (1) set."""
    positions = np.zeros((26, 3), dtype=np.float32)
    orientations = np.tile(np.array([0, 0, 0, 1], dtype=np.float32), (26, 1))
    positions[1] = wrist_position
    orientations[1] = wrist_rotation.as_quat()
    is_valid = np.zeros(26, dtype=np.uint8)
    is_valid[1] = valid
    return SimpleNamespace(
        joints=SimpleNamespace(positions=positions, orientations=orientations, is_valid=is_valid)
    )


def raw(
    full_body=None,
    head=None,
    left_controller=None,
    right_controller=None,
    left_hand=None,
    right_hand=None,
):
    """One ``IsaacTeleopClient._get_tracker_data()`` result."""
    return dict(
        full_body=full_body,
        head=head,
        left_controller=left_controller,
        right_controller=right_controller,
        left_hand=left_hand,
        right_hand=right_hand,
    )


class FakeClient:
    """``IsaacTeleopClient`` with a session that is always open and never polled."""

    has_session = True

    def start_runtime(self):
        pass

    def open_session(self, stop_event):
        pass

    def _get_tracker_data(self):
        return None

    def close(self):
        pass


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
