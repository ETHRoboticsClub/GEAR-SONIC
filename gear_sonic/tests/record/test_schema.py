"""Contract tests for the frozen G1 recording schema. No hardware and no lerobot."""

from pathlib import Path
import subprocess
import sys

from gear_sonic.data.record.schema import (
    ACTION_MAX_AGE_NS,
    ACTION_MISSING,
    BODY_DIM,
    CAMERA_STALE,
    CAMERA_STALE_NS,
    FPS,
    IMAGE_HEIGHT,
    IMAGE_WIDTH,
    PICO_DIM,
    PICO_INTERP,
    PICO_INTERP_MAX_NS,
    PICO_NAMES,
    PICO_STALE,
    STATE_INTERP,
    STATE_INTERP_MAX_NS,
    TOKEN_DIM,
    action_max_age_ns,
    features,
    iter_records,
    load_layout,
    pack_record,
    period_ns,
)


def test_schema_imports_without_lerobot():
    # A fresh interpreter: other tests in this process import LeRobot.
    root = Path(__file__).resolve().parents[3]
    script = """
import sys
assert "lerobot" not in sys.modules
import gear_sonic.data.record.schema as schema
assert "lerobot" not in sys.modules
assert schema.FPS == 50
assert "CALLS" in (schema.__doc__ or "")
assert 'zmq_sources.subscribe_state(host="127.0.0.1", port=5557)' in schema.__doc__
assert "convert.segment_invalid(rows_valid_mask, *, min_segment_frames)" in schema.__doc__
"""
    completed = subprocess.run(
        [sys.executable, "-c", script],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr


def test_pico_names_length_and_key_names():
    assert len(PICO_NAMES) == PICO_DIM == 209
    assert len(set(PICO_NAMES)) == 209
    assert PICO_NAMES[:8] == (
        "head_tracked",
        "head_x",
        "head_y",
        "head_z",
        "head_qx",
        "head_qy",
        "head_qz",
        "head_qw",
    )
    left = PICO_NAMES[8:24]
    assert left == (
        "left_controller_tracked",
        "left_controller_grip_x",
        "left_controller_grip_y",
        "left_controller_grip_z",
        "left_controller_grip_qx",
        "left_controller_grip_qy",
        "left_controller_grip_qz",
        "left_controller_grip_qw",
        "left_controller_trigger",
        "left_controller_squeeze",
        "left_controller_stick_x",
        "left_controller_stick_y",
        "left_controller_primary",
        "left_controller_secondary",
        "left_controller_menu",
        "left_controller_stick_click",
    )
    assert PICO_NAMES[24] == "right_controller_tracked"
    assert PICO_NAMES[39] == "right_controller_stick_click"
    assert PICO_NAMES[40] == "body_tracked"
    assert PICO_NAMES[41] == "body_pos_0_x"
    assert PICO_NAMES[112] == "body_pos_23_z"
    assert PICO_NAMES[113] == "body_rot_0_qx"
    assert PICO_NAMES[-4:] == (
        "body_rot_23_qx",
        "body_rot_23_qy",
        "body_rot_23_qz",
        "body_rot_23_qw",
    )


def test_flag_bits_are_distinct():
    flags = [STATE_INTERP, ACTION_MISSING, CAMERA_STALE, PICO_INTERP, PICO_STALE]
    assert flags == [1 << 0, 1 << 1, 1 << 2, 1 << 3, 1 << 4]
    assert len(set(flags)) == len(flags)
    assert ACTION_MAX_AGE_NS == action_max_age_ns() == 30_000_000 == period_ns() * 3 // 2
    assert period_ns(FPS) == 20_000_000
    assert STATE_INTERP_MAX_NS == 60_000_000
    assert CAMERA_STALE_NS == 100_000_000
    assert PICO_INTERP_MAX_NS == 150_000_000


def test_pack_record_roundtrip(tmp_path):
    path = tmp_path / "session.bin"
    first = b"\x00\x01msgpack"
    second = b""
    path.write_bytes(pack_record(10, 20, first) + pack_record(30, 40, second))
    assert list(iter_records(path)) == [(10, 20, first), (30, 40, second)]


def test_iter_records_stops_on_truncated_tail(tmp_path):
    path = tmp_path / "session.bin"
    complete = pack_record(1, 2, b"ok")
    partial_header = complete[:10]
    path.write_bytes(complete + partial_header)
    assert list(iter_records(path)) == [(1, 2, b"ok")]

    long_record = pack_record(5, 6, b"abcdef")
    path.write_bytes(complete + long_record[:-3])
    assert list(iter_records(path)) == [(1, 2, b"ok")]

    path.write_bytes(b"G1RS")
    assert list(iter_records(path)) == []


def test_features_shapes_match_layout_and_defaults():
    layout = load_layout()
    spec = features()
    state = spec["observation.state"]
    assert state["dtype"] == "float32"
    assert tuple(state["shape"]) == (layout["state_dim"],)
    assert list(state["names"]) == list(layout["joint_names"])

    video = spec["observation.images.ego_view"]
    assert video["dtype"] == "video"
    assert tuple(video["shape"]) == (IMAGE_HEIGHT, IMAGE_WIDTH, 3) == (360, 640, 3)
    assert video["names"] == ["height", "width", "channel"]

    token = spec["action.motion_token"]
    assert token["dtype"] == "float32"
    assert tuple(token["shape"]) == (TOKEN_DIM,) == (64,)
    assert token["names"] == [f"token_{index}" for index in range(64)]

    velocity = spec["observation.joint_velocity"]
    assert tuple(velocity["shape"]) == (BODY_DIM,)
    assert velocity["names"] == list(layout["body_joint_names"])
    assert spec["quality.valid"]["dtype"] == "int64"
    assert spec["quality.flags"]["dtype"] == "int64"
    assert tuple(spec["quality.valid"]["shape"]) == (1,)
    assert tuple(spec["observation.pico"]["shape"]) == (209,)
    assert spec["observation.pico"]["names"] == list(PICO_NAMES)
    assert spec["observation.base_quat"]["names"] == ["w", "x", "y", "z"]
    assert spec["observation.sample_age_s"]["names"] == ["robot", "pico", "camera"]
    assert spec["action.left_hand"]["names"] == [f"left_hand_motor_{index}" for index in range(7)]
    assert spec["observation.left_hand_dq"]["names"] == [f"motor_{index}" for index in range(7)]

    resized = features(image_height=10, image_width=20)
    assert tuple(resized["observation.images.ego_view"]["shape"]) == (10, 20, 3)


def test_layout_groups_are_contiguous_and_do_not_overlap():
    layout = load_layout()
    names = list(layout["joint_names"])
    state_dim = layout["state_dim"]
    assert state_dim == len(names)
    assert isinstance(state_dim, int) and state_dim > 0

    order = [
        "left_leg",
        "right_leg",
        "waist",
        "left_arm",
        "right_arm",
        "left_hand",
        "right_hand",
    ]
    groups = layout["groups"]
    assert list(groups) == order

    occupied = {}
    for group_name in order:
        group = groups[group_name]
        start, end = group["start"], group["end"]
        assert isinstance(start, int) and isinstance(end, int)
        assert 0 <= start < end <= state_dim
        assert group["names"] == names[start:end]
        assert len(group["names"]) == end - start
        for index in range(start, end):
            assert index not in occupied, (names[index], occupied.get(index), group_name)
            occupied[index] = group_name

    assert len(groups["left_leg"]["names"]) == 6
    assert len(groups["right_leg"]["names"]) == 6
    assert all(
        name.startswith("left_") and "hand" not in name for name in groups["left_leg"]["names"]
    )
    assert all(
        name.startswith("right_") and "hand" not in name for name in groups["right_leg"]["names"]
    )
    assert groups["left_hand"]["names"]
    assert all(name.startswith("left_hand_") for name in groups["left_hand"]["names"])
    assert all(name.startswith("right_hand_") for name in groups["right_hand"]["names"])
    assert len(groups["left_hand"]["names"]) == 7
    assert len(groups["right_hand"]["names"]) == 7
    assert "left_hip_pitch_joint" in groups["left_leg"]["names"]
    assert "right_ankle_roll_joint" in groups["right_leg"]["names"]

    assigned = [name for group_name in order for name in groups[group_name]["names"]]
    assert len(assigned) == len(set(assigned))
    unassigned = list(layout["unassigned"])
    assert [name for name in names if name not in set(assigned)] == unassigned
    if unassigned:
        note = layout.get("unassigned_note", "")
        assert isinstance(note, str) and note.strip()
        for name in unassigned:
            assert name in note

    body_names = list(layout["body_joint_names"])
    assert len(body_names) == 29
    non_hand = [name for name in names if "hand" not in name]
    assert body_names == non_hand
    source = layout["body_joint_names_source"]
    assert "MuJoCo" in source
