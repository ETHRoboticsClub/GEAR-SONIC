"""The fidelity report tells a wrong SMPL mapping from faithful tracking, on bodies of known truth."""

import msgpack
import numpy as np
from scipy.spatial.transform import Rotation
import torch

from gear_sonic.scripts import analyze_pico_fidelity as fidelity
from gear_sonic.scripts.pico_manager_thread_server import SMPL_PARENT_INDICES
from gear_sonic.tests.teleop import fakes
from gear_sonic.trl.utils.torch_transform import compute_human_joints

MS = 1_000_000
HALF_TURN = Rotation.from_euler("y", 180, degrees=True)
Y_UP_TO_Z_UP = Rotation.from_euler("x", 90, degrees=True)


def tracked_body(seed=0, scale=1.1):
    """A posed body as PICO would report it when its joints sit exactly where its
    rotations put them: ``(24, 7)`` OpenXR poses of an operator ``scale`` times SMPL."""
    rng = np.random.default_rng(seed)
    root = Rotation.from_euler("y", rng.uniform(-180, 180), degrees=True)
    local = Rotation.from_rotvec(rng.uniform(-0.6, 0.6, (23, 3)))
    world = [root]
    for joint in range(1, 24):
        world.append(world[SMPL_PARENT_INDICES[joint]] * local[joint - 1])
    joints = compute_human_joints(
        body_pose=torch.from_numpy(local[:21].as_rotvec().reshape(1, 63)).float(),
        global_orient=torch.from_numpy((Y_UP_TO_Z_UP * root).as_rotvec()[None]).float(),
    )[0].numpy()
    body = np.zeros((24, 7), dtype=np.float32)
    body[:, :3] = scale * Y_UP_TO_Z_UP.inv().apply(joints) + [0.3, 0.9, -1.2]
    body[:, 3:] = np.array([(rotation * HALF_TURN).as_quat() for rotation in world])
    return body


def test_faithful_tracking_agrees_on_every_bone():
    result = fidelity.skeleton_agreement([tracked_body(seed) for seed in range(5)])
    assert result["frames"] == 5
    assert len(result["bones"]) == 23
    for name, bone in result["bones"].items():
        assert bone["direction_error_deg"]["max"] < 0.5, name
        assert abs(bone["length_pico_over_smpl"] - 1.1) < 0.01, name


def test_wrongly_mapped_joint_is_singled_out():
    bodies = []
    for seed in range(5):
        body = tracked_body(seed)
        elbow = fidelity.BODY_JOINT_NAMES.index("left_elbow")
        # The elbow's rotation is misread by 40 degrees; PICO's positions are untouched.
        twisted = Rotation.from_quat(body[elbow, 3:]) * Rotation.from_euler("y", 40, degrees=True)
        body[elbow, 3:] = twisted.as_quat()
        bodies.append(body)
    bones = fidelity.skeleton_agreement(bodies)["bones"]
    assert bones["left_wrist"]["direction_error_deg"]["p50"] > 30  # The forearm it poses.
    for name in ("left_elbow", "right_wrist", "left_knee", "head"):
        assert bones[name]["direction_error_deg"]["max"] < 0.5, name


def tracking_frame(arrival_ns, sequence, device_time_ns=None, body=None):
    frame = {
        **fakes.raw(body=fakes.body(*((body[:, :3], body[:, 3:]) if body is not None
                                      else (np.zeros((24, 3)),)))),
        "sample_time_ns": int(arrival_ns),
        "body_update_sequence": sequence,
    }  # fmt: skip
    if device_time_ns is not None:
        frame["device_time_ns"] = device_time_ns
    return frame


def test_body_timing_reports_the_jitter_a_headset_clock_removes():
    rng = np.random.default_rng(0)
    capture = np.arange(720) * round(1e9 / 72)
    delay = 4 * MS + rng.uniform(0, 10 * MS, len(capture))
    delay[3] = 4 * MS
    frames = [
        tracking_frame(c + d, i + 1, [0, int(c) + 7_000_000_000])
        for i, (c, d) in enumerate(zip(capture, delay))
    ]
    frames.insert(10, {**frames[9]})  # A bridge heartbeat repeats a body.
    timing = fidelity.body_timing(frames)
    assert timing["fresh_bodies"] == 720
    assert abs(timing["rate_hz"] - 72) < 0.5
    assert timing["headset_body_timestamp"] == {"present": False}
    packet = timing["headset_packet_timestamp"]
    assert packet["present"] and packet["advancing"]
    assert abs(packet["rate_vs_pc_clock"] - 1) < 0.01
    assert 9 < packet["arrival_jitter_ms"]["max"] <= 10

    relayed = fidelity.body_timing([tracking_frame(c, None) for c in capture])
    assert relayed["headset_packet_timestamp"] == {"present": False}


def test_wrist_following_finds_the_robot_delay():
    times = np.arange(0, 20, 0.02)
    feedback = []
    for t in times:
        target, measured = np.zeros(29), np.zeros(29)
        target[19] = 0.5 * np.sin(2 * np.pi * 0.5 * t)
        measured[19] = 0.5 * np.sin(2 * np.pi * 0.5 * (t - 0.06))
        feedback.append((t, {"body_q_target": target.tolist(), "body_q_measured": measured.tolist()}))
    joints = fidelity.wrist_following(feedback)["joints"]
    assert joints["left_wrist_roll"]["best_delay_ms"] == 60
    assert joints["left_wrist_roll"]["rms_error_at_best_delay_deg"] < 0.1
    assert joints["left_wrist_roll"]["rms_error_deg"] > 3
    assert "note" in joints["right_wrist_yaw"]  # Never moved.


def test_report_reads_a_capture(tmp_path):
    rows = []
    for i in range(6):
        frame = tracking_frame(1_000 * MS + i * 14 * MS, i + 1, body=tracked_body(i))
        rows.append((i * 0.014, b"tracking" + msgpack.packb(frame, use_bin_type=True)))
        rows.append((i * 0.014, b"pose" + b"ignored"))
    capture = tmp_path / "capture.msgpack"
    capture.write_bytes(msgpack.packb(rows, use_bin_type=True))
    result = fidelity.report(capture)
    assert result["skeleton_agreement"]["frames"] == 6
    worst = max(b["direction_error_deg"]["max"] for b in result["skeleton_agreement"]["bones"].values())
    assert worst < 0.5
    assert result["body_timing"]["fresh_bodies"] == 6
    assert result["wrist_following"] == {"feedback_frames": 0}
    fidelity.print_report(result)
