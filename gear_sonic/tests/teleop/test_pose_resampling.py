from types import SimpleNamespace

import numpy as np
import pytest
from scipy.spatial.transform import Rotation
import torch

from gear_sonic.scripts import pico_manager_thread_server as server
from gear_sonic.tests.teleop import fakes
from gear_sonic.trl.utils.torch_transform import compute_human_joints

MS = 1_000_000


def push(resampler, time_ms):
    # A slow one-axis turn has an independently known angular trajectory.
    pose = np.zeros((21, 3))
    pose[0, 0] = time_ms / 10_000
    return resampler.push(
        time_ms * MS, pose,
        np.array([1., 0., 0., 0.]),
    )


def test_jittered_tracking_produces_evenly_spaced_pose_frames():
    r = server.UniformPoseResampler(50, 5)
    frames = []
    for stamp in (0, 14, 28, 41, 56, 71, 84, 98, 111):
        frames.extend(push(r, stamp))
    assert [f[0] for f in frames] == list(range(6))
    assert [f[1] for f in frames] == [i * 20 * MS for i in range(6)]
    for index, stamp, pose, quat in frames:
        np.testing.assert_allclose(pose[0], [stamp / MS / 10_000, 0., 0.], atol=1e-7)
        assert np.isfinite(pose).all()
        assert np.linalg.norm(quat) == pytest.approx(1.)


def test_delayed_packet_skips_old_work_and_keeps_only_newest_window():
    r = server.UniformPoseResampler(50, 5)
    push(r, 0)
    push(r, 14)
    frames = push(r, 211)
    assert [f[0] for f in frames] == [6, 7, 8, 9, 10]
    assert [f[1] // MS for f in frames] == [120, 140, 160, 180, 200]
    for _, stamp, pose, _ in frames:
        np.testing.assert_allclose(pose[0], [stamp / MS / 10_000, 0., 0.], atol=1e-7)
    assert push(r, 211) == []
    assert push(r, 100) == []
    assert push(r, 218) == []  # No extrapolation before the next grid point.
    assert push(r, 226)[0][1] == 220 * MS


def test_reset_starts_a_fresh_timeline():
    r = server.UniformPoseResampler(50, 5)
    push(r, 1000)
    push(r, 1100)
    r.reset()
    assert push(r, 2000)[0][:2] == (0, 2000 * MS)


def test_duplicate_sample_does_not_repeat_smpl_mapping(monkeypatch):
    s = server.PoseStreamer.__new__(server.PoseStreamer)
    s.reader = SimpleNamespace(get_latest=lambda: {"timestamp_ns": 123})
    s.last_processed_stamp_ns = 123
    monkeypatch.setattr(server, "compute_from_body_poses", lambda *args: pytest.fail("duplicate mapping"))
    s.run_once()


def test_streamer_publishes_overlapping_windows_on_the_50_hz_grid():
    reader = fakes.reader()
    socket = fakes.RecordingSocket()
    three_point = SimpleNamespace(
        enable_smpl_vis=False,
        process_smpl_pose=lambda *a, **kw: np.tile([0., 0., 0., 1., 0., 0., 0.], (3, 1)),
    )
    s = server.PoseStreamer(socket, reader, three_point, 5, 50, False, "", "npz")
    frame = fakes.raw(body=fakes.body(np.tile([0., 1., 0.], (24, 1))))
    for i in range(73):
        stamp = 1_000 * MS + round(i * 1e9 / 72)
        fakes.feed(reader, frame, stamp)
        s.run_once()
    assert len(socket.sent) == 47  # 51 grid points, first 4 fill the initial window.
    windows = [fakes.decode(m, b"pose")[1] for m in socket.sent]
    for f in windows:
        np.testing.assert_array_equal(np.diff(f["frame_timestamp_ns"]), [20 * MS] * 4)
        assert f["smpl_pose"].shape == (5, 21, 3)
        assert np.isfinite(f["joint_pos"]).all()
        # The wire feeds SONIC's trained encoder, not the geometric pelvis-local API.
        expected_origin_length = np.linalg.norm(server.prepare_cpu_kinematics()[0][0])
        np.testing.assert_allclose(
            np.linalg.norm(f['smpl_joints'][:, 0], axis=1), expected_origin_length, atol=2e-6,
        )
        assert f["prep_timing_ms"].shape == (3,)
        assert 0 <= f["prep_timing_ms"][0] <= f["prep_timing_ms"][1]
        np.testing.assert_array_equal(np.diff(f["frame_index"]), [1] * 4)
    np.testing.assert_array_equal(windows[0]["frame_index"], [0, 1, 2, 3, 4])
    np.testing.assert_array_equal(windows[1]["frame_index"], [1, 2, 3, 4, 5])
    s.on_mode_exit()
    assert not s.frame_buffer
    assert s.last_processed_stamp_ns is None


@pytest.mark.parametrize('start_deg,end_deg', [(0., 20.), (0., 120.), (170., -170.)])
def test_resampled_wire_joints_match_the_wire_smpl_rotations(start_deg, end_deg):
    reader, socket = fakes.reader(), fakes.RecordingSocket()
    three_point = SimpleNamespace(
        enable_smpl_vis=False,
        process_smpl_pose=lambda *a, **kw: np.tile([0., 0., 0., 1., 0., 0., 0.], (3, 1)),
    )
    streamer = server.PoseStreamer(socket, reader, three_point, 5, 50, False, '', 'npz')
    for stamp_ms, angle in [(1000, start_deg), (1080, end_deg)]:
        orientations = np.tile([0., 0., 0., 1.], (24, 1))
        orientations[18] = Rotation.from_euler('z', angle, degrees=True).as_quat()
        frame = fakes.raw(body=fakes.body(np.tile([0., 1., 0.], (24, 1)), orientations))
        fakes.feed(reader, frame, stamp_ms * MS)
        streamer.run_once()
    assert len(socket.sent) == 1
    _, fields = fakes.decode(socket.sent[0], b'pose')
    np.testing.assert_array_equal(fields['frame_index'], np.arange(5))
    # Independent generic torch FK, including the trained encoder's rest origin.
    root = Rotation.from_quat(fields['body_quat_w'], scalar_first=True)
    fk_root = root * Rotation.from_quat([.5, .5, .5, .5])
    world = compute_human_joints(
        torch.from_numpy(fields['smpl_pose'].reshape(5, 63).copy()),
        torch.from_numpy(fk_root.as_rotvec().astype(np.float32)),
    ).numpy()
    expected = np.einsum('nij,nkj->nki', root.inv().as_matrix(), world)
    np.testing.assert_allclose(fields['smpl_joints'], expected, atol=3e-6, rtol=1e-5)
