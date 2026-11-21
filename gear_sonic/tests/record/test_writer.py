"""Reload one tiny LeRobot episode written by writer.write_episode. No hardware."""

import inspect
import json
import os

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"

import numpy as np
import pytest
from lerobot.datasets import LeRobotDataset

from gear_sonic.data.record.schema import (
    BODY_DIM,
    CAMERA_STALE,
    HAND_DIM,
    PICO_DIM,
    REPO_ID,
    STATE_INTERP,
    TOKEN_DIM,
    Row,
    features,
    load_layout,
)
from gear_sonic.data.record.writer import append_episode, write_episode

_HEIGHT = 16
_WIDTH = 16
_FPS = 50


def _zeros(length: int) -> tuple[float, ...]:
    return tuple(0.0 for _ in range(length))


def _row(state_dim: int, *, image_index: int, valid: int, flags: int, stamp: float) -> Row:
    return Row(
        t_ns=1,
        state=_zeros(state_dim),
        projected_gravity=(0.0, 0.0, -1.0),
        base_quat=(1.0, 0.0, 0.0, 0.0),
        base_ang_vel=(0.0, 0.0, 0.0),
        joint_velocity=_zeros(BODY_DIM),
        left_hand_dq=_zeros(HAND_DIM),
        right_hand_dq=_zeros(HAND_DIM),
        action=_zeros(BODY_DIM),
        left_hand_action=_zeros(HAND_DIM),
        right_hand_action=_zeros(HAND_DIM),
        token=tuple(stamp for _ in range(TOKEN_DIM)),
        pico=_zeros(PICO_DIM),
        pico_sequence=0,
        g1_index=0,
        ages=(0.0, 0.0, 0.0),
        valid=valid,
        flags=flags,
        image_index=image_index,
    )


def _frames() -> list[np.ndarray]:
    frames = []
    for value in (20, 40, 60):
        image = np.zeros((_HEIGHT, _WIDTH, 3), dtype=np.uint8)
        image[..., 1] = value
        frames.append(image)
    return frames


def test_write_episode_reloads_four_frames(tmp_path):
    state_dim = int(load_layout()["state_dim"])
    rows = [
        _row(state_dim, image_index=0, valid=1, flags=0, stamp=0.0),
        _row(state_dim, image_index=1, valid=1, flags=STATE_INTERP, stamp=1.0),
        _row(state_dim, image_index=-1, valid=0, flags=CAMERA_STALE, stamp=2.0),
        _row(state_dim, image_index=2, valid=1, flags=0, stamp=3.0),
    ]
    metadata = {
        "camera_offset_ns": 123,
        "source_commit": "abc123",
        "success": True,
        "drops": {"state": 0, "tracking": 1, "jpeg": 2},
    }
    t0_ns = 1_700_000_000_000_000_000
    root = tmp_path / "episode"
    path = write_episode(
        rows,
        _frames(),
        root,
        fps=_FPS,
        task="pick up the cup",
        t0_ns=t0_ns,
        metadata=metadata,
        image_hw=(_HEIGHT, _WIDTH),
    )

    assert path == root
    assert metadata["video_encode"] in {"streaming_encoding=True", "streaming_encoding=False"}

    clock = json.loads((root / "meta" / "grid_clock.json").read_text(encoding="utf-8"))
    assert clock == {
        "clock": "CLOCK_MONOTONIC",
        "grid_origin_monotonic_ns": t0_ns,
        "fps": _FPS,
        "camera_offset_ns": 123,
        "source_commit": "abc123",
    }

    quality = json.loads(
        (root / "meta" / "quality" / "episode_000000.json").read_text(encoding="utf-8")
    )
    assert quality["n_rows"] == 4
    assert quality["n_invalid"] == 1
    assert quality["pct_invalid"] == 25.0
    assert quality["pct_state_interp"] == 25.0
    assert quality["gaps"] == [{"k": 2, "flags": CAMERA_STALE}]
    assert quality["success"] is True
    assert quality["drops"] == {"state": 0, "tracking": 1, "jpeg": 2}
    assert quality["video_encode"] == metadata["video_encode"]

    spec = features(_HEIGHT, _WIDTH)
    kwargs = {"repo_id": REPO_ID, "root": root}
    if "video_backend" in inspect.signature(LeRobotDataset.__init__).parameters:
        kwargs["video_backend"] = "pyav"
    dataset = LeRobotDataset(**kwargs)
    assert set(spec) <= set(dataset.features)
    for key, feature in spec.items():
        stored = dataset.features[key]
        assert stored["dtype"] == feature["dtype"]
        assert tuple(stored["shape"]) == tuple(feature["shape"])

    assert len(dataset) == 4
    for index in range(4):
        item = dataset[index]
        stamp = item["timestamp"]
        stamp = float(stamp.reshape(-1)[0])
        assert abs(stamp - index / _FPS) <= 1e-4
        token = item["action.motion_token"]
        assert tuple(token.shape) == (TOKEN_DIM,)

    stale_flags = dataset[2]["quality.flags"]
    assert int(stale_flags.reshape(-1)[0]) == CAMERA_STALE


def test_append_episode_twice_and_rejects_a_different_layout(tmp_path):
    state_dim = int(load_layout()["state_dim"])
    root = tmp_path / "dataset"
    metadata = {"source_commit": "abc123", "success": True, "camera_offset_ns": 5}
    first = append_episode(
        _four_rows(state_dim),
        _frames(),
        root,
        fps=_FPS,
        task="pick up the cup",
        t0_ns=1_000,
        metadata=metadata,
        image_hw=(_HEIGHT, _WIDTH),
    )
    second = append_episode(
        _four_rows(state_dim),
        _frames(),
        root,
        fps=_FPS,
        task="pick up the cup",
        t0_ns=2_000,
        metadata={"source_commit": "def456", "success": False},
        image_hw=(_HEIGHT, _WIDTH),
    )
    assert (first, second) == (0, 1)

    for index, origin, commit, success in (
        (0, 1_000, "abc123", True),
        (1, 2_000, "def456", False),
    ):
        clock = json.loads(
            (root / "meta" / "grid_clock" / f"episode_{index:06d}.json").read_text(encoding="utf-8")
        )
        quality = json.loads(
            (root / "meta" / "quality" / f"episode_{index:06d}.json").read_text(encoding="utf-8")
        )
        assert clock["grid_origin_monotonic_ns"] == origin
        assert clock["fps"] == _FPS
        assert clock["source_commit"] == commit
        assert quality["n_rows"] == 4
        assert quality["success"] is success

    info = json.loads((root / "meta" / "info.json").read_text(encoding="utf-8"))
    assert info["video_files_size_in_mb"] == 100
    assert info["data_files_size_in_mb"] == 50
    assert info["total_episodes"] == 2

    kwargs = {"repo_id": REPO_ID, "root": root}
    if "video_backend" in inspect.signature(LeRobotDataset.__init__).parameters:
        kwargs["video_backend"] = "pyav"
    dataset = LeRobotDataset(**kwargs)
    assert dataset.num_episodes == 2
    assert len(dataset) == 8
    starts = []
    for index in range(len(dataset)):
        item = dataset[index]
        episode = int(np.asarray(item["episode_index"]).reshape(-1)[0])
        frame = int(np.asarray(item["frame_index"]).reshape(-1)[0])
        stamp = float(np.asarray(item["timestamp"]).reshape(-1)[0])
        assert abs(stamp - frame / _FPS) <= 1e-4
        if frame == 0:
            starts.append((episode, stamp))
    assert sorted(starts) == [(0, 0.0), (1, 0.0)]

    with pytest.raises(ValueError):
        append_episode(
            _four_rows(state_dim),
            _frames(),
            root,
            fps=_FPS + 1,
            task="pick up the cup",
            t0_ns=3_000,
            metadata={"success": True},
            image_hw=(_HEIGHT, _WIDTH),
        )
    with pytest.raises(ValueError):
        append_episode(
            _four_rows(state_dim),
            _frames(),
            root,
            fps=_FPS,
            task="pick up the cup",
            t0_ns=3_000,
            metadata={"success": True},
            image_hw=(_HEIGHT // 2, _WIDTH // 2),
        )
    info = json.loads((root / "meta" / "info.json").read_text(encoding="utf-8"))
    assert info["total_episodes"] == 2


def _four_rows(state_dim: int) -> list[Row]:
    return [
        _row(state_dim, image_index=0, valid=1, flags=0, stamp=0.0),
        _row(state_dim, image_index=1, valid=1, flags=0, stamp=1.0),
        _row(state_dim, image_index=2, valid=1, flags=0, stamp=2.0),
        _row(state_dim, image_index=0, valid=1, flags=0, stamp=3.0),
    ]
