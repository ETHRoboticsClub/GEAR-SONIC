"""Reload one tiny LeRobot episode written by writer.write_episode. No hardware."""

import inspect
import json
import os

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"

import numpy as np
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
from gear_sonic.data.record.writer import write_episode

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
