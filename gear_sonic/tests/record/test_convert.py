"""Tests for invalid-row segmentation and GR00T modality metadata."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil

import numpy as np
import pytest

from gear_sonic.data.record.convert import segment_invalid
from gear_sonic.data.record.convert.groot_meta import build_modality, validate_modality, write_modality
from gear_sonic.data.record.convert.segment import write_segmented_dataset
from gear_sonic.data.record.schema import FPS, REPO_ID, ROBOT_TYPE, features, load_layout
from gear_sonic.scripts.convert_to_groot import main


def test_segment_invalid_gap_and_short_run():
    gap = [True, True, True, False, True, True, True]
    assert segment_invalid(gap, min_segment_frames=1) == [(0, 3), (4, 7)]
    assert segment_invalid([[1], [1], [1], [0], [1], [1], [1]], min_segment_frames=1) == [
        (0, 3),
        (4, 7),
    ]

    short = [0, 1, 1, 0, 1, 1, 1]
    assert segment_invalid(short, min_segment_frames=3) == [(4, 7)]
    assert segment_invalid([1, 1], min_segment_frames=3) == []
    assert segment_invalid([1, 1, 0, 1, 1], min_segment_frames=3) == []


def test_modality_matches_layout_and_features(tmp_path):
    layout = load_layout()
    modality = build_modality(layout)
    assert list(modality["state"]) == [
        "left_leg",
        "right_leg",
        "waist",
        "left_arm",
        "right_arm",
        "left_hand",
        "right_hand",
        "projected_gravity",
    ]
    for name, group in layout["groups"].items():
        assert modality["state"][name]["start"] == group["start"]
        assert modality["state"][name]["end"] == group["end"]
        assert modality["state"][name]["original_key"] == "observation.state"
    assert modality["state"]["left_hand"]["start"] == 22
    assert modality["state"]["left_hand"]["end"] == 29
    assert modality["state"]["right_arm"]["start"] == 29
    assert modality["state"]["right_arm"]["end"] == 36
    assert list(modality["state"]).index("right_arm") < list(modality["state"]).index("left_hand")
    assert modality["state"]["projected_gravity"] == {
        "start": 0,
        "end": 3,
        "original_key": "observation.projected_gravity",
    }
    assert list(modality["action"]) == ["motion_token", "left_hand_joints", "right_hand_joints"]
    assert modality["action"]["motion_token"] == {
        "start": 0,
        "end": 64,
        "original_key": "action.motion_token",
    }
    assert modality["action"]["left_hand_joints"] == {
        "start": 0,
        "end": 7,
        "original_key": "action.left_hand",
    }
    assert modality["action"]["right_hand_joints"] == {
        "start": 0,
        "end": 7,
        "original_key": "action.right_hand",
    }
    assert modality["video"]["ego_view"]["original_key"] == "observation.images.ego_view"
    assert modality["annotation"]["human.task_description"]["original_key"] == "task_index"

    validate_modality(features(), modality)
    path = tmp_path / "modality.json"
    write_modality(path, modality)
    text = path.read_text(encoding="utf-8")
    assert text.index('"right_arm"') < text.index('"left_hand"')

    too_wide = json.loads(text)
    too_wide["state"]["right_arm"]["end"] = 100
    with pytest.raises(ValueError, match="right_arm"):
        validate_modality(features(), too_wide)


def test_segment_dataset_splits_invalid_middle_and_leaves_source(tmp_path):
    source = tmp_path / "source"
    _write_tiny_dataset(source, valid=(1, 0, 1, 1))
    before = _fingerprint(source)

    destination = tmp_path / "segmented"
    segments = write_segmented_dataset(source, destination, min_segment_frames=1)

    assert _fingerprint(source) == before
    assert segments == [
        {"episode_index": 0, "source_episode": 0, "start": 0, "end": 1},
        {"episode_index": 1, "source_episode": 0, "start": 2, "end": 4},
    ]
    _assert_two_kept_episodes(destination)


def test_convert_skips_v2_subprocess_without_ffmpeg(tmp_path, capsys):
    if shutil.which("ffmpeg") is not None:
        pytest.skip("ffmpeg is installed; the blocked path is covered when it is missing")

    source = tmp_path / "source"
    _write_tiny_dataset(source, valid=(1, 0, 1, 1))
    before = _fingerprint(source)
    destination = tmp_path / "out"
    code = main([str(source), str(destination), "--min-segment-s", f"{1 / FPS}"])
    captured = capsys.readouterr()

    assert code == 1
    assert "ffmpeg" in captured.err
    blocked = destination / "CONVERT_BLOCKED.txt"
    assert blocked.is_file()
    assert "ffmpeg" in blocked.read_text(encoding="utf-8")
    assert _fingerprint(source) == before
    info = json.loads((destination / "meta" / "info.json").read_text(encoding="utf-8"))
    assert info["codebase_version"] == "v3.0"
    assert info["fps"] == FPS
    assert not (destination / "meta" / "modality.json").exists()
    payload = json.loads(captured.out)
    assert payload["blocked"] is True
    assert payload["v2_written"] is False
    if shutil.which("ffprobe") is None:
        assert payload["codec_probe"] == "skipped: ffprobe is not installed"
    _assert_two_kept_episodes(destination)


def _write_tiny_dataset(root: Path, valid: tuple[int, ...]) -> None:
    from lerobot.configs import RGBEncoderConfig
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    spec = features(image_height=16, image_width=16)
    dataset = LeRobotDataset.create(
        repo_id=REPO_ID,
        fps=FPS,
        features=spec,
        root=root,
        robot_type=ROBOT_TYPE,
        use_videos=True,
        video_backend="pyav",
        rgb_encoder=RGBEncoderConfig(vcodec="h264", g=1),
    )
    try:
        for index, flag in enumerate(valid):
            frame = {"task": "walk"}
            image = np.zeros((16, 16, 3), dtype=np.uint8)
            image[..., 1] = 20 * index
            for key, feature in spec.items():
                if feature["dtype"] == "video":
                    frame[key] = image
                    continue
                value = np.zeros(tuple(feature["shape"]), dtype=feature["dtype"])
                if key == "quality.valid":
                    value[:] = flag
                elif key == "observation.state":
                    value[0] = index
                frame[key] = value
            dataset.add_frame(frame)
        dataset.save_episode()
    finally:
        dataset.finalize()


def _assert_two_kept_episodes(root: Path) -> None:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    loaded = LeRobotDataset(REPO_ID, root=root, video_backend="pyav")
    assert loaded.num_episodes == 2
    assert loaded.num_frames == 3
    table = loaded.hf_dataset
    order = sorted(
        range(len(table)),
        key=lambda index: (
            int(np.asarray(table[index]["episode_index"]).reshape(-1)[0]),
            int(np.asarray(table[index]["frame_index"]).reshape(-1)[0]),
        ),
    )
    states = [float(np.asarray(table[index]["observation.state"]).reshape(-1)[0]) for index in order]
    valids = [int(np.asarray(table[index]["quality.valid"]).reshape(-1)[0]) for index in order]
    episodes = [int(np.asarray(table[index]["episode_index"]).reshape(-1)[0]) for index in order]
    assert states == [0.0, 2.0, 3.0]
    assert valids == [1, 1, 1]
    assert episodes == [0, 1, 1]


def _fingerprint(root: Path) -> list[tuple[str, str]]:
    rows = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames.sort()
        for name in sorted(filenames):
            path = Path(dirpath) / name
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            rows.append((path.relative_to(root).as_posix(), digest))
    return rows
