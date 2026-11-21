"""Two append rounds keep one continuous GR00T v2.1 dataset."""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"

import pyarrow.parquet as pq

from gear_sonic.data.record.convert.append import append_v21
from gear_sonic.data.record.convert.segment import convert_dataset
from gear_sonic.data.record.schema import load_layout
from gear_sonic.data.record.writer import append_episode
from gear_sonic.tests.record.test_writer import _frames, _row

_FPS = 50


def test_two_rounds_are_continuous_and_deduplicate_tasks(tmp_path):
    v3 = tmp_path / "v3"
    groot = tmp_path / "groot"
    state_dim = int(load_layout()["state_dim"])
    first = _append_v3(v3, state_dim, task="pick up the cup", t0_ns=1_000)
    _convert_into(v3, groot, [first], tmp_path / "round0")
    second = _append_v3(v3, state_dim, task="pick up the cup", t0_ns=2_000)
    _convert_into(v3, groot, [second], tmp_path / "round1")

    info = json.loads((groot / "meta" / "info.json").read_text(encoding="utf-8"))
    tasks = _jsonl(groot / "meta" / "tasks.jsonl")
    segments = _jsonl(groot / "meta" / "segments.jsonl")
    assert info["codebase_version"] == "v2.1"
    assert info["fps"] == _FPS
    assert info["total_episodes"] == 2
    assert info["total_frames"] == 8
    assert info["total_tasks"] == 1
    assert info["total_videos"] == 2
    assert info["total_chunks"] == 1
    assert info["splits"]["train"] == "0:2"
    assert [record["task"] for record in tasks] == ["pick up the cup"]
    assert [(record["episode_index"], record["source_episode"]) for record in segments] == [
        (0, 0),
        (1, 1),
    ]
    assert not (groot / "meta" / "stats.json").exists()
    assert not (groot / "meta" / "relative_stats.json").exists()

    videos = sorted((groot / "videos").rglob("*.mp4"))
    assert len(videos) == 2
    rows = []
    for episode in (0, 1):
        parquet = _episode_parquet(groot, info, episode)
        table = pq.read_table(parquet)
        episode_index = table.column("episode_index").to_pylist()
        index = table.column("index").to_pylist()
        task_index = table.column("task_index").to_pylist()
        assert table.num_rows == 4
        assert _flat(episode_index) == [episode] * 4
        assert _flat(task_index) == [0] * 4
        described = table.column("annotation.human.task_description").to_pylist()
        assert _flat(described) == [0] * 4
        rows.append(_flat(index))
        video = _episode_video(groot, info, episode)
        assert _frame_count(video) == table.num_rows
    assert rows == [list(range(0, 4)), list(range(4, 8))]


def _append_v3(root: Path, state_dim: int, *, task: str, t0_ns: int) -> int:
    rows = [
        _row(state_dim, image_index=index % 3, valid=1, flags=0, stamp=float(index))
        for index in range(4)
    ]
    return append_episode(
        rows,
        _frames(),
        root,
        fps=_FPS,
        task=task,
        t0_ns=t0_ns,
        metadata={"source_commit": "abc", "success": True},
        image_hw=(16, 16),
    )


def _convert_into(v3: Path, groot: Path, episodes: list[int], scratch: Path) -> None:
    result = convert_dataset(
        v3,
        scratch,
        min_segment_frames=1,
        episodes=episodes,
    )
    assert result.blocked is False
    assert result.v2_written is True
    append_v21(groot, result.out_root)


def _episode_parquet(root: Path, info: dict, episode: int) -> Path:
    return root / info["data_path"].format(
        episode_index=episode,
        episode_chunk=episode // int(info["chunks_size"]),
    )


def _episode_video(root: Path, info: dict, episode: int) -> Path:
    return root / info["video_path"].format(
        episode_index=episode,
        episode_chunk=episode // int(info["chunks_size"]),
        video_key="observation.images.ego_view",
    )


def _frame_count(path: Path) -> int:
    completed = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-count_frames",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=nb_read_frames",
            "-of",
            "default=nokey=1:noprint_wrappers=1",
            str(path),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    return int(completed.stdout.strip())


def _jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _flat(values) -> list[int]:
    flat = []
    for value in values:
        if isinstance(value, list):
            flat.append(int(value[0]))
        else:
            flat.append(int(value))
    return flat
