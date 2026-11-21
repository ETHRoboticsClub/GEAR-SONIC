"""CPU tests for the G1 GR00T fine-tuning entry point: no GPU, no GR00T, no network.

    python -m pytest ethrc/groot_finetune/tests -q

The fake dataset has the layout the G1 recorder's export writes (LeRobot v2.1 with
meta/modality.json and meta/segments.jsonl, plus the v3 twin's meta/sources.jsonl);
parquet and video files are placeholders, since selection only copies them.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys

import pytest

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))

import dataset as D  # noqa: E402
import train as T  # noqa: E402

MODALITY = {
    "state": {
        "left_leg": {"start": 0, "end": 6, "original_key": "observation.state"},
        "right_leg": {"start": 6, "end": 12, "original_key": "observation.state"},
        "waist": {"start": 12, "end": 15, "original_key": "observation.state"},
        "left_arm": {"start": 15, "end": 22, "original_key": "observation.state"},
        "right_arm": {"start": 29, "end": 36, "original_key": "observation.state"},
        "left_hand": {"start": 22, "end": 29, "original_key": "observation.state"},
        "right_hand": {"start": 36, "end": 43, "original_key": "observation.state"},
        "projected_gravity": {"start": 0, "end": 3, "original_key": "observation.projected_gravity"},
    },
    "action": {
        "motion_token": {"start": 0, "end": 64, "original_key": "action.motion_token"},
        "left_hand_joints": {"start": 0, "end": 7, "original_key": "action.left_hand"},
        "right_hand_joints": {"start": 0, "end": 7, "original_key": "action.right_hand"},
    },
    "video": {"ego_view": {"original_key": "observation.images.ego_view"}},
    "annotation": {"human.task_description": {"original_key": "task_index"}},
}


def _jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))


def make_groot(root: Path, lengths: list[int], sources: list[int], version: str = "v2.1") -> Path:
    """A recorder-style _gr00t dataset: episode i has lengths[i] frames, from v3 sources[i]."""
    meta = root / "meta"
    meta.mkdir(parents=True)
    info = {
        "codebase_version": version,
        "fps": 50,
        "total_episodes": len(lengths),
        "total_frames": sum(lengths),
        "total_tasks": 1,
        "chunks_size": 1000,
        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
        "robot_type": "g1",
        "splits": {"train": f"0:{len(lengths)}"},
        "total_chunks": 1,
        "total_videos": len(lengths),
        "features": {
            "observation.state": {"dtype": "float32"},
            "action.motion_token": {"dtype": "float32"},
            "observation.images.ego_view": {"dtype": "video"},
        },
    }
    (meta / "info.json").write_text(json.dumps(info))
    (meta / "modality.json").write_text(json.dumps(MODALITY))
    _jsonl(meta / "tasks.jsonl", [{"task_index": 0, "task": "grasp the bottle"}])
    _jsonl(
        meta / "episodes.jsonl",
        [{"episode_index": i, "tasks": ["grasp the bottle"], "length": n} for i, n in enumerate(lengths)],
    )
    _jsonl(meta / "episodes_stats.jsonl", [{"episode_index": i, "stats": {}} for i in range(len(lengths))])
    _jsonl(
        meta / "segments.jsonl",
        [{"episode_index": i, "source_episode": s, "start": 0, "end": n}
         for i, (n, s) in enumerate(zip(lengths, sources))],
    )  # fmt: skip
    (meta / "stats.json").write_text("{}")
    for i in range(len(lengths)):
        parquet = root / f"data/chunk-000/episode_{i:06d}.parquet"
        video = root / f"videos/chunk-000/observation.images.ego_view/episode_{i:06d}.mp4"
        for path in (parquet, video):
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(f"{path.name}".encode())
    return root


def make_sources(root: Path, success: list[bool | None]) -> Path:
    (root / "meta").mkdir(parents=True)
    _jsonl(root / "meta" / "sources.jsonl", [{"episode_index": i, "success": s} for i, s in enumerate(success)])
    return root


@pytest.fixture
def bottle(tmp_path):
    """The 2026-11-21 shape: 5 takes, the first failed; take 3 cut into two segments."""
    groot = make_groot(tmp_path / "grasp_bottle_g1_gr00t", [1544, 2122, 900, 30, 1025, 1114], [0, 1, 2, 2, 3, 4])
    v3 = make_sources(tmp_path / "grasp_bottle_g1", [False, True, True, True, None])
    return groot, v3


def select(groot: Path, v3: Path | None, out: Path, policy: str = "not-failed") -> dict:
    labels = D.read_sources(D.fetch_sources(str(v3))) if v3 else None
    return D.select_episodes(D.resolve(str(groot)), out, policy=policy, labels=labels, labels_from=str(v3) if v3 else None)


def kept(report: dict) -> list[int]:
    return [row["episode"] for row in report["kept"]]


def test_not_failed_drops_failed_and_too_short_episodes(bottle, tmp_path):
    groot, v3 = bottle
    out = tmp_path / "run" / "data"
    report = select(groot, v3, out)
    assert kept(report) == [1, 2, 4, 5]
    reasons = {row["episode"]: row["reason"] for row in report["dropped"]}
    assert reasons[0] == "marked failed"
    assert "action horizon" in reasons[3]
    # Unlabeled (take 4, success None) is kept by not-failed.
    assert {row["episode"]: row["success"] for row in report["kept"]}[5] is None


def test_selected_copy_holds_only_kept_episodes_with_their_indices(bottle, tmp_path):
    groot, v3 = bottle
    out = tmp_path / "data"
    select(groot, v3, out)
    parquets = sorted(p.name for p in (out / "data/chunk-000").iterdir())
    videos = sorted(p.name for p in (out / "videos/chunk-000/observation.images.ego_view").iterdir())
    assert parquets == [f"episode_{i:06d}.parquet" for i in (1, 2, 4, 5)]
    assert videos == [f"episode_{i:06d}.mp4" for i in (1, 2, 4, 5)]
    episodes = D.read_jsonl(out / "meta/episodes.jsonl")
    assert [e["episode_index"] for e in episodes] == [1, 2, 4, 5]
    assert [r["episode_index"] for r in D.read_jsonl(out / "meta/episodes_stats.jsonl")] == [1, 2, 4, 5]
    assert [r["episode_index"] for r in D.read_jsonl(out / "meta/segments.jsonl")] == [1, 2, 4, 5]
    info = json.loads((out / "meta/info.json").read_text())
    assert info["total_episodes"] == 4
    assert info["total_frames"] == 2122 + 900 + 1025 + 1114
    assert info["total_videos"] == 4
    assert info["splits"] == {"train": "0:4"}
    # GR00T recomputes statistics from the kept episodes.
    assert not (out / "meta/stats.json").exists()
    assert (out / "meta/modality.json").read_text() == (groot / "meta/modality.json").read_text()
    assert json.loads((out / "meta" / D.REPORT).read_text())["policy"] == "not-failed"


def test_successful_needs_a_true_label(bottle, tmp_path):
    groot, v3 = bottle
    report = select(groot, v3, tmp_path / "data", policy="successful")
    assert kept(report) == [1, 2, 4]
    assert {r["episode"]: r["reason"] for r in report["dropped"]}[5] == "no success label"


def test_all_keeps_failed_but_never_too_short(bottle, tmp_path):
    groot, v3 = bottle
    report = select(groot, v3, tmp_path / "data", policy="all")
    assert kept(report) == [0, 1, 2, 4, 5]


def test_without_labels_not_failed_keeps_everything_long_enough(bottle, tmp_path):
    groot, _ = bottle
    report = select(groot, None, tmp_path / "data")
    assert kept(report) == [0, 1, 2, 4, 5]


def test_nothing_left_is_an_error(tmp_path):
    groot = make_groot(tmp_path / "g", [1000], [0])
    v3 = make_sources(tmp_path / "v", [False])
    with pytest.raises(D.DatasetError, match="no episode left"):
        select(groot, v3, tmp_path / "data")


def test_rerun_replaces_the_previous_selection(bottle, tmp_path):
    groot, v3 = bottle
    out = tmp_path / "data"
    select(groot, v3, out, policy="all")
    select(groot, v3, out, policy="successful")
    assert not (out / "data/chunk-000/episode_000000.parquet").exists()


def test_v3_dataset_is_refused_with_a_hint(tmp_path):
    groot = make_groot(tmp_path / "grasp_bottle_g1", [1000], [0], version="v3.0")
    with pytest.raises(D.DatasetError, match="_gr00t"):
        D.validate(groot)


def test_dataset_without_g1_sonic_keys_is_refused(tmp_path):
    groot = make_groot(tmp_path / "g", [1000], [0])
    modality = json.loads((groot / "meta/modality.json").read_text())
    del modality["action"]["motion_token"]
    (groot / "meta/modality.json").write_text(json.dumps(modality))
    with pytest.raises(D.DatasetError, match="action.motion_token"):
        D.validate(groot)


def test_missing_episode_file_is_reported(bottle, tmp_path):
    groot, v3 = bottle
    (groot / "videos/chunk-000/observation.images.ego_view/episode_000002.mp4").unlink()
    with pytest.raises(D.DatasetError, match="episode 2"):
        select(groot, v3, tmp_path / "data")


def test_labels_dataset_defaults_to_the_v3_twin():
    assert D.default_labels_dataset("ETHRC/grasp_bottle_g1_gr00t") == "ETHRC/grasp_bottle_g1"
    assert D.default_labels_dataset("ETHRC/other") is None
    args = T.parse(["--dataset", "ETHRC/grasp_bottle_g1_gr00t", "--prepare-only"])
    assert T.labels_dataset(args) == "ETHRC/grasp_bottle_g1"
    args = T.parse(["--dataset", "ETHRC/grasp_bottle_g1_gr00t", "--labels", "none", "--prepare-only"])
    assert T.labels_dataset(args) is None


def test_model_repo_is_required_unless_nothing_is_uploaded():
    with pytest.raises(SystemExit):
        T.parse(["--dataset", "ETHRC/x_gr00t"])
    T.parse(["--dataset", "ETHRC/x_gr00t", "--no-upload"])
    T.parse(["--dataset", "ETHRC/x_gr00t", "--prepare-only"])


def test_finetune_command_targets_g1_sonic(tmp_path):
    args = T.parse(["--dataset", "ETHRC/x_gr00t", "--model-repo", "ETHRC/m", "--max-steps", "500"])
    cmd = T.finetune_command(args, tmp_path / "data", tmp_path / "output", 1, resume=False)
    assert cmd[0] == "gr00t/experiment/launch_finetune.py"
    assert cmd[cmd.index("--embodiment_tag") + 1] == "UNITREE_G1_SONIC"
    assert cmd[cmd.index("--max_steps") + 1] == "500"
    assert "--resume_from_checkpoint" not in cmd
    assert "--resume_from_checkpoint" in T.finetune_command(args, tmp_path, tmp_path, 1, resume=True)


def test_pinned_revision_is_a_full_sha():
    assert len(T.ISAAC_GR00T_REVISION) == 40
    int(T.ISAAC_GR00T_REVISION, 16)


def test_a_run_is_finished_only_with_a_model_and_the_same_settings(tmp_path):
    from dataclasses import asdict, replace

    run = T.RunSettings("ETHRC/x_gr00t", "abc", "ETHRC/x", "not-failed", T.BASE_MODEL,
                        T.ISAAC_GR00T_REVISION, 100, 32, 1e-4, 50, 1)  # fmt: skip
    output = tmp_path / "output"
    output.mkdir()
    (output / "ethrc_run.json").write_text(json.dumps({"settings": asdict(run), "finished": True}))
    assert not T.is_finished(output, run)
    (output / "model.safetensors").write_bytes(b"")
    assert T.is_finished(output, run)
    assert not T.is_finished(output, replace(run, max_steps=200))


def test_model_card_names_dataset_selection_and_revision(bottle, tmp_path):
    groot, v3 = bottle
    report = select(groot, v3, tmp_path / "data")
    run = T.RunSettings("ETHRC/grasp_bottle_g1_gr00t", "abc", str(v3), "not-failed", T.BASE_MODEL,
                        T.ISAAC_GR00T_REVISION, 100, 32, 1e-4, 50, 1)  # fmt: skip
    card = T.model_card("ETHRC/gr00t-g1-grasp-bottle", run, report)
    assert "`ETHRC/grasp_bottle_g1_gr00t` @ `abc`" in card
    assert "4 used" in card and "2 dropped" in card and "3 marked successful" in card
    assert T.ISAAC_GR00T_REVISION in card
    assert "`grasp the bottle`" in card
