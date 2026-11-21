"""Read a G1 GR00T dataset from the robot's export and pick the episodes to train on.

The G1 recorder (``docs/g1_teleop_recording.md``) publishes every task twice:

* ``<name>``: LeRobot v3.0, the full 50 Hz grid; ``meta/sources.jsonl`` holds each
  episode's ``success`` (the operator's y/n after the take).
* ``<name>_gr00t``: LeRobot v2.1 with ``meta/modality.json`` for GR00T's
  ``UNITREE_G1_SONIC`` embodiment. Invalid rows are cut out, so one take can become
  several episodes; ``meta/segments.jsonl`` maps each one back to its v3 episode
  (``source_episode``).

Training reads the ``_gr00t`` dataset. :func:`select_episodes` writes a copy that holds
only the chosen episodes (by default every episode not marked as failed), keeping their
files and indices as they are: GR00T's loader walks ``meta/episodes.jsonl`` and opens
each episode by its own ``episode_index``, and its statistics read the parquet files
present, so dropping an episode means not copying it.

No GR00T import here: this module runs, and is tested, without a GPU or GR00T install.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import shutil

# GR00T's UNITREE_G1_SONIC embodiment (gr00t/configs/data/embodiment_configs.py at the
# pinned revision). The dataset's meta/modality.json must name at least these keys.
STATE_KEYS = (
    "left_leg",
    "right_leg",
    "waist",
    "left_arm",
    "right_arm",
    "left_hand",
    "right_hand",
    "projected_gravity",
)
ACTION_KEYS = ("motion_token", "left_hand_joints", "right_hand_joints")
VIDEO_KEYS = ("ego_view",)
TOKEN_DIM = 64
# UNITREE_G1_SONIC predicts 40 future steps (delta_indices 0..39). GR00T's sampler gives
# one training step per frame that has this many frames ahead of it, so an episode
# shorter than this gives none.
ACTION_HORIZON = 40
CODEBASE_VERSION = "v2.1"
GROOT_SUFFIX = "_gr00t"
EPISODE_POLICIES = ("not-failed", "successful", "all")
REPORT = "ethrc_selection.json"

# Copied verbatim; episodes.jsonl, episodes_stats.jsonl, segments.jsonl and info.json are
# rewritten for the kept episodes. stats.json / relative_stats.json are not copied:
# GR00T recomputes them from the kept episodes.
_META_COPIED = ("modality.json", "tasks.jsonl")


class DatasetError(RuntimeError):
    """The dataset cannot be trained on as it is; the message says why and what to do."""


@dataclass(frozen=True)
class Source:
    """Where a dataset was read from: a local directory, or a Hub repo at a commit."""

    path: Path
    name: str
    revision: str | None  # Hub commit sha; None for a local directory


def _read_json(path: Path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise DatasetError(f"{path} is missing") from None
    except json.JSONDecodeError as error:
        raise DatasetError(f"{path} is not valid JSON: {error}") from None


def read_jsonl(path: Path) -> list[dict]:
    """Every JSON object in a .jsonl file; blank lines are skipped."""
    rows = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError as error:
            raise DatasetError(f"{path}:{number} is not valid JSON: {error}") from None
    return rows


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def resolve(dataset: str, revision: str | None = None, *, cache_dir: Path | None = None) -> Source:
    """A local directory as it is, or a Hub dataset downloaded at ``revision``."""
    local = Path(dataset).expanduser()
    if local.exists():
        if not local.is_dir():
            raise DatasetError(f"{local} is not a directory")
        return Source(local.resolve(), local.name, None)
    if dataset.count("/") != 1:
        raise DatasetError(
            f"{dataset!r} is neither a local directory nor a Hub dataset id (namespace/name)"
        )
    from huggingface_hub import HfApi, snapshot_download

    info = HfApi().dataset_info(dataset, revision=revision)
    path = snapshot_download(
        dataset, repo_type="dataset", revision=info.sha, cache_dir=cache_dir
    )
    return Source(Path(path), dataset.split("/")[-1], info.sha)


def validate(root: Path) -> dict:
    """Check that ``root`` is a GR00T v2.1 dataset for UNITREE_G1_SONIC; return info.json."""
    meta = root / "meta"
    info = _read_json(meta / "info.json")
    version = info.get("codebase_version")
    if version != CODEBASE_VERSION:
        hint = (
            " This looks like the LeRobot v3.0 copy: train on the `_gr00t` dataset instead."
            if str(version).startswith("v3")
            else ""
        )
        raise DatasetError(f"{root} is LeRobot {version}, GR00T needs {CODEBASE_VERSION}.{hint}")
    modality = _read_json(meta / "modality.json")
    missing = [
        f"{group}.{key}"
        for group, keys in (("state", STATE_KEYS), ("action", ACTION_KEYS), ("video", VIDEO_KEYS))
        for key in keys
        if key not in modality.get(group, {})
    ]
    if missing:
        raise DatasetError(
            f"{meta / 'modality.json'} lacks {', '.join(missing)}: it is not a UNITREE_G1_SONIC "
            "dataset from the G1 recorder"
        )
    token = modality["action"]["motion_token"]
    if token.get("end", 0) - token.get("start", 0) != TOKEN_DIM:
        raise DatasetError(f"action.motion_token is not {TOKEN_DIM} wide in modality.json")
    for name in ("episodes.jsonl", "tasks.jsonl"):
        if not (meta / name).is_file():
            raise DatasetError(f"{meta / name} is missing")
    for key in ("data_path", "video_path", "chunks_size"):
        if key not in info:
            raise DatasetError(f"{meta / 'info.json'} has no {key!r}")
    return info


def default_labels_dataset(dataset: str) -> str | None:
    """``ns/name_gr00t`` -> ``ns/name``, the v3 dataset that holds the success labels."""
    if dataset.endswith(GROOT_SUFFIX):
        return dataset[: -len(GROOT_SUFFIX)]
    return None


def read_sources(path: Path) -> dict[int, bool | None]:
    """v3 ``meta/sources.jsonl``: v3 episode index -> success (True, False, or None)."""
    labels: dict[int, bool | None] = {}
    for row in read_jsonl(path):
        success = row.get("success")
        labels[int(row["episode_index"])] = success if isinstance(success, bool) else None
    return labels


def fetch_sources(labels_dataset: str, revision: str | None = None) -> Path:
    """``meta/sources.jsonl`` of a local v3 dataset, or downloaded from the Hub."""
    local = Path(labels_dataset).expanduser()
    if local.exists():
        path = local / "meta" / "sources.jsonl"
        if not path.is_file():
            raise DatasetError(f"{path} is missing: {local} is not the recorder's v3 dataset")
        return path
    from huggingface_hub import hf_hub_download

    return Path(
        hf_hub_download(
            labels_dataset, "meta/sources.jsonl", repo_type="dataset", revision=revision
        )
    )


@dataclass(frozen=True)
class Episode:
    index: int
    length: int
    tasks: tuple[str, ...]
    source_episode: int | None
    success: bool | None


def episodes(root: Path, labels: dict[int, bool | None] | None) -> list[Episode]:
    """The dataset's episodes with their v3 source episode and success label, if known."""
    meta = root / "meta"
    segments = {}
    if (meta / "segments.jsonl").is_file():
        segments = {int(r["episode_index"]): r for r in read_jsonl(meta / "segments.jsonl")}
    found = []
    for row in read_jsonl(meta / "episodes.jsonl"):
        index = int(row["episode_index"])
        source = segments.get(index, {}).get("source_episode")
        source = int(source) if source is not None else None
        success = None
        if labels is not None and source is not None:
            success = labels.get(source)
        found.append(
            Episode(
                index=index,
                length=int(row["length"]),
                tasks=tuple(row.get("tasks", ())),
                source_episode=source,
                success=success,
            )
        )
    return found


def keep_reason(episode: Episode, policy: str) -> str | None:
    """None to keep the episode, else why it is dropped."""
    if policy not in EPISODE_POLICIES:
        raise ValueError(f"unknown episode policy {policy!r}")
    if episode.length < ACTION_HORIZON:
        return f"{episode.length} frames, fewer than the {ACTION_HORIZON}-step action horizon"
    if policy == "not-failed" and episode.success is False:
        return "marked failed"
    if policy == "successful" and episode.success is not True:
        return "marked failed" if episode.success is False else "no success label"
    return None


def _episode_files(root: Path, info: dict, index: int) -> list[Path]:
    """Every file of one episode that exists: its parquet, each video, each mask."""
    chunk = index // int(info["chunks_size"])
    paths = [root / info["data_path"].format(episode_chunk=chunk, episode_index=index)]
    video_keys = [k for k, v in info.get("features", {}).items() if v.get("dtype") == "video"]
    for key in video_keys:
        paths.append(
            root / info["video_path"].format(episode_chunk=chunk, episode_index=index, video_key=key)
        )
    mask_path = info.get("mask_path")
    if mask_path:
        for key in video_keys:
            candidate = root / mask_path.format(
                episode_chunk=chunk, episode_index=index, video_key=key
            )
            if candidate.exists():
                paths.append(candidate)
    return paths


def select_episodes(
    source: Source,
    out: Path,
    *,
    policy: str = "not-failed",
    labels: dict[int, bool | None] | None = None,
    labels_from: str | None = None,
) -> dict:
    """Write to ``out`` a copy of the dataset holding only the episodes ``policy`` keeps.

    Returns the selection report, also written to ``out/meta/ethrc_selection.json``.
    ``out`` is replaced if it exists.
    """
    root = source.path
    info = validate(root)
    all_episodes = episodes(root, labels)
    kept, dropped = [], []
    for episode in all_episodes:
        reason = keep_reason(episode, policy)
        (dropped if reason else kept).append((episode, reason))
    if not kept:
        reasons = sorted({reason for _, reason in dropped})
        raise DatasetError(
            f"no episode left to train on with --episodes {policy} "
            f"({len(dropped)} dropped: {'; '.join(reasons)})"
        )

    if out.exists():
        shutil.rmtree(out)
    (out / "meta").mkdir(parents=True)
    for name in _META_COPIED:
        shutil.copy2(root / "meta" / name, out / "meta" / name)

    kept_indices = {episode.index for episode, _ in kept}
    for episode, _ in kept:
        for path in _episode_files(root, info, episode.index):
            if not path.is_file():
                raise DatasetError(f"episode {episode.index}: {path} is missing")
            target = out / path.relative_to(root)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, target)

    meta = root / "meta"
    _write_jsonl(
        out / "meta" / "episodes.jsonl",
        [r for r in read_jsonl(meta / "episodes.jsonl") if int(r["episode_index"]) in kept_indices],
    )
    for name in ("episodes_stats.jsonl", "segments.jsonl"):
        if (meta / name).is_file():
            _write_jsonl(
                out / "meta" / name,
                [r for r in read_jsonl(meta / name) if int(r["episode_index"]) in kept_indices],
            )

    new_info = dict(info)
    new_info["total_episodes"] = len(kept)
    new_info["total_frames"] = sum(episode.length for episode, _ in kept)
    video_keys = [k for k, v in info.get("features", {}).items() if v.get("dtype") == "video"]
    new_info["total_videos"] = len(kept) * len(video_keys)
    chunks = {episode.index // int(info["chunks_size"]) for episode, _ in kept}
    new_info["total_chunks"] = len(chunks)
    new_info["splits"] = {"train": f"0:{len(kept)}"}
    (out / "meta" / "info.json").write_text(json.dumps(new_info, indent=4), encoding="utf-8")

    fps = float(info.get("fps", 50))
    report = {
        "source": {"dataset": source.name, "path": str(root), "revision": source.revision},
        "labels_from": labels_from,
        "policy": policy,
        "fps": fps,
        "kept": [
            {"episode": e.index, "frames": e.length, "source_episode": e.source_episode,
             "success": e.success}
            for e, _ in kept
        ],  # fmt: skip
        "dropped": [
            {"episode": e.index, "frames": e.length, "source_episode": e.source_episode,
             "success": e.success, "reason": reason}
            for e, reason in dropped
        ],  # fmt: skip
        "total_frames": new_info["total_frames"],
        "tasks": sorted({task for e, _ in kept for task in e.tasks}),
    }
    (out / "meta" / REPORT).write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report
