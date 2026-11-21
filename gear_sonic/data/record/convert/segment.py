"""Split recorded episodes on ``quality.valid`` and convert them for GR00T.

``segment_invalid`` is the schema CALLS function. ``convert_dataset`` copies a
LeRobot v3 dataset, writes one new episode per kept valid run, and, when the
``ffmpeg`` binary is on PATH, runs GR00T's unmodified v3-to-v2 script against
a second copy.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
import inspect
import json
import math
import os
from pathlib import Path
import shutil
import subprocess

import numpy as np

GROOT_PYTHON = Path("/home/unitree/.venvs/groot-convert/bin/python")
GROOT_CONVERT_SCRIPT = Path(
    "/home/unitree/ETHRC-Humanoid-Isaac-GR00T/scripts/lerobot_conversion/convert_v3_to_v2.py"
)
BLOCKED_NAME = "CONVERT_BLOCKED.txt"
V21 = "v2.1"
V30 = "v3.0"

_BLOCKED_TEXT = """\
ffmpeg is required to finish this conversion.

The segmented LeRobot v3 dataset in this directory is complete. GR00T's
convert_v3_to_v2.py was not run: it shells out to the ffmpeg binary to split
episode videos, and no ffmpeg executable is on PATH (shutil.which("ffmpeg")
is None). Install ffmpeg and rerun convert_to_groot.py. This directory is
still codebase v3.0, not a v2.1 dataset.

meta/segments.jsonl maps each new episode index to the source episode and
the half-open frame range [start, end) that was kept.
"""


@dataclass(frozen=True)
class ConvertResult:
    """Outcome of ``convert_dataset``.

    ``codec_probe`` is ``"skipped: ffprobe is not installed"`` when ``ffprobe``
    is not on PATH. A missing probe is not a validation failure.
    """

    out_root: Path
    blocked: bool
    v2_written: bool
    codebase_version: str
    fps: float
    fps_matches: bool
    modality_ok: bool
    codec_probe: str
    segments: list[dict]
    message: str

    def to_dict(self) -> dict:
        return {
            "out_root": str(self.out_root),
            "blocked": self.blocked,
            "v2_written": self.v2_written,
            "codebase_version": self.codebase_version,
            "fps": self.fps,
            "fps_matches": self.fps_matches,
            "modality_ok": self.modality_ok,
            "codec_probe": self.codec_probe,
            "segments": self.segments,
            "message": self.message,
        }


def segment_invalid(rows_valid_mask, *, min_segment_frames: int) -> list[tuple[int, int]]:
    """Return half-open ranges of contiguous valid rows.

    A row is valid when its flag is true or non-zero. Runs shorter than
    ``min_segment_frames`` are dropped. ``rows_valid_mask`` may be a flat
    sequence, a column of shape ``(N, 1)``, or a numpy array.
    """
    if isinstance(min_segment_frames, bool) or not isinstance(min_segment_frames, int):
        raise ValueError(f"min_segment_frames must be a positive int, got {min_segment_frames!r}")
    if min_segment_frames < 1:
        raise ValueError(f"min_segment_frames must be a positive int, got {min_segment_frames!r}")

    flags = _valid_flags(rows_valid_mask)
    ranges: list[tuple[int, int]] = []
    start: int | None = None
    for index, flag in enumerate(flags):
        if flag:
            if start is None:
                start = index
            continue
        if start is not None:
            if index - start >= min_segment_frames:
                ranges.append((start, index))
            start = None
    if start is not None and len(flags) - start >= min_segment_frames:
        ranges.append((start, len(flags)))
    return ranges


def write_segmented_dataset(
    v3_root: Path | str,
    out_root: Path | str,
    *,
    min_segment_s: float = 1.0,
    min_segment_frames: int | None = None,
    source_frames: Mapping | None = None,
    repo_id: str | None = None,
    episodes: Iterable[int] | None = None,
) -> list[dict]:
    """Write a new LeRobot v3 dataset containing only the kept valid runs.

    The source directory is read, not modified. Videos are re-encoded with
    ``RGBEncoderConfig(vcodec="h264", g=1)``. The reader is opened with
    ``video_backend="pyav"`` when that constructor accepts it.

    ``source_frames`` is optional and keyed by source episode index. Each value
    is a sequence of RGB frames aligned with that episode, or a dict of video
    key to such a sequence. It is used when the caller already has the pixels.
    The CLI does not pass it: frames are decoded with PyAV from the mp4.
    torchcodec does not load on this board, so it is never selected.

    ``episodes`` selects accumulated v3 episode indices. ``None`` keeps every
    episode. ``source_episode`` in each returned record is that v3 index, not
    the new episode index inside ``out_root``.

    Returns records ``{episode_index, source_episode, start, end}``. Also
    writes them to ``meta/segments.jsonl`` inside ``out_root``.
    """
    from gear_sonic.data.record.convert.groot_meta import write_segments_jsonl
    from gear_sonic.data.record.schema import REPO_ID, ROBOT_TYPE

    source = Path(v3_root)
    destination = Path(out_root)
    if destination.exists():
        raise FileExistsError(f"refusing to write segmented dataset over {destination}")
    dataset = _open_reader(source, repo_id or REPO_ID)
    fps = int(dataset.meta.fps)
    min_frames = _resolve_min_frames(fps, min_segment_s, min_segment_frames)
    features = _user_features(dataset.meta.features)
    robot_type = getattr(dataset.meta.info, "robot_type", None) or ROBOT_TYPE
    grouped = _group_rows(dataset.hf_dataset)
    if not grouped:
        raise ValueError(f"no frames in {source}")
    grouped = _select_episodes(grouped, episodes, source)

    segments: list[dict] = []
    writer = _create_writer(
        destination,
        repo_id or REPO_ID,
        fps,
        features,
        robot_type,
        use_videos=any(spec["dtype"] == "video" for spec in features.values()),
    )
    try:
        new_index = 0
        for episode in sorted(grouped):
            rows = grouped[episode]
            mask = [int(_scalar(row["quality.valid"])) != 0 for row in rows]
            ranges = segment_invalid(mask, min_segment_frames=min_frames)
            if not ranges:
                continue
            videos = _episode_videos(dataset, episode, rows, source_frames)
            for start, end in ranges:
                for offset in range(start, end):
                    writer.add_frame(_frame_dict(rows[offset], videos, offset, features, dataset))
                writer.save_episode()
                segments.append(
                    {
                        "episode_index": new_index,
                        "source_episode": int(episode),
                        "start": int(start),
                        "end": int(end),
                    }
                )
                new_index += 1
    finally:
        writer.finalize()

    if not segments:
        raise ValueError(
            f"no valid run of at least {min_frames} frame(s) in {source} "
            f"(min_segment_s={min_segment_s})"
        )
    write_segments_jsonl(destination / "meta" / "segments.jsonl", segments)
    return segments


def convert_dataset(
    v3_root: Path | str,
    out_root: Path | str,
    *,
    min_segment_s: float = 1.0,
    min_segment_frames: int | None = None,
    source_frames: Mapping | None = None,
    repo_id: str | None = None,
    episodes: Iterable[int] | None = None,
) -> ConvertResult:
    """Copy ``v3_root``, segment it, and convert the copy to GR00T v2.1.

    The caller's directory is never moved and never converted in place.
    GR00T's script joins ``--root / repo_id`` and then moves that directory,
    so the subprocess receives the parent of a disposable copy.

    When ``ffmpeg`` is missing, the segmented v3 dataset is still published to
    ``out_root`` with ``CONVERT_BLOCKED.txt``, and ``v2_written`` is false.

    ``episodes`` limits the copy to those accumulated v3 episode indices.
    ``None`` converts the whole dataset. Segment ``source_episode`` values
    stay those v3 indices.
    """
    from gear_sonic.data.record.convert.groot_meta import (
        add_task_description_column,
        build_modality,
        delete_stats_files,
        validate_modality,
        write_modality,
        write_segments_jsonl,
    )
    from gear_sonic.data.record.schema import REPO_ID

    source = Path(v3_root).resolve()
    destination = Path(out_root).resolve()
    chosen_repo = repo_id or REPO_ID
    _reject_overlapping_paths(source, destination)
    source_info = _read_info(source)
    if source_info.get("codebase_version") != V30:
        raise ValueError(
            f"{source} codebase_version is {source_info.get('codebase_version')!r}, expected {V30!r}"
        )
    source_fps = source_info["fps"]
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)

    with _temp_dir() as raw_temporary:
        temporary = Path(raw_temporary)
        copied = temporary / "v3_copy"
        shutil.copytree(source, copied, symlinks=False)
        segmented = temporary / "segmented_v3"
        segments = write_segmented_dataset(
            copied,
            segmented,
            min_segment_s=min_segment_s,
            min_segment_frames=min_segment_frames,
            source_frames=source_frames,
            repo_id=chosen_repo,
            episodes=episodes,
        )
        _publish_tree(segmented, destination)
        modality = build_modality()
        published = _read_info(destination)
        validate_modality(published["features"], modality)
        if published["fps"] != source_fps:
            raise ValueError(f"segmented fps {published['fps']!r} does not match source fps {source_fps!r}")

        if shutil.which("ffmpeg") is None:
            (destination / BLOCKED_NAME).write_text(_BLOCKED_TEXT, encoding="utf-8")
            message = (
                "ffmpeg is required to convert the segmented v3 dataset to GR00T v2.1. "
                f"Segmented v3 was written to {destination}. See {BLOCKED_NAME}."
            )
            return ConvertResult(
                out_root=destination,
                blocked=True,
                v2_written=False,
                codebase_version=str(published["codebase_version"]),
                fps=published["fps"],
                fps_matches=True,
                modality_ok=True,
                codec_probe=_probe_codec(destination),
                segments=segments,
                message=message,
            )

        # convert_v3_to_v2.py sets root = Path(root) / repo_id, then moves that
        # directory aside. Point it at a copy whose parent we can throw away.
        staged_parent = temporary / "groot_root"
        staged = staged_parent.joinpath(*chosen_repo.split("/"))
        staged.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(destination, staged, symlinks=False)
        _run_groot_convert(staged_parent, chosen_repo)
        if not (staged / "meta" / "info.json").is_file():
            raise RuntimeError(f"GR00T conversion did not leave a dataset at {staged}")
        write_modality(staged / "meta" / "modality.json", modality)
        add_task_description_column(staged)
        delete_stats_files(staged)
        write_segments_jsonl(staged / "meta" / "segments.jsonl", segments)
        v2_info = _read_info(staged)
        validate_modality(v2_info["features"], modality)
        if v2_info.get("codebase_version") != V21:
            raise ValueError(
                f"expected codebase_version {V21!r} after v3-to-v2, "
                f"found {v2_info.get('codebase_version')!r}"
            )
        if v2_info["fps"] != source_fps:
            raise ValueError(f"v2 fps {v2_info['fps']!r} does not match source fps {source_fps!r}")
        codec_probe = _probe_codec(staged)
        _replace_tree(staged, destination)
        return ConvertResult(
            out_root=destination,
            blocked=False,
            v2_written=True,
            codebase_version=V21,
            fps=v2_info["fps"],
            fps_matches=True,
            modality_ok=True,
            codec_probe=codec_probe,
            segments=segments,
            message=f"Wrote GR00T v2.1 dataset to {destination}.",
        )


def _temp_dir():
    import tempfile

    return tempfile.TemporaryDirectory(prefix="g1-convert-")


def _reject_overlapping_paths(source: Path, destination: Path) -> None:
    if source == destination or _is_relative_to(destination, source) or _is_relative_to(source, destination):
        raise ValueError(f"out_root {destination} must be outside the source dataset {source}")


def _is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return path != parent


def _read_info(root: Path) -> dict:
    path = root / "meta" / "info.json"
    if not path.is_file():
        raise FileNotFoundError(f"missing {path}")
    info = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(info, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return info


def _select_episodes(grouped: dict[int, list[dict]], episodes: Iterable[int] | None, source: Path) -> dict:
    """Keep ``episodes`` when set. Keys stay the accumulated v3 episode indices."""
    if episodes is None:
        return grouped
    selected: set[int] = set()
    for episode in episodes:
        if isinstance(episode, bool) or not isinstance(episode, (int, np.integer)):
            raise TypeError(f"episodes must be ints, got {episode!r}")
        selected.add(int(episode))
    if not selected:
        raise ValueError("episodes is empty")
    missing = sorted(selected - set(grouped))
    if missing:
        raise ValueError(f"episodes {missing} are not in {source}")
    return {episode: grouped[episode] for episode in selected}


def _resolve_min_frames(fps: int, min_segment_s: float, min_segment_frames: int | None) -> int:
    if min_segment_frames is not None:
        if isinstance(min_segment_frames, bool) or not isinstance(min_segment_frames, int):
            raise ValueError(f"min_segment_frames must be a positive int, got {min_segment_frames!r}")
        if min_segment_frames < 1:
            raise ValueError(f"min_segment_frames must be a positive int, got {min_segment_frames!r}")
        return min_segment_frames
    if isinstance(min_segment_s, bool) or not isinstance(min_segment_s, (int, float)):
        raise ValueError(f"min_segment_s must be a non-negative number, got {min_segment_s!r}")
    if not math.isfinite(float(min_segment_s)) or float(min_segment_s) < 0:
        raise ValueError(f"min_segment_s must be a non-negative number, got {min_segment_s!r}")
    return max(1, int(round(float(fps) * float(min_segment_s))))


def _publish_tree(source: Path, destination: Path) -> None:
    if destination.exists():
        raise FileExistsError(destination)
    shutil.copytree(source, destination, symlinks=False)


def _replace_tree(source: Path, destination: Path) -> None:
    temporary = destination.with_name(destination.name + ".replacing")
    backup = destination.with_name(destination.name + ".previous")
    if temporary.exists():
        shutil.rmtree(temporary)
    if backup.exists():
        shutil.rmtree(backup)
    shutil.copytree(source, temporary, symlinks=False)
    destination.rename(backup)
    try:
        temporary.rename(destination)
    except Exception:
        if not destination.exists():
            backup.rename(destination)
        raise
    shutil.rmtree(backup)


def _run_groot_convert(root_parent: Path, repo_id: str) -> None:
    if not GROOT_PYTHON.is_file():
        raise FileNotFoundError(f"groot-convert interpreter not found: {GROOT_PYTHON}")
    if not GROOT_CONVERT_SCRIPT.is_file():
        raise FileNotFoundError(f"GR00T convert script not found: {GROOT_CONVERT_SCRIPT}")
    env = os.environ.copy()
    env.setdefault("HF_HUB_OFFLINE", "1")
    env.setdefault("HF_DATASETS_OFFLINE", "1")
    completed = subprocess.run(
        [
            str(GROOT_PYTHON),
            str(GROOT_CONVERT_SCRIPT),
            "--repo-id",
            repo_id,
            "--root",
            str(root_parent),
        ],
        check=False,
        capture_output=True,
        text=True,
        env=env,
    )
    if completed.returncode != 0:
        detail = (completed.stderr or completed.stdout or "").strip()
        raise RuntimeError(
            f"convert_v3_to_v2.py exited {completed.returncode} for repo {repo_id} "
            f"under {root_parent}.\n{detail[-4000:]}"
        )


def _probe_codec(root: Path) -> str:
    """Return the first video codec, or say that ffprobe is absent.

    A missing ``ffprobe`` does not fail validation. ``ffmpeg`` and ``ffprobe``
    are both absent on this board, so the usual result is the skipped message.
    """
    if shutil.which("ffprobe") is None:
        return "skipped: ffprobe is not installed"
    videos_root = root / "videos"
    videos = sorted(videos_root.rglob("*.mp4")) if videos_root.is_dir() else []
    if not videos:
        return "skipped: no mp4 to probe"
    completed = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=codec_name",
            "-of",
            "csv=p=0",
            str(videos[0]),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        detail = (completed.stderr or "").strip()
        return f"ffprobe failed: {detail}" if detail else "ffprobe failed"
    codec = completed.stdout.strip()
    return codec or "ffprobe returned no codec"


def _open_reader(root: Path, repo_id: str):
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    kwargs = _video_backend_kwargs(LeRobotDataset.__init__)
    if "download_videos" in inspect.signature(LeRobotDataset.__init__).parameters:
        kwargs["download_videos"] = False
    return LeRobotDataset(repo_id, root=root, **kwargs)


def _create_writer(root: Path, repo_id: str, fps: int, features: dict, robot_type: str, *, use_videos: bool):
    from lerobot.configs import RGBEncoderConfig
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    kwargs = _video_backend_kwargs(LeRobotDataset.create)
    signature = inspect.signature(LeRobotDataset.create)
    if "rgb_encoder" in signature.parameters:
        kwargs["rgb_encoder"] = RGBEncoderConfig(vcodec="h264", g=1)
    return LeRobotDataset.create(
        repo_id=repo_id,
        fps=fps,
        features=features,
        root=root,
        robot_type=robot_type,
        use_videos=use_videos,
        **kwargs,
    )


def _video_backend_kwargs(callable_) -> dict:
    if "video_backend" in inspect.signature(callable_).parameters:
        return {"video_backend": "pyav"}
    return {}


def _user_features(features: dict) -> dict:
    from lerobot.utils.constants import DEFAULT_FEATURES

    cleaned = {}
    for key, feature in features.items():
        if key in DEFAULT_FEATURES:
            continue
        spec = {
            "dtype": feature["dtype"],
            "shape": tuple(int(dim) for dim in feature["shape"]),
        }
        names = feature.get("names")
        if names is not None:
            spec["names"] = list(names) if isinstance(names, (list, tuple)) else names
        cleaned[key] = spec
    return cleaned


def _group_rows(hf_dataset) -> dict[int, list[dict]]:
    grouped: dict[int, list[dict]] = {}
    for position in range(len(hf_dataset)):
        row = hf_dataset[position]
        if "quality.valid" not in row:
            raise ValueError("dataset is missing the quality.valid column")
        episode = int(_scalar(row["episode_index"]))
        grouped.setdefault(episode, []).append(row)
    for rows in grouped.values():
        rows.sort(key=lambda row: int(_scalar(row["frame_index"])))
    return grouped


def _episode_videos(dataset, episode: int, rows: list[dict], source_frames: Mapping | None) -> dict:
    video_keys = list(dataset.meta.video_keys)
    if not video_keys:
        return {}
    if source_frames is not None and episode in source_frames:
        return _frames_from_caller(source_frames[episode], video_keys, len(rows))
    loaded = {}
    errors: list[str] = []
    for key in video_keys:
        try:
            loaded[key] = _decode_with_pyav_backend(dataset, episode, key, rows)
            continue
        except Exception as exc:
            errors.append(f"{key} via video_backend=pyav: {exc}")
        try:
            loaded[key] = _decode_with_av_package(dataset, episode, key, rows)
        except Exception as exc:
            errors.append(f"{key} via av: {exc}")
            raise RuntimeError(
                "Could not decode video without torchcodec. torchcodec does not load "
                "on this board. The CLI reads frames with PyAV (the av package) from "
                "the mp4. Pass source_frames to write_segmented_dataset to rebuild "
                "segments from frames the caller already has.\n" + "\n".join(errors)
            ) from exc
    return loaded


def _frames_from_caller(provided, video_keys: list[str], length: int) -> dict[str, list[np.ndarray]]:
    if isinstance(provided, Mapping) and set(video_keys).issubset(provided):
        payload = {key: list(provided[key]) for key in video_keys}
    else:
        if len(video_keys) != 1:
            raise ValueError(
                "source_frames entries must map each video key to frames when the dataset "
                f"has cameras {video_keys}"
            )
        payload = {video_keys[0]: list(provided)}
    images: dict[str, list[np.ndarray]] = {}
    for key, frames in payload.items():
        if len(frames) != length:
            raise ValueError(f"source_frames[{key!r}] has {len(frames)} frames, episode has {length}")
        images[key] = [_as_uint8_hwc(frame) for frame in frames]
    return images


def _decode_with_pyav_backend(dataset, episode: int, video_key: str, rows: list[dict]) -> list[np.ndarray]:
    from lerobot.datasets.video_utils import decode_video_frames

    record = _episode_record(dataset, episode)
    from_timestamp = float(_scalar(record[f"videos/{video_key}/from_timestamp"]))
    timestamps = [from_timestamp + float(_scalar(row["timestamp"])) for row in rows]
    path = dataset.root / dataset.meta.get_video_file_path(episode, video_key)
    decoded = decode_video_frames(
        path,
        timestamps,
        dataset.tolerance_s,
        "pyav",
        return_uint8=True,
    )
    images = _chw_batch_to_hwc(decoded)
    if len(images) != len(rows):
        raise RuntimeError(f"decoded {len(images)} frames from {path}, episode has {len(rows)}")
    return images


def _decode_with_av_package(dataset, episode: int, video_key: str, rows: list[dict]) -> list[np.ndarray]:
    import av

    record = _episode_record(dataset, episode)
    from_timestamp = float(_scalar(record[f"videos/{video_key}/from_timestamp"]))
    path = dataset.root / dataset.meta.get_video_file_path(episode, video_key)
    frames: list[np.ndarray] = []
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        for frame in container.decode(stream):
            frames.append(frame.to_ndarray(format="rgb24"))
    start = int(round(from_timestamp * float(dataset.meta.fps)))
    selected = frames[start : start + len(rows)]
    if len(selected) != len(rows):
        raise RuntimeError(
            f"av read {len(frames)} frames from {path}; needed {len(rows)} starting at {start} "
            f"(from_timestamp={from_timestamp})"
        )
    return selected


def _chw_batch_to_hwc(decoded) -> list[np.ndarray]:
    if hasattr(decoded, "detach"):
        decoded = decoded.detach().cpu().numpy()
    array = np.asarray(decoded)
    if array.ndim != 4:
        raise RuntimeError(f"expected a batch of frames, got shape {array.shape}")
    images = []
    for frame in array:
        images.append(_as_uint8_hwc(frame))
    return images


def _episode_record(dataset, episode: int) -> dict:
    episodes = dataset.meta.episodes
    record = episodes[episode]
    if int(_scalar(record["episode_index"])) == episode:
        return record
    for index in range(len(episodes)):
        candidate = episodes[index]
        if int(_scalar(candidate["episode_index"])) == episode:
            return candidate
    raise KeyError(f"episode {episode} is not in dataset metadata")


def _frame_dict(row: dict, videos: dict, offset: int, features: dict, dataset) -> dict:
    task_index = int(_scalar(row["task_index"]))
    frame = {"task": _task_name(dataset, task_index)}
    for key, feature in features.items():
        if feature["dtype"] in ("image", "video"):
            frame[key] = videos[key][offset]
            continue
        frame[key] = _feature_array(row[key], feature)
    return frame


def _task_name(dataset, task_index: int) -> str:
    tasks = dataset.meta.tasks
    if tasks is None:
        return "task"
    matched = tasks.index[tasks["task_index"] == task_index]
    if len(matched) == 0:
        raise KeyError(f"task_index {task_index} is not in meta/tasks")
    return str(matched[0])


def _feature_array(value, feature: dict) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    dtype = np.dtype(feature["dtype"])
    array = np.asarray(value, dtype=dtype)
    return np.reshape(array, tuple(int(dim) for dim in feature["shape"]))


def _as_uint8_hwc(frame) -> np.ndarray:
    if hasattr(frame, "detach"):
        frame = frame.detach().cpu().numpy()
    array = np.asarray(frame)
    if array.ndim != 3:
        raise ValueError(f"expected an HxWx3 frame, got shape {array.shape}")
    if array.shape[0] == 3 and array.shape[-1] != 3:
        array = np.transpose(array, (1, 2, 0))
    if array.dtype != np.uint8:
        scale = 255.0 if array.size and float(np.nanmax(array)) <= 1.0 else 1.0
        array = np.clip(array * scale, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(array)


def _scalar(value):
    if hasattr(value, "detach"):
        value = value.detach().cpu()
        if getattr(value, "ndim", 0) == 0:
            return value.item()
        return value.reshape(-1)[0].item()
    array = np.asarray(value)
    if array.shape == ():
        item = array.item()
        return item.item() if hasattr(item, "item") else item
    item = array.reshape(-1)[0]
    return item.item() if hasattr(item, "item") else item


def _valid_flags(rows_valid_mask) -> list[bool]:
    if hasattr(rows_valid_mask, "detach"):
        rows_valid_mask = rows_valid_mask.detach().cpu().numpy()
    if hasattr(rows_valid_mask, "to_pylist") and not isinstance(
        rows_valid_mask, (str, bytes, np.ndarray, Sequence)
    ):
        return _valid_flags(rows_valid_mask.to_pylist())
    if isinstance(rows_valid_mask, (str, bytes)):
        raise TypeError("rows_valid_mask must be a sequence of valid flags")

    if _object_mask(rows_valid_mask):
        array = np.asarray(rows_valid_mask, dtype=object)
    else:
        array = np.asarray(rows_valid_mask)
    if array.dtype == object:
        flags: list[bool] = []
        flat = array.reshape(-1) if array.ndim else [array.item()]
        for item in flat:
            if _is_nested_flag(item):
                flags.extend(_valid_flags(item))
            else:
                flags.append(_as_flag(item))
        return flags
    if array.shape == ():
        return [_as_flag(array.item())]
    return [_as_flag(item) for item in array.reshape(-1).tolist()]


def _object_mask(rows_valid_mask) -> bool:
    if isinstance(rows_valid_mask, np.ndarray):
        return rows_valid_mask.dtype == object
    if isinstance(rows_valid_mask, Sequence) and not isinstance(rows_valid_mask, (str, bytes)):
        return any(_is_nested_flag(item) or hasattr(item, "detach") for item in rows_valid_mask)
    return False


def _is_nested_flag(item) -> bool:
    if isinstance(item, (str, bytes)):
        return False
    if isinstance(item, np.ndarray):
        return item.ndim > 0
    return isinstance(item, Sequence)


def _as_flag(item) -> bool:
    if isinstance(item, (bool, np.bool_)):
        return bool(item)
    if isinstance(item, (int, np.integer)):
        return int(item) != 0
    if isinstance(item, (float, np.floating)):
        number = float(item)
        if math.isnan(number):
            return False
        return number != 0.0
    raise TypeError(f"valid flag must be numeric or bool, got {item!r}")
