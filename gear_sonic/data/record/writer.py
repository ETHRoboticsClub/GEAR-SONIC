"""Write G1 teleop episodes as a LeRobot v3 dataset.

``write_episode`` is the ``writer.write_episode`` call in ``schema.py``. It does
not import torchcodec. Encoding goes through PyAV. ``create`` is tried with
``streaming_encoding=True`` and, if that call fails while importing torchcodec,
once more with ``streaming_encoding=False`` and the same H.264 encoder.

``append_episode`` creates that dataset the first time, then resumes it. The
first create also bounds parquet and video files so a later upload can resend
them in pieces. Per-episode clock and quality files live under
``meta/grid_clock/`` and ``meta/quality/``.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np

from gear_sonic.data.record import schema

_VIDEO_KEY = "observation.images.ego_view"
_INT64_COLUMNS = (
    ("observation.g1_debug_index", "g1_index"),
    ("observation.pico_sequence", "pico_sequence"),
    ("quality.valid", "valid"),
    ("quality.flags", "flags"),
)
_FLOAT32_COLUMNS = (
    ("observation.state", "state"),
    ("observation.projected_gravity", "projected_gravity"),
    ("observation.base_quat", "base_quat"),
    ("observation.base_ang_vel", "base_ang_vel"),
    ("observation.joint_velocity", "joint_velocity"),
    ("observation.left_hand_dq", "left_hand_dq"),
    ("observation.right_hand_dq", "right_hand_dq"),
    ("observation.pico", "pico"),
    ("observation.sample_age_s", "ages"),
    ("action", "action"),
    ("action.left_hand", "left_hand_action"),
    ("action.right_hand", "right_hand_action"),
    ("action.motion_token", "token"),
)
_METADATA_KEYS = frozenset({"camera_offset_ns", "source_commit", "success", "video_encode"})
_STREAMING = "streaming_encoding=True"
_BATCH = "streaming_encoding=False"


def write_episode(
    rows,
    frames_rgb,
    root,
    *,
    fps,
    task,
    t0_ns,
    metadata,
    image_hw,
) -> Path:
    """Write ``rows`` and RGB frames into a new LeRobot v3 dataset at ``root``.

    Returns the dataset root. Also writes ``meta/grid_clock.json`` and
    ``meta/quality/episode_000000.json``. ``metadata['video_encode']`` records
    whether streaming encoding or the non-streaming retry produced the dataset.
    """
    rows, image_height, image_width = _prepare_episode(rows, fps, metadata, image_hw)
    root = Path(root)
    existed = root.exists()
    features = schema.features(image_height, image_width)
    dataset, video_encode = _create_dataset(root, existed, fps, features)
    metadata["video_encode"] = video_encode
    _add_rows(
        dataset,
        rows,
        frames_rgb,
        task=task,
        image_height=image_height,
        image_width=image_width,
    )
    dataset.save_episode()
    dataset.finalize()

    written = Path(dataset.root)
    _write_sidecars(
        written,
        rows,
        fps=fps,
        t0_ns=t0_ns,
        metadata=metadata,
        video_encode=video_encode,
        episode_index=None,
    )
    return written


def append_episode(
    rows,
    frames_rgb,
    root,
    *,
    fps,
    task,
    t0_ns,
    metadata,
    image_hw,
) -> int:
    """Append one episode to a LeRobot v3 dataset and return its index.

    A root without ``meta/info.json`` is created the same way as
    ``write_episode``, with ``video_files_size_in_mb=100`` and
    ``data_files_size_in_mb=50``. An existing dataset is resumed. ``fps`` and
    the ego-view shape in ``info.json`` must match this call. Sidecars are
    ``meta/grid_clock/episode_NNNNNN.json`` and
    ``meta/quality/episode_NNNNNN.json``.
    """
    rows, image_height, image_width = _prepare_episode(rows, fps, metadata, image_hw)
    root = Path(root)
    features = schema.features(image_height, image_width)
    info_path = root / "meta" / "info.json"
    if not info_path.is_file():
        dataset, video_encode = _create_dataset(
            root,
            root.exists(),
            fps,
            features,
            video_files_size_in_mb=100,
            data_files_size_in_mb=50,
        )
    else:
        _require_compatible(info_path, fps, image_height, image_width)
        dataset, video_encode = _resume_dataset(root)
    episode_index = int(dataset.meta.total_episodes)
    metadata["video_encode"] = video_encode
    _add_rows(
        dataset,
        rows,
        frames_rgb,
        task=task,
        image_height=image_height,
        image_width=image_width,
    )
    dataset.save_episode()
    dataset.finalize()
    _write_sidecars(
        Path(dataset.root),
        rows,
        fps=fps,
        t0_ns=t0_ns,
        metadata=metadata,
        video_encode=video_encode,
        episode_index=episode_index,
    )
    return episode_index


def _prepare_episode(rows, fps, metadata, image_hw):
    if not isinstance(metadata, dict):
        raise TypeError(f"metadata must be a dict, got {type(metadata).__name__}")
    if isinstance(fps, bool) or not isinstance(fps, int) or fps <= 0:
        raise ValueError(f"fps must be a positive int, got {fps!r}")
    rows = list(rows)
    if not rows:
        raise ValueError("rows must contain at least one row")
    image_height, image_width = _image_hw(image_hw)
    return rows, image_height, image_width


def _require_compatible(info_path: Path, fps: int, image_height: int, image_width: int) -> None:
    info = json.loads(info_path.read_text(encoding="utf-8"))
    if not isinstance(info, dict):
        raise ValueError(f"{info_path} must contain a JSON object")
    stored_fps = info.get("fps")
    if isinstance(stored_fps, bool) or not isinstance(stored_fps, (int, float)):
        raise ValueError(f"{info_path} fps is {stored_fps!r}, not {fps}")
    if float(stored_fps) != float(fps):
        raise ValueError(f"{info_path} fps is {stored_fps!r}, not {fps}")
    features = info.get("features")
    if not isinstance(features, dict) or _VIDEO_KEY not in features:
        raise ValueError(f"{info_path} is missing {_VIDEO_KEY}")
    shape = features[_VIDEO_KEY].get("shape") if isinstance(features[_VIDEO_KEY], dict) else None
    expected = [image_height, image_width, 3]
    if not isinstance(shape, (list, tuple)) or [int(dim) for dim in shape] != expected:
        raise ValueError(f"{info_path} {_VIDEO_KEY} shape is {shape!r}, not {expected}")


def _image_hw(image_hw) -> tuple[int, int]:
    try:
        image_height, image_width = image_hw
    except (TypeError, ValueError) as exc:
        raise ValueError("image_hw must be (height, width)") from exc
    return image_height, image_width


def _create_dataset(root, existed, fps, features, **limits):
    try:
        return _open_dataset(root, fps, features, streaming=True, **limits), _STREAMING
    except Exception as exc:
        if not _torchcodec_failure(exc):
            raise
        if not existed and root.exists():
            shutil.rmtree(root)
        try:
            dataset = _open_dataset(root, fps, features, streaming=False, **limits)
        except Exception as retry_exc:
            raise retry_exc from exc
        return dataset, _BATCH


def _open_dataset(root, fps, features, *, streaming: bool, **limits):
    from lerobot.configs.video import RGBEncoderConfig
    from lerobot.datasets import LeRobotDataset

    # g=1 makes every frame a keyframe so a later ffmpeg stream-copy can cut
    # on any row. create() has no vcodec argument; the encoder config carries it.
    encoder = RGBEncoderConfig(vcodec="h264", g=1)
    return LeRobotDataset.create(
        repo_id=schema.REPO_ID,
        fps=fps,
        features=features,
        root=root,
        robot_type=schema.ROBOT_TYPE,
        use_videos=True,
        rgb_encoder=encoder,
        streaming_encoding=streaming,
        **limits,
    )


def _resume_dataset(root):
    from lerobot.configs.video import RGBEncoderConfig
    from lerobot.datasets import LeRobotDataset

    def open_resume(streaming: bool):
        return LeRobotDataset.resume(
            repo_id=schema.REPO_ID,
            root=root,
            rgb_encoder=RGBEncoderConfig(vcodec="h264", g=1),
            streaming_encoding=streaming,
        )

    try:
        return open_resume(True), _STREAMING
    except Exception as exc:
        if not _torchcodec_failure(exc):
            raise
        try:
            dataset = open_resume(False)
        except Exception as retry_exc:
            raise retry_exc from exc
        return dataset, _BATCH


def _torchcodec_failure(exc: BaseException) -> bool:
    seen: set[int] = set()
    current: BaseException | None = exc
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        text = f"{type(current).__module__}.{type(current).__name__}: {current}"
        lowered = text.lower()
        if "torchcodec" in lowered or "libnppicc" in lowered:
            return True
        current = current.__cause__ or current.__context__
    return False


def _add_rows(dataset, rows, frames_rgb, *, task, image_height, image_width) -> None:
    previous = None
    for index, row in enumerate(rows):
        image = _image_for_row(
            row,
            frames_rgb,
            previous,
            image_height=image_height,
            image_width=image_width,
            index=index,
        )
        previous = image
        dataset.add_frame(_frame_dict(row, image, task=task))


def _image_for_row(row, frames_rgb, previous, *, image_height, image_width, index):
    if row.image_index >= 0:
        try:
            source = frames_rgb[row.image_index]
        except (IndexError, TypeError) as exc:
            raise IndexError(
                f"row {index} image_index {row.image_index} is outside frames_rgb"
            ) from exc
        image = np.asarray(source)
        if image.dtype != np.uint8 or image.shape != (image_height, image_width, 3):
            raise ValueError(
                f"row {index} frame must be uint8 HWC {(image_height, image_width, 3)}, "
                f"got dtype {image.dtype} shape {image.shape}"
            )
        if not image.flags["C_CONTIGUOUS"]:
            image = np.ascontiguousarray(image)
        return image

    if (row.flags & schema.CAMERA_STALE) == 0:
        raise ValueError(
            f"row {index} image_index is {row.image_index} but CAMERA_STALE is unset"
        )
    if previous is not None:
        return previous
    return np.zeros((image_height, image_width, 3), dtype=np.uint8)


def _frame_dict(row, image, *, task) -> dict:
    frame = {"task": task, _VIDEO_KEY: image}
    for key, attr in _FLOAT32_COLUMNS:
        frame[key] = np.asarray(getattr(row, attr), dtype=np.float32)
    for key, attr in _INT64_COLUMNS:
        frame[key] = np.asarray([getattr(row, attr)], dtype=np.int64)
    return frame


def _write_sidecars(root, rows, *, fps, t0_ns, metadata, video_encode, episode_index) -> None:
    meta = root / "meta"
    clock = {
        "clock": "CLOCK_MONOTONIC",
        "grid_origin_monotonic_ns": int(t0_ns),
        "fps": int(fps),
        "camera_offset_ns": _metadata_int(metadata, "camera_offset_ns", 0),
        "source_commit": _metadata_str(metadata, "source_commit", ""),
    }
    quality = _quality_report(rows, metadata, video_encode)
    if episode_index is None:
        quality_dir = meta / "quality"
        quality_dir.mkdir(parents=True, exist_ok=True)
        _dump_json(meta / "grid_clock.json", clock)
        _dump_json(quality_dir / "episode_000000.json", quality)
        return
    name = f"episode_{int(episode_index):06d}.json"
    grid_dir = meta / "grid_clock"
    quality_dir = meta / "quality"
    grid_dir.mkdir(parents=True, exist_ok=True)
    quality_dir.mkdir(parents=True, exist_ok=True)
    _dump_json(grid_dir / name, clock)
    _dump_json(quality_dir / name, quality)


def _quality_report(rows, metadata, video_encode) -> dict:
    n_rows = len(rows)
    n_invalid = sum(row.valid == 0 for row in rows)
    n_state_interp = sum((row.flags & schema.STATE_INTERP) != 0 for row in rows)
    success = metadata["success"] if "success" in metadata else None
    return {
        "n_rows": n_rows,
        "n_invalid": n_invalid,
        "pct_invalid": _percent(n_invalid, n_rows),
        "pct_state_interp": _percent(n_state_interp, n_rows),
        "gaps": [
            {"k": index, "flags": int(row.flags)}
            for index, row in enumerate(rows)
            if row.valid == 0
        ],
        "success": success,
        "drops": _drop_counts(metadata),
        "video_encode": video_encode,
    }


def _percent(count: int, total: int) -> float:
    if total <= 0:
        return 0.0
    return 100.0 * count / total


def _drop_counts(metadata: dict) -> dict:
    drops: dict = {}
    for key, value in metadata.items():
        if key in _METADATA_KEYS or "drop" not in key.lower():
            continue
        if key in {"drops", "drop_counts"} and isinstance(value, dict):
            for name, count in value.items():
                drops[str(name)] = _jsonable(count)
            continue
        drops[key] = _jsonable(value)
    return drops


def _metadata_int(metadata: dict, key: str, default: int) -> int:
    value = metadata.get(key, default)
    if value is None:
        value = default
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"metadata[{key!r}] must be an int, got {type(value).__name__}")
    return int(value)


def _metadata_str(metadata: dict, key: str, default: str) -> str:
    value = metadata.get(key, default)
    if value is None:
        return default
    if not isinstance(value, str):
        raise TypeError(f"metadata[{key!r}] must be a str, got {type(value).__name__}")
    return value


def _jsonable(value):
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


def _dump_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
