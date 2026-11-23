"""Append one freshly converted GR00T v2.1 dataset onto an accumulated one.

Episode parquet files and mp4s are written before any metadata is replaced.
Each metadata file is written to a temporary sibling and moved into place with
``os.replace``. ``stats.json`` and ``relative_stats.json`` are not kept.
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
import shutil

import pyarrow as pa
import pyarrow.parquet as pq

_V21 = "v2.1"
_TASK_INDEX = "task_index"
_TASK_TEXT = "annotation.human.task_description"
_EPISODE_INDEX = "episode_index"
_GLOBAL_INDEX = "index"
_STATS_NAMES = ("stats.json", "relative_stats.json")


def append_v21(dst_root, src_root) -> list[int]:
    """Merge ``src_root`` into ``dst_root`` and return the new episode indices.

    When ``dst_root`` has no dataset yet, ``src_root`` is moved there. Otherwise
    the two datasets must share modality, fps, features, and codebase version
    ``v2.1``. New ``episode_index`` values are shifted by the destination's
    ``total_episodes``, and the global ``index`` column by ``total_frames``.
    """
    destination = Path(dst_root)
    source = Path(src_root)
    if not (source / "meta" / "info.json").is_file():
        raise FileNotFoundError(f"{source} has no meta/info.json")
    if destination.exists() and destination.resolve() == source.resolve():
        raise ValueError("dst_root and src_root are the same dataset")
    source_info = _read_info(source)
    if source_info.get("codebase_version") != _V21:
        raise ValueError(
            f"{source} codebase_version is {source_info.get('codebase_version')!r}, expected {_V21!r}"
        )
    if not (destination / "meta" / "info.json").is_file():
        if destination.exists():
            raise FileExistsError(f"{destination} exists but has no meta/info.json")
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(source), str(destination))
        _remove_stats(destination)
        return _episode_indices(destination)

    return _merge(destination, source, source_info)


def _merge(destination: Path, source: Path, source_info: dict) -> list[int]:
    dest_info = _read_info(destination)
    _require_same_dataset(destination, source, dest_info, source_info)
    episode_offset = int(dest_info["total_episodes"])
    frame_offset = int(dest_info["total_frames"])
    task_map, new_tasks = _task_mapping(source, destination)
    episodes = _read_jsonl(source / "meta" / "episodes.jsonl")
    stats_rows = _read_jsonl(source / "meta" / "episodes_stats.jsonl")
    segments = _read_jsonl(source / "meta" / "segments.jsonl")
    if not episodes:
        raise ValueError(f"{source} has no episodes to append")
    if not segments:
        raise ValueError(f"{source / 'meta' / 'segments.jsonl'} has no segments")
    episodes = sorted(episodes, key=lambda record: _plain_int(record[_EPISODE_INDEX]))
    stats_by_episode = {_plain_int(record[_EPISODE_INDEX]): record for record in stats_rows}
    episode_ids = [_plain_int(record[_EPISODE_INDEX]) for record in episodes]
    if set(stats_by_episode) != set(episode_ids):
        raise ValueError(f"{source} episodes.jsonl and episodes_stats.jsonl do not list the same episodes")

    video_keys = _video_keys(dest_info)
    dest_chunk = int(dest_info.get("chunks_size") or 1000)
    source_chunk = int(source_info.get("chunks_size") or 1000)
    plan = [
        _plan_episode(
            source,
            destination,
            record,
            source_info=source_info,
            dest_info=dest_info,
            episode_offset=episode_offset,
            video_keys=video_keys,
            source_chunk=source_chunk,
            dest_chunk=dest_chunk,
        )
        for record in episodes
    ]
    written_rows: list[int] = []
    for item in plan:
        written_rows.append(
            _write_episode_files(
                item,
                task_map=task_map,
                episode_offset=episode_offset,
                frame_offset=frame_offset,
            )
        )

    new_indices = [item["new_index"] for item in plan]
    _append_jsonl(destination / "meta" / "tasks.jsonl", new_tasks)
    _append_jsonl(
        destination / "meta" / "episodes.jsonl",
        [_shift_episode_record(record, episode_offset) for record in episodes],
    )
    _append_jsonl(
        destination / "meta" / "episodes_stats.jsonl",
        [
            _shift_stats(stats_by_episode[old_index], episode_offset, frame_offset)
            for old_index in episode_ids
        ],
    )
    _append_jsonl(
        destination / "meta" / "segments.jsonl",
        [_shift_segment(record, episode_offset) for record in segments],
    )
    _write_updated_info(
        destination,
        dest_info,
        added_episodes=len(plan),
        added_frames=sum(written_rows),
        task_count=_task_count(destination),
        video_keys=len(video_keys),
    )
    _remove_stats(destination)
    return new_indices


def _require_same_dataset(destination: Path, source: Path, dest_info: dict, source_info: dict) -> None:
    if dest_info.get("codebase_version") != _V21:
        raise ValueError(
            f"{destination} codebase_version is {dest_info.get('codebase_version')!r}, expected {_V21!r}"
        )
    if float(dest_info["fps"]) != float(source_info["fps"]):
        raise ValueError(
            f"fps differs: destination {dest_info['fps']!r}, source {source_info['fps']!r}"
        )
    dest_features = _canon(dest_info.get("features"))
    source_features = _canon(source_info.get("features"))
    if dest_features != source_features:
        where = _first_diff(dest_features, source_features) or "features"
        raise ValueError(f"features differ at {where}")
    dest_modality = _read_json(destination / "meta" / "modality.json")
    source_modality = _read_json(source / "meta" / "modality.json")
    if _canon(dest_modality) != _canon(source_modality):
        where = _first_diff(_canon(dest_modality), _canon(source_modality)) or "modality.json"
        raise ValueError(f"modality.json differs at {where}")


def _task_mapping(source: Path, destination: Path) -> tuple[dict[int, int], list[dict]]:
    dest_tasks = _read_jsonl(destination / "meta" / "tasks.jsonl")
    source_tasks = _read_jsonl(source / "meta" / "tasks.jsonl")
    by_name: dict[str, int] = {}
    for record in dest_tasks:
        by_name.setdefault(str(record["task"]), _plain_int(record[_TASK_INDEX]))
    mapping: dict[int, int] = {}
    added: list[dict] = []
    next_index = max((_plain_int(record[_TASK_INDEX]) for record in dest_tasks), default=-1) + 1
    for record in source_tasks:
        name = str(record["task"])
        source_index = _plain_int(record[_TASK_INDEX])
        if name in by_name:
            mapping[source_index] = by_name[name]
            continue
        mapping[source_index] = next_index
        added.append({_TASK_INDEX: next_index, "task": name})
        by_name[name] = next_index
        next_index += 1
    return mapping, added


def _plan_episode(
    source,
    destination,
    record,
    *,
    source_info,
    dest_info,
    episode_offset,
    video_keys,
    source_chunk,
    dest_chunk,
) -> dict:
    old_index = _plain_int(record[_EPISODE_INDEX])
    new_index = old_index + episode_offset
    src_parquet = source / _render(
        source_info["data_path"],
        episode_index=old_index,
        chunks_size=source_chunk,
    )
    dst_parquet = destination / _render(
        dest_info["data_path"],
        episode_index=new_index,
        chunks_size=dest_chunk,
    )
    videos = []
    for key in video_keys:
        src_video = source / _render(
            source_info["video_path"],
            episode_index=old_index,
            chunks_size=source_chunk,
            video_key=key,
        )
        dst_video = destination / _render(
            dest_info["video_path"],
            episode_index=new_index,
            chunks_size=dest_chunk,
            video_key=key,
        )
        videos.append((src_video, dst_video))
    for path in (src_parquet, *(src for src, _dst in videos)):
        if not path.is_file():
            raise FileNotFoundError(f"missing episode file {path}")
    for path in (dst_parquet, *(dst for _src, dst in videos)):
        if path.exists():
            raise FileExistsError(f"refusing to overwrite {path}")
    return {
        "old_index": old_index,
        "new_index": new_index,
        "src_parquet": src_parquet,
        "dst_parquet": dst_parquet,
        "videos": videos,
    }


def _write_episode_files(item, *, task_map, episode_offset, frame_offset) -> int:
    table = pq.read_table(item["src_parquet"])
    table = _shift_column(table, _EPISODE_INDEX, episode_offset)
    table = _shift_column(table, _GLOBAL_INDEX, frame_offset)
    table = _remap_column(table, _TASK_INDEX, task_map)
    if _TASK_TEXT in table.column_names:
        table = _remap_column(table, _TASK_TEXT, task_map)
    item["dst_parquet"].parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, item["dst_parquet"])
    for src, dst in item["videos"]:
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
    return int(table.num_rows)


def _shift_column(table, name: str, offset: int):
    if name not in table.column_names:
        raise ValueError(f"parquet is missing the {name} column")
    values = _walk(table.column(name).to_pylist(), lambda value: _plain_int(value) + offset)
    return _replace_column(table, name, values)


def _remap_column(table, name: str, mapping: dict[int, int]):
    if name not in table.column_names:
        raise ValueError(f"parquet is missing the {name} column")

    def remap(value):
        source_index = _plain_int(value)
        if source_index not in mapping:
            raise KeyError(f"{name} value {source_index} is not in the task map")
        return mapping[source_index]

    values = _walk(table.column(name).to_pylist(), remap)
    return _replace_column(table, name, values)


def _walk(values, fn):
    def convert(value):
        if isinstance(value, list):
            return [convert(item) for item in value]
        if value is None:
            return None
        return fn(value)

    return [convert(value) for value in values]


def _replace_column(table, name: str, values):
    index = table.schema.get_field_index(name)
    field = table.schema.field(name)
    try:
        array = pa.array(values, type=field.type)
    except (pa.ArrowInvalid, pa.ArrowTypeError, TypeError):
        array = pa.array(values).cast(field.type)
    return table.set_column(index, name, array)


def _shift_episode_record(record: dict, episode_offset: int) -> dict:
    updated = dict(record)
    updated[_EPISODE_INDEX] = _shift_numbers(record[_EPISODE_INDEX], episode_offset)
    return updated


def _shift_stats(record: dict, episode_offset: int, frame_offset: int) -> dict:
    updated = json.loads(json.dumps(record))
    if _EPISODE_INDEX in updated:
        updated[_EPISODE_INDEX] = _shift_numbers(updated[_EPISODE_INDEX], episode_offset)
    stats = updated.get("stats")
    if isinstance(stats, dict):
        _shift_minmax(stats.get(_EPISODE_INDEX), episode_offset)
        _shift_minmax(stats.get(_GLOBAL_INDEX), frame_offset)
    return updated


def _shift_minmax(node, offset: int) -> None:
    if not isinstance(node, dict):
        return
    for key in ("min", "max"):
        if key in node:
            node[key] = _shift_numbers(node[key], offset)


def _shift_numbers(value, offset: int):
    if isinstance(value, list):
        return [_shift_numbers(item, offset) for item in value]
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"cannot offset {value!r}")
    shifted = value + offset
    if isinstance(value, int) and isinstance(offset, int):
        return int(shifted)
    if isinstance(shifted, float) and shifted.is_integer() and float(value).is_integer():
        return int(shifted)
    return shifted


def _shift_segment(record: dict, episode_offset: int) -> dict:
    updated = dict(record)
    updated[_EPISODE_INDEX] = _plain_int(record[_EPISODE_INDEX]) + episode_offset
    return updated


def _write_updated_info(root, info, *, added_episodes, added_frames, task_count, video_keys) -> None:
    updated = dict(info)
    total_episodes = int(info["total_episodes"]) + int(added_episodes)
    total_frames = int(info["total_frames"]) + int(added_frames)
    chunks_size = int(info.get("chunks_size") or 1000)
    updated["total_episodes"] = total_episodes
    updated["total_frames"] = total_frames
    updated["total_tasks"] = int(task_count)
    updated["total_videos"] = total_episodes * int(video_keys)
    updated["total_chunks"] = math.ceil(total_episodes / chunks_size) if total_episodes else 0
    splits = dict(updated.get("splits") or {})
    splits["train"] = f"0:{total_episodes}"
    updated["splits"] = splits
    _atomic_write(root / "meta" / "info.json", json.dumps(updated, indent=4, ensure_ascii=False) + "\n")


def _task_count(root: Path) -> int:
    return len(_read_jsonl(root / "meta" / "tasks.jsonl"))


def _video_keys(info: dict) -> list[str]:
    features = info.get("features")
    if not isinstance(features, dict):
        raise ValueError("info.json features must be an object")
    keys = [name for name, spec in features.items() if isinstance(spec, dict) and spec.get("dtype") == "video"]
    if keys and not info.get("video_path"):
        raise ValueError("info.json has video features but no video_path")
    if "data_path" not in info:
        raise ValueError("info.json is missing data_path")
    return keys


def _render(template: str, *, episode_index: int, chunks_size: int, video_key: str | None = None) -> str:
    episode_chunk = int(episode_index) // int(chunks_size)
    try:
        return template.format(
            episode_index=episode_index,
            episode_chunk=episode_chunk,
            chunk_index=episode_chunk,
            file_index=0,
            video_key=video_key or "",
        )
    except (IndexError, KeyError, ValueError) as exc:
        raise ValueError(f"cannot format dataset path {template!r}") from exc


def _episode_indices(root: Path) -> list[int]:
    path = root / "meta" / "episodes.jsonl"
    if path.is_file():
        return [_plain_int(record[_EPISODE_INDEX]) for record in _read_jsonl(path)]
    info = _read_info(root)
    return list(range(int(info.get("total_episodes") or 0)))


def _remove_stats(root: Path) -> None:
    meta = root / "meta"
    for name in _STATS_NAMES:
        path = meta / name
        if path.is_file() or path.is_symlink():
            path.unlink()


def _append_jsonl(path: Path, records: list[dict]) -> None:
    if not records:
        return
    existing = path.read_text(encoding="utf-8") if path.is_file() else ""
    if existing and not existing.endswith("\n"):
        existing += "\n"
    body = "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records)
    _atomic_write(path, existing + body)


def _atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def _read_info(root: Path) -> dict:
    return _read_json(root / "meta" / "info.json")


def _read_json(path: Path):
    if not path.is_file():
        raise FileNotFoundError(f"missing {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return payload


def _read_jsonl(path: Path) -> list[dict]:
    if not path.is_file():
        raise FileNotFoundError(f"missing {path}")
    records = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        record = json.loads(line)
        if not isinstance(record, dict):
            raise ValueError(f"{path}:{line_number} must be a JSON object")
        records.append(record)
    return records


def _plain_int(value) -> int:
    if isinstance(value, bool):
        raise TypeError(f"expected an int, got {value!r}")
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    if isinstance(value, list) and len(value) == 1:
        return _plain_int(value[0])
    raise TypeError(f"expected an int, got {value!r}")


def _canon(value):
    if isinstance(value, dict):
        return {str(key): _canon(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_canon(item) for item in value]
    if isinstance(value, float) and value.is_integer():
        return int(value)
    return value


def _first_diff(left, right, path: str = "") -> str | None:
    if left == right:
        return None
    if isinstance(left, dict) and isinstance(right, dict):
        for key in list(left) + [key for key in right if key not in left]:
            child = f"{path}.{key}" if path else str(key)
            if key not in left or key not in right:
                return child
            found = _first_diff(left[key], right[key], child)
            if found:
                return found
        return None
    if isinstance(left, list) and isinstance(right, list) and len(left) == len(right):
        for index, (item, other) in enumerate(zip(left, right)):
            found = _first_diff(item, other, f"{path}[{index}]")
            if found:
                return found
        return None
    return path or "<root>"
