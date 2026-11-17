"""GR00T ``meta/modality.json`` for the UNITREE_G1_SONIC embodiment."""

from __future__ import annotations

import json
from pathlib import Path

from gear_sonic.data.record.schema import load_layout

# Embodiment config lists right_arm before left_hand. Key order follows that
# list. start/end still address the real observation.state slices, so
# left_hand stays 22:29 and right_arm stays 29:36.
_STATE_GROUP_ORDER = (
    "left_leg",
    "right_leg",
    "waist",
    "left_arm",
    "right_arm",
    "left_hand",
    "right_hand",
)

_STATE_COLUMN = "observation.state"
_DEFAULT_ORIGINAL_KEY = {
    "state": _STATE_COLUMN,
    "action": "action",
}


def build_modality(layout: dict | None = None) -> dict:
    """Build the modality dict for ``meta/modality.json``.

    Body groups are slices of ``observation.state``. ``projected_gravity`` and
    the action groups point at their own columns. Video and language use the
    original keys GR00T's loader expects.
    """
    if layout is None:
        layout = load_layout()
    groups = layout["groups"]
    state: dict[str, dict] = {}
    for name in _STATE_GROUP_ORDER:
        group = groups[name]
        state[name] = {
            "start": int(group["start"]),
            "end": int(group["end"]),
            "original_key": _STATE_COLUMN,
        }
    state["projected_gravity"] = {
        "start": 0,
        "end": 3,
        "original_key": "observation.projected_gravity",
    }
    return {
        "state": state,
        "action": {
            "motion_token": {
                "start": 0,
                "end": 64,
                "original_key": "action.motion_token",
            },
            "left_hand_joints": {
                "start": 0,
                "end": 7,
                "original_key": "action.left_hand",
            },
            "right_hand_joints": {
                "start": 0,
                "end": 7,
                "original_key": "action.right_hand",
            },
        },
        "video": {
            "ego_view": {"original_key": "observation.images.ego_view"},
        },
        "annotation": {
            "human.task_description": {"original_key": "task_index"},
        },
    }


def write_modality(path: Path, modality: dict | None = None) -> dict:
    """Write ``modality`` as JSON. Returns the dict that was written."""
    payload = build_modality() if modality is None else modality
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload, indent=4) + "\n", encoding="utf-8")
    return payload


def validate_modality(info_features: dict, modality: dict) -> None:
    """Raise ``ValueError`` if a modality slice does not fit its column.

    Entries without ``start`` and ``end`` (video, annotation) are not sliced.
    A missing ``original_key`` uses ``observation.state`` for state and
    ``action`` for action. ``end`` is exclusive and may equal the column width.
    """
    if not isinstance(info_features, dict):
        raise TypeError(f"info_features must be a dict, got {type(info_features).__name__}")
    if not isinstance(modality, dict):
        raise TypeError(f"modality must be a dict, got {type(modality).__name__}")

    for section, groups in modality.items():
        if not isinstance(groups, dict):
            continue
        for name, spec in groups.items():
            if not isinstance(spec, dict) or "start" not in spec or "end" not in spec:
                continue
            start = spec["start"]
            end = spec["end"]
            label = f"{section}.{name}"
            if isinstance(start, bool) or isinstance(end, bool):
                raise ValueError(f"{label} start/end must be ints, got {start!r}:{end!r}")
            if not isinstance(start, int) or not isinstance(end, int):
                raise ValueError(f"{label} start/end must be ints, got {start!r}:{end!r}")
            original = spec.get("original_key", _DEFAULT_ORIGINAL_KEY.get(section))
            if not original:
                raise ValueError(f"{label} slice has no original_key")
            if original not in info_features:
                raise ValueError(f"{label} original_key {original!r} is not an info.json feature")
            shape = info_features[original].get("shape")
            if not shape:
                raise ValueError(f"{label} column {original!r} has no shape")
            width = int(shape[0])
            if start < 0 or start >= end or end > width:
                raise ValueError(
                    f"{label} slice [{start}:{end}) exceeds {original} width {width}"
                )


def write_segments_jsonl(path: Path, segments: list[dict]) -> None:
    """Write ``episode_index -> source_episode, [start, end)`` records."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    lines = [json.dumps(record, separators=(",", ":")) for record in segments]
    text = "\n".join(lines)
    if text:
        text += "\n"
    destination.write_text(text, encoding="utf-8")


def add_task_description_column(root: Path) -> None:
    """Add ``annotation.human.task_description`` equal to ``task_index`` if absent."""
    import pyarrow.parquet as pq

    data_root = Path(root) / "data"
    if not data_root.is_dir():
        raise FileNotFoundError(f"no data directory under {root}")
    paths = sorted(data_root.rglob("*.parquet"))
    if not paths:
        raise FileNotFoundError(f"no parquet files under {data_root}")
    column = "annotation.human.task_description"
    for path in paths:
        table = pq.read_table(path)
        if column in table.column_names:
            continue
        if "task_index" not in table.column_names:
            raise ValueError(f"{path} has no task_index column")
        table = table.append_column(column, table.column("task_index"))
        pq.write_table(table, path)


def delete_stats_files(root: Path) -> None:
    """Remove dataset-level stats GR00T recomputes for this embodiment."""
    meta = Path(root) / "meta"
    for name in ("stats.json", "relative_stats.json"):
        path = meta / name
        if path.is_file():
            path.unlink()
