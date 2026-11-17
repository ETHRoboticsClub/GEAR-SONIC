"""Hugging Face publish helpers for the accumulated G1 datasets.

The Hub client is injected. Tests pass a fake. ``read_token`` never puts the
token text into an error or a warning.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import stat
import sys

_SOURCE_KEYS = (
    "episode_index",
    "session_id",
    "start_ns",
    "stop_ns",
    "success",
    "task",
    "source_commit",
    "rows",
    "valid_rows",
)


def push_dataset(api, folder, repo_id, *, private, tag, commit_message) -> str:
    """Create or reuse a dataset repo, upload ``folder``, and move ``tag``.

    An existing repo with the other visibility raises before any upload.
    ``tag`` is deleted when it already exists, then created on the new commit.
    Returns the dataset repo URL.
    """
    from huggingface_hub.errors import RevisionNotFoundError

    created = api.create_repo(
        repo_id,
        repo_type="dataset",
        private=private,
        exist_ok=True,
    )
    info = api.repo_info(repo_id, repo_type="dataset")
    actual = getattr(info, "private", None)
    if actual is None:
        raise RuntimeError(f"could not read whether {repo_id} is private")
    if bool(actual) != bool(private):
        current = "private" if actual else "public"
        requested = "private" if private else "public"
        raise RuntimeError(f"{repo_id} is already {current}; refusing to publish it as {requested}")
    commit = api.upload_folder(
        folder_path=folder,
        repo_id=repo_id,
        repo_type="dataset",
        commit_message=commit_message,
    )
    try:
        api.delete_tag(repo_id, tag=tag, repo_type="dataset")
    except RevisionNotFoundError:
        pass
    api.create_tag(repo_id, tag=tag, revision=getattr(commit, "oid", None), repo_type="dataset")
    url = getattr(created, "url", None)
    if url:
        return str(url)
    text = str(created)
    if text.startswith("http://") or text.startswith("https://"):
        return text
    return f"https://huggingface.co/datasets/{repo_id}"


def read_token(path) -> str:
    """Return the stripped token stored at ``path``.

    A missing or empty file raises. A mode wider than ``600`` is warned on
    stderr. The token itself is not included in either message.
    """
    file_path = Path(path).expanduser()
    if not file_path.is_file():
        raise FileNotFoundError(
            f"Hugging Face token file is missing ({file_path}). "
            "Create that file with mode 600 before publishing."
        )
    token = file_path.read_text(encoding="utf-8").strip()
    if not token:
        raise ValueError(f"Hugging Face token file is empty ({file_path})")
    mode = stat.S_IMODE(file_path.stat().st_mode)
    if mode & ~0o600:
        print(
            f"warning: {file_path} mode is {mode:04o}, wider than 600",
            file=sys.stderr,
        )
    return token


def sources_path(root) -> Path:
    return Path(root) / "meta" / "sources.jsonl"


def read_sources(root) -> list[dict]:
    """Return the appended-episode ledger, or an empty list when it is absent."""
    path = sources_path(root)
    if not path.is_file():
        return []
    records = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        record = json.loads(line)
        if not isinstance(record, dict):
            raise ValueError(f"{path}:{line_number} must be a JSON object")
        records.append(record)
    return records


def window_already_appended(root, session_id, start_ns, stop_ns) -> bool:
    """Whether this raw session window is already a line in ``sources.jsonl``."""
    for record in read_sources(root):
        if record.get("session_id") != session_id:
            continue
        if int(record["start_ns"]) == int(start_ns) and int(record["stop_ns"]) == int(stop_ns):
            return True
    return False


def append_source(root, record: dict) -> None:
    """Append one ledger line. The file is replaced atomically."""
    missing = [key for key in _SOURCE_KEYS if key not in record]
    if missing:
        raise ValueError(f"source record is missing {missing}")
    payload = {key: record[key] for key in _SOURCE_KEYS}
    path = sources_path(root)
    existing = path.read_text(encoding="utf-8") if path.is_file() else ""
    if existing and not existing.endswith("\n"):
        existing += "\n"
    line = json.dumps(payload, ensure_ascii=False) + "\n"
    _atomic_write(path, existing + line)


def unconverted_episodes(v3_root, groot_root) -> list[int]:
    """v3 episodes in ``sources.jsonl`` that no GR00T segment comes from yet."""
    converted = set()
    segments = Path(groot_root) / "meta" / "segments.jsonl"
    if segments.is_file():
        for line in segments.read_text(encoding="utf-8").splitlines():
            if line.strip():
                converted.add(int(json.loads(line)["source_episode"]))
    appended = sorted({int(record["episode_index"]) for record in read_sources(v3_root)})
    return [index for index in appended if index not in converted]


def pushed_state_path(root) -> Path:
    """Beside the dataset, so recording a push never changes what was pushed."""
    dataset = Path(root)
    return dataset.with_name(f"{dataset.name}.pushed.json")


def needs_push(root, repo_id) -> bool:
    """Whether ``root`` holds episodes its last successful push to ``repo_id`` did not."""
    info_path = Path(root) / "meta" / "info.json"
    if not info_path.is_file():
        return False
    total = json.loads(info_path.read_text(encoding="utf-8")).get("total_episodes")
    state_path = pushed_state_path(root)
    if not state_path.is_file():
        return True
    state = json.loads(state_path.read_text(encoding="utf-8"))
    return state.get("repo_id") != repo_id or state.get("total_episodes") != total


def mark_pushed(root, repo_id, url) -> None:
    info = json.loads((Path(root) / "meta" / "info.json").read_text(encoding="utf-8"))
    state = {"repo_id": repo_id, "total_episodes": info.get("total_episodes"), "url": url}
    _atomic_write(pushed_state_path(root), json.dumps(state, indent=2) + "\n")


def write_dataset_card(root, *, title: str) -> None:
    """Write a short ``README.md`` when the dataset does not already have one."""
    dataset = Path(root)
    path = dataset / "README.md"
    if path.is_file():
        return
    info = json.loads((dataset / "meta" / "info.json").read_text(encoding="utf-8"))
    features = info.get("features") if isinstance(info.get("features"), dict) else {}
    tasks = _task_names(dataset)
    task_line = ", ".join(tasks) if tasks else "(none yet)"
    text = "\n".join(
        [
            f"# {title}",
            "",
            "Robot: Unitree G1, PICO teleop, ZED left eye.",
            "",
            f"fps: {info.get('fps')}",
            f"tasks: {task_line}",
            f"features: {_feature_summary(features)}",
            "",
            "Rows with quality.valid 0 were cut in the GR00T copy.",
            "",
        ]
    )
    path.write_text(text, encoding="utf-8")


def _task_names(root: Path) -> list[str]:
    jsonl = root / "meta" / "tasks.jsonl"
    if jsonl.is_file():
        names = []
        for line in jsonl.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            record = json.loads(line)
            names.append(str(record["task"]))
        return names
    parquet = root / "meta" / "tasks.parquet"
    if not parquet.is_file():
        return []
    import pandas as pd

    frame = pd.read_parquet(parquet)
    if "task" in frame.columns:
        return [str(task) for task in frame["task"].tolist()]
    return [str(task) for task in frame.index.tolist()]


def _feature_summary(features: dict) -> str:
    parts = []
    for name, spec in features.items():
        if not isinstance(spec, dict):
            parts.append(str(name))
            continue
        dtype = spec.get("dtype", "?")
        shape = spec.get("shape")
        shape_text = "" if shape is None else " " + str(tuple(shape))
        if dtype == "video":
            parts.append(f"{name} video{shape_text}")
        else:
            parts.append(f"{name} {dtype}{shape_text}")
    return ", ".join(parts)


def _atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)
