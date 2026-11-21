"""Build one LeRobot episode per start/stop pair in a raw session log.

Usage::

    python gear_sonic/scripts/build_episodes.py <session_dir> <out_dir> [--fps 50] [--height 360] [--width 640]

Each episode is written to its own dataset root (``out_dir/episode_000000``,
``out_dir/episode_000001``, ...) because ``write_episode`` creates a new
dataset. ``out_dir/manifest.json`` lists those directories.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
import sys

import cv2
import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from gear_sonic.data.record.build import build_rows
from gear_sonic.data.record.schema import ImageRef, load_layout
from gear_sonic.data.record.session_log import SessionReader
from gear_sonic.data.record.writer import write_episode
from gear_sonic.data.record.zmq_sources import decode_g1_debug, decode_tracking


def episode_windows(events, *, end_ns: int | None = None) -> list[dict]:
    """Pair each ``episode_stop`` with the preceding ``episode_start``.

    ``success`` is taken from the stop event when that event carries it
    (``mark_event(..., extra={"success": bool})`` stores the field on the
    event itself). A stop with no start is ignored. A start that is still
    open at the end of the log is closed at ``end_ns`` with ``success``
    None, so an interrupted recording is still built.
    """
    windows: list[dict] = []
    pending = None
    for event in events:
        if not isinstance(event, dict):
            continue
        name = event.get("name")
        if name == "episode_start":
            pending = event
            continue
        if name != "episode_stop" or pending is None:
            continue
        start_ns = int(pending["stamp_ns"])
        stop_ns = int(event["stamp_ns"])
        success = event["success"] if "success" in event else None
        pending = None
        if stop_ns < start_ns:
            continue
        windows.append({"start_ns": start_ns, "stop_ns": stop_ns, "success": success})
    if pending is not None and end_ns is not None:
        start_ns = int(pending["stamp_ns"])
        stop_ns = int(end_ns)
        if stop_ns >= start_ns:
            windows.append({"start_ns": start_ns, "stop_ns": stop_ns, "success": None})
    return windows


def assemble_episode(session_dir, start_ns: int, stop_ns: int, *, height: int, width: int) -> dict:
    """Decode one episode window from a session directory.

    State and tracking membership uses the stamp stored in the log. ``decode_g1_debug``
    resets ``stamp_ns`` to 0, so the log stamp replaces it. Tracking keeps the
    ``sample_time_ns`` stamp from ``decode_tracking``. JPEG rows with ``error``
    set, or bytes that do not decode, are omitted. Kept frames stay in log
    order so a late arrival remains aligned with its ``ImageRef`` index;
    ``build_rows`` sorts those refs by stamp.
    """
    reader = SessionReader(Path(session_dir))
    state = []
    for stamp_ns, _recv_ns, payload in reader.iter_state():
        if not _inside(stamp_ns, start_ns, stop_ns):
            continue
        decoded = decode_g1_debug(payload)
        if decoded is None:
            continue
        state.append(replace(decoded, stamp_ns=int(stamp_ns)))

    tracking = []
    for stamp_ns, _recv_ns, payload in reader.iter_tracking():
        if not _inside(stamp_ns, start_ns, stop_ns):
            continue
        decoded = decode_tracking(payload)
        if decoded is None:
            continue
        tracking.append(decoded)

    images, frames_rgb = _decode_images(
        reader.iter_jpeg(),
        start_ns=start_ns,
        stop_ns=stop_ns,
        height=height,
        width=width,
    )
    return {
        "state": state,
        "tracking": tracking,
        "images": images,
        "frames_rgb": frames_rgb,
        "t0_ns": _origin_ns(state, images, start_ns),
    }


def _last_stamp(reader: SessionReader) -> int | None:
    """Latest stamp in the session, so an interrupted episode still has an end."""
    stamps: list[int] = []
    stamps.extend(stamp for stamp, _recv, _payload in reader.iter_state())
    stamps.extend(stamp for stamp, _recv, _payload in reader.iter_tracking())
    stamps.extend(stamp for stamp, _sequence, _blob, _error in reader.iter_jpeg())
    if not stamps:
        return None
    return max(stamps)


def build_episodes(
    session_dir,
    out_dir,
    *,
    fps: int = 50,
    height: int = 360,
    width: int = 640,
    task: str | None = None,
) -> list[dict]:
    """Write one dataset per episode and ``manifest.json`` under ``out_dir``.

    Prints one line per episode with ``rows``, ``invalid``, and ``pct_invalid``
    taken from that dataset's quality json.
    """
    if isinstance(fps, bool) or not isinstance(fps, int) or fps <= 0:
        raise ValueError(f"fps must be a positive int, got {fps!r}")
    if isinstance(height, bool) or isinstance(width, bool):
        raise ValueError(f"height and width must be positive ints, got {height!r}, {width!r}")
    if not isinstance(height, int) or not isinstance(width, int) or height <= 0 or width <= 0:
        raise ValueError(f"height and width must be positive ints, got {height!r}, {width!r}")

    session_dir = Path(session_dir)
    out_dir = Path(out_dir)
    info = _read_session(session_dir)
    chosen_task = task if task is not None else info.get("task")
    if not isinstance(chosen_task, str) or not chosen_task:
        raise ValueError(f"{session_dir / 'session.json'} is missing task")
    source_commit = info.get("source_commit", "")
    if not isinstance(source_commit, str):
        source_commit = ""

    out_dir.mkdir(parents=True, exist_ok=True)
    layout = load_layout()
    reader = SessionReader(session_dir)
    windows = episode_windows(reader.iter_events(), end_ns=_last_stamp(reader))
    written_rows: list[dict] = []
    next_index = _next_episode_index(out_dir)

    for window in windows:
        assembled = assemble_episode(
            session_dir,
            window["start_ns"],
            window["stop_ns"],
            height=height,
            width=width,
        )
        rows = build_rows(
            assembled["state"],
            assembled["tracking"],
            assembled["images"],
            fps=fps,
            t0_ns=assembled["t0_ns"],
            layout=layout,
        )
        if not rows:
            continue
        episode_dir = out_dir / f"episode_{next_index:06d}"
        next_index += 1
        metadata = {
            "source_commit": source_commit,
            "success": window["success"],
        }
        written = write_episode(
            rows,
            assembled["frames_rgb"],
            episode_dir,
            fps=fps,
            task=chosen_task,
            t0_ns=assembled["t0_ns"],
            metadata=metadata,
            image_hw=(height, width),
        )
        quality = _read_quality(written)
        print(
            f"{written.name} rows={quality['n_rows']} invalid={quality['n_invalid']} "
            f"pct_invalid={quality['pct_invalid']}",
            flush=True,
        )
        written_rows.append(
            {
                "dir": written.name,
                "path": written,
                "task": chosen_task,
                "success": window["success"],
                "t0": assembled["t0_ns"],
                "n_rows": quality["n_rows"],
                "n_invalid": quality["n_invalid"],
                "pct_invalid": quality["pct_invalid"],
                "image_sequences": [image.sequence for image in assembled["images"]],
                "image_stamps": [image.stamp_ns for image in assembled["images"]],
            }
        )

    manifest = {
        "episodes": [
            {
                "dir": row["dir"],
                "task": row["task"],
                "success": row["success"],
                "t0": row["t0"],
            }
            for row in written_rows
        ]
    }
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )
    return written_rows


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("session_dir", type=Path, help="Raw session directory to read.")
    parser.add_argument("out_dir", type=Path, help="Directory that receives episode datasets.")
    parser.add_argument("--fps", type=int, default=50, help="Grid rate passed to build_rows (default: 50).")
    parser.add_argument("--height", type=int, default=360, help="Encoded image height (default: 360).")
    parser.add_argument("--width", type=int, default=640, help="Encoded image width (default: 640).")
    args = parser.parse_args(argv)
    try:
        build_episodes(
            args.session_dir,
            args.out_dir,
            fps=args.fps,
            height=args.height,
            width=args.width,
        )
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


def _inside(stamp_ns: int, start_ns: int, stop_ns: int) -> bool:
    return start_ns <= int(stamp_ns) <= stop_ns


def _origin_ns(state, images, start_ns: int) -> int:
    """``max(first state stamp, first image stamp)``, or whichever one exists."""
    state_stamp = min((sample.stamp_ns for sample in state), default=None)
    image_stamp = min((image.stamp_ns for image in images), default=None)
    if state_stamp is None and image_stamp is None:
        return int(start_ns)
    if state_stamp is None:
        return int(image_stamp)
    if image_stamp is None:
        return int(state_stamp)
    return max(int(state_stamp), int(image_stamp))


def _decode_images(jpegs, *, start_ns: int, stop_ns: int, height: int, width: int):
    images: list[ImageRef] = []
    frames: list[np.ndarray] = []
    for stamp_ns, sequence, blob, error in jpegs:
        if not _inside(stamp_ns, start_ns, stop_ns) or error:
            continue
        frame = _jpeg_to_rgb(blob, height, width)
        if frame is None:
            continue
        images.append(ImageRef(stamp_ns=int(stamp_ns), sequence=int(sequence), valid=True))
        frames.append(frame)
    return images, frames


def _jpeg_to_rgb(blob: bytes, height: int, width: int) -> np.ndarray | None:
    if not blob:
        return None
    decoded = cv2.imdecode(np.frombuffer(blob, dtype=np.uint8).copy(), cv2.IMREAD_COLOR)
    if decoded is None or decoded.ndim != 3 or decoded.shape[2] != 3:
        return None
    if decoded.shape[0] != height or decoded.shape[1] != width:
        decoded = cv2.resize(decoded, (width, height), interpolation=cv2.INTER_AREA)
    rgb = cv2.cvtColor(decoded, cv2.COLOR_BGR2RGB)
    if rgb.dtype != np.uint8 or rgb.shape != (height, width, 3):
        return None
    return np.ascontiguousarray(rgb)


def _read_session(session_dir: Path) -> dict:
    path = session_dir / "session.json"
    if not path.is_file():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return data


def _read_quality(episode_dir: Path) -> dict:
    path = episode_dir / "meta" / "quality" / "episode_000000.json"
    quality = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(quality, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return quality


def _next_episode_index(out_dir: Path) -> int:
    highest = -1
    for path in out_dir.glob("episode_*"):
        suffix = path.name.removeprefix("episode_")
        if suffix.isdigit():
            highest = max(highest, int(suffix))
    return highest + 1


if __name__ == "__main__":
    raise SystemExit(main())
