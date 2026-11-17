"""Record ZMQ state, ZMQ tracking, and an optional ZED camera into a session log.

Usage::

    python gear_sonic/scripts/run_pico_session_recorder.py --task "pick up the cup"

Defaults: state port 5557, tracking port 5558, no camera, and a new directory
under ``/home/unitree/datasets/raw/<UTC timestamp>``. Enter toggles
``episode_start`` / ``episode_stop``. After a stop, ``y`` or ``n`` is stored
on that stop event as ``success``.

``capture_streams`` is the non-interactive core. Tests call it with finite
iterators. This command passes the live subscribers, and ``iter_frames`` only
when ``--camera`` is set.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import queue
import select
import subprocess
import sys
import threading
import time

import cv2

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from gear_sonic.data.record.session_log import SessionWriter
from gear_sonic.data.record.zmq_sources import subscribe_state, subscribe_tracking

_RAW_ROOT = Path("/home/unitree/datasets/raw")


def capture_streams(writer, state_iter, tracking_iter, frames, stop) -> None:
    """Enqueue state, tracking, and optional camera frames until they end.

    ``state_iter`` and ``tracking_iter`` yield ``(recv_ns, payload)``. The log
    stamp for both streams is that receive time. ``frames`` yields objects with
    ``stamp_ns``, ``sequence``, ``yuyv``, and ``error``, or is None when the
    camera is off. The camera thread only queues those objects. Another thread
    converts YUYV with ``left_eye_rgb`` and ``cv2.imencode``.

    Each iterator is read to completion, so a finite iterator makes this
    function return even when ``stop`` is never set. ``stop`` is checked between
    items. A subscriber blocked in ``recv`` observes it on the next yielded frame.
    """
    if stop is None:
        raise TypeError("stop is required")
    errors: list[BaseException] = []
    error_lock = threading.Lock()

    def record_error(exc: BaseException) -> None:
        with error_lock:
            errors.append(exc)
        stop.set()

    threads = [
        threading.Thread(
            target=_guard(
                lambda: _pump_pairs(
                    state_iter,
                    stop,
                    lambda stamp, payload: writer.append_state(stamp, stamp, payload),
                ),
                record_error,
            ),
            name="record-state",
            daemon=True,
        ),
        threading.Thread(
            target=_guard(
                lambda: _pump_pairs(
                    tracking_iter,
                    stop,
                    lambda stamp, payload: writer.append_tracking(stamp, stamp, payload),
                ),
                record_error,
            ),
            name="record-tracking",
            daemon=True,
        ),
    ]
    if frames is not None:
        encoded: queue.Queue = queue.Queue()
        camera_done = threading.Event()
        threads.append(
            threading.Thread(
                target=_guard(lambda: _pump_frames(frames, encoded, camera_done, stop), record_error),
                name="record-camera",
                daemon=True,
            )
        )
        threads.append(
            threading.Thread(
                target=_guard(lambda: _encode_frames(writer, encoded, camera_done, stop), record_error),
                name="record-jpeg",
                daemon=True,
            )
        )
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    if errors:
        raise errors[0]


def run_until(writer, stop: threading.Event) -> None:
    """Print ``health()`` every second and toggle episodes from Enter.

    A blank line starts an episode, or stops the open one. After a stop, ``y``
    or ``n`` sets ``success`` on that ``episode_stop`` event. EOF sets ``stop``.
    """
    episode_open = False
    next_health = 0.0
    while not stop.is_set():
        now = time.monotonic()
        if now >= next_health:
            print(writer.health(), flush=True)
            next_health = now + 1.0
        timeout = max(0.0, next_health - time.monotonic())
        readable, _, _ = select.select([sys.stdin], [], [], timeout)
        if not readable:
            continue
        line = sys.stdin.readline()
        if line == "":
            stop.set()
            return
        if line.strip() != "":
            continue
        if not episode_open:
            stamp_ns = time.monotonic_ns()
            writer.mark_event("episode_start", stamp_ns)
            episode_open = True
            print(f"episode_start {stamp_ns}", flush=True)
            continue
        stamp_ns = time.monotonic_ns()
        success = _read_success()
        writer.mark_event("episode_stop", stamp_ns, {"success": success})
        episode_open = False
        print(f"episode_stop {stamp_ns} success={success}", flush=True)


def write_session_json(root, *, task: str, source_commit: str, settings: dict) -> Path:
    """Write ``session.json`` with the monotonic clock, task, commit, and settings."""
    payload = {
        "clock": "CLOCK_MONOTONIC",
        "task": task,
        "source_commit": source_commit,
        "settings": settings,
    }
    path = Path(root) / "session.json"
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return path


def default_session_root(moment: datetime | None = None) -> Path:
    """Return ``/home/unitree/datasets/raw/<UTC timestamp>``."""
    current = moment if moment is not None else datetime.now(timezone.utc)
    if current.tzinfo is None:
        current = current.replace(tzinfo=timezone.utc)
    stamp = current.astimezone(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return _RAW_ROOT / stamp


def read_source_commit(repo: Path | None = None) -> str:
    """Return ``git -C <repo> rev-parse HEAD`` for this worktree."""
    root = _REPO_ROOT if repo is None else Path(repo)
    completed = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--task", required=True, help="Task name stored in session.json and episode metadata.")
    parser.add_argument("--host", default="127.0.0.1", help="ZMQ host to connect to (default: 127.0.0.1).")
    parser.add_argument("--state-port", type=int, default=5557, help="g1_debug SUB port (default: 5557).")
    parser.add_argument("--tracking-port", type=int, default=5558, help="tracking SUB port (default: 5558).")
    parser.add_argument(
        "--camera",
        default="",
        help="V4L2 device for the ZED. Empty skips the camera (default: empty).",
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=None,
        help="Session directory. Default: /home/unitree/datasets/raw/<UTC timestamp>.",
    )
    args = parser.parse_args(argv)
    root = Path(args.root) if args.root is not None else default_session_root()
    stop = threading.Event()
    writer = SessionWriter(root, queue_size=8)
    worker = None
    try:
        settings = {
            "host": args.host,
            "state_port": args.state_port,
            "tracking_port": args.tracking_port,
            "camera": args.camera,
            "root": str(root),
        }
        write_session_json(
            root,
            task=args.task,
            source_commit=read_source_commit(),
            settings=settings,
        )
        frames = None
        if args.camera:
            from gear_sonic.data.record.zed_v4l2 import iter_frames

            frames = iter_frames(args.camera)
        worker = threading.Thread(
            target=capture_streams,
            args=(
                writer,
                subscribe_state(args.host, args.state_port),
                subscribe_tracking(args.host, args.tracking_port),
                frames,
                stop,
            ),
            name="record-capture",
            daemon=True,
        )
        worker.start()
        try:
            run_until(writer, stop)
        except KeyboardInterrupt:
            stop.set()
    finally:
        stop.set()
        if worker is not None:
            worker.join(timeout=1.0)
        writer.close()
    return 0


def _pump_pairs(iterator, stop: threading.Event, append) -> None:
    try:
        for recv_ns, payload in iterator:
            if stop.is_set():
                return
            append(int(recv_ns), bytes(payload))
    finally:
        _close_iterator(iterator)


def _pump_frames(frames, encoded: queue.Queue, camera_done: threading.Event, stop: threading.Event) -> None:
    try:
        for frame in frames:
            if stop.is_set():
                return
            encoded.put(frame)
    finally:
        camera_done.set()
        _close_iterator(frames)


def _encode_frames(writer, encoded: queue.Queue, camera_done: threading.Event, stop: threading.Event) -> None:
    while True:
        if camera_done.is_set() and encoded.empty():
            return
        try:
            frame = encoded.get(timeout=0.05)
        except queue.Empty:
            if stop.is_set() and camera_done.is_set():
                return
            continue
        _append_encoded(writer, frame)


def _append_encoded(writer, frame) -> None:
    stamp_ns = int(frame.stamp_ns)
    sequence = int(frame.sequence)
    if bool(frame.error):
        writer.append_jpeg(stamp_ns, sequence, b"", error=True)
        return
    from gear_sonic.data.record.zed_v4l2 import HEIGHT, WIDTH, left_eye_rgb

    width = int(getattr(frame, "width", WIDTH))
    height = int(getattr(frame, "height", HEIGHT))
    try:
        rgb = left_eye_rgb(frame.yuyv, width, height)
    except ValueError:
        writer.append_jpeg(stamp_ns, sequence, b"", error=True)
        return
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    ok, encoded = cv2.imencode(".jpg", bgr)
    if not ok:
        writer.append_jpeg(stamp_ns, sequence, b"", error=True)
        return
    writer.append_jpeg(stamp_ns, sequence, encoded.tobytes(), error=False)


def _read_success() -> bool:
    while True:
        print("success? [y/n]", flush=True)
        line = sys.stdin.readline()
        if line == "":
            return False
        answer = line.strip().lower()
        if answer in {"y", "yes"}:
            return True
        if answer in {"n", "no"}:
            return False


def _guard(fn, record_error):
    def run() -> None:
        try:
            fn()
        except Exception as exc:
            record_error(exc)

    return run


def _close_iterator(iterator) -> None:
    close = getattr(iterator, "close", None)
    if not callable(close):
        return
    try:
        close()
    except RuntimeError:
        # The iterator has already finished, or this thread is not the one
        # that is executing it. The consumer's own finally still runs.
        return


if __name__ == "__main__":
    raise SystemExit(main())
