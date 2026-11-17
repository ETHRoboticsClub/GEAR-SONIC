"""Append-only raw session log for one teleop recording.

Capture threads call :meth:`SessionWriter.append_state`,
:meth:`SessionWriter.append_tracking`, :meth:`SessionWriter.append_jpeg`,
and :meth:`SessionWriter.mark_event`. Those calls only enqueue. They do not
touch the filesystem and they do not wait for a full queue: a full queue
increments that stream's drop count and returns. One background thread
appends to disk and flushes every record.

Files in ``root``:

* ``g1_debug.log``, ``tracking.log``: concatenated :func:`schema.pack_record`
  frames.
* ``zed_left.mjpeg``: raw JPEG bytes, one frame after another.
* ``zed_left.idx``: little-endian ``<QQQII>`` records
  ``(stamp_ns, sequence, file_offset, length, flags)``. Bit 0 of ``flags``
  means the frame was captured with an error.
* ``events.jsonl``: one JSON object per line, ``name``, ``stamp_ns``, then
  any extra fields.

Construction refuses to start when fewer than 2 GiB are free and reports
``health()["disk_warning"]`` when free space is below 5 GiB.
:class:`SessionReader` uses :func:`schema.iter_records`, so a truncated final
state or tracking record is ignored. JPEG index rows that start past EOF, or
whose byte length would run past EOF, are ignored too.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
import errno
import json
from pathlib import Path
import queue
import shutil
import struct
import threading

from gear_sonic.data.record.schema import iter_records, pack_record

_GIB = 1024**3
_REFUSE_BELOW = 2 * _GIB
_WARN_BELOW = 5 * _GIB

_STREAMS = ("state", "tracking", "jpeg", "event")
_INDEX = struct.Struct("<QQQII")
_JPEG_ERROR = 1

__all__ = ["SessionReader", "SessionWriter"]


def _emit(handle, data: bytes | str) -> None:
    """Write one chunk and flush it. Only the writer thread calls this."""
    handle.write(data)
    handle.flush()


class SessionWriter:
    """Bounded, non-blocking queues in front of append-only session files.

    ``disk_free_bytes`` defaults to ``shutil.disk_usage(root).free``. Tests
    pass a callable so they do not read the real disk. Each stream has its
    own queue of ``queue_size`` records so a full JPEG queue cannot drop
    state or tracking.
    """

    def __init__(
        self,
        root: Path,
        *,
        queue_size: int = 8,
        disk_free_bytes: Callable[[], int] | None = None,
    ) -> None:
        if isinstance(queue_size, bool) or not isinstance(queue_size, int) or queue_size < 1:
            raise ValueError(f"queue_size must be a positive int, got {queue_size!r}")
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        if disk_free_bytes is None:
            free = shutil.disk_usage(self.root).free
        else:
            free = disk_free_bytes()
        if free < _REFUSE_BELOW:
            raise OSError(
                errno.ENOSPC,
                f"refusing to record with {free} bytes free; need at least {_REFUSE_BELOW} bytes",
                str(self.root),
            )
        self._disk_warning = free < _WARN_BELOW
        self._lock = threading.Lock()
        self._closed = False
        self._queues = {name: queue.Queue(maxsize=queue_size) for name in _STREAMS}
        self._ready: queue.Queue[str | None] = queue.Queue()
        self._appended = {name: 0 for name in _STREAMS}
        self._dropped = {name: 0 for name in _STREAMS}
        self._state_file = None
        self._tracking_file = None
        self._mjpeg_file = None
        self._index_file = None
        self._events_file = None
        self._jpeg_offset = 0
        self._open_files()
        self._thread = threading.Thread(target=self._run, name="session-log", daemon=True)
        try:
            self._thread.start()
        except Exception:
            self._close_files()
            raise

    def append_state(self, stamp_ns: int, recv_ns: int, payload: bytes) -> None:
        """Enqueue one g1_debug frame. No disk IO."""
        self._enqueue("state", ("state", stamp_ns, recv_ns, bytes(payload)))

    def append_tracking(self, stamp_ns: int, recv_ns: int, payload: bytes) -> None:
        """Enqueue one tracking frame. No disk IO."""
        self._enqueue("tracking", ("tracking", stamp_ns, recv_ns, bytes(payload)))

    def append_jpeg(
        self,
        stamp_ns: int,
        sequence: int,
        jpeg_bytes: bytes,
        error: bool = False,
    ) -> None:
        """Enqueue one left-eye JPEG and its index row. No disk IO."""
        self._enqueue("jpeg", ("jpeg", stamp_ns, sequence, bytes(jpeg_bytes), bool(error)))

    def mark_event(self, name: str, stamp_ns: int, extra: dict | None = None) -> None:
        """Enqueue one events.jsonl object. No disk IO."""
        copied = None if extra is None else dict(extra)
        self._enqueue("event", ("event", name, stamp_ns, copied))

    def health(self) -> dict:
        """Return appended and dropped counts, queue depth, and disk_warning.

        ``appended`` and ``dropped`` are per stream (``state``, ``tracking``,
        ``jpeg``, ``event``). ``appended`` counts records accepted onto a
        queue. ``dropped`` counts records refused because that queue was full.
        ``queue_depth`` is the number of records still waiting in each queue.
        ``disk_warning`` is true when free space at construction was at least
        2 GiB and below 5 GiB.
        """
        with self._lock:
            return {
                "appended": dict(self._appended),
                "dropped": dict(self._dropped),
                "queue_depth": {name: self._queues[name].qsize() for name in _STREAMS},
                "disk_warning": self._disk_warning,
            }

    def close(self) -> None:
        """Flush queued records, join the writer thread, and close the files."""
        with self._lock:
            already = self._closed
            self._closed = True
            if not already:
                self._ready.put(None)
        self._thread.join()

    def __enter__(self) -> SessionWriter:
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def _enqueue(self, stream: str, item: tuple) -> None:
        with self._lock:
            if self._closed:
                raise RuntimeError("session log is closed")
            try:
                self._queues[stream].put_nowait(item)
            except queue.Full:
                self._dropped[stream] += 1
                return
            self._appended[stream] += 1
            self._ready.put(stream)

    def _open_files(self) -> None:
        try:
            self._state_file = (self.root / "g1_debug.log").open("ab")
            self._tracking_file = (self.root / "tracking.log").open("ab")
            self._mjpeg_file = (self.root / "zed_left.mjpeg").open("ab")
            self._index_file = (self.root / "zed_left.idx").open("ab")
            self._events_file = (self.root / "events.jsonl").open("a", encoding="utf-8", newline="\n")
            self._jpeg_offset = self._mjpeg_file.tell()
        except Exception:
            self._close_files()
            raise

    def _close_files(self) -> None:
        for handle in (
            self._state_file,
            self._tracking_file,
            self._mjpeg_file,
            self._index_file,
            self._events_file,
        ):
            if handle is None or handle.closed:
                continue
            try:
                handle.flush()
            except Exception:
                pass
            try:
                handle.close()
            except Exception:
                pass

    def _run(self) -> None:
        try:
            while True:
                token = self._ready.get()
                if token is None:
                    return
                item = self._queues[token].get_nowait()
                self._persist(item)
        finally:
            self._close_files()

    def _persist(self, item: tuple) -> None:
        kind = item[0]
        if kind == "state":
            _, stamp_ns, recv_ns, payload = item
            _emit(self._state_file, pack_record(stamp_ns, recv_ns, payload))
        elif kind == "tracking":
            _, stamp_ns, recv_ns, payload = item
            _emit(self._tracking_file, pack_record(stamp_ns, recv_ns, payload))
        elif kind == "jpeg":
            _, stamp_ns, sequence, blob, error = item
            self._write_jpeg(stamp_ns, sequence, blob, error)
        elif kind == "event":
            _, name, stamp_ns, extra = item
            self._write_event(name, stamp_ns, extra)
        else:
            raise RuntimeError(f"unknown session record {kind!r}")

    def _write_jpeg(self, stamp_ns: int, sequence: int, blob: bytes, error: bool) -> None:
        offset = self._jpeg_offset
        _emit(self._mjpeg_file, blob)
        self._jpeg_offset = offset + len(blob)
        flags = _JPEG_ERROR if error else 0
        record = _INDEX.pack(int(stamp_ns), int(sequence), offset, len(blob), flags)
        _emit(self._index_file, record)

    def _write_event(self, name: str, stamp_ns: int, extra: dict | None) -> None:
        body: dict = {"name": name, "stamp_ns": stamp_ns}
        if extra:
            for key, value in extra.items():
                if key not in body:
                    body[key] = value
        _emit(self._events_file, json.dumps(body) + "\n")


class SessionReader:
    """Read a session directory, skipping a torn tail."""

    def __init__(self, root: Path) -> None:
        self.root = Path(root)

    def iter_state(self) -> Iterator[tuple[int, int, bytes]]:
        """Yield ``(stamp_ns, recv_ns, payload)`` from ``g1_debug.log``."""
        yield from self._iter_log("g1_debug.log")

    def iter_tracking(self) -> Iterator[tuple[int, int, bytes]]:
        """Yield ``(stamp_ns, recv_ns, payload)`` from ``tracking.log``."""
        yield from self._iter_log("tracking.log")

    def iter_jpeg(self) -> Iterator[tuple[int, int, bytes, bool]]:
        """Yield ``(stamp_ns, sequence, jpeg_bytes, error)`` for in-range index rows."""
        index_path = self.root / "zed_left.idx"
        mjpeg_path = self.root / "zed_left.mjpeg"
        if not index_path.is_file():
            return
        mjpeg_size = mjpeg_path.stat().st_size if mjpeg_path.is_file() else 0
        mjpeg = mjpeg_path.open("rb") if mjpeg_path.is_file() else None
        try:
            with index_path.open("rb") as index:
                while True:
                    chunk = index.read(_INDEX.size)
                    if len(chunk) < _INDEX.size:
                        return
                    stamp_ns, sequence, file_offset, length, flags = _INDEX.unpack(chunk)
                    end = file_offset + length
                    if file_offset > mjpeg_size or end > mjpeg_size:
                        continue
                    if length == 0:
                        blob = b""
                    elif mjpeg is None:
                        continue
                    else:
                        mjpeg.seek(file_offset)
                        blob = mjpeg.read(length)
                        if len(blob) != length:
                            continue
                    yield stamp_ns, sequence, blob, bool(flags & _JPEG_ERROR)
        finally:
            if mjpeg is not None:
                mjpeg.close()

    def iter_events(self) -> Iterator[dict]:
        """Yield one dict per complete events.jsonl line. A torn last line stops."""
        path = self.root / "events.jsonl"
        if not path.is_file():
            return
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                text = line.strip()
                if not text:
                    continue
                try:
                    yield json.loads(text)
                except json.JSONDecodeError:
                    return

    def _iter_log(self, name: str) -> Iterator[tuple[int, int, bytes]]:
        path = self.root / name
        if not path.is_file():
            return
        yield from iter_records(path)
