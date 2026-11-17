"""Session log: non-blocking capture queues and crash-tolerant reads. No hardware."""

import json
import struct
import threading
import time

import pytest

from gear_sonic.data.record.schema import pack_record
from gear_sonic.data.record.session_log import SessionReader, SessionWriter

_GIB = 1024**3
_INDEX = struct.Struct("<QQQII")


def _threads_named_session_log():
    return [thread for thread in threading.enumerate() if thread.name == "session-log"]


def test_round_trip_state_tracking_jpeg_and_event(tmp_path):
    root = tmp_path / "session"
    jpeg_ok = b"\xff\xd8\xff\xd9ok"
    jpeg_bad = b"\xff\xd8bad"
    with SessionWriter(root, disk_free_bytes=lambda: 10 * _GIB) as writer:
        writer.append_state(10, 11, b"one")
        writer.append_state(12, 13, b"")
        writer.append_tracking(20, 21, b"track")
        writer.append_jpeg(30, 1, jpeg_ok, error=False)
        writer.append_jpeg(31, 2, jpeg_bad, error=True)
        writer.mark_event("episode_start", 700, {"success": True})
        writer.mark_event("episode_stop", 800, None)
        assert writer.health()["appended"] == {
            "state": 2,
            "tracking": 1,
            "jpeg": 2,
            "event": 2,
        }
        assert writer.health()["dropped"] == {
            "state": 0,
            "tracking": 0,
            "jpeg": 0,
            "event": 0,
        }
        assert writer.health()["disk_warning"] is False

    assert not _threads_named_session_log()
    assert (root / "g1_debug.log").read_bytes() == pack_record(10, 11, b"one") + pack_record(12, 13, b"")
    assert (root / "tracking.log").read_bytes() == pack_record(20, 21, b"track")
    assert (root / "zed_left.mjpeg").read_bytes() == jpeg_ok + jpeg_bad

    index = (root / "zed_left.idx").read_bytes()
    assert len(index) == _INDEX.size * 2
    assert _INDEX.unpack_from(index, 0) == (30, 1, 0, len(jpeg_ok), 0)
    assert _INDEX.unpack_from(index, _INDEX.size) == (31, 2, len(jpeg_ok), len(jpeg_bad), 1)

    events = [json.loads(line) for line in (root / "events.jsonl").read_text(encoding="utf-8").splitlines()]
    assert events == [
        {"name": "episode_start", "stamp_ns": 700, "success": True},
        {"name": "episode_stop", "stamp_ns": 800},
    ]

    reader = SessionReader(root)
    assert list(reader.iter_state()) == [(10, 11, b"one"), (12, 13, b"")]
    assert list(reader.iter_tracking()) == [(20, 21, b"track")]
    assert list(reader.iter_jpeg()) == [
        (30, 1, jpeg_ok, False),
        (31, 2, jpeg_bad, True),
    ]
    assert list(reader.iter_events()) == events


def test_truncated_tail_is_ignored(tmp_path):
    root = tmp_path / "session"
    jpeg_ok = b"\xff\xd8full"
    jpeg_cut = b"\xff\xd8torn-frame"
    with SessionWriter(root, disk_free_bytes=lambda: 10 * _GIB) as writer:
        writer.append_state(1, 2, b"ok")
        writer.append_state(3, 4, b"also")
        writer.append_tracking(5, 6, b"pico")
        writer.append_jpeg(7, 1, jpeg_ok, error=False)
        writer.append_jpeg(8, 2, jpeg_cut, error=False)
        writer.mark_event("mark", 9, {"k": 1})

    state_path = root / "g1_debug.log"
    state_path.write_bytes(state_path.read_bytes() + pack_record(9, 9, b"partial-tail")[:10])
    tracking_path = root / "tracking.log"
    tracking_path.write_bytes(tracking_path.read_bytes() + b"G1RS")
    mjpeg_path = root / "zed_left.mjpeg"
    mjpeg_path.write_bytes(jpeg_ok)
    (root / "zed_left.idx").write_bytes((root / "zed_left.idx").read_bytes() + b"\x00" * 10)
    with (root / "events.jsonl").open("a", encoding="utf-8") as handle:
        handle.write('{"name": "torn"')

    reader = SessionReader(root)
    assert list(reader.iter_state()) == [(1, 2, b"ok"), (3, 4, b"also")]
    assert list(reader.iter_tracking()) == [(5, 6, b"pico")]
    assert list(reader.iter_jpeg()) == [(7, 1, jpeg_ok, False)]
    assert list(reader.iter_events()) == [{"name": "mark", "stamp_ns": 9, "k": 1}]

    state_path.write_bytes(b"XXXX" + pack_record(1, 2, b"ok")[4:])
    with pytest.raises(ValueError):
        list(reader.iter_state())


def test_full_queue_drops_without_blocking(tmp_path, monkeypatch):
    import gear_sonic.data.record.session_log as session_log

    release = threading.Event()
    entered = threading.Event()
    original = session_log._emit

    def blocking_emit(handle, data):
        if not release.is_set():
            entered.set()
            release.wait(timeout=3)
        return original(handle, data)

    monkeypatch.setattr(session_log, "_emit", blocking_emit)
    writer = SessionWriter(tmp_path, queue_size=1, disk_free_bytes=lambda: 10 * _GIB)
    try:
        writer.append_state(1, 1, b"a")
        assert entered.wait(timeout=2)
        writer.append_state(2, 2, b"b")
        assert writer.health()["queue_depth"]["state"] == 1
        assert writer.health()["dropped"]["state"] == 0
        started = time.perf_counter()
        writer.append_state(3, 3, b"c")
        elapsed = time.perf_counter() - started
        assert elapsed < 0.5
        assert writer.health()["dropped"]["state"] == 1
        assert writer.health()["appended"]["state"] == 2
    finally:
        release.set()
        writer.close()

    assert list(SessionReader(tmp_path).iter_state()) == [(1, 1, b"a"), (2, 2, b"b")]


def test_disk_free_below_2gib_raises_before_thread(tmp_path):
    assert _threads_named_session_log() == []
    with pytest.raises(OSError, match="refusing to record"):
        SessionWriter(tmp_path, disk_free_bytes=lambda: 2 * _GIB - 1)
    assert _threads_named_session_log() == []
    assert list(tmp_path.iterdir()) == []


def test_disk_warning_between_2_and_5_gib(tmp_path):
    with SessionWriter(tmp_path / "edge", disk_free_bytes=lambda: 2 * _GIB) as writer:
        assert writer.health()["disk_warning"] is True
    with SessionWriter(tmp_path / "mid", disk_free_bytes=lambda: 5 * _GIB - 1) as writer:
        assert writer.health()["disk_warning"] is True
    with SessionWriter(tmp_path / "ok", disk_free_bytes=lambda: 5 * _GIB) as writer:
        assert writer.health()["disk_warning"] is False
