"""Session capture through episode build, stall segmentation, and convert.

No teleop, no motors, and no V4L2 device. ZMQ publishers bind an ephemeral port.
"""

from __future__ import annotations

import hashlib
import inspect
import json
import os
from pathlib import Path
import shutil
import threading
import time

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"

import cv2
import msgpack
import numpy as np
import zmq

from gear_sonic.data.record.build import build_rows
from gear_sonic.data.record.convert.segment import convert_dataset, segment_invalid
from gear_sonic.data.record.schema import CAMERA_STALE, REPO_ID, load_layout
from gear_sonic.data.record.session_log import SessionReader, SessionWriter
from gear_sonic.scripts.build_episodes import assemble_episode, build_episodes, episode_windows
from gear_sonic.scripts.run_pico_session_recorder import capture_streams, write_session_json

_RESERVED_PORTS = (5556, 5557, 5558, 60061)
_STEADY = 36
_PERIOD_S = 0.02
_HOLE_NS = 200_000_000
_EARLY_SEQUENCE = 1
_LATE_SEQUENCE = 2
_CORRUPT_SEQUENCE = 7777
_TASK = "walk to the table"
_COMMIT = "test-commit"


def test_capture_streams_finite_iterators_finish(tmp_path):
    """Finite iterators return, and JPEG encoding stays off the capture thread."""
    root = tmp_path / "finite"
    width, height = 8, 2
    yuyv = _stereo_yuyv((10, 180, 30), (200, 20, 220), width, height)
    good = _Frame(15, 4, yuyv, error=False, width=width, height=height)
    bad = _Frame(16, 5, b"", error=True, width=width, height=height)
    stop = threading.Event()
    errors: list[BaseException] = []

    def run() -> None:
        try:
            with SessionWriter(root, queue_size=8) as writer:
                capture_streams(
                    writer,
                    iter([(10, b"state-a"), (30, b"state-b")]),
                    iter([(20, b"track-a")]),
                    iter([good, bad]),
                    stop,
                )
        except BaseException as exc:
            errors.append(exc)

    worker = threading.Thread(target=run, name="finite-capture", daemon=True)
    worker.start()
    worker.join(timeout=5)
    assert not worker.is_alive(), "capture_streams did not return for finite iterators"
    if errors:
        raise errors[0]
    assert not stop.is_set()

    reader = SessionReader(root)
    assert list(reader.iter_state()) == [(10, 10, b"state-a"), (30, 30, b"state-b")]
    assert list(reader.iter_tracking()) == [(20, 20, b"track-a")]
    jpegs = list(reader.iter_jpeg())
    assert [(stamp, sequence, error) for stamp, sequence, _blob, error in jpegs] == [
        (15, 4, False),
        (16, 5, True),
    ]
    decoded = cv2.imdecode(np.frombuffer(jpegs[0][2], dtype=np.uint8), cv2.IMREAD_COLOR)
    assert decoded is not None
    assert decoded.shape == (height, width // 2, 3)


def test_end_to_end_stall_is_segmented(tmp_path, capsys):
    session = tmp_path / "session"
    out_dir = tmp_path / "episodes"
    converted = tmp_path / "converted"
    stop = threading.Event()
    state_seen: list[tuple[int, bytes]] = []
    tracking_seen: list[tuple[int, bytes]] = []
    lock = threading.Lock()
    errors: list[BaseException] = []
    state_pub = tracking_pub = None
    state_ctx = tracking_ctx = None
    worker = None
    writer = None
    g1_on_tracking = _g1_payload(424242)
    wakeup_state = _g1_payload(999999)
    wakeup_tracking = _tracking_payload(999999, 1)

    try:
        state_ctx, state_pub, state_port = _bind_publisher()
        tracking_ctx, tracking_pub, tracking_port = _bind_publisher()
        assert state_port not in _RESERVED_PORTS
        assert tracking_port not in _RESERVED_PORTS
        assert state_port != tracking_port

        writer = SessionWriter(session, queue_size=8)
        write_session_json(
            session,
            task=_TASK,
            source_commit=_COMMIT,
            settings={
                "host": "127.0.0.1",
                "state_port": state_port,
                "tracking_port": tracking_port,
                "camera": "",
                "root": str(session),
            },
        )
        state_iter = _stopping_iter(
            _subscribe_state(state_port),
            stop,
            state_seen,
            lock,
        )
        tracking_iter = _stopping_iter(
            _subscribe_tracking(tracking_port),
            stop,
            tracking_seen,
            lock,
        )

        def run() -> None:
            try:
                capture_streams(writer, state_iter, tracking_iter, None, stop)
            except BaseException as exc:
                errors.append(exc)

        worker = threading.Thread(target=run, name="e2e-capture", daemon=True)
        worker.start()
        _publish_until_connected(
            state_pub,
            tracking_pub,
            state_seen,
            tracking_seen,
            lock,
            g1_on_tracking,
        )
        _wait_idle(writer)
        time.sleep(0.05)
        with lock:
            state_base = len(state_seen)
            tracking_base = len(tracking_seen)
        appended_base = writer.health()["appended"]

        for index in range(_STEADY):
            state_pub.send(b"g1_debug" + _g1_payload(1000 + index))
            tracking_pub.send(b"tracking" + _tracking_payload(5000 + index, time.monotonic_ns()))
            if index == _STEADY // 2:
                tracking_pub.send(b"g1_debug" + g1_on_tracking)
            time.sleep(_PERIOD_S)

        _wait_for(
            lambda: (
                len(state_seen) >= state_base + _STEADY
                and len(tracking_seen) >= tracking_base + _STEADY
                and writer.health()["appended"]["state"] >= appended_base["state"] + _STEADY
                and writer.health()["appended"]["tracking"] >= appended_base["tracking"] + _STEADY
            ),
            timeout=5,
            detail=lambda: (len(state_seen), len(tracking_seen), writer.health()),
        )
        _wait_idle(writer)
        time.sleep(0.05)
        stop.set()
        state_pub.send(b"g1_debug" + wakeup_state)
        tracking_pub.send(b"tracking" + wakeup_tracking)
        worker.join(timeout=5)
        assert not worker.is_alive(), "capture_streams did not return after subscribers ended"
        if errors:
            raise errors[0]

        steady = state_seen[state_base : state_base + _STEADY]
        stamps = [stamp for stamp, _payload in steady]
        assert [ _index_of(payload) for _stamp, payload in steady ] == list(range(1000, 1000 + _STEADY))
        assert stamps == sorted(stamps)
        span = stamps[-1] - stamps[0]
        assert span >= 400_000_000, span

        start_ns, stop_ns = stamps[0], stamps[-1]
        image_stamps = _image_stamps(stamps)
        early, late = image_stamps[0], image_stamps[-1]
        hole_left, hole_right = _largest_gap(image_stamps)
        assert hole_right - hole_left >= _HOLE_NS
        corrupt_stamp = hole_left + (hole_right - hole_left) // 2
        assert hole_left < corrupt_stamp < hole_right

        _append_jpeg(writer, late, _LATE_SEQUENCE, _jpeg(255))
        _append_jpeg(writer, early, _EARLY_SEQUENCE, _jpeg(0))
        _append_jpeg(writer, corrupt_stamp, _CORRUPT_SEQUENCE, b"\x00\x01not-a-jpeg", error=True)
        for index, stamp in enumerate(image_stamps):
            if stamp in (early, late):
                continue
            _append_jpeg(writer, stamp, 100 + index, _jpeg(128))
        writer.mark_event("episode_start", start_ns)
        writer.mark_event("episode_stop", stop_ns, {"success": True})
        dropped = writer.health()["dropped"]
        writer.close()
        assert dropped == {"state": 0, "tracking": 0, "jpeg": 0, "event": 0}

        for name in ("g1_debug.log", "tracking.log", "zed_left.mjpeg", "zed_left.idx", "events.jsonl", "session.json"):
            path = session / name
            assert path.is_file() and path.stat().st_size > 0, name

        reader = SessionReader(session)
        tracking_payloads = [payload for _stamp, _recv, payload in reader.iter_tracking()]
        assert g1_on_tracking not in tracking_payloads
        assert all(not payload.startswith(b"g1_debug") for payload in tracking_payloads)
        assert all(not payload.startswith(b"tracking") for payload in tracking_payloads)
        decoded_tracking = [payload for payload in tracking_payloads if _decodes_tracking(payload)]
        assert decoded_tracking, "subscriber output did not contain a tracking frame"
        assert all(_index_of(payload) != 424242 for _stamp, _recv, payload in reader.iter_state())
        assert all(_index_of(payload) != 999999 for _stamp, _recv, payload in reader.iter_state())

        session_before = _fingerprint(session)
        assembled = assemble_episode(session, start_ns, stop_ns, height=16, width=16)
        assert  _CORRUPT_SEQUENCE not in [image.sequence for image in assembled["images"]]
        assert [image.sequence for image in assembled["images"][:2]] == [_LATE_SEQUENCE, _EARLY_SEQUENCE]
        assert assembled["images"][0].stamp_ns > assembled["images"][1].stamp_ns
        assert len(assembled["frames_rgb"]) == len(assembled["images"])
        rows = build_rows(
            assembled["state"],
            assembled["tracking"],
            assembled["images"],
            fps=50,
            t0_ns=assembled["t0_ns"],
            layout=load_layout(),
        )
        assert rows[0].image_index == 1
        assert assembled["images"][rows[0].image_index].sequence == _EARLY_SEQUENCE
        assert float(assembled["frames_rgb"][1][:, :, 1].mean()) < 5
        assert float(assembled["frames_rgb"][0][:, :, 1].mean()) > 250
        assert any(sample.stamp_ns == start_ns for sample in assembled["state"])
        assert all(sample.stamp_ns != 0 for sample in assembled["state"])
        assert any(row.flags & CAMERA_STALE for row in rows), _flag_summary(rows)
        assert any(row.valid == 1 for row in rows), _flag_summary(rows)

        built = build_episodes(session, out_dir, fps=50, height=16, width=16)
        printed = capsys.readouterr().out
        assert len(built) == 1
        episode = out_dir / "episode_000000"
        assert built[0]["path"] == episode
        assert _CORRUPT_SEQUENCE not in built[0]["image_sequences"]
        quality = json.loads((episode / "meta" / "quality" / "episode_000000.json").read_text(encoding="utf-8"))
        assert quality["n_rows"] == built[0]["n_rows"]
        assert quality["n_invalid"] == built[0]["n_invalid"] > 0
        assert quality["pct_invalid"] == built[0]["pct_invalid"]
        assert f"rows={quality['n_rows']}" in printed
        assert f"invalid={quality['n_invalid']}" in printed
        assert f"pct_invalid={quality['pct_invalid']}" in printed
        assert any(int(gap["flags"]) & CAMERA_STALE for gap in quality["gaps"])
        assert quality["success"] is True

        manifest = json.loads((out_dir / "manifest.json").read_text(encoding="utf-8"))
        assert manifest["episodes"] == [
            {"dir": "episode_000000", "task": _TASK, "success": True, "t0": assembled["t0_ns"]}
        ]
        clock = json.loads((episode / "meta" / "grid_clock.json").read_text(encoding="utf-8"))
        assert clock["clock"] == "CLOCK_MONOTONIC"
        assert clock["grid_origin_monotonic_ns"] == assembled["t0_ns"]
        assert clock["fps"] == 50
        assert clock["source_commit"] == _COMMIT

        valid_flags, stale_at = _valid_and_stale(episode)
        assert stale_at
        assert all(valid_flags[index] == 0 for index in stale_at)
        ranges = segment_invalid(valid_flags, min_segment_frames=1)
        covered = {index for start, end in ranges for index in range(start, end)}
        assert covered.isdisjoint(stale_at)
        assert min(covered) < min(stale_at)
        assert max(covered) > max(stale_at)

        episode_before = _fingerprint(episode)
        result = convert_dataset(episode, converted, min_segment_frames=1)
        if shutil.which("ffmpeg") is None:
            assert result.blocked is True
            blocked = Path(result.out_root) / "CONVERT_BLOCKED.txt"
            assert blocked.is_file()
            assert "ffmpeg" in blocked.read_text(encoding="utf-8")
        assert (episode / "meta" / "info.json").is_file()
        assert (episode / "meta" / "quality" / "episode_000000.json").is_file()
        assert _fingerprint(session) == session_before
        assert _fingerprint(episode) == episode_before
        assert episode.exists()
    finally:
        stop.set()
        if writer is not None:
            writer.close()
        if state_pub is not None:
            try:
                state_pub.send(b"g1_debug" + wakeup_state, flags=zmq.NOBLOCK)
            except Exception:
                pass
        if tracking_pub is not None:
            try:
                tracking_pub.send(b"tracking" + wakeup_tracking, flags=zmq.NOBLOCK)
            except Exception:
                pass
        if worker is not None:
            worker.join(timeout=5)
        for pub in (state_pub, tracking_pub):
            if pub is not None:
                pub.close(linger=0)
        for ctx in (state_ctx, tracking_ctx):
            if ctx is not None:
                ctx.term()


class _Frame:
    def __init__(self, stamp_ns: int, sequence: int, yuyv: bytes, *, error: bool, width: int, height: int) -> None:
        self.stamp_ns = stamp_ns
        self.sequence = sequence
        self.yuyv = yuyv
        self.error = error
        self.width = width
        self.height = height


def _subscribe_state(port: int):
    from gear_sonic.data.record.zmq_sources import subscribe_state

    return subscribe_state("127.0.0.1", port)


def _subscribe_tracking(port: int):
    from gear_sonic.data.record.zmq_sources import subscribe_tracking

    return subscribe_tracking("127.0.0.1", port)


def _stopping_iter(inner, stop: threading.Event, seen: list, lock: threading.Lock):
    try:
        for item in inner:
            if stop.is_set():
                return
            with lock:
                seen.append(item)
            yield item
    finally:
        close = getattr(inner, "close", None)
        if callable(close):
            close()


def _bind_publisher():
    context = zmq.Context()
    publisher = None
    try:
        for _ in range(8):
            publisher = context.socket(zmq.PUB)
            publisher.setsockopt(zmq.LINGER, 0)
            publisher.setsockopt(zmq.SNDHWM, 1000)
            publisher.bind("tcp://127.0.0.1:0")
            endpoint = publisher.getsockopt(zmq.LAST_ENDPOINT)
            if isinstance(endpoint, bytes):
                endpoint = endpoint.decode()
            port = int(str(endpoint).rsplit(":", 1)[-1])
            if port not in _RESERVED_PORTS:
                return context, publisher, port
            publisher.close(linger=0)
            publisher = None
        raise AssertionError("ephemeral bind landed on a reserved recorder port")
    except Exception:
        if publisher is not None:
            publisher.close(linger=0)
        context.term()
        raise


def _publish_until_connected(state_pub, tracking_pub, state_seen, tracking_seen, lock, g1_on_tracking) -> None:
    warmup_state = _g1_payload(7)
    warmup_tracking = _tracking_payload(1, time.monotonic_ns())
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        state_pub.send(b"g1_debug" + warmup_state)
        tracking_pub.send(b"tracking" + warmup_tracking)
        tracking_pub.send(b"g1_debug" + g1_on_tracking)
        with lock:
            if state_seen and tracking_seen:
                return
        time.sleep(0.01)
    raise AssertionError("subscriber never received a frame")


def _g1_payload(index: int) -> bytes:
    data = {
        "index": index,
        "body_q": [0.0] * 29,
        "body_dq": [0.0] * 29,
        "last_action": [0.1] * 29,
        "left_hand_q": [0.0] * 7,
        "left_hand_dq": [0.0] * 7,
        "right_hand_q": [0.0] * 7,
        "right_hand_dq": [0.0] * 7,
        "last_left_hand_action": [0.0] * 7,
        "last_right_hand_action": [0.0] * 7,
        "base_quat": [1.0, 0.0, 0.0, 0.0],
        "base_ang_vel": [0.0, 0.0, 0.0],
        "token_state": [0.25] * 64,
    }
    return msgpack.packb(data, use_bin_type=True)


def _controller() -> dict:
    return {
        "grip": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
        "trigger": 0.0,
        "squeeze": 0.0,
        "stick_x": 0.0,
        "stick_y": 0.0,
        "primary": False,
        "secondary": False,
        "menu": False,
        "stick_click": False,
    }


def _tracking_payload(sequence: int, sample_time_ns: int) -> bytes:
    positions = [0.0] * 72
    rotations: list[float] = []
    for _joint in range(24):
        rotations.extend((0.0, 0.0, 0.0, 1.0))
    frame = {
        "sequence": sequence,
        "sample_time_ns": int(sample_time_ns),
        "body_update_sequence": int(sequence),
        "head": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
        "left_controller": _controller(),
        "right_controller": None,
        "body": {"positions": positions, "rotations": rotations},
    }
    return msgpack.packb(frame, use_bin_type=True)


def _index_of(payload: bytes) -> int | None:
    try:
        data = msgpack.unpackb(payload, raw=False)
    except (msgpack.UnpackException, msgpack.ExtraData, ValueError, TypeError):
        return None
    if not isinstance(data, dict):
        return None
    index = data.get("index")
    if isinstance(index, bool) or not isinstance(index, int):
        return None
    return index


def _decodes_tracking(payload: bytes) -> bool:
    from gear_sonic.data.record.zmq_sources import decode_tracking

    return decode_tracking(payload) is not None


def _image_stamps(stamps: list[int]) -> list[int]:
    start = stamps[0]
    hole_lo = start + 120_000_000
    if hole_lo + _HOLE_NS > stamps[-1] - 80_000_000:
        hole_lo = start + max(0, (stamps[-1] - start - _HOLE_NS) // 2)
    kept = [stamp for stamp in stamps if stamp <= hole_lo or stamp >= hole_lo + _HOLE_NS]
    assert kept[0] <= hole_lo
    assert kept[-1] >= hole_lo + _HOLE_NS
    return kept


def _largest_gap(stamps: list[int]) -> tuple[int, int]:
    pairs = list(zip(stamps, stamps[1:]))
    return max(pairs, key=lambda pair: pair[1] - pair[0])


def _jpeg(green: int) -> bytes:
    image = np.zeros((16, 16, 3), dtype=np.uint8)
    image[:, :, 1] = green
    ok, encoded = cv2.imencode(".jpg", image)
    assert ok
    return encoded.tobytes()


def _append_jpeg(writer, stamp_ns: int, sequence: int, blob: bytes, error: bool = False) -> None:
    writer.append_jpeg(stamp_ns, sequence, blob, error=error)
    _wait_idle(writer)


def _wait_idle(writer, timeout: float = 2.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        depth = writer.health()["queue_depth"]
        if all(value == 0 for value in depth.values()):
            return
        time.sleep(0.002)
    raise AssertionError(writer.health())


def _wait_for(predicate, timeout: float, detail) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.01)
    raise AssertionError(detail())


def _stereo_yuyv(left_rgb: tuple[int, int, int], right_rgb: tuple[int, int, int], width: int, height: int) -> bytes:
    eye_w = width // 2
    left = np.empty((height, eye_w, 3), dtype=np.uint8)
    left[:, :] = left_rgb
    right = np.empty((height, eye_w, 3), dtype=np.uint8)
    right[:, :] = right_rgb
    left_packed = cv2.cvtColor(left, cv2.COLOR_RGB2YUV_YUYV).reshape(height, eye_w * 2)
    right_packed = cv2.cvtColor(right, cv2.COLOR_RGB2YUV_YUYV).reshape(height, eye_w * 2)
    frame = np.concatenate([left_packed, right_packed], axis=1)
    assert frame.shape == (height, width * 2)
    return np.ascontiguousarray(frame).tobytes()


def _flag_summary(rows) -> dict:
    summary: dict[int, int] = {}
    for row in rows:
        summary[int(row.flags)] = summary.get(int(row.flags), 0) + 1
    return summary


def _valid_and_stale(episode: Path) -> tuple[list[int], list[int]]:
    from lerobot.datasets import LeRobotDataset

    kwargs = {"repo_id": REPO_ID, "root": episode}
    if "video_backend" in inspect.signature(LeRobotDataset.__init__).parameters:
        kwargs["video_backend"] = "pyav"
    dataset = LeRobotDataset(**kwargs)
    try:
        table = dataset.hf_dataset
        valid_flags: list[int] = []
        stale_at: list[int] = []
        for index in range(len(table)):
            row = table[index]
            valid = int(np.asarray(row["quality.valid"]).reshape(-1)[0])
            flags = int(np.asarray(row["quality.flags"]).reshape(-1)[0])
            valid_flags.append(valid)
            if flags & CAMERA_STALE:
                stale_at.append(index)
        return valid_flags, stale_at
    finally:
        del dataset


def test_an_interrupted_episode_is_closed_at_the_last_stamp():
    windows = episode_windows(
        [{"name": "episode_start", "stamp_ns": 10}],
        end_ns=40,
    )
    assert windows == [{"start_ns": 10, "stop_ns": 40, "success": None}]


def _fingerprint(root: Path) -> list[tuple[str, str]]:
    rows = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames.sort()
        for name in sorted(filenames):
            path = Path(dirpath) / name
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            rows.append((path.relative_to(root).as_posix(), digest))
    return rows
