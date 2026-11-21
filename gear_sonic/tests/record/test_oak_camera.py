"""The OAK-D W recorder path: capture settings, stamps, and logging.

No device is opened: ``oak_mjpeg._sdk`` is replaced by a small fake of the
DepthAI calls the module makes. The real camera was checked on the G1 by hand.
"""

from __future__ import annotations

import datetime as dt
import threading
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from gear_sonic.data.record import oak_mjpeg
from gear_sonic.data.record.session_log import SessionReader, SessionWriter
from gear_sonic.scripts.run_pico_session_recorder import (
    _append_encoded,
    capture_streams,
    open_camera,
)


def _jpeg(width: int, height: int, seed: int = 0) -> bytes:
    rng = np.random.default_rng(seed)
    gradient = np.linspace(0, 255, width, dtype=np.float32)
    image = np.stack(
        [
            np.tile(gradient, (height, 1)),
            np.tile(gradient[::-1], (height, 1)),
            rng.uniform(0, 255, size=(height, width)),
        ],
        axis=-1,
    ).astype(np.uint8)
    ok, encoded = cv2.imencode(".jpg", image)
    assert ok
    return encoded.tobytes()


# --- pure helpers -------------------------------------------------------------


@pytest.mark.parametrize("value", ["oakd", "OAKD", "oak", "oakd:19443010313B604800"])
def test_oak_names(value):
    assert oak_mjpeg.is_oak(value)


@pytest.mark.parametrize("value", ["", "/dev/video0", "zed", "oak_mono"])
def test_other_cameras_are_not_oak(value):
    assert not oak_mjpeg.is_oak(value)


def test_parse_camera_device_id():
    assert oak_mjpeg.parse_camera("oakd") is None
    assert oak_mjpeg.parse_camera("oakd:ABC") == "ABC"
    with pytest.raises(ValueError):
        oak_mjpeg.parse_camera("/dev/video0")


def test_sizes():
    assert oak_mjpeg.parse_size("640x360") == (640, 360)
    assert oak_mjpeg.parse_size("1280X720") == (1280, 720)
    for bad in ("640x400", "1920x1080", "640", "axb"):
        with pytest.raises(ValueError):
            oak_mjpeg.parse_size(bad)


def test_timedelta_ns_is_exact():
    # About 11.6 days of uptime, where a float of seconds loses microseconds.
    value = dt.timedelta(days=11, seconds=51_234, microseconds=987_654)
    assert oak_mjpeg.timedelta_ns(value) == ((11 * 86_400 + 51_234) * 1_000_000 + 987_654) * 1_000
    with pytest.raises(TypeError):
        oak_mjpeg.timedelta_ns(1.5)


# --- recorder -----------------------------------------------------------------


class _Writer:
    def __init__(self):
        self.rows = []

    def append_jpeg(self, stamp_ns, sequence, blob, error):
        self.rows.append((stamp_ns, sequence, blob, error))


def test_device_jpeg_is_logged_unchanged():
    blob = _jpeg(640, 360)
    writer = _Writer()
    _append_encoded(writer, oak_mjpeg.Frame(stamp_ns=5, sequence=9, jpeg=blob, error=False))
    _append_encoded(writer, oak_mjpeg.Frame(stamp_ns=6, sequence=10, jpeg=b"", error=True))
    assert writer.rows == [(5, 9, blob, False), (6, 10, b"", True)]


def test_capture_streams_logs_oak_frames(tmp_path):
    blobs = [_jpeg(640, 360, seed) for seed in range(3)]
    frames = [
        oak_mjpeg.Frame(stamp_ns=100 + i, sequence=i, jpeg=blob, error=False)
        for i, blob in enumerate(blobs)
    ]
    root = tmp_path / "oak"
    with SessionWriter(root, queue_size=8) as writer:
        capture_streams(writer, iter([]), iter([]), iter(frames), threading.Event())
    logged = list(SessionReader(root).iter_jpeg())
    assert logged == [(100 + i, i, blob, False) for i, blob in enumerate(blobs)]


def test_open_camera_without_device():
    camera = open_camera("")
    assert camera.frames is None and camera.info == {"kind": "none"}
    # A V4L2 path stays lazy: nothing is opened until the camera thread reads.
    zed = open_camera("/dev/video-does-not-exist")
    assert zed.info["kind"] == "zed_v4l2"
    zed.frames.close()


# --- OakCapture against a fake SDK -------------------------------------------------


class _Node:
    def __init__(self, calls, kind):
        self.calls, self.kind = calls, kind
        self.video = SimpleNamespace(link=lambda target: calls.append((kind, "link")))
        self.bitstream = self.video
        self.input = object()

    def __getattr__(self, name):
        def record(*args):
            self.calls.append((self.kind, name, args))

        return record


class _Queue:
    def __init__(self, messages, device):
        self.messages, self.device = list(messages), device

    def get(self):
        if self.messages:
            return self.messages.pop(0)
        # Like DepthAI: block until the device closes, then raise.
        self.device.closed.wait(5)
        raise RuntimeError("Communication exception - device closed")


class _Message:
    def __init__(self, stamp, sequence, data):
        self.stamp, self.sequence, self.data = stamp, sequence, data

    def getTimestamp(self, offset):
        assert offset == "MIDDLE"
        return self.stamp

    def getSequenceNum(self):
        return self.sequence

    def getData(self):
        return np.frombuffer(self.data, dtype=np.uint8)


def _fake_sdk(messages, devices=("MXID-1",)):
    calls: list = []

    class Pipeline:
        def create(self, node):
            return _Node(calls, node)

    class Device:
        @staticmethod
        def getAllAvailableDevices():
            return [SimpleNamespace(getMxId=lambda m=m: m, name=f"usb-{m}") for m in devices]

        def __init__(self, pipeline, info):
            calls.append(("device", info.getMxId()))
            self.closed = threading.Event()

        def getOutputQueue(self, name, maxSize, blocking):
            calls.append(("queue", name, maxSize, blocking))
            return _Queue(messages, self)

        def getCameraSensorNames(self):
            return {"CameraBoardSocket.CAM_A": "OV9782"}

        def getProductName(self):
            return "OAK-D-W-97"

        def getUsbSpeed(self):
            return "UsbSpeed.SUPER"

        def close(self):
            calls.append(("close",))
            self.closed.set()

    sdk = SimpleNamespace(
        Pipeline=Pipeline,
        Device=Device,
        node=SimpleNamespace(ColorCamera="color", VideoEncoder="encoder", XLinkOut="xlink"),
        CameraBoardSocket=SimpleNamespace(CAM_A="CAM_A"),
        ColorCameraProperties=SimpleNamespace(SensorResolution=SimpleNamespace(THE_720_P="720P")),
        VideoEncoderProperties=SimpleNamespace(Profile=SimpleNamespace(MJPEG="MJPEG")),
        CameraExposureOffset=SimpleNamespace(MIDDLE="MIDDLE"),
    )
    return sdk, calls


def test_capture_settings_and_frames(monkeypatch):
    blob = _jpeg(640, 360)
    stamp = dt.timedelta(seconds=1234, microseconds=567)
    sdk, calls = _fake_sdk([_Message(stamp, 41, blob), _Message(stamp, 42, b"")])
    monkeypatch.setattr(oak_mjpeg, "_sdk", lambda: sdk)

    capture = oak_mjpeg.OakCapture()
    assert ("color", "setResolution", ("720P",)) in calls
    assert ("color", "setIspScale", (1, 2)) in calls
    assert ("color", "setVideoSize", (640, 360)) in calls
    assert ("color", "setFps", (30.0,)) in calls
    assert ("encoder", "setQuality", (90,)) in calls
    assert ("queue", "mjpeg", 30, False) in calls
    assert capture.info["sensor"] == "OV9782" and capture.info["mxid"] == "MXID-1"

    frames = iter(capture)
    first, second = next(frames), next(frames)
    assert first == oak_mjpeg.Frame(1_234_000_567_000, 41, blob, False)
    assert second.error and second.jpeg == b""

    # A reader blocked on the next frame ends when another thread closes the device.
    done = threading.Event()

    def drain():
        for _ in frames:
            pass
        done.set()

    reader = threading.Thread(target=drain, daemon=True)
    reader.start()
    capture.close()
    capture.close()
    assert done.wait(2)
    assert calls.count(("close",)) == 1


def test_full_size_skips_the_isp_scale(monkeypatch):
    sdk, calls = _fake_sdk([])
    monkeypatch.setattr(oak_mjpeg, "_sdk", lambda: sdk)
    oak_mjpeg.OakCapture(width=1280, height=720).close()
    assert not any(call[:2] == ("color", "setIspScale") for call in calls)
    assert ("color", "setVideoSize", (1280, 720)) in calls


def test_missing_device_is_an_os_error(monkeypatch):
    sdk, _ = _fake_sdk([], devices=())
    monkeypatch.setattr(oak_mjpeg, "_sdk", lambda: sdk)
    with pytest.raises(OSError, match="no OAK camera found"):
        oak_mjpeg.OakCapture()
    sdk, _ = _fake_sdk([], devices=("A",))
    monkeypatch.setattr(oak_mjpeg, "_sdk", lambda: sdk)
    with pytest.raises(OSError, match="not found"):
        oak_mjpeg.OakCapture("B")
