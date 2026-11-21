"""OAK-D W head camera: color frames JPEG-encoded on the device, for the recorder.

The G1's crown mount holds a Luxonis OAK-D W (color sensor OV9782 on ``CAM_A``,
global shutter, native 1280x720 mode). The device scales and JPEG-encodes the
color stream itself, so the host only copies bytes into the session log:

* sensor mode ``THE_720_P`` (1280x720, 16:9, the dataset's aspect ratio),
* ISP scale 1/2 to 640x360 by default, the size ``build_episodes.py`` encodes,
  so nothing the dataset keeps is thrown away and the raw log stays small
  (about 115 MB per minute at quality 90, against about 300 for the ZED),
* 30 fps, MJPEG from the device's hardware encoder.

``Frame.stamp_ns`` is the middle of the exposure on the host's steady clock.
DepthAI keeps the device clock synced to ``std::chrono::steady_clock``, which is
``CLOCK_MONOTONIC`` on Linux: the clock the recorder stamps everything else with.

``timedelta_ns``, ``parse_camera``, ``parse_size``, and ``check_size`` are pure.
Tests call those; only ``OakCapture`` opens a device.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
import datetime as _dt
import threading

# Defaults: the dataset grid's image size, the camera's frame rate, and a JPEG
# quality well above what the 1.5 Mbit/s training video keeps.
WIDTH = 640
HEIGHT = 360
FPS = 30
QUALITY = 90
SENSOR_WIDTH = 1280
SENSOR_HEIGHT = 720
# Frames the host queue holds before the oldest is dropped: one second.
QUEUE_FRAMES = FPS
STREAM_NAME = "mjpeg"
# ``--camera`` values that select this camera. ``oakd:<mxid>`` picks a device.
CAMERA_NAMES = ("oakd", "oak")


@dataclass(frozen=True)
class Frame:
    """One encoded color frame. ``error`` frames are logged with no bytes."""

    stamp_ns: int
    sequence: int
    jpeg: bytes
    error: bool
    width: int = WIDTH
    height: int = HEIGHT


def is_oak(camera: str) -> bool:
    """Return True when a ``--camera`` value names the OAK rather than a V4L2 path."""
    name = camera.split(":", 1)[0].strip().lower()
    return name in CAMERA_NAMES


def parse_camera(camera: str) -> str | None:
    """Return the device id in ``oakd:<id>``, or None for the first OAK found."""
    if not is_oak(camera):
        raise ValueError(f"{camera!r} does not name an OAK camera ({', '.join(CAMERA_NAMES)})")
    _, _, device_id = camera.partition(":")
    device_id = device_id.strip()
    return device_id or None


def parse_size(text: str) -> tuple[int, int]:
    """Parse ``WIDTHxHEIGHT`` and check it with :func:`check_size`."""
    try:
        width_text, height_text = text.lower().split("x")
        width, height = int(width_text), int(height_text)
    except ValueError as exc:
        raise ValueError(f"camera size must look like 640x360, got {text!r}") from exc
    check_size(width, height)
    return width, height


def check_size(width: int, height: int) -> None:
    """Accept the two sizes the device produces without cropping: 1280x720 and 640x360."""
    if (width, height) not in {(SENSOR_WIDTH, SENSOR_HEIGHT), (SENSOR_WIDTH // 2, SENSOR_HEIGHT // 2)}:
        raise ValueError(
            f"OAK camera size must be {SENSOR_WIDTH}x{SENSOR_HEIGHT} or "
            f"{SENSOR_WIDTH // 2}x{SENSOR_HEIGHT // 2}, got {width}x{height}"
        )


def timedelta_ns(value: _dt.timedelta) -> int:
    """Return a DepthAI timestamp (a ``timedelta`` since the steady clock's epoch) in ns.

    Integer arithmetic only: ``total_seconds()`` is a float and loses
    microseconds past about 2**53 ns.
    """
    if not isinstance(value, _dt.timedelta):
        raise TypeError(f"expected datetime.timedelta, got {type(value).__name__}")
    return (value.days * 86_400 + value.seconds) * 1_000_000_000 + value.microseconds * 1_000


def _sdk():
    import depthai  # noqa: PLC0415 - optional: only the OAK path needs it

    return depthai


def _pipeline(dai, width: int, height: int, fps: int, quality: int):
    pipeline = dai.Pipeline()
    camera = pipeline.create(dai.node.ColorCamera)
    camera.setBoardSocket(dai.CameraBoardSocket.CAM_A)
    camera.setResolution(dai.ColorCameraProperties.SensorResolution.THE_720_P)
    camera.setFps(float(fps))
    if (width, height) != (SENSOR_WIDTH, SENSOR_HEIGHT):
        camera.setIspScale(1, 2)
    camera.setVideoSize(width, height)
    encoder = pipeline.create(dai.node.VideoEncoder)
    encoder.setDefaultProfilePreset(fps, dai.VideoEncoderProperties.Profile.MJPEG)
    encoder.setQuality(quality)
    camera.video.link(encoder.input)
    link = pipeline.create(dai.node.XLinkOut)
    link.setStreamName(STREAM_NAME)
    encoder.bitstream.link(link.input)
    return pipeline


def _find_device(dai, device_id: str | None):
    devices = list(dai.Device.getAllAvailableDevices())
    if not devices:
        raise OSError(
            "no OAK camera found (lsusb should list 03e7:2485 or 03e7:f63b); "
            "check the cable, and stop any other program holding it (the ETHRC cameras container)"
        )
    if device_id is None:
        return devices[0]
    for info in devices:
        if device_id in {info.getMxId(), info.name}:
            return info
    found = ", ".join(f"{info.getMxId()} ({info.name})" for info in devices)
    raise OSError(f"OAK camera {device_id!r} not found; available: {found}")


class OakCapture:
    """An open OAK-D streaming on-device JPEG. Iterate it for :class:`Frame` objects.

    Opening boots the device (about 2 s) and raises ``OSError`` when no camera
    answers, so a recorder fails before it writes anything. ``info`` describes
    what was opened, for ``session.json``. ``close`` stops the device.
    """

    def __init__(
        self,
        device_id: str | None = None,
        *,
        width: int = WIDTH,
        height: int = HEIGHT,
        fps: int = FPS,
        quality: int = QUALITY,
    ) -> None:
        check_size(width, height)
        if not 1 <= int(quality) <= 100:
            raise ValueError(f"JPEG quality must be 1-100, got {quality}")
        if not 1 <= int(fps) <= 60:
            raise ValueError(f"fps must be 1-60, got {fps}")
        dai = _sdk()
        self._dai = dai
        self.width, self.height = int(width), int(height)
        info = _find_device(dai, device_id)
        try:
            self._device = dai.Device(_pipeline(dai, self.width, self.height, int(fps), int(quality)), info)
        except RuntimeError as exc:
            raise OSError(f"cannot open OAK camera {info.getMxId()}: {exc}") from exc
        self._queue = self._device.getOutputQueue(STREAM_NAME, maxSize=QUEUE_FRAMES, blocking=False)
        sensors = {str(k).rsplit(".", 1)[-1]: v for k, v in self._device.getCameraSensorNames().items()}
        try:
            product = self._device.getProductName()
        except RuntimeError:
            product = ""
        self.info = {
            "kind": "oakd",
            "product": product,
            "mxid": info.getMxId(),
            "usb_speed": str(self._device.getUsbSpeed()).rsplit(".", 1)[-1],
            "socket": "CAM_A",
            "sensor": sensors.get("CAM_A", ""),
            "sensor_mode": f"{SENSOR_WIDTH}x{SENSOR_HEIGHT}",
            "width": self.width,
            "height": self.height,
            "fps": int(fps),
            "jpeg_quality": int(quality),
            "stamp": "exposure middle, host CLOCK_MONOTONIC",
        }
        self._closed = False
        self._close_lock = threading.Lock()

    def __iter__(self) -> Iterator[Frame]:
        middle = self._dai.CameraExposureOffset.MIDDLE
        while not self._closed:
            try:
                message = self._queue.get()
            except RuntimeError as exc:
                if self._closed:
                    return
                raise OSError(f"OAK camera stopped answering: {exc}") from exc
            if message is None:
                continue
            stamp_ns = timedelta_ns(message.getTimestamp(middle))
            data = bytes(message.getData())
            yield Frame(
                stamp_ns=stamp_ns,
                sequence=int(message.getSequenceNum()),
                jpeg=data,
                error=not data,
                width=self.width,
                height=self.height,
            )

    def close(self) -> None:
        """Stop the device. Safe from any thread, and more than once."""
        with self._close_lock:
            if self._closed:
                return
            self._closed = True
        self._device.close()

    def __enter__(self) -> OakCapture:
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()


def iter_frames(
    device_id: str | None = None,
    *,
    width: int = WIDTH,
    height: int = HEIGHT,
    fps: int = FPS,
    quality: int = QUALITY,
) -> Iterator[Frame]:
    """Yield frames from an OAK until the caller closes this generator."""
    capture = OakCapture(device_id, width=width, height=height, fps=fps, quality=quality)
    try:
        yield from capture
    finally:
        capture.close()
