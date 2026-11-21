"""ZED side-by-side YUYV capture and left-eye RGB conversion.

Schema calls:

* ``left_eye_rgb`` turns one packed 2560x720 YUYV frame into the left eye.
* ``iter_frames`` negotiates that format on a V4L2 device and yields every
  dequeued buffer. It does not drop short or flagged frames.

``check_format``, ``classify_buffer``, and ``frame_stamp_ns`` are pure.
Tests call those, not ``iter_frames``, and they never open a device.
"""

from __future__ import annotations

from collections.abc import Iterator
import ctypes
from dataclasses import dataclass
import errno
import fcntl
import mmap
import os
import select

import cv2
import numpy as np

# videodev2.h, LP64. ioctl numbers are derived from these structure sizes.
BUF_TYPE_VIDEO_CAPTURE = 1
MEMORY_MMAP = 1
FIELD_NONE = 1
TIMESTAMP_MASK = 0x0000E000
TIMESTAMP_MONOTONIC = 0x00002000
BUF_FLAG_ERROR = 0x00000040

WIDTH = 2560
HEIGHT = 720
FPS = 30
BUFFER_COUNT = 4
BYTES_PER_PIXEL = 2
FRAME_BYTES = WIDTH * HEIGHT * BYTES_PER_PIXEL

_POLL_SECONDS = 0.5


def fourcc(code: str) -> int:
    """Pack a 4-character pixel format the way ``v4l2_fourcc`` does."""
    if len(code) != 4:
        raise ValueError(f"fourcc must be 4 characters, got {code!r}")
    a, b, c, d = (ord(ch) for ch in code)
    return a | (b << 8) | (c << 16) | (d << 24)


PIX_FMT_YUYV = fourcc("YUYV")


class _PixFormat(ctypes.Structure):
    _fields_ = [
        ("width", ctypes.c_uint32),
        ("height", ctypes.c_uint32),
        ("pixelformat", ctypes.c_uint32),
        ("field", ctypes.c_uint32),
        ("bytesperline", ctypes.c_uint32),
        ("sizeimage", ctypes.c_uint32),
        ("colorspace", ctypes.c_uint32),
        ("priv", ctypes.c_uint32),
        ("flags", ctypes.c_uint32),
        ("ycbcr_enc", ctypes.c_uint32),
        ("quantization", ctypes.c_uint32),
        ("xfer_func", ctypes.c_uint32),
    ]


class _FormatUnion(ctypes.Union):
    # v4l2_window holds pointers, so the format union is 8-byte aligned.
    _fields_ = [
        ("pix", _PixFormat),
        ("raw_data", ctypes.c_uint8 * 200),
        ("_align", ctypes.c_uint64),
    ]


class _Format(ctypes.Structure):
    _fields_ = [("type", ctypes.c_uint32), ("fmt", _FormatUnion)]


class _Fract(ctypes.Structure):
    _fields_ = [("numerator", ctypes.c_uint32), ("denominator", ctypes.c_uint32)]


class _CaptureParm(ctypes.Structure):
    _fields_ = [
        ("capability", ctypes.c_uint32),
        ("capturemode", ctypes.c_uint32),
        ("timeperframe", _Fract),
        ("extendedmode", ctypes.c_uint32),
        ("readbuffers", ctypes.c_uint32),
        ("reserved", ctypes.c_uint32 * 4),
    ]


class _ParmUnion(ctypes.Union):
    _fields_ = [("capture", _CaptureParm), ("raw_data", ctypes.c_uint8 * 200)]


class _StreamParm(ctypes.Structure):
    _fields_ = [("type", ctypes.c_uint32), ("parm", _ParmUnion)]


class _RequestBuffers(ctypes.Structure):
    _fields_ = [
        ("count", ctypes.c_uint32),
        ("type", ctypes.c_uint32),
        ("memory", ctypes.c_uint32),
        ("capabilities", ctypes.c_uint32),
        ("reserved", ctypes.c_uint32),
    ]


class _Timeval(ctypes.Structure):
    _fields_ = [("tv_sec", ctypes.c_long), ("tv_usec", ctypes.c_long)]


class _Timecode(ctypes.Structure):
    _fields_ = [
        ("type", ctypes.c_uint32),
        ("flags", ctypes.c_uint32),
        ("frames", ctypes.c_uint8),
        ("seconds", ctypes.c_uint8),
        ("minutes", ctypes.c_uint8),
        ("hours", ctypes.c_uint8),
        ("userbits", ctypes.c_uint8 * 4),
    ]


class _BufferMemory(ctypes.Union):
    _fields_ = [
        ("offset", ctypes.c_uint32),
        ("userptr", ctypes.c_ulong),
        ("fd", ctypes.c_int32),
    ]


class _Buffer(ctypes.Structure):
    _fields_ = [
        ("index", ctypes.c_uint32),
        ("type", ctypes.c_uint32),
        ("bytesused", ctypes.c_uint32),
        ("flags", ctypes.c_uint32),
        ("field", ctypes.c_uint32),
        ("timestamp", _Timeval),
        ("timecode", _Timecode),
        ("sequence", ctypes.c_uint32),
        ("memory", ctypes.c_uint32),
        ("m", _BufferMemory),
        ("length", ctypes.c_uint32),
        ("reserved2", ctypes.c_uint32),
        ("request_fd", ctypes.c_int32),
    ]


def _ioc(direction: int, number: int, size: int) -> int:
    return (direction << 30) | (size << 16) | (ord("V") << 8) | number


_READ = 2
_WRITE = 1
S_FMT = _ioc(_READ | _WRITE, 5, ctypes.sizeof(_Format))
REQBUFS = _ioc(_READ | _WRITE, 8, ctypes.sizeof(_RequestBuffers))
QUERYBUF = _ioc(_READ | _WRITE, 9, ctypes.sizeof(_Buffer))
QBUF = _ioc(_READ | _WRITE, 15, ctypes.sizeof(_Buffer))
DQBUF = _ioc(_READ | _WRITE, 17, ctypes.sizeof(_Buffer))
STREAMON = _ioc(_WRITE, 18, ctypes.sizeof(ctypes.c_int))
STREAMOFF = _ioc(_WRITE, 19, ctypes.sizeof(ctypes.c_int))
S_PARM = _ioc(_READ | _WRITE, 22, ctypes.sizeof(_StreamParm))


def _check_abi() -> None:
    """Refuse to run if the ctypes layout drifts from LP64 videodev2.h."""
    sizes = {
        "format": ctypes.sizeof(_Format),
        "parm": ctypes.sizeof(_StreamParm),
        "req": ctypes.sizeof(_RequestBuffers),
        "buffer": ctypes.sizeof(_Buffer),
    }
    if sizes != {"format": 208, "parm": 204, "req": 20, "buffer": 88}:
        raise RuntimeError(f"V4L2 structure sizes do not match videodev2.h: {sizes}")
    if _Format.fmt.offset != 8 or _StreamParm.parm.offset != 4:
        raise RuntimeError("V4L2 structure alignment does not match videodev2.h")
    if (
        _Buffer.timestamp.offset != 24
        or _Buffer.sequence.offset != 56
        or _Buffer.m.offset != 64
        or _Buffer.length.offset != 72
    ):
        raise RuntimeError("v4l2_buffer field offsets do not match videodev2.h")
    codes = {
        "S_FMT": (S_FMT, 0xC0D05605),
        "REQBUFS": (REQBUFS, 0xC0145608),
        "QUERYBUF": (QUERYBUF, 0xC0585609),
        "QBUF": (QBUF, 0xC058560F),
        "DQBUF": (DQBUF, 0xC0585611),
        "STREAMON": (STREAMON, 0x40045612),
        "STREAMOFF": (STREAMOFF, 0x40045613),
        "S_PARM": (S_PARM, 0xC0CC5616),
    }
    bad = {name: got for name, (got, want) in codes.items() if got != want}
    if bad:
        raise RuntimeError(f"V4L2 ioctl codes do not match videodev2.h: {bad}")


_check_abi()


@dataclass(frozen=True)
class Frame:
    """One dequeued V4L2 buffer. ``error`` does not remove it from the stream."""

    stamp_ns: int
    sequence: int
    yuyv: bytes
    error: bool


def left_eye_rgb(
    yuyv: bytes | memoryview,
    width: int = WIDTH,
    height: int = HEIGHT,
) -> np.ndarray:
    """Return the left eye of a side-by-side YUYV frame as uint8 RGB.

    Each row is ``width * 2`` bytes. The left eye is the first
    ``(width // 2) * 2`` bytes of every row (1280 pixels, 2560 bytes at the
    default size). That is not a prefix of the whole frame: the right eye
    occupies the rest of each row. YUYV packs ``Y0, U, Y1, V``. The crop is
    converted with BT.601 via ``cv2.COLOR_YUV2RGB_YUYV``.

    A buffer whose length is not ``width * height * 2`` raises ValueError.
    """
    if isinstance(width, bool) or isinstance(height, bool):
        raise ValueError(f"width and height must be ints, got {width!r}, {height!r}")
    if not isinstance(width, int) or not isinstance(height, int):
        raise ValueError(f"width and height must be ints, got {width!r}, {height!r}")
    if width <= 0 or height <= 0 or width % 4 != 0:
        raise ValueError(
            f"YUYV width must be a positive multiple of 4 and height positive, got {width}x{height}"
        )
    expected = width * height * BYTES_PER_PIXEL
    try:
        raw = np.frombuffer(yuyv, dtype=np.uint8)
    except (TypeError, BufferError) as exc:
        raise ValueError("yuyv must be a bytes-like YUYV buffer") from exc
    if raw.size != expected:
        raise ValueError(
            f"YUYV buffer is {raw.size} bytes, expected {expected} for {width}x{height}"
        )
    row_bytes = width * BYTES_PER_PIXEL
    left_width = width // 2
    left_bytes = left_width * BYTES_PER_PIXEL
    rows = raw.reshape(height, row_bytes)
    # Copy the left half of every row. A single prefix of the frame would
    # mix the right eye of one row into the next left-eye row.
    left = np.ascontiguousarray(rows[:, :left_bytes]).reshape(height, left_width, 2)
    rgb = cv2.cvtColor(left, cv2.COLOR_YUV2RGB_YUYV)
    if rgb.dtype != np.uint8 or rgb.shape != (height, left_width, 3):
        raise RuntimeError(
            f"RGB conversion returned {getattr(rgb, 'dtype', None)} {getattr(rgb, 'shape', None)}"
        )
    return rgb


def check_format(
    width: int,
    height: int,
    pixelformat: int,
    *,
    bytesperline: int,
    sizeimage: int,
    interval_numerator: int,
    interval_denominator: int,
    buffer_count: int,
) -> None:
    """Raise ValueError unless the driver granted the ZED capture contract.

    The contract is 2560x720 packed YUYV (5120 bytes per line, ``FRAME_BYTES``
    image size), 30 fps as a time-per-frame of ``n / (30 n)``, and 4 buffers.
    """
    problems: list[str] = []
    if (width, height) != (WIDTH, HEIGHT):
        problems.append(f"size {width}x{height}, expected {WIDTH}x{HEIGHT}")
    if pixelformat != PIX_FMT_YUYV:
        problems.append(f"pixelformat {pixelformat:#x}, expected YUYV {PIX_FMT_YUYV:#x}")
    row_bytes = WIDTH * BYTES_PER_PIXEL
    if bytesperline != row_bytes:
        problems.append(f"bytesperline {bytesperline}, expected {row_bytes}")
    if sizeimage != FRAME_BYTES:
        problems.append(f"sizeimage {sizeimage}, expected {FRAME_BYTES}")
    if interval_numerator <= 0 or interval_denominator != FPS * interval_numerator:
        problems.append(
            f"interval {interval_numerator}/{interval_denominator}, expected 1/{FPS}"
        )
    if buffer_count != BUFFER_COUNT:
        problems.append(f"buffer count {buffer_count}, expected {BUFFER_COUNT}")
    if problems:
        raise ValueError("refusing V4L2 format: " + "; ".join(problems))


def classify_buffer(bytesused: int, flags: int, expected_bytes: int) -> bool:
    """Return True when a dequeued buffer must be yielded with ``error=True``.

    Short means ``bytesused < expected_bytes``. The V4L2 error flag is
    ``V4L2_BUF_FLAG_ERROR`` (``BUF_FLAG_ERROR``). Other flag bits, including
    the monotonic timestamp bit, are not errors by themselves.
    """
    if isinstance(bytesused, bool) or isinstance(expected_bytes, bool):
        raise ValueError("byte counts must be ints")
    if not isinstance(bytesused, int) or not isinstance(expected_bytes, int):
        raise ValueError("byte counts must be ints")
    if bytesused < 0 or expected_bytes < 0:
        raise ValueError(f"byte counts must be >= 0, got {bytesused}, {expected_bytes}")
    short = bytesused < expected_bytes
    flagged = (int(flags) & BUF_FLAG_ERROR) != 0
    return bool(short or flagged)


def frame_stamp_ns(flags: int, tv_sec: int, tv_usec: int) -> int:
    """Convert a V4L2 buffer timestamp to nanoseconds.

    Raises RuntimeError unless the timestamp type field is
    ``V4L2_BUF_FLAG_TIMESTAMP_MONOTONIC``. Other flag bits may be set.
    """
    if (int(flags) & TIMESTAMP_MASK) != TIMESTAMP_MONOTONIC:
        raise RuntimeError(
            "V4L2 buffer timestamp is not V4L2_BUF_FLAG_TIMESTAMP_MONOTONIC "
            f"(flags=0x{int(flags) & 0xFFFFFFFF:x})"
        )
    return int(tv_sec) * 1_000_000_000 + int(tv_usec) * 1_000


def _ioctl(fd: int, request: int, arg: object) -> None:
    fcntl.ioctl(fd, request, arg, True)


class _Capture:
    """One mmap capture queue. ``read_frame`` yields every DQBUF, errors included."""

    def __init__(self, device: str) -> None:
        self._fd = os.open(device, os.O_RDWR | os.O_NONBLOCK | os.O_CLOEXEC)
        self._maps: list[mmap.mmap] = []
        self._streaming = False
        self._sizeimage = FRAME_BYTES
        try:
            self._negotiate()
        except BaseException:
            self.close()
            raise

    def _negotiate(self) -> None:
        fmt = _Format(type=BUF_TYPE_VIDEO_CAPTURE)
        fmt.fmt.pix.width = WIDTH
        fmt.fmt.pix.height = HEIGHT
        fmt.fmt.pix.pixelformat = PIX_FMT_YUYV
        fmt.fmt.pix.field = FIELD_NONE
        _ioctl(self._fd, S_FMT, fmt)
        pix = fmt.fmt.pix

        parm = _StreamParm(type=BUF_TYPE_VIDEO_CAPTURE)
        parm.parm.capture.timeperframe = _Fract(1, FPS)
        _ioctl(self._fd, S_PARM, parm)
        granted = parm.parm.capture.timeperframe

        request = _RequestBuffers(
            count=BUFFER_COUNT,
            type=BUF_TYPE_VIDEO_CAPTURE,
            memory=MEMORY_MMAP,
        )
        _ioctl(self._fd, REQBUFS, request)
        check_format(
            int(pix.width),
            int(pix.height),
            int(pix.pixelformat),
            bytesperline=int(pix.bytesperline),
            sizeimage=int(pix.sizeimage),
            interval_numerator=int(granted.numerator),
            interval_denominator=int(granted.denominator),
            buffer_count=int(request.count),
        )
        self._sizeimage = int(pix.sizeimage)
        for index in range(int(request.count)):
            buf = _Buffer(index=index, type=BUF_TYPE_VIDEO_CAPTURE, memory=MEMORY_MMAP)
            _ioctl(self._fd, QUERYBUF, buf)
            self._maps.append(
                mmap.mmap(
                    self._fd,
                    int(buf.length),
                    mmap.MAP_SHARED,
                    mmap.PROT_READ,
                    offset=int(buf.m.offset),
                )
            )
            _ioctl(self._fd, QBUF, buf)
        _ioctl(self._fd, STREAMON, ctypes.c_int(BUF_TYPE_VIDEO_CAPTURE))
        self._streaming = True

    def read_frame(self) -> Frame:
        while True:
            try:
                ready, _, _ = select.select([self._fd], [], [], _POLL_SECONDS)
            except InterruptedError:
                continue
            if not ready:
                continue
            buf = _Buffer(type=BUF_TYPE_VIDEO_CAPTURE, memory=MEMORY_MMAP)
            try:
                _ioctl(self._fd, DQBUF, buf)
            except OSError as exc:
                if exc.errno in (errno.EAGAIN, errno.EWOULDBLOCK, errno.EINTR):
                    continue
                raise
            try:
                index = int(buf.index)
                bytesused = int(buf.bytesused)
                flags = int(buf.flags)
                sequence = int(buf.sequence)
                tv_sec = int(buf.timestamp.tv_sec)
                tv_usec = int(buf.timestamp.tv_usec)
                if index < 0 or index >= len(self._maps):
                    raise RuntimeError(f"V4L2 buffer index {index} is outside the mapped queue")
                mapping = self._maps[index]
                usable = min(bytesused, len(mapping)) if bytesused > 0 else 0
                # Copy before QBUF: the driver may reuse the mapping immediately.
                payload = bytes(mapping[:usable])
            finally:
                _ioctl(self._fd, QBUF, buf)
            return Frame(
                stamp_ns=frame_stamp_ns(flags, tv_sec, tv_usec),
                sequence=sequence,
                yuyv=payload,
                error=classify_buffer(bytesused, flags, self._sizeimage),
            )

    def close(self) -> None:
        if self._fd < 0:
            return
        fd = self._fd
        self._fd = -1
        try:
            if self._streaming:
                try:
                    _ioctl(fd, STREAMOFF, ctypes.c_int(BUF_TYPE_VIDEO_CAPTURE))
                except OSError:
                    pass
        finally:
            for mapping in self._maps:
                mapping.close()
            self._maps.clear()
            os.close(fd)


def iter_frames(device: str) -> Iterator[Frame]:
    """Yield every frame from ``device`` until the caller closes this generator.

    Negotiates 2560x720 ``V4L2_PIX_FMT_YUYV`` at 30 fps with 4 mmap buffers.
    A short ``bytesused`` or ``V4L2_BUF_FLAG_ERROR`` sets ``Frame.error`` and
    the frame is still yielded. A buffer whose timestamp type is not
    ``V4L2_BUF_FLAG_TIMESTAMP_MONOTONIC`` raises RuntimeError. Closing the
    generator (or dropping it) stops the stream and closes the device.
    """
    capture = _Capture(device)
    try:
        while True:
            yield capture.read_frame()
    finally:
        capture.close()
