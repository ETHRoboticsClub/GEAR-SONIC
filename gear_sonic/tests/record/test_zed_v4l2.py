"""Left-eye YUYV conversion and pure V4L2 checks. No device is opened.

``iter_frames`` is live-only. These tests call ``left_eye_rgb``,
``check_format``, ``classify_buffer``, and ``frame_stamp_ns``.
"""

from __future__ import annotations

import inspect
import sys

import cv2
import numpy as np
import pytest

import gear_sonic.data.record.zed_v4l2 as zed_v4l2
from gear_sonic.data.record.zed_v4l2 import (
    BUF_FLAG_ERROR,
    BUFFER_COUNT,
    FPS,
    FRAME_BYTES,
    HEIGHT,
    PIX_FMT_YUYV,
    TIMESTAMP_MASK,
    TIMESTAMP_MONOTONIC,
    WIDTH,
    Frame,
    check_format,
    classify_buffer,
    frame_stamp_ns,
    left_eye_rgb,
)

LEFT_RGB = (220, 40, 30)
RIGHT_RGB = (30, 40, 220)
_QUEUE_FLAGS = TIMESTAMP_MONOTONIC | 0x1 | 0x4 | 0x00010000


def _solid_yuyv(rgb: tuple[int, int, int], width: int, height: int) -> np.ndarray:
    image = np.empty((height, width, 3), dtype=np.uint8)
    image[:, :] = rgb
    packed = cv2.cvtColor(image, cv2.COLOR_RGB2YUV_YUYV)
    assert packed.shape == (height, width, 2)
    assert packed.dtype == np.uint8
    return packed


def _stereo_yuyv(
    left_rgb: tuple[int, int, int],
    right_rgb: tuple[int, int, int],
    width: int = WIDTH,
    height: int = HEIGHT,
) -> bytes:
    """Pack side-by-side eyes: each row is left bytes, then right bytes."""
    eye_w = width // 2
    left = _solid_yuyv(left_rgb, eye_w, height).reshape(height, eye_w * 2)
    right = _solid_yuyv(right_rgb, eye_w, height).reshape(height, eye_w * 2)
    frame = np.concatenate([left, right], axis=1)
    assert frame.shape == (height, width * 2)
    return np.ascontiguousarray(frame).tobytes()


def _accepted_format(**overrides: int) -> dict[str, int]:
    values = {
        "width": WIDTH,
        "height": HEIGHT,
        "pixelformat": PIX_FMT_YUYV,
        "bytesperline": WIDTH * 2,
        "sizeimage": FRAME_BYTES,
        "interval_numerator": 1,
        "interval_denominator": FPS,
        "buffer_count": BUFFER_COUNT,
    }
    values.update(overrides)
    return values


def test_import_does_not_open_a_device_or_import_camera_drivers():
    assert not any(name == "ethrc_cameras" or name.startswith("ethrc_cameras.") for name in sys.modules)
    source = inspect.getsource(zed_v4l2)
    assert "ethrc_cameras" not in source
    assert "iter_frames" in source
    assert PIX_FMT_YUYV == (ord("Y") | (ord("U") << 8) | (ord("Y") << 16) | (ord("V") << 24))
    assert BUF_FLAG_ERROR == 0x40
    assert TIMESTAMP_MONOTONIC == 0x2000
    assert TIMESTAMP_MASK == 0xE000
    assert FRAME_BYTES == 2560 * 720 * 2


def test_left_eye_rgb_shape_and_color():
    raw = _stereo_yuyv(LEFT_RGB, RIGHT_RGB)
    assert len(raw) == FRAME_BYTES
    # The right eye starts at byte 2560 of row 0, not at byte 2560 of the frame.
    assert raw[2560:2564] != raw[0:4]

    rgb = left_eye_rgb(raw)
    from_view = left_eye_rgb(memoryview(raw))
    assert rgb.shape == (720, 1280, 3)
    assert rgb.dtype == np.uint8
    assert from_view.shape == rgb.shape
    np.testing.assert_array_equal(from_view, rgb)

    delta = np.abs(rgb.astype(np.int16) - np.array(LEFT_RGB, dtype=np.int16))
    assert int(delta.max()) <= 2
    # Row 1 is still the left eye. A contiguous prefix of the frame would
    # place the right eye on that row.
    right = np.array(RIGHT_RGB, dtype=np.int16)
    for y, x in ((0, 0), (1, 0), (360, 640), (719, 1279)):
        pixel = rgb[y, x].astype(np.int16)
        assert int(np.abs(pixel - right).max()) > 2
        assert not np.array_equal(rgb[y, x], np.array(RIGHT_RGB, dtype=np.uint8))


def test_left_eye_rgb_crops_each_row_on_a_small_frame():
    raw = _stereo_yuyv(LEFT_RGB, RIGHT_RGB, width=8, height=2)
    rgb = left_eye_rgb(raw, width=8, height=2)
    assert rgb.shape == (2, 4, 3)
    delta = np.abs(rgb.astype(np.int16) - np.array(LEFT_RGB, dtype=np.int16))
    assert int(delta.max()) <= 2
    assert int(np.abs(rgb[1, 0].astype(np.int16) - np.array(RIGHT_RGB, dtype=np.int16)).max()) > 2


def test_left_eye_rgb_short_buffer_raises():
    full = FRAME_BYTES
    with pytest.raises(ValueError):
        left_eye_rgb(b"")
    with pytest.raises(ValueError):
        left_eye_rgb(b"\x00" * (full - 1))
    with pytest.raises(ValueError):
        left_eye_rgb(b"\x00" * (full + 1))
    # Left-eye byte count alone is not a full side-by-side frame.
    with pytest.raises(ValueError):
        left_eye_rgb(b"\x00" * (1280 * 720 * 2))
    with pytest.raises(ValueError):
        left_eye_rgb(memoryview(b"\x00" * 100))


def test_check_format_accepts_contract_and_rejects_mismatch():
    assert check_format(**_accepted_format()) is None
    assert check_format(**_accepted_format(interval_numerator=2, interval_denominator=60)) is None
    rejected = (
        {"width": 1280},
        {"height": 360},
        {"pixelformat": PIX_FMT_YUYV + 1},
        {"bytesperline": WIDTH * 2 + 32},
        {"sizeimage": FRAME_BYTES - 1},
        {"interval_numerator": 1, "interval_denominator": 15},
        {"interval_numerator": 1001, "interval_denominator": 30000},
        {"interval_numerator": 0, "interval_denominator": 30},
        {"buffer_count": 3},
        {"buffer_count": 0},
        {"buffer_count": 8},
    )
    for overrides in rejected:
        with pytest.raises(ValueError):
            check_format(**_accepted_format(**overrides))


def test_classify_buffer_marks_short_and_error_flag():
    flags = _QUEUE_FLAGS
    assert classify_buffer(FRAME_BYTES, flags, FRAME_BYTES) is False
    assert classify_buffer(FRAME_BYTES + 1, flags, FRAME_BYTES) is False
    assert classify_buffer(FRAME_BYTES - 1, flags, FRAME_BYTES) is True
    assert classify_buffer(0, flags, FRAME_BYTES) is True
    assert classify_buffer(FRAME_BYTES, flags | BUF_FLAG_ERROR, FRAME_BYTES) is True
    assert classify_buffer(FRAME_BYTES - 1, flags | BUF_FLAG_ERROR, FRAME_BYTES) is True
    assert classify_buffer(0, BUF_FLAG_ERROR, FRAME_BYTES) is True


def test_frame_stamp_requires_monotonic_and_converts_timeval():
    assert frame_stamp_ns(_QUEUE_FLAGS, 3, 250) == 3_000_000_000 + 250_000
    # Error flag does not change the clock type.
    assert frame_stamp_ns(_QUEUE_FLAGS | BUF_FLAG_ERROR, 1, 0) == 1_000_000_000
    with pytest.raises(RuntimeError):
        frame_stamp_ns(0, 3, 250)
    with pytest.raises(RuntimeError):
        frame_stamp_ns(0x00004000, 3, 250)  # TIMESTAMP_COPY
    # A non-zero monotonic bit is not enough when the type field is mixed.
    with pytest.raises(RuntimeError):
        frame_stamp_ns(TIMESTAMP_MONOTONIC | 0x00004000, 1, 0)


def test_frame_fields():
    frame = Frame(stamp_ns=5, sequence=7, yuyv=b"yuyv", error=False)
    assert frame.stamp_ns == 5
    assert frame.sequence == 7
    assert frame.yuyv == b"yuyv"
    assert frame.error is False
