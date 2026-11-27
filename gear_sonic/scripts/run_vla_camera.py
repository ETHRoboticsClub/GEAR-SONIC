"""Publish the G1 head camera for ``run_vla_inference.py`` as the recorder saw it.

    python gear_sonic/scripts/run_vla_camera.py --camera oakd        # or /dev/video0 (ZED)

A VLA trained on ``run_pico_session_recorder.py`` takes learned from the frames that
recorder kept: the OAK-D W's on-device JPEG at 640x360, quality 90 (``oak_mjpeg``), or
the ZED's left eye. ``composed_camera`` opens the OAK at 640x480 instead, another field of
view. This server opens the camera the recorder's way and publishes each frame, still
JPEG, as the ``ego_view`` of an ``ImageMessageSchema`` on ZMQ PUB (``--port``, 5555), which
``ComposedCameraClientSensor`` decodes. A ZED's left eye is scaled to the same size and
encoded at the same quality. The timestamp is the wall time of the frame's arrival, which
the client compares with its own clock.
"""

from __future__ import annotations

from dataclasses import dataclass
import time

import cv2
import msgpack
import zmq

from gear_sonic.data.record import oak_mjpeg

EGO_VIEW = "ego_view"
# Frames ZMQ keeps for a slow subscriber before it drops them: the newest matter.
SEND_HWM = 20


@dataclass
class CameraConfig:
    camera: str = "oakd"
    """``oakd`` (first OAK), ``oakd:<mxid>``, or a ZED's V4L2 device such as /dev/video0."""

    port: int = 5555
    size: str = f"{oak_mjpeg.WIDTH}x{oak_mjpeg.HEIGHT}"
    """Frame size, as the recorder's ``--camera-size``."""

    quality: int = oak_mjpeg.QUALITY
    """JPEG quality, as the recorder's ``--camera-quality``."""


def oak_frames(config: CameraConfig):
    width, height = oak_mjpeg.parse_size(config.size)
    capture = oak_mjpeg.OakCapture(
        oak_mjpeg.parse_camera(config.camera), width=width, height=height, quality=config.quality
    )
    print(f"camera: {capture.info}", flush=True)
    with capture:
        for frame in capture:
            if not frame.error:
                yield frame.jpeg


def zed_frames(config: CameraConfig):
    from gear_sonic.data.record import zed_v4l2

    width, height = oak_mjpeg.parse_size(config.size)
    params = [int(cv2.IMWRITE_JPEG_QUALITY), int(config.quality)]
    print(f"camera: ZED left eye from {config.camera}, sent at {width}x{height}", flush=True)
    for frame in zed_v4l2.iter_frames(config.camera):
        if frame.error:
            continue
        rgb = zed_v4l2.left_eye_rgb(frame.yuyv, zed_v4l2.WIDTH, zed_v4l2.HEIGHT)
        bgr = cv2.resize(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), (width, height), interpolation=cv2.INTER_AREA)
        ok, encoded = cv2.imencode(".jpg", bgr, params)
        if ok:
            yield encoded.tobytes()


def main(config: CameraConfig) -> None:
    frames = oak_frames(config) if oak_mjpeg.is_oak(config.camera) else zed_frames(config)
    context = zmq.Context()
    socket = context.socket(zmq.PUB)
    socket.setsockopt(zmq.SNDHWM, SEND_HWM)
    socket.setsockopt(zmq.LINGER, 0)
    socket.bind(f"tcp://*:{config.port}")
    print(f"publishing {EGO_VIEW} on tcp://*:{config.port}", flush=True)
    sent, dropped, report = 0, 0, time.monotonic()
    try:
        for jpeg in frames:
            message = {"timestamps": {EGO_VIEW: time.time()}, "images": {EGO_VIEW: jpeg}}
            try:
                socket.send(msgpack.packb(message, use_bin_type=True), flags=zmq.NOBLOCK)
                sent += 1
            except zmq.Again:
                dropped += 1
            if time.monotonic() - report >= 5.0:
                print(f"sent {sent} frames ({len(jpeg)} B last), dropped {dropped}", flush=True)
                report = time.monotonic()
    finally:
        socket.close()
        context.term()


def parse(argv: list[str] | None = None) -> CameraConfig:
    # argparse, not tyro: this runs in the recorder's venv, which has no tyro.
    import argparse

    defaults = CameraConfig()
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--camera", default=defaults.camera)
    parser.add_argument("--port", type=int, default=defaults.port)
    parser.add_argument("--size", default=defaults.size)
    parser.add_argument("--quality", type=int, default=defaults.quality)
    return CameraConfig(**vars(parser.parse_args(argv)))


if __name__ == "__main__":
    main(parse())
