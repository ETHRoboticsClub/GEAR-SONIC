"""The VLA camera server's frames reach run_vla_inference.py's camera client decoded."""

from __future__ import annotations

import threading
import time

import cv2
import numpy as np
import pytest
import zmq

from gear_sonic.camera.composed_camera import ComposedCameraClientSensor
from gear_sonic.scripts import run_vla_camera as V


def free_port() -> int:
    socket = zmq.Context.instance().socket(zmq.PUB)
    port = socket.bind_to_random_port("tcp://127.0.0.1")
    socket.close(linger=0)
    return port


def test_an_oak_jpeg_arrives_as_the_rgb_ego_view(monkeypatch):
    bgr = np.zeros((360, 640, 3), np.uint8)
    bgr[:, :320] = (0, 0, 255)  # left half red in BGR
    ok, jpeg = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
    assert ok
    stop = threading.Event()

    def frames(_config):
        while not stop.is_set():
            yield jpeg.tobytes()
            time.sleep(0.02)

    monkeypatch.setattr(V, "oak_frames", frames)
    port = free_port()
    server = threading.Thread(target=V.main, args=(V.CameraConfig(port=port),), daemon=True)
    server.start()
    client = ComposedCameraClientSensor(server_ip="127.0.0.1", port=port)
    try:
        deadline, message = time.monotonic() + 5.0, None
        while message is None and time.monotonic() < deadline:
            message = client.read()
            time.sleep(0.02)
    finally:
        stop.set()
        server.join(timeout=2.0)
    assert message is not None, "no frame reached the client"
    image = message["images"][V.EGO_VIEW]
    assert image.shape == (360, 640, 3) and image.dtype == np.uint8
    # RGB on the client side: red is channel 0.
    assert image[180, 100].argmax() == 0 and image[180, 500].max() < 30
    assert abs(message["timestamps"][V.EGO_VIEW] - time.time()) < 5.0


@pytest.mark.parametrize(
    ("argv", "expected"),
    [
        ([], V.CameraConfig()),
        (["--camera", "/dev/video0", "--quality", "80"], V.CameraConfig(camera="/dev/video0", quality=80)),
    ],
)
def test_the_defaults_are_the_recorders(argv, expected):
    assert V.parse(argv) == expected
    assert V.CameraConfig().size == "640x360" and V.CameraConfig().quality == 90
