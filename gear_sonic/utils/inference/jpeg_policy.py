"""JPEG camera frames between the VLA client and an Isaac-GR00T policy server.

GR00T's ``PolicyClient`` sends each observation's images as raw uint8 arrays: about 690 KB
for one 640x360 RGB frame, which a lab uplink of about 1 MB/s turns into most of a second
per inference. ``JpegPolicyClient`` speaks the same protocol (ZMQ REQ/REP, msgpack with
msgpack-numpy, ``{"endpoint", "data"}`` requests) but sends every ``video`` entry as JPEG,
about 40 KB a frame. ``JpegDecodingPolicy`` wraps the real policy on the server and turns
the JPEGs back into the arrays the policy expects, so the model sees the same layout either
way. Neither side imports gr00t here: the client runs on the robot without Isaac-GR00T,
and ``gear_sonic/scripts/serve_gr00t_jpeg.py`` serves a ``Gr00tPolicy`` through it.

A JPEG frame is ``{JPEG_KEY: [bytes, ...], "shape": [..., H, W, 3]}``: one JPEG per image in
the array's leading dimensions (batch, time), in C order.
"""

from __future__ import annotations

from typing import Any

import cv2
import msgpack
import msgpack_numpy as mnp
import numpy as np
import zmq

JPEG_KEY = "__jpeg__"
# Visually lossless for a 640x360 head camera; the frame is about 40 KB.
DEFAULT_JPEG_QUALITY = 90
# GR00T's PolicyClient default: a reply later than this is a failed call.
DEFAULT_TIMEOUT_MS = 15000


def encode_frames(images: np.ndarray, quality: int = DEFAULT_JPEG_QUALITY) -> dict:
    """``images`` (..., H, W, 3) uint8 RGB as one JPEG per image."""
    images = np.asarray(images)
    if images.dtype != np.uint8 or images.ndim < 3 or images.shape[-1] != 3:
        raise ValueError(f"expected uint8 (..., H, W, 3) RGB images, got {images.dtype} {images.shape}")
    flat = images.reshape(-1, *images.shape[-3:])
    params = [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)]
    encoded = []
    for image in flat:
        ok, buffer = cv2.imencode(".jpg", cv2.cvtColor(image, cv2.COLOR_RGB2BGR), params)
        if not ok:
            raise ValueError("JPEG encoding failed")
        encoded.append(buffer.tobytes())
    return {JPEG_KEY: encoded, "shape": list(images.shape)}


def decode_frames(frame: dict) -> np.ndarray:
    """The uint8 RGB array ``encode_frames`` was given (up to JPEG loss)."""
    shape = tuple(int(n) for n in frame["shape"])
    images = []
    for data in frame[JPEG_KEY]:
        bgr = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
        if bgr is None:
            raise ValueError("a JPEG frame does not decode")
        images.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    out = np.stack(images)
    if out.shape[-3:] != shape[-3:] or out.shape[0] != int(np.prod(shape[:-3], dtype=int)):
        raise ValueError(f"decoded frames {out.shape} do not make the declared shape {shape}")
    return out.reshape(shape)


def is_jpeg_frame(value: Any) -> bool:
    return isinstance(value, dict) and JPEG_KEY in value


def encode_observation(observation: dict, quality: int = DEFAULT_JPEG_QUALITY) -> dict:
    """``observation`` with every ``video`` array as JPEG; everything else as it was."""
    video = observation.get("video")
    if not video:
        return observation
    return {**observation, "video": {k: encode_frames(v, quality) for k, v in video.items()}}


def decode_observation(observation: dict) -> dict:
    """``observation`` with every JPEG ``video`` entry decoded; raw arrays pass through."""
    video = observation.get("video")
    if not video:
        return observation
    return {
        **observation,
        "video": {k: decode_frames(v) if is_jpeg_frame(v) else v for k, v in video.items()},
    }


def _pack(data: Any) -> bytes:
    return msgpack.packb(data, default=mnp.encode, use_bin_type=True)


def _unpack(data: bytes) -> Any:
    return msgpack.unpackb(data, object_hook=mnp.decode, raw=False)


class JpegPolicyClient:
    """GR00T's ``PolicyClient`` calls (``ping``, ``get_action``, ``reset``) with JPEG frames."""

    def __init__(
        self,
        host: str = "localhost",
        port: int = 5555,
        quality: int = DEFAULT_JPEG_QUALITY,
        timeout_ms: int = DEFAULT_TIMEOUT_MS,
    ):
        self.host, self.port, self.quality, self.timeout_ms = host, port, quality, timeout_ms
        self._context = zmq.Context()
        self._open()

    def _open(self) -> None:
        self._socket = self._context.socket(zmq.REQ)
        self._socket.setsockopt(zmq.RCVTIMEO, self.timeout_ms)
        self._socket.setsockopt(zmq.SNDTIMEO, self.timeout_ms)
        self._socket.setsockopt(zmq.LINGER, 0)
        self._socket.connect(f"tcp://{self.host}:{self.port}")

    def call(self, endpoint: str, data: dict | None = None) -> Any:
        request: dict = {"endpoint": endpoint}
        if data is not None:
            request["data"] = data
        try:
            self._socket.send(_pack(request))
            reply = _unpack(self._socket.recv())
        except zmq.error.Again:
            # A REQ socket that timed out waits for a reply forever: start a fresh one.
            self._socket.close()
            self._open()
            raise
        if isinstance(reply, dict) and "error" in reply:
            raise RuntimeError(f"policy server error: {reply['error']}")
        return reply

    def ping(self) -> bool:
        try:
            self.call("ping")
        except zmq.error.ZMQError:
            return False
        return True

    def get_action(self, observation: dict, options: dict | None = None) -> tuple[dict, dict]:
        data = {"observation": encode_observation(observation, self.quality), "options": options}
        action, info = self.call("get_action", data)
        return action, info

    def reset(self, options: dict | None = None) -> Any:
        return self.call("reset", {"options": options})

    def close(self) -> None:
        self._socket.close()
        self._context.term()


class JpegDecodingPolicy:
    """The server side: decodes a request's JPEG frames, then asks ``policy`` (a GR00T
    ``BasePolicy``: ``get_action``, ``reset``, ``get_modality_config``)."""

    def __init__(self, policy: Any):
        self.policy = policy

    def get_action(self, observation: dict, options: dict | None = None):
        return self.policy.get_action(decode_observation(observation), options)

    def reset(self, options: dict | None = None):
        return self.policy.reset(options)

    def get_modality_config(self):
        return self.policy.get_modality_config()
