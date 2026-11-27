"""JPEG frames between the VLA client and a GR00T policy server.

The server double answers as GR00T's ``PolicyServer`` does (``gr00t/policy/server_client.py``
at ethrc/groot_finetune/ISAAC_GR00T_REVISION): a ZMQ REP socket, msgpack with
msgpack-numpy, ``{"endpoint", "data"}`` requests, the handler's result as the reply and
``{"error": ...}`` when it raises.
"""

from __future__ import annotations

import threading

import msgpack
import msgpack_numpy as mnp
import numpy as np
import pytest
import zmq

from gear_sonic.utils.inference import jpeg_policy as J

HEIGHT, WIDTH = 360, 640


def head_frame() -> np.ndarray:
    """A camera-like frame: smooth gradients with a few sharp edges."""
    y, x = np.mgrid[0:HEIGHT, 0:WIDTH]
    frame = np.stack([x * 255 // WIDTH, y * 255 // HEIGHT, (x + y) * 255 // (WIDTH + HEIGHT)], -1)
    frame[100:200, 300:400] = (240, 30, 30)
    return frame.astype(np.uint8)


def test_a_frame_survives_jpeg_with_its_shape_and_little_loss():
    images = head_frame()[np.newaxis, np.newaxis]
    encoded = J.encode_frames(images)
    assert len(encoded[J.JPEG_KEY]) == 1 and encoded["shape"] == [1, 1, HEIGHT, WIDTH, 3]
    assert len(encoded[J.JPEG_KEY][0]) < images.nbytes / 10
    decoded = J.decode_frames(encoded)
    assert decoded.dtype == np.uint8 and decoded.shape == images.shape
    error = np.abs(decoded.astype(int) - images.astype(int))
    assert error.mean() < 2.0
    # Colors keep their channel: the red square stays red.
    assert decoded[0, 0, 150, 350].argmax() == 0


def test_several_images_keep_their_order():
    frames = np.stack([np.full((8, 16, 3), v, np.uint8) for v in (10, 120, 240)])[np.newaxis]
    decoded = J.decode_frames(J.encode_frames(frames))
    assert [int(decoded[0, i].mean().round()) for i in range(3)] == [10, 120, 240]


@pytest.mark.parametrize(
    "images",
    [np.zeros((4, 4, 3), np.float32), np.zeros((4, 4), np.uint8), np.zeros((4, 4, 4), np.uint8)],
)
def test_only_uint8_rgb_images_are_encoded(images):
    with pytest.raises(ValueError, match="RGB"):
        J.encode_frames(images)


def test_a_frame_that_does_not_make_its_shape_is_refused():
    encoded = J.encode_frames(np.zeros((1, 1, 8, 8, 3), np.uint8))
    encoded["shape"] = [1, 1, 8, 16, 3]
    with pytest.raises(ValueError, match="declared shape"):
        J.decode_frames(encoded)


def test_only_the_video_entries_change():
    observation = {
        "video": {"ego_view": head_frame()[np.newaxis, np.newaxis]},
        "state": {"left_arm": np.zeros((1, 1, 7))},
        "language": {"annotation.human.task_description": [["grasp the bottle"]]},
    }
    encoded = J.encode_observation(observation)
    assert J.is_jpeg_frame(encoded["video"]["ego_view"])
    assert encoded["state"] is observation["state"] and encoded["language"] is observation["language"]
    decoded = J.decode_observation(encoded)
    assert decoded["video"]["ego_view"].shape == (1, 1, HEIGHT, WIDTH, 3)
    # A raw frame (GR00T's own client) passes through the server side unchanged.
    assert J.decode_observation(observation)["video"]["ego_view"] is observation["video"]["ego_view"]


class Recorder:
    """A policy that keeps what it was asked and answers a fixed action chunk."""

    def __init__(self):
        self.observations = []

    def get_action(self, observation, options=None):
        if observation["language"]["annotation.human.task_description"] == [["fail"]]:
            raise RuntimeError("the policy refused")
        self.observations.append(observation)
        tokens = np.arange(40 * 64, dtype=np.float32).reshape(1, 40, 64)
        return {"motion_token": tokens}, {"latency_s": 0.1}

    def reset(self, options=None):
        return {"reset": True}

    def get_modality_config(self):
        return {}


@pytest.fixture
def server():
    """A GR00T-protocol server around ``JpegDecodingPolicy(Recorder())`` on a free port."""
    policy = Recorder()
    wrapped = J.JpegDecodingPolicy(policy)
    context = zmq.Context()
    socket = context.socket(zmq.REP)
    port = socket.bind_to_random_port("tcp://127.0.0.1")
    stop = threading.Event()
    handlers = {
        "ping": lambda: {"status": "ok"},
        "get_action": wrapped.get_action,
        "reset": wrapped.reset,
    }

    def serve():
        poller = zmq.Poller()
        poller.register(socket, zmq.POLLIN)
        while not stop.is_set():
            if not poller.poll(50):
                continue
            request = msgpack.unpackb(socket.recv(), object_hook=mnp.decode, raw=False)
            handler = handlers[request.get("endpoint", "get_action")]
            try:
                result = handler(**request["data"]) if "data" in request else handler()
            except Exception as error:  # noqa: BLE001 - the server double replies like GR00T's
                result = {"error": str(error)}
            socket.send(msgpack.packb(result, default=mnp.encode, use_bin_type=True))

    thread = threading.Thread(target=serve, daemon=True)
    thread.start()
    yield port, policy
    stop.set()
    thread.join()
    socket.close(linger=0)
    context.term()


def observation(task: str = "grasp the bottle") -> dict:
    return {
        "video": {"ego_view": head_frame()[np.newaxis, np.newaxis]},
        "state": {"left_arm": np.full((1, 1, 7), 0.25, np.float32)},
        "language": {"annotation.human.task_description": [[task]]},
    }


def test_the_policy_gets_decoded_frames_and_the_client_its_action(server):
    port, policy = server
    client = J.JpegPolicyClient(port=port)
    try:
        assert client.ping()
        action, info = client.get_action(observation())
        assert client.reset() == {"reset": True}
    finally:
        client.close()
    np.testing.assert_array_equal(action["motion_token"], np.arange(40 * 64, dtype=np.float32).reshape(1, 40, 64))
    assert info == {"latency_s": 0.1}
    (seen,) = policy.observations
    frame = seen["video"]["ego_view"]
    assert frame.dtype == np.uint8 and frame.shape == (1, 1, HEIGHT, WIDTH, 3)
    np.testing.assert_array_equal(seen["state"]["left_arm"], np.full((1, 1, 7), 0.25))


def test_a_server_error_reaches_the_caller(server):
    port, _ = server
    client = J.JpegPolicyClient(port=port)
    try:
        with pytest.raises(RuntimeError, match="the policy refused"):
            client.get_action(observation("fail"))
        # The socket is usable after an error reply.
        assert client.ping()
    finally:
        client.close()


def test_a_silent_server_times_out_and_the_client_can_call_again(server):
    port, _ = server
    silent = zmq.Context().socket(zmq.REP)
    silent_port = silent.bind_to_random_port("tcp://127.0.0.1")
    client = J.JpegPolicyClient(port=silent_port, timeout_ms=200)
    try:
        assert not client.ping()
        assert not client.ping(), "a timed-out REQ socket must be replaced, not reused"
    finally:
        client.close()
        silent.close(linger=0)


def test_gr00ts_own_serializer_reads_the_request():
    gr00t = pytest.importorskip("gr00t.policy.server_client", reason="needs Isaac-GR00T")
    request = {"endpoint": "get_action", "data": {"observation": J.encode_observation(observation())}}
    decoded = gr00t.MsgSerializer.from_bytes(J._pack(request))
    frame = decoded["data"]["observation"]["video"]["ego_view"]
    assert J.decode_frames(frame).shape == (1, 1, HEIGHT, WIDTH, 3)
