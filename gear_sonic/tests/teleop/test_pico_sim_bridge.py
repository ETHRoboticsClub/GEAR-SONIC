import numpy as np
import pytest
from types import SimpleNamespace

from gear_sonic.scripts.pico_sim_bridge import FreshPoses, operator_poses, sample
from gear_sonic.utils.teleop.input_readers import TrackingReader


def test_openxr_axes_and_orientation_reach_reader_unchanged():
    # OpenXR right/up/back maps to operator -left/up/-forward.
    sdk = np.array([[1., 2., 3., 0., 0., 0., 1.]])
    wire = operator_poses(sdk)
    np.testing.assert_allclose(wire[0, :3], [-3., -1., 2.])
    reader = TrackingReader(clock=lambda: 1_000_000_000)
    body = np.repeat(wire, 24, axis=0)
    reader._update({
        "sample_time_ns": 1_000_000_000,
        "body": {"positions": body[:, :3].ravel().tolist(),
                 "rotations": body[:, 3:].ravel().tolist()},
    }, 1_000_000_000)
    actual = reader.get_latest()["body_poses_np"]
    np.testing.assert_allclose(actual[:, :3], np.repeat(sdk[:, :3], 24, axis=0))
    np.testing.assert_allclose(np.abs(actual[:, 6]), 1., atol=1e-6)
    np.testing.assert_allclose(actual[:, 3:6], 0., atol=1e-6)


def test_cached_tracking_expires_and_new_data_recovers():
    fresh = FreshPoses()
    pose = [[0., 1., 0., 0., 0., 0., 1.]]
    assert fresh.read("head", pose, 0., stamp=1) is not None
    assert fresh.read("head", pose, 0.3, stamp=1) is None
    pose[0][0] = 0.01
    assert fresh.read("head", pose, 0.4, stamp=2) is not None
    assert fresh.read("body", pose, 0., stamp=10) is not None
    assert fresh.read("body", pose, 0.3, stamp=10) is None
    assert fresh.read("body", pose, 0.4, stamp=11) is not None


@pytest.mark.parametrize("pose", [
    [[0., 0., 0., 0., 0., 0., 0.]],
    [[float("nan"), 0., 0., 0., 0., 0., 1.]],
    [[0., 0., 0.]],
])
def test_invalid_sdk_pose_is_untracked(pose):
    assert FreshPoses().read("head", pose, 0., stamp=1) is None


def test_body_updates_without_vendor_timestamp():
    class SDK:
        def __init__(self):
            self.body = np.tile([0., 1., 0., 0., 0., 0., 1.], (24, 1))
            self.updates = dict(head=0, left=0, right=0, body=1)

        def get_input_update_sequences(self):
            return self.updates

        def get_headset_pose(self):
            return [0.] * 7

        get_left_controller_pose = get_headset_pose
        get_right_controller_pose = get_headset_pose

        def is_body_data_available(self):
            return True

        def get_body_joints_pose(self):
            return self.body

        def get_body_timestamp_ns(self):
            return 0

    sdk, fresh = SDK(), FreshPoses()
    frame = sample(sdk, fresh, 0, 0.)
    assert frame["body"] is not None
    assert frame["body_update_sequence"] == 1
    assert frame["input_update_sequences"] == sdk.updates
    sdk.get_body_input_timing = lambda: dict(body_sequence=1, headset_frame_unix_ns=123)
    assert sample(sdk, fresh, 0, 0.)['body_input_timing']['headset_frame_unix_ns'] == 123
    sdk.get_body_input_timing = lambda: dict(body_sequence=2, headset_frame_unix_ns=456)
    assert 'body_input_timing' not in sample(sdk, fresh, 0, 0.)
    sdk.body[0, 0] = .1
    sdk.updates["body"] += 1
    assert frame["input_update_sequences"]["body"] == 1  # Snapshot, not the mutable SDK dictionary.
    assert sample(sdk, fresh, 1, .3)["body"] is not None
    assert sample(sdk, fresh, 2, .6)["body"] is None


def test_stationary_controllers_keep_buttons_until_packets_stop():
    updates = dict(head=1, left=1, right=1, body=0)
    buttons = dict(A=False, B=False, X=False, Y=False)
    sdk = SimpleNamespace(
        get_input_update_sequences=lambda: updates,
        get_headset_pose=lambda: [0., 1.6, 0., 0., 0., 0., 1.],
        is_body_data_available=lambda: False,
    )
    for side in ("left", "right"):
        setattr(sdk, f"get_{side}_controller_pose", lambda: [0., 1., 0., 0., 0., 0., 1.])
        setattr(sdk, f"get_{side}_axis", lambda: [0., 0.])
        for name in ("menu_button", "axis_click", "trigger", "grip"):
            setattr(sdk, f"get_{side}_{name}", lambda: 0)
    for name in buttons:
        setattr(sdk, f"get_{name}_button", lambda name=name: buttons[name])
    fresh = FreshPoses()
    sample(sdk, fresh, 0, 0.)
    buttons.update(A=True, X=True)
    # Hold A+X for much longer than the stale timeout without moving either hand.
    for sequence in range(1, 11):
        updates.update(head=sequence+1, left=sequence+1, right=sequence+1)
        frame = sample(sdk, fresh, sequence, sequence*.1)
        assert frame["left_controller"]["primary"]
        assert frame["right_controller"]["primary"]
    # A fresh packet carrying release must clear the buttons immediately.
    buttons.update(A=False, X=False)
    updates.update(left=12, right=12)
    frame = sample(sdk, fresh, 12, 1.1)
    assert not frame["left_controller"]["primary"]
    assert not frame["right_controller"]["primary"]
    # Head traffic must not keep disconnected controllers fresh.
    updates["head"] = 99
    frame = sample(sdk, fresh, 13, 1.4)
    assert frame["head"] is not None
    assert frame["left_controller"] is None
    assert frame["right_controller"] is None


def test_no_packets_never_counts_as_tracked():
    assert FreshPoses().read("head", [[0., 1., 0., 0., 0., 0., 1.]], 0., stamp=0) is None


def test_bridge_retains_sdk_stall_until_next_publication(monkeypatch):
    import msgpack
    import sys
    import zmq
    from gear_sonic.scripts import pico_sim_bridge as bridge

    clock = [0]
    calls = [0]
    sleeps = [0]
    sent, closed = [], []

    def updates():
        calls[0] += 1
        if calls[0] == 2:
            clock[0] += 25_000_000  # SDK getter stalls while body counter stays fixed.
        return dict(head=0, left=0, right=0, body=2 if calls[0] == 4 else 1)

    def sleep(duration):
        clock[0] += round(duration * 1e9)
        sleeps[0] += 1
        if sleeps[0] == 4:
            raise KeyboardInterrupt

    def no_feedback(flags):
        raise zmq.Again()

    publisher = SimpleNamespace(setsockopt=lambda *a: None, bind=lambda *a: None,
                                send=sent.append, send_multipart=sent.append, close=lambda **kw: None)
    feedback = SimpleNamespace(setsockopt=lambda *a: None, connect=lambda *a: None,
                               recv_multipart=no_feedback, close=lambda **kw: None)
    context = SimpleNamespace(socket=lambda kind: publisher if kind == zmq.PUB else feedback,
                              term=lambda: None)
    sdk = SimpleNamespace(
        init=lambda: None, close=lambda: closed.append(True),
        get_input_update_sequences=updates,
        get_headset_pose=lambda: [0., 0., 0., 0., 0., 0., 1.],
        get_left_controller_pose=lambda: [0., 0., 0., 0., 0., 0., 1.],
        get_right_controller_pose=lambda: [0., 0., 0., 0., 0., 0., 1.],
        is_body_data_available=lambda: True,
        get_body_joints_pose=lambda: np.tile([0., 1., 0., 0., 0., 0., 1.], (24, 1)),
    )
    monkeypatch.setitem(sys.modules, "xrobotoolkit_sdk", sdk)
    monkeypatch.setattr(bridge.zmq, "Context", lambda: context)
    monkeypatch.setattr(bridge, "time", SimpleNamespace(monotonic_ns=lambda: clock[0], sleep=sleep))
    bridge.main()
    frames = [msgpack.unpackb(raw[len(bridge.TRACKING_TOPIC):], raw=False) for raw in sent]
    assert [frame["body_update_sequence"] for frame in frames] == [1, 2]
    assert frames[-1]["sdk_counter_read_max_ms"] == 25
    assert frames[-1]["bridge_poll_interval_max_ms"] == 26
    assert frames[-1]["body"] is not None
    assert closed == [True]
