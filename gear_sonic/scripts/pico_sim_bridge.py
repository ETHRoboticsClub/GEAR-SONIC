"""Local PICO input for the standalone MuJoCo deployment.

The fork's manager consumes ETHRC-format tracking, while the standalone C++
controller publishes only robot feedback. This simulation adapter combines
XRoboToolkit tracking and unmodified C++ feedback on localhost:5558. The manager
must use --zmq_feedback_port 5558; the C++ controller keeps its default 5557.
No ETHRC API or robot connection is used here.
"""

import logging
import time

import msgpack
import numpy as np
from scipy.spatial.transform import Rotation
import zmq

from gear_sonic.utils.teleop.input_readers import (
    INPUT_STALE_AFTER_S,
    OPERATOR_FROM_OPENXR,
    QUAT_NORM_TOLERANCE,
    TRACKING_TOPIC,
)

LOG = logging.getLogger(__name__)
OPERATOR_ROTATION = Rotation.from_matrix(OPERATOR_FROM_OPENXR)


def operator_poses(rows):
    """Convert SDK OpenXR xyz/xyzw poses to the reader's REP-103 wire frame."""
    poses = np.asarray(rows, dtype=float)
    if poses.ndim != 2 or poses.shape[1] != 7 or not np.isfinite(poses).all():
        raise ValueError("invalid SDK pose shape or non-finite value")
    if np.any(np.abs(np.linalg.norm(poses[:, 3:], axis=1) - 1) > QUAT_NORM_TOLERANCE):
        raise ValueError("SDK quaternion is not unit length")
    return np.concatenate(
        (poses[:, :3] @ OPERATOR_FROM_OPENXR.T,
         (OPERATOR_ROTATION * Rotation.from_quat(poses[:, 3:])).as_quat()), axis=1
    )


class FreshPoses:
    """Expire cached values when device packets stop, even if other devices update."""

    def __init__(self):
        self.previous = {}

    def read(self, name, rows, now, stamp):
        if stamp <= 0:
            return None
        try:
            poses = operator_poses(rows)
        except (ValueError, TypeError):
            return None
        key = stamp
        old_key, changed_at = self.previous.get(name, (None, now))
        if key != old_key:
            changed_at = now
        self.previous[name] = key, changed_at
        return poses if now - changed_at <= INPUT_STALE_AFTER_S else None


def controller(xrt, side, poses):
    if poses is None:
        return None
    left = side == "left"
    stick = getattr(xrt, f"get_{side}_axis")()
    return {
        "grip": poses[0].tolist(), "aim": None,
        "primary": bool((xrt.get_X_button if left else xrt.get_A_button)()),
        "secondary": bool((xrt.get_Y_button if left else xrt.get_B_button)()),
        "menu": bool(getattr(xrt, f"get_{side}_menu_button")()),
        "stick_click": bool(getattr(xrt, f"get_{side}_axis_click")()),
        "trigger": float(getattr(xrt, f"get_{side}_trigger")()),
        "squeeze": float(getattr(xrt, f"get_{side}_grip")()),
        "stick_x": float(stick[0]), "stick_y": float(stick[1]),
    }


def sample(xrt, fresh, sequence, now, updates=None):
    updates = xrt.get_input_update_sequences() if updates is None else updates
    head = fresh.read("head", [xrt.get_headset_pose()], now, updates["head"])
    frame = {
        "sequence": sequence, "sample_time_ns": time.monotonic_ns(),
        "body_update_sequence": int(updates["body"]),
        "input_update_sequences": {key: int(updates[key]) for key in ("head", "left", "right", "body")},
        "head": None if head is None else head[0].tolist(),
        "body": None,
    }
    for side in ("left", "right"):
        poses = fresh.read(side, [getattr(xrt, f"get_{side}_controller_pose")()], now, updates[side])
        frame[f"{side}_controller"] = controller(xrt, side, poses)
    if xrt.is_body_data_available():
        # Use SDK receipt counters: some clients omit Body.timeStampNs, and
        # stationary devices may send identical poses in successive packets.
        body = fresh.read("body", xrt.get_body_joints_pose(), now,
                          stamp=updates["body"])
        if body is not None and body.shape == (24, 7):
            frame["body"] = {
                "positions": body[:, :3].ravel().tolist(),
                "rotations": body[:, 3:].ravel().tolist(),
            }
            if hasattr(xrt, "get_body_input_timing"):
                timing = xrt.get_body_input_timing()
                # SDK callbacks can advance while Python reads separate getters.
                # Never attribute another sample's timestamp to this body sequence.
                if timing.get("body_sequence") == frame["body_update_sequence"]:
                    frame["body_input_timing"] = timing
    return frame


def main():
    import xrobotoolkit_sdk as xrt

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    context = zmq.Context()
    publisher = context.socket(zmq.PUB)
    publisher.setsockopt(zmq.SNDHWM, 10)
    publisher.bind("tcp://127.0.0.1:5558")
    feedback = context.socket(zmq.SUB)
    feedback.setsockopt(zmq.SUBSCRIBE, b"")
    feedback.connect("tcp://127.0.0.1:5557")
    fresh = FreshPoses()
    sequence = 0
    last_stamp = None
    last_status = None
    last_sample = 0.0
    last_poll_ns = None
    poll_interval_max_ms = sdk_read_max_ms = 0.0
    xrt.init()
    LOG.info("PICO simulation bridge ready; waiting for headset data. Feedback :5557 -> :5558")
    try:
        while True:
            # Forward the original byte frames, including robot configuration.
            for _ in range(100):
                try:
                    publisher.send_multipart(feedback.recv_multipart(zmq.NOBLOCK))
                except zmq.Again:
                    break
            poll_started_ns = time.monotonic_ns()
            updates = xrt.get_input_update_sequences()
            poll_finished_ns = time.monotonic_ns()
            sdk_read_max_ms = max(sdk_read_max_ms, (poll_finished_ns - poll_started_ns) / 1e6)
            if last_poll_ns is not None:
                poll_interval_max_ms = max(poll_interval_max_ms, (poll_finished_ns - last_poll_ns) / 1e6)
            last_poll_ns = poll_finished_ns
            now = poll_finished_ns / 1e9
            stamp = tuple(updates[key] for key in ("head", "left", "right", "body"))
            if stamp != last_stamp or now - last_sample >= 0.05:
                frame = sample(xrt, fresh, sequence, now, updates)
                # Preserve short polling/SDK stalls even if they end between publications.
                frame["bridge_poll_interval_max_ms"] = poll_interval_max_ms
                frame["sdk_counter_read_max_ms"] = sdk_read_max_ms
                publisher.send(TRACKING_TOPIC + msgpack.packb(frame, use_bin_type=True))
                poll_interval_max_ms = sdk_read_max_ms = 0.0
                sequence += 1
                last_stamp, last_sample = stamp, now
                status = tuple(frame.get(k) is not None for k in
                               ("head", "left_controller", "right_controller", "body"))
                if status != last_status:
                    LOG.info("Tracked head=%s left=%s right=%s body=%s", *status)
                    last_status = status
            time.sleep(0.001)
    except KeyboardInterrupt:
        pass
    finally:
        xrt.close()
        feedback.close(linger=0)
        publisher.close(linger=0)
        context.term()


if __name__ == "__main__":
    main()
