"""Latency compensation for the VR three-point targets.

The targets reach the policy tens of milliseconds after the headset sampled
them (tunnel, manager, control tick), and the policy then trails them by its
own tracking lag. ``ThreePointPredictor`` extrapolates each point along its
current velocity by a fixed lead, so a steady hand motion arrives where the
hand is by then. A reversal overshoots by at most the clamp, which bounds the
error the lead can add.

Velocities are least-squares slopes over the samples of the last
``window_s``: a longer window smooths tracking noise and lags the estimate.
The estimate restarts after a gap longer than the window, so a stalled stream
is never extrapolated.
"""

from collections import deque

import numpy as np
from scipy.spatial.transform import Rotation

# Samples of the last 50 ms: about 4 at the headsets' 72-90 Hz, enough for a
# least-squares slope, short enough to follow a reaching motion's acceleration.
DEFAULT_WINDOW_S = 0.05
# The farthest a lead may move a target (m, rad): a few centimetres and about
# 20 degrees bound the overshoot of a reversal at the leads this is meant for.
DEFAULT_MAX_SHIFT_M = 0.08
DEFAULT_MAX_TURN_RAD = 0.35
# Two samples closer than this give no usable velocity (duplicate stamps).
MIN_SPAN_S = 0.004


class ThreePointPredictor:
    """Extrapolates ``(3, 7)`` poses ``[x, y, z, qw, qx, qy, qz]`` (left wrist,
    right wrist, neck) ``lead_s`` past their sample time."""

    def __init__(
        self,
        lead_s: float,
        window_s: float = DEFAULT_WINDOW_S,
        max_shift_m: float = DEFAULT_MAX_SHIFT_M,
        max_turn_rad: float = DEFAULT_MAX_TURN_RAD,
    ):
        if not (lead_s > 0 and window_s > 0 and max_shift_m >= 0 and max_turn_rad >= 0):
            raise ValueError("lead and window must be positive, the clamps non-negative")
        self.lead_s = lead_s
        self.window_ns = round(window_s * 1e9)
        self.max_shift_m = max_shift_m
        self.max_turn_rad = max_turn_rad
        self._samples: deque[tuple[int, np.ndarray]] = deque()

    def reset(self) -> None:
        self._samples.clear()

    def __call__(self, stamp_ns: int, pose: np.ndarray) -> np.ndarray:
        """The pose ``lead_s`` ahead; the pose itself until two samples span the window."""
        samples = self._samples
        if samples and (stamp_ns <= samples[-1][0] or stamp_ns - samples[-1][0] > self.window_ns):
            samples.clear()
        samples.append((stamp_ns, np.array(pose, dtype=np.float64)))
        while stamp_ns - samples[0][0] > self.window_ns:
            samples.popleft()
        span_s = (stamp_ns - samples[0][0]) * 1e-9
        if span_s < MIN_SPAN_S:
            return pose

        t = np.array([(s - stamp_ns) * 1e-9 for s, _ in samples])
        positions = np.stack([p[:, :3] for _, p in samples])  # (k, 3, 3)
        t_centered = t - t.mean()
        velocity = np.einsum("k,kij->ij", t_centered, positions - positions.mean(axis=0))
        velocity /= np.dot(t_centered, t_centered)
        shift = velocity * self.lead_s
        norms = np.linalg.norm(shift, axis=1, keepdims=True)
        shift *= np.minimum(1.0, self.max_shift_m / np.maximum(norms, 1e-12))

        newest = Rotation.from_quat(pose[:, 3:], scalar_first=True)
        oldest = Rotation.from_quat(samples[0][1][:, 3:], scalar_first=True)
        omega = (newest * oldest.inv()).as_rotvec() / span_s  # world-frame rad/s
        turn = omega * self.lead_s
        angles = np.linalg.norm(turn, axis=1, keepdims=True)
        turn *= np.minimum(1.0, self.max_turn_rad / np.maximum(angles, 1e-12))

        predicted = np.empty_like(pose)
        predicted[:, :3] = pose[:, :3] + shift
        predicted[:, 3:] = (Rotation.from_rotvec(turn) * newest).as_quat(scalar_first=True)
        return predicted
