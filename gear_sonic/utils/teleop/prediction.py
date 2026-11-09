"""Latency compensation for the VR three-point targets.

The targets reach the policy tens of milliseconds after the headset sampled
them (tunnel, manager, control tick), and the policy then trails them by its
own tracking lag. ``ThreePointPredictor`` extrapolates each point along its
current velocity by a fixed lead, so a steady hand motion arrives where the
hand is by then. A reversal overshoots by at most the clamp, which bounds the
error the lead can add.

Velocities are least-squares slopes over the samples of the last
``window_s``: a longer window smooths tracking noise and lags the estimate.
The slopes run on a regular time axis, each sample one mean interval after the
last: the stamps are arrival times, and the headset samples at a fixed rate
that Wi-Fi and TCP deliver in bursts, so slopes against the stamps themselves
turn that jitter into velocity noise (twice the output jitter in
offline replays of bursty arrivals). The estimate restarts after a gap longer
than the window, so a stalled stream is never extrapolated.
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
# Weight of each new arrival interval in the mean interval: about 20 samples
# (a quarter second at 90 Hz) average out bursts and follow a rate change.
INTERVAL_SMOOTHING = 0.05


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
        self.window_s = window_s
        self.window_ns = round(window_s * 1e9)
        self.max_shift_m = max_shift_m
        self.max_turn_rad = max_turn_rad
        self._samples: deque[np.ndarray] = deque()
        self._last_ns: int | None = None
        self._interval_s: float | None = None
        self._intervals = 0

    def reset(self) -> None:
        self._samples.clear()
        self._last_ns = None
        self._interval_s = None
        self._intervals = 0

    def __call__(self, stamp_ns: int, pose: np.ndarray) -> np.ndarray:
        """The pose ``lead_s`` ahead; the pose itself until the input's rate is known."""
        samples, last_ns = self._samples, self._last_ns
        if last_ns is not None and not 0 < stamp_ns - last_ns <= self.window_ns:
            samples.clear()
        elif last_ns is not None:
            # The mean of the intervals so far, then a moving average of them.
            self._intervals += 1
            gap_s = (stamp_ns - last_ns) * 1e-9
            weight = max(INTERVAL_SMOOTHING, 1.0 / self._intervals)
            interval_s = self._interval_s or 0.0
            self._interval_s = interval_s + weight * (gap_s - interval_s)
        self._last_ns = stamp_ns
        samples.append(np.array(pose, dtype=np.float64))
        interval_s = self._interval_s
        if interval_s is None:
            return pose
        while len(samples) > max(2, round(self.window_s / interval_s) + 1):
            samples.popleft()
        if len(samples) < 2:
            return pose

        count = len(samples)
        t_centered = (np.arange(count) - (count - 1) / 2) * interval_s
        positions = np.stack([p[:, :3] for p in samples])  # (k, 3, 3)
        velocity = np.einsum("k,kij->ij", t_centered, positions - positions.mean(axis=0))
        velocity /= np.dot(t_centered, t_centered)
        shift = velocity * self.lead_s
        norms = np.linalg.norm(shift, axis=1, keepdims=True)
        shift *= np.minimum(1.0, self.max_shift_m / np.maximum(norms, 1e-12))

        newest = Rotation.from_quat(pose[:, 3:], scalar_first=True)
        oldest = Rotation.from_quat(samples[0][:, 3:], scalar_first=True)
        omega = (newest * oldest.inv()).as_rotvec() / ((count - 1) * interval_s)  # world rad/s
        turn = omega * self.lead_s
        angles = np.linalg.norm(turn, axis=1, keepdims=True)
        turn *= np.minimum(1.0, self.max_turn_rad / np.maximum(angles, 1e-12))

        predicted = np.empty_like(pose)
        predicted[:, :3] = pose[:, :3] + shift
        predicted[:, 3:] = (Rotation.from_rotvec(turn) * newest).as_quat(scalar_first=True)
        return predicted
