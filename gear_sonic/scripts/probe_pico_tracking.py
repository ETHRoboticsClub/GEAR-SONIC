"""Measure PICO SDK update gaps independently of the teleop/simulator processes.

Requires the local XRoboToolkit PC service and a connected, streaming headset.
Measures PC observations of SDK receipt counters, never headset-to-PC latency.
"""

import argparse
from datetime import datetime
import json
import math
from pathlib import Path
import socket
import time

import numpy as np


DEVICES = ("head", "left", "right", "body")


def stats(values):
    if not values:
        return None
    return dict(zip(("p50", "p95", "max"), np.percentile(values, [50, 95, 100]).tolist()))


class TrackingGapProbe:
    def __init__(self, threshold_ms=20.0):
        self.threshold_ms = threshold_ms
        self.sequences = dict.fromkeys(DEVICES, 0)
        self.last_update_ns = dict.fromkeys(DEVICES)
        self.update_counts = dict.fromkeys(DEVICES, 0)
        self.skipped_updates = dict.fromkeys(DEVICES, 0)
        self.gaps = {device: [] for device in DEVICES}
        self.poll_intervals_ms = []
        self.sdk_read_ms = []
        self.body_gaps = []
        self.first_ns = self.last_ns = None
        self.counter_resets = 0
        self._reset_body_interval()

    def _reset_body_interval(self):
        self.body_poll_max_ms = self.body_read_max_ms = 0.0
        self.peer_progress = dict.fromkeys(DEVICES[:-1], 0)

    def observe(self, started_ns, finished_ns, sequences):
        current = {key: int(sequences[key]) for key in DEVICES}
        read_ms = (finished_ns - started_ns) / 1e6
        self.sdk_read_ms.append(read_ms)
        self.body_read_max_ms = max(self.body_read_max_ms, read_ms)
        if self.first_ns is None:
            self.first_ns = finished_ns
        if self.last_ns is not None:
            interval_ms = (finished_ns - self.last_ns) / 1e6
            self.poll_intervals_ms.append(interval_ms)
            self.body_poll_max_ms = max(self.body_poll_max_ms, interval_ms)
        self.last_ns = finished_ns
        if any(current[key] < self.sequences[key] for key in DEVICES):
            # Reconnection/counter reset must not be counted as an enormous gap.
            self.counter_resets += 1
            self.last_update_ns = dict.fromkeys(DEVICES)
            self.sequences = dict.fromkeys(DEVICES, 0)
            self._reset_body_interval()
        body_changed = current["body"] > self.sequences["body"]
        if not body_changed and self.last_update_ns["body"] is not None:
            for key in self.peer_progress:
                self.peer_progress[key] += max(0, current[key] - self.sequences[key])
        for key in DEVICES:
            advance = current[key] - self.sequences[key]
            if advance <= 0:
                continue
            previous_ns = self.last_update_ns[key]
            self.update_counts[key] += 1
            if self.sequences[key] > 0:
                self.skipped_updates[key] += max(0, advance - 1)
            if previous_ns is not None:
                gap_ms = (finished_ns - previous_ns) / 1e6
                self.gaps[key].append(gap_ms)
                if key == "body" and gap_ms > self.threshold_ms:
                    self.body_gaps.append({
                        "gap_ms": gap_ms,
                        "start_monotonic_ns": previous_ns,
                        "end_monotonic_ns": finished_ns,
                        "max_poll_interval_ms": self.body_poll_max_ms,
                        "max_sdk_read_ms": self.body_read_max_ms,
                        # Excludes peers advancing only in the poll ending the gap.
                        "peer_updates_observed_while_body_unchanged": self.peer_progress.copy(),
                        "body_counter_advance": advance,
                    })
            self.last_update_ns[key] = finished_ns
        if body_changed:
            self._reset_body_interval()
        self.sequences = current

    def summary(self):
        elapsed = 0 if self.first_ns is None else (self.last_ns - self.first_ns) / 1e9
        return {
            "observed_seconds": elapsed,
            "threshold_ms": self.threshold_ms,
            "poll_interval_ms": stats(self.poll_intervals_ms),
            "sdk_counter_read_ms": stats(self.sdk_read_ms),
            "counter_resets": self.counter_resets,
            "devices": {
                key: {
                    "observed_updates": self.update_counts[key],
                    "updates_between_polls": self.skipped_updates[key],
                    "update_gap_ms": stats(self.gaps[key]),
                    "gaps_over_threshold": sum(g > self.threshold_ms for g in self.gaps[key]),
                    "ending_silence_ms": None if self.last_update_ns[key] is None else
                        (self.last_ns - self.last_update_ns[key]) / 1e6,
                } for key in DEVICES
            },
            "long_body_gaps": self.body_gaps,
            "note": (
                "PC-observed SDK receipt counters; no capture-to-PC latency measurement. "
                "Peer progress during a body gap separates body-section pauses from shared pauses; "
                "it does not identify headset, network or PC-service root cause. Large poll/read "
                "times mean this observer may have missed intervening updates. Updates between "
                "polls are not proof of network packet loss. Ending silence is separate from completed gaps."
            ),
        }


def capture(sdk, seconds, wait, poll_s):
    probe = TrackingGapProbe()
    waiting_until = time.monotonic() + wait
    deadline = None
    while True:
        started_ns = time.monotonic_ns()
        sequences = sdk.get_input_update_sequences()
        finished_ns = time.monotonic_ns()
        now = time.monotonic()
        if deadline is None:
            if sequences["body"] > 0:
                probe = TrackingGapProbe()  # Keep waiting time out of active body statistics.
                deadline = now + seconds
                print(f"Recording {seconds:g}s of SDK counters; keep body tracking active.", flush=True)
        probe.observe(started_ns, finished_ns, sequences)
        if deadline is None and now >= waiting_until:
            result = probe.summary()
            result["status"] = "no_body_tracking"
            return result
        if deadline is not None and now >= deadline:
            result = probe.summary()
            result["status"] = "recorded"
            return result
        time.sleep(poll_s)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seconds", type=float, default=60)
    parser.add_argument("--wait", type=float, default=120)
    parser.add_argument("--poll-ms", type=float, default=1)
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parents[2] /
                        ".local/pico-sim/logs" / datetime.now().strftime("sdk-gaps-%Y%m%d-%H%M%S.json"))
    args = parser.parse_args()
    if not all(math.isfinite(v) and v > 0 for v in (args.seconds, args.wait, args.poll_ms)):
        parser.error("--seconds, --wait and --poll-ms must be finite and positive")
    if args.output.exists():
        parser.error(f"Output already exists: {args.output}")
    try:
        with socket.create_connection(("127.0.0.1", 60061), timeout=.5):
            pass
    except OSError:
        parser.exit(2, "XRoboToolkit PC service is stopped. Start it before probing tracking.\n")
    import xrobotoolkit_sdk as sdk

    if not hasattr(sdk, "get_input_update_sequences"):
        parser.exit(2, "This probe requires this checkout's SDK receipt-counter bindings.\n")
    print("Waiting for SDK body updates. This probe issues no robot or simulator commands.", flush=True)
    sdk.init()
    try:
        result = capture(sdk, args.seconds, args.wait, args.poll_ms / 1000)
    finally:
        sdk.close()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as output:
        json.dump(result, output, indent=2)
        output.write("\n")
    print(f"Saved {args.output}; status={result['status']}")
    print(json.dumps({key: result[key] for key in
                      ("observed_seconds", "poll_interval_ms", "sdk_counter_read_ms", "devices")}, indent=2))
    return 0 if result["status"] == "recorded" else 2


if __name__ == "__main__":
    raise SystemExit(main())
