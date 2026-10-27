"""Capture setup must not time out early or hide the stalls being measured."""

from types import SimpleNamespace
import json
import subprocess
import sys

import msgpack
import numpy as np

from gear_sonic.scripts import measure_pico_sim_latency as capture
from gear_sonic.utils.teleop.zmq.zmq_planner_sender import pack_pose_message


class ScheduledSocket:
    def __init__(self, clock, events, interrupt_at=None):
        self.clock, self.events, self.index = clock, events, 0
        self.interrupt_at = interrupt_at

    def poll(self, timeout):
        end = self.clock.now + timeout / 1000
        if self.interrupt_at is not None and end >= self.interrupt_at:
            self.clock.now = self.interrupt_at
            raise KeyboardInterrupt
        if self.index < len(self.events) and self.events[self.index][0] <= end:
            self.clock.now = max(self.clock.now, self.events[self.index][0])
            return True
        self.clock.now = end
        return False

    def recv(self):
        raw = self.events[self.index][1]
        self.index += 1
        return raw


def traffic(at=0, mode=1):
    events = []
    for i in range(60):
        stamp = at + i * .02
        tracking = {'head': {}, 'left_controller': {}, 'right_controller': {},
                    'body': {} if i < 20 else None, 'sample_time_ns': int(stamp * 1e9)}
        manager = pack_pose_message({'stream_mode': np.array([mode], dtype=np.int32)},
                                    topic='manager_state')
        events.extend((stamp, raw) for raw in (
            b'tracking' + msgpack.packb(tracking), b'g1_debug' + msgpack.packb({}),
            b'pose', manager,
        ))
    return events


def run(monkeypatch, events, *, wait=0, seconds=.6, interrupt_at=None, max_bytes=1000000):
    clock = SimpleNamespace(now=0.)
    monkeypatch.setattr(capture.time, 'monotonic', lambda: clock.now)
    socket = ScheduledSocket(clock, events, interrupt_at)
    return capture.record(socket, seconds, wait, .1, lambda *_: None, max_bytes)


def test_waits_beyond_old_timeout_then_retains_tracking_loss(monkeypatch):
    rows, start, elapsed, reason = run(monkeypatch, traffic(301))
    assert start >= 301.1  # Operator can take more than three minutes to get ready.
    assert reason == 'complete' and elapsed >= .6
    assert rows[0][1].startswith(b'manager_state')
    body_states = [msgpack.unpackb(raw[8:], raw=False)['body']
                   for _, raw in rows if raw.startswith(b'tracking')]
    assert {} in body_states and None in body_states  # Faults after start remain in the data.


def test_planner_does_not_start_a_pose_measurement(monkeypatch):
    rows, start, _, reason = run(monkeypatch, traffic(mode=2), wait=.3)
    assert rows == [] and start is None and reason == 'wait_timeout'


def test_interrupt_retains_partial_capture(monkeypatch):
    rows, start, _, reason = run(monkeypatch, traffic(), interrupt_at=.3)
    assert rows and start is not None and reason == 'interrupted'


def test_size_limit_bounds_capture_memory(monkeypatch):
    rows, start, _, reason = run(monkeypatch, traffic(), max_bytes=2000)
    assert start is not None and reason == 'size_limit'
    assert sum(len(raw) for _, raw in rows) <= 2000


def test_offline_analysis_uses_sdk_counters_and_matches_latest_input(tmp_path):
    # Wire-format fixture: poses stay identical while receipt counters advance.
    # The body updates every other tick; the head/controllers update every tick.
    source = (capture.ROOT / 'gear_sonic_deploy/src/g1/g1_deploy_onnx_ref/include/policy_parameters.hpp').read_text()
    import re
    mapping = list(map(int, re.search(r'isaaclab_to_mujoco\s*=\s*\{([^}]+)', source)
                       .group(1).replace(',', ' ').split()))
    rows = []
    for i in range(60):
        stamp = 1 + i * .02
        manager = pack_pose_message({'stream_mode': np.array([1], dtype=np.int32)}, topic='manager_state')
        tracking = {'head': {}, 'left_controller': {}, 'right_controller': {}, 'body': {},
                    'sample_time_ns': int((stamp - .001) * 1e9),
                    'input_update_sequences': dict(head=i+1, left=i+1, right=i+1, body=i//2+1),
                    'body_input_timing': {'body_sequence': i//2+1,
                                         'headset_frame_unix_ns': 1_700_000_000_000_000_000 + (i//2)*40_000_000,
                                         'sdk_receipt_monotonic_ns': int((1 + (i//2)*.04 - .001)*1e9)}}
        indices = np.arange(i-4, i+1, dtype=np.int64)
        joints = np.array([index / 100 + np.arange(29) / 1000 for index in indices])
        pose = pack_pose_message({'frame_index': indices, 'joint_pos': joints,
                                  'timestamp_monotonic': np.array([stamp-.005]),
                                  'prep_timing_ms': np.array([2., 4., .5])})
        feedback = {'body_q_target': joints[1][mapping].tolist()}
        rows.extend(((stamp, manager),
                     (stamp+.001, b'tracking' + msgpack.packb(tracking)),
                     (stamp+.002, pose),
                     (stamp+.010, b'g1_debug' + msgpack.packb(feedback))))
    path = tmp_path / 'capture.msgpack'
    path.write_bytes(msgpack.packb(rows, use_bin_type=True))
    subprocess.run([sys.executable, str(capture.ROOT / 'gear_sonic/scripts/analyze_pico_sim_latency.py'),
                    str(path)], check=True, capture_output=True, text=True)
    result = json.loads((tmp_path / 'capture-components.json').read_text())
    assert result['intervals']['fresh_head']['count'] == 60
    assert result['intervals']['fresh_body']['count'] == 30
    assert result['intervals']['fresh_body']['gap_ms']['p50'] == 40
    assert result['sample_count'] > 50
    assert result['timings']['newest_encoder_sample_receipt_to_feedback_ms']['p50'] == 15
    assert result['tracking_message_timing']['samples'] == 30
    assert result['tracking_message_timing']['headset_message_interval_ms']['p50'] == 40
    assert result['tracking_message_timing']['relative_transit_excess_ms']['max'] <= .001
