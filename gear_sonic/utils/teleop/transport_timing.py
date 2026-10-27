"""Passive TCP observations and clock-aware tracking-message timing analysis."""

import json
from pathlib import Path
import re
import subprocess
import threading
import time

import numpy as np

_FILTER = ('( src 10.42.0.0/24 or dst 10.42.0.0/24 or sport = :60061 or dport = :60061 '
           'or sport = :5556 or dport = :5556 or sport = :5558 or dport = :5558 )')


class MessageTransitTracker:
    """Additional message transit relative to the fastest observed delivery."""

    def __init__(self):
        self.previous = None
        self.minimum = None

    def observe(self, timing):
        if not isinstance(timing, dict):
            return None
        source = timing.get('headset_frame_unix_ns')
        receipt = timing.get('sdk_receipt_monotonic_ns')
        if not all(isinstance(value, int) and value > 0 for value in (source, receipt)):
            return None
        if self.previous is not None:
            source_gap = source - self.previous[0]
            receipt_gap = receipt - self.previous[1]
            if source_gap <= 0 or receipt_gap < 0 or abs(source_gap - receipt_gap) > 1_000_000_000:
                self.minimum = None
        self.previous = source, receipt
        difference = receipt - source
        self.minimum = difference if self.minimum is None else min(self.minimum, difference)
        return difference - self.minimum


def parse_tcp_info(text):
    connections, current = [], None
    for line in text.splitlines():
        fields = line.split()
        if len(fields) >= 5 and fields[1].isdigit() and fields[2].isdigit():
            current = {'state': fields[0], 'recv_queue_bytes': int(fields[1]),
                       'send_queue_bytes': int(fields[2]), 'local': fields[3], 'peer': fields[4]}
            current['scope'] = 'hotspot' if any(v.startswith('10.42.') for v in fields[3:5]) else 'loopback'
            connections.append(current)
        elif current is not None:
            for name, pattern, cast in (
                ('rtt_ms', r'\brtt:([\d.]+)', float),
                ('rtt_variation_ms', r'\brtt:[\d.]+/([\d.]+)', float),
                ('retransmissions_total', r'\bretrans:\d+/(\d+)', int),
                ('bytes_retransmitted_total', r'\bbytes_retrans:(\d+)', int),
                ('notsent_bytes', r'\bnotsent:(\d+)', int),
            ):
                match = re.search(pattern, line)
                if match:
                    current[name] = cast(match[1])
    return connections


class TcpMonitor:
    """Run ss off the packet recorder thread; no socket or network settings change."""

    def __init__(self, path, interval=1.):
        self.path = Path(path)
        self.interval = interval
        self.recording = threading.Event()
        self.stopped = threading.Event()
        self.samples = []
        self.thread = threading.Thread(target=self._run, name='tcp-timing', daemon=True)

    def start(self):
        self.thread.start()

    def _run(self):
        while not self.stopped.wait(self.interval):
            if not self.recording.is_set():
                continue
            sample = {'monotonic_ns': time.monotonic_ns()}
            try:
                result = subprocess.run(['ss', '-H', '-t', '-i', '-n', _FILTER],
                                        capture_output=True, text=True, timeout=.75)
                sample['connections'] = parse_tcp_info(result.stdout)
                if result.returncode:
                    sample['error'] = result.stderr.strip()
            except (OSError, subprocess.TimeoutExpired) as exc:
                sample['error'] = str(exc)
            sample['observation_work_ms'] = (time.monotonic_ns() - sample['monotonic_ns']) / 1e6
            self.samples.append(sample)

    def close(self):
        self.recording.clear()
        self.stopped.set()
        self.thread.join(timeout=2)
        with self.path.open('w') as stream:
            for sample in self.samples:
                stream.write(json.dumps(sample) + '\n')

    def summary(self):
        result = {'samples': len(self.samples), 'errors': sum('error' in s for s in self.samples),
                  'note': 'TCP RTT is round trip, not tracking-message one-way latency. '
                          'Queue sizes and RTT are sampled snapshots; they can miss short stalls.'}
        for scope in ('hotspot', 'loopback'):
            rows = [c for s in self.samples for c in s.get('connections', []) if c['scope'] == scope]
            rtt = [c['rtt_ms'] for c in rows if 'rtt_ms' in c]
            result[scope] = {'connection_observations': len(rows), 'rtt_ms': stats(rtt),
                             'max_recv_queue_bytes': max((c['recv_queue_bytes'] for c in rows), default=0),
                             'max_send_queue_bytes': max((c['send_queue_bytes'] for c in rows), default=0)}
            connections = {}
            for row in rows:
                key = (row['local'], row['peer'])
                count = row.get('retransmissions_total', 0)
                if key not in connections:
                    connections[key] = [count, count]
                else:
                    connections[key][1] = max(connections[key][1], count)
            result[scope]['retransmissions_between_observations'] = sum(
                high - initial for initial, high in connections.values())
        return result


def stats(values):
    return dict(zip(('p50', 'p95', 'max'), np.percentile(values, [50, 95, 100]).round(3).tolist())) if len(values) else None


def summarize_message_timing(samples):
    """Source wall clock vs PC callback clocks, without assuming clock agreement."""
    result = {'samples': len(samples), 'note':
              'Headset timestamps mark tracking-message preparation, not sensor capture. '
              'Raw wall-clock differences include unknown clock offset. Relative transit '
              'excess subtracts the smallest observed difference and assumes no clock step; '
              'it cannot measure constant one-way delay or replace clock synchronization.'}
    if len(samples) < 2:
        return result
    source = np.array([s['headset_frame_unix_ns'] for s in samples], dtype=np.int64)
    receipt = np.array([s['sdk_receipt_monotonic_ns'] for s in samples], dtype=np.int64)
    source_gaps = np.diff(source) / 1e6
    receipt_gaps = np.diff(receipt) / 1e6
    result['headset_message_interval_ms'] = stats(source_gaps)
    result['sdk_receipt_interval_ms'] = stats(receipt_gaps)
    result['clock_discontinuity_or_long_pause'] = bool(
        np.any(source_gaps <= 0) or np.any(receipt_gaps < 0)
        or np.any(np.abs(source_gaps - receipt_gaps) > 1000))
    if not result['clock_discontinuity_or_long_pause']:
        differences = receipt - source
        excess = (differences - differences.min()) / 1e6
        result['relative_transit_excess_ms'] = stats(excess)
    return result
