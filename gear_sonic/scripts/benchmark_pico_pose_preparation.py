"""Replay saved tracking through old/new pose preparation, without running a sim."""

import argparse
import importlib.util
import json
from pathlib import Path
import time

import msgpack
import numpy as np

from gear_sonic.scripts import pico_manager_thread_server as current
from gear_sonic.tests.teleop.fakes import decode
from gear_sonic.utils.teleop.input_readers import TrackingReader


class Sink:
    def __init__(self):
        self.messages = []

    def send(self, raw):
        self.messages.append(raw)


def run(module, frames):
    clock = [frames[0]['sample_time_ns']]
    reader = TrackingReader(clock=lambda: clock[0])
    sink = Sink()
    three_point = module.ThreePointPose()
    streamer = module.PoseStreamer(sink, reader, three_point, 5, 50, False, '', 'npz')
    reader._update(frames[0], clock[0])
    three_point.calibrate_now(reader.get_latest()['body_poses_np'])
    elapsed = []
    for frame in frames:
        clock[0] = frame['sample_time_ns']
        reader._update(frame, clock[0])
        start = time.perf_counter_ns()
        streamer.run_once()
        elapsed.append((time.perf_counter_ns() - start) / 1e6)
    three_point.close()
    return [decode(raw, b'pose')[1] for raw in sink.messages], elapsed


def stats(values):
    return dict(zip(('p50', 'p95', 'max'), np.percentile(values, [50, 95, 100]).tolist()))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--capture', type=Path, required=True)
    parser.add_argument('--baseline-script', type=Path, required=True,
                        help='Trusted saved pre-change pico_manager_thread_server.py')
    parser.add_argument('--frames', type=int, default=600)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Output exists; choose another name to preserve results.')
    spec = importlib.util.spec_from_file_location('pico_baseline', args.baseline_script)
    baseline = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(baseline)
    rows = msgpack.unpackb(args.capture.read_bytes(), raw=False)
    frames = []
    for _, raw in rows:
        if raw.startswith(b'tracking'):
            frame = msgpack.unpackb(raw[8:], raw=False)
            if frame.get('body') is not None:
                frames.append(frame)
                if len(frames) == args.frames:
                    break
    if len(frames) < 20:
        parser.error('Need at least 20 tracking frames.')
    # Warm both implementations, then bracket the comparison to expose order effects.
    run(baseline, frames[:30])
    run(current, frames[:30])
    runs, reference = [], None
    errors = {}
    for name, module in [('before', baseline), ('after', current), ('after', current), ('before', baseline)]:
        packets, times = run(module, frames)
        if reference is None:
            reference = packets
        if len(reference) != len(packets):
            raise AssertionError('Publication count changed')
        for original, actual in zip(reference, packets):
            np.testing.assert_array_equal(original['frame_index'], actual['frame_index'])
            np.testing.assert_array_equal(original['frame_timestamp_ns'], actual['frame_timestamp_ns'])
            for key in ('smpl_pose', 'smpl_joints', 'body_quat_w', 'joint_pos', 'vr_position',
                        'vr_orientation', 'left_hand_joints', 'right_hand_joints', 'heading_increment'):
                error = float(np.max(np.abs(original[key] - actual[key])))
                errors[key] = max(errors.get(key, 0), error)
                np.testing.assert_allclose(original[key], actual[key], atol=2e-5, rtol=1e-5)
        runs.append({'implementation': name, 'published_packets': len(packets),
                     'run_once_ms': stats(times[10:]),
                     'preparation_ms': stats([p['prep_timing_ms'][1] for p in packets[5:]]),
                     'mapping_ms': stats([p['prep_timing_ms'][0] for p in packets[5:]])})
    result = {'tracking_frames': len(frames), 'capture': str(args.capture),
              'baseline_script': str(args.baseline_script), 'runs': runs,
              'output_max_absolute_errors': errors,
              'note': 'Offline CPU replay with calibrated 3-point processing and no sleep. '
                      'Not a live end-to-end latency measurement. Model context and reference timestamps stay unchanged.'}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
