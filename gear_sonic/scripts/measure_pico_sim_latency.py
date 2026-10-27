"""Arm a bounded PC-local capture, then record once live POSE input is ready.

The raw list of (PC monotonic receipt time, wire bytes) is compatible with
analyze_pico_sim_latency.py. This observer never sends robot commands.
"""

import argparse
from collections import Counter, deque
from datetime import datetime
import hashlib
import importlib.util
import json
from pathlib import Path
import signal
import shutil
import subprocess
import time

import msgpack
import zmq

from gear_sonic.tests.teleop.fakes import decode
from gear_sonic.utils.teleop.transport_timing import TcpMonitor

ROOT = Path(__file__).resolve().parents[2]
TOPICS = (b'tracking', b'manager_state', b'pose', b'g1_debug', b'robot_config')


def write_json(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def command(*args):
    try:
        result = subprocess.run(args, cwd=ROOT, capture_output=True, text=True, timeout=3)
        return result.stdout.strip() if result.returncode == 0 else None
    except (OSError, subprocess.TimeoutExpired):
        return None


def metadata():
    binary = ROOT / 'gear_sonic_deploy/target/release/g1_deploy_onnx_ref'
    digest = hashlib.sha256()
    if binary.is_file():
        with binary.open('rb') as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b''):
                digest.update(chunk)
    sources = ['gear_sonic/scripts/pico_manager_thread_server.py',
               'gear_sonic/scripts/pico_sim_bridge.py', 'gear_sonic/utils/teleop/smpl_cpu.py',
               'gear_sonic/utils/teleop/input_readers.py', 'gear_sonic/utils/teleop/tracking_timing.py',
               'gear_sonic/utils/teleop/transport_timing.py']
    fingerprints = {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
                    for name in sources if (ROOT / name).is_file()}
    sdk = importlib.util.find_spec('xrobotoolkit_sdk')
    if sdk is not None and sdk.origin and Path(sdk.origin).is_file():
        fingerprints[sdk.origin] = hashlib.sha256(Path(sdk.origin).read_bytes()).hexdigest()
    return {
        'created_at': datetime.now().astimezone().isoformat(),
        'branch': command('git', 'branch', '--show-current'),
        'commit': command('git', 'rev-parse', 'HEAD'),
        'worktree_status': command('git', 'status', '--short'),
        'controller_sha256': digest.hexdigest() if binary.is_file() else None,
        'runtime_file_sha256': fingerprints,
        'hotspot_band_channel': command('nmcli', '-g',
                                       '802-11-wireless.band,802-11-wireless.channel',
                                       'connection', 'show', 'teleop-sim'),
        'expected_launcher_configuration': 'policy/low_latency; 4 reference frames; 50 Hz policy',
        'clock': 'PC monotonic receipt; not headset capture time',
    }


def record(socket, seconds, wait, warmup, report, max_bytes=256 * 1024 * 1024):
    """Wait without retaining traffic; retain a finite capture, including faults."""
    rows = []
    armed_at = time.monotonic()
    start = warm_since = None
    mode = None
    tracked = False
    received = {}
    byte_count = 0
    reason = 'complete'
    last_report, last_state = float('-inf'), None
    try:
        while True:
            now = time.monotonic()
            if start is not None and now - start >= seconds:
                break
            if start is None and wait > 0 and now - armed_at >= wait:
                reason = 'wait_timeout'
                break
            ready = (mode == 1 and tracked and all(
                0 <= now - received.get(topic, float('-inf')) <= .2
                for topic in (b'tracking', b'manager_state', b'pose', b'g1_debug')
            ))
            if start is None:
                warm_since = (now if warm_since is None else warm_since) if ready else None
            state = 'recording' if start is not None else ('warming_up' if ready else 'waiting')
            if state != last_state or now - last_report >= 1:
                report(state, 0 if start is None else now - start)
                last_report, last_state = now, state
            if not socket.poll(50):
                continue
            raw = socket.recv()
            now = time.monotonic()
            topic = next((topic for topic in TOPICS if raw.startswith(topic)), None)
            if topic is None:
                continue
            try:
                if topic == b'manager_state':
                    _, fields = decode(raw, topic)
                    mode = int(fields['stream_mode'][0])
                elif topic == b'tracking':
                    frame = msgpack.unpackb(raw[len(topic):], raw=False)
                    tracked = all(frame.get(key) is not None for key in
                                  ('head', 'left_controller', 'right_controller', 'body'))
            except (ValueError, KeyError, TypeError, AssertionError, msgpack.UnpackException):
                continue
            received[topic] = now
            if start is None:
                # Start on a fresh POSE marker so offline segmentation has context.
                if topic != b'manager_state' or mode != 1 or warm_since is None:
                    continue
                if now - warm_since < warmup:
                    continue
                if not tracked or any(now - received.get(key, float('-inf')) > .2
                                      for key in (b'tracking', b'pose', b'g1_debug')):
                    continue
                start = now
                print(f'Recording {seconds:g} seconds. Keep POSE active and move gently.', flush=True)
            if byte_count + len(raw) > max_bytes:
                reason = 'size_limit'
                break
            rows.append((now, raw))
            byte_count += len(raw)
    except KeyboardInterrupt:
        reason = 'interrupted'
    elapsed = 0 if start is None else time.monotonic() - start
    return rows, start, elapsed, reason


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seconds', type=float, default=60)
    parser.add_argument('--wait', type=float, default=0, help='Wait limit; 0 keeps the recorder armed.')
    parser.add_argument('--warmup', type=float, default=5, help='Seconds of live POSE before capture.')
    parser.add_argument('--name', default=datetime.now().strftime('pico-latency-%Y%m%d-%H%M%S-%f'))
    parser.add_argument('--output-dir', type=Path, default=ROOT / '.local/pico-sim/logs')
    parser.add_argument('--pose-port', type=int, default=5556)
    parser.add_argument('--feedback-port', type=int, default=5558)
    parser.add_argument('--no-dds', action='store_true', help='Omit simulation speed sampling.')
    args = parser.parse_args()
    if args.seconds <= 0 or args.wait < 0 or args.warmup < 0:
        parser.error('seconds must be positive; wait and warmup must be nonnegative')
    if Path(args.name).name != args.name or '.' in args.name:
        parser.error('name must be a filename stem without periods')
    args.output_dir.mkdir(parents=True, exist_ok=True)
    base = args.output_dir / args.name
    if base.with_suffix('.msgpack').exists() or base.with_suffix('.json').exists():
        parser.error('Capture name already exists; choose a new name to preserve previous data.')
    info = {**metadata(), 'name': args.name, 'state': 'waiting',
            'requested_seconds': args.seconds, 'warmup_seconds': args.warmup,
            'raw_capture': str(base.with_suffix('.msgpack').resolve()),
            'summary': str(base.with_suffix('.json').resolve()),
            'components': str(base.with_name(base.name + '-components.json').resolve())}
    latest = args.output_dir / 'latest-latency-capture.json'
    write_json(base.with_suffix('.json'), info)
    transport = TcpMonitor(base.with_name(base.name + '-transport.jsonl'))

    def report(state, elapsed):
        if state == 'recording':
            transport.recording.set()
        else:
            transport.recording.clear()
        info.update(state=state, elapsed_seconds=round(elapsed, 1))
        write_json(latest, info)

    report('waiting', 0)
    print('Armed: waiting for live POSE, tracked devices, and policy feedback. '
          f'Warm-up {args.warmup:g}s; capture {args.seconds:g}s.', flush=True)
    print(f'Capture prefix: {base}', flush=True)
    context = zmq.Context()
    socket = context.socket(zmq.SUB)
    socket.setsockopt(zmq.RCVHWM, 10000)
    for topic in TOPICS:
        socket.setsockopt(zmq.SUBSCRIBE, topic)
    for port in (args.pose_port, args.feedback_port):
        socket.connect(f'tcp://127.0.0.1:{port}')
    lowstate = deque(maxlen=max(1000, int((args.seconds + 5) * 1000)))
    subscriber = None
    previous_handler = signal.getsignal(signal.SIGTERM)
    previous_hup_handler = signal.getsignal(signal.SIGHUP)

    def interrupted(signum, frame):
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGHUP, interrupted)
    transport.start()
    try:
        if not args.no_dds:
            from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelSubscriber
            from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowState_

            ChannelFactoryInitialize(0, 'lo')
            subscriber = ChannelSubscriber('rt/lowstate', LowState_)
            subscriber.Init(lambda message: lowstate.append((time.monotonic(), message.tick)), 10)
        rows, start, elapsed, reason = record(socket, args.seconds, args.wait, args.warmup, report)
    finally:
        transport.close()
        if subscriber is not None:
            subscriber.Close()
        socket.close(linger=0)
        context.term()
        signal.signal(signal.SIGTERM, previous_handler)
        signal.signal(signal.SIGHUP, previous_hup_handler)
    if rows:
        with base.with_suffix('.msgpack').open('wb') as stream:
            msgpack.pack(rows, stream, use_bin_type=True)
    counts = Counter(next(topic.decode() for topic in TOPICS if raw.startswith(topic))
                     for _, raw in rows)
    info.update(state='complete' if reason == 'complete' else ('partial' if rows else reason),
                stop_reason=reason, elapsed_seconds=round(elapsed, 3), packets=len(rows),
                rates_hz={key: round(count / elapsed, 2) for key, count in counts.items()},
                note='Run analyze_pico_sim_latency.py on the raw capture for latency components. '
                     'PC receipt to feedback excludes headset capture and visible robot response.')
    info['transport_capture'] = str(transport.path.resolve())
    info['tcp_transport_summary'] = transport.summary()
    samples = [sample for sample in lowstate
               if start is not None and start <= sample[0] <= start + elapsed]
    if samples:
        lowstate_path = base.with_name(base.name + '-lowstate.msgpack')
        with lowstate_path.open('wb') as stream:
            msgpack.pack(samples, stream, use_bin_type=True)
        info['lowstate_capture'] = str(lowstate_path.resolve())
    if len(samples) > 1 and samples[-1][0] > samples[0][0]:
        info['sim_realtime_factor'] = ((samples[-1][1] - samples[0][1]) / 1000
                                       / (samples[-1][0] - samples[0][0]))
    if rows:
        log_dir = base.with_name(base.name + '-logs')
        log_dir.mkdir(exist_ok=True)
        for component in ('controller', 'manager', 'simulator', 'bridge', 'service'):
            source = args.output_dir / (component + '.log')
            if source.is_file():
                shutil.copy2(source, log_dir / source.name)
        info['component_logs'] = str(log_dir.resolve())
    write_json(base.with_suffix('.json'), info)
    write_json(latest, info)
    print(json.dumps(info, indent=2), flush=True)
    return 0 if rows else 1


if __name__ == '__main__':
    raise SystemExit(main())
