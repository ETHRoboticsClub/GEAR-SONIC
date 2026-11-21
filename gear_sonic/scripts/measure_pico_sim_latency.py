"""Record a live POSE session and compare frame age with the previous capture."""
import argparse
from collections import Counter, defaultdict
from datetime import datetime
import json
from pathlib import Path
import re
import time

import msgpack
import numpy as np
from scipy.spatial import cKDTree
import zmq
from gear_sonic.tests.teleop.fakes import decode
from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelSubscriber
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowState_

parser = argparse.ArgumentParser()
parser.add_argument('--seconds', type=float, default=60)
parser.add_argument('--wait', type=float, default=180)
parser.add_argument('--name', default=datetime.now().strftime('pico-latency-%Y%m%d-%H%M%S'))
args = parser.parse_args()
root = Path(__file__).resolve().parents[2]
base = root / '.local/pico-sim/logs' / args.name
base.parent.mkdir(parents=True, exist_ok=True)
if base.with_suffix('.msgpack').exists() or base.with_suffix('.json').exists():
    raise SystemExit('Capture name already exists; choose a new --name to preserve previous data.')
context = zmq.Context()
socket = context.socket(zmq.SUB)
socket.setsockopt(zmq.SUBSCRIBE, b'')
for port in (5556, 5558):
    socket.connect(f'tcp://127.0.0.1:{port}')
lowstate = []
ChannelFactoryInitialize(0, 'lo')
subscriber = ChannelSubscriber('rt/lowstate', LowState_)
subscriber.Init(lambda m: lowstate.append((time.monotonic(), m.tick)), 10)
rows = []
times = defaultdict(list)
changed = defaultdict(list)
previous = {}
tracking = Counter()
modes = Counter()
relay_age = []
start = None
mode = None
all_tracked = False
deadline = time.monotonic() + args.wait
print('Waiting for POSE mode with head, controllers and body tracked...', flush=True)
while True:
    now = time.monotonic()
    if start is None and now > deadline:
        raise SystemExit('POSE test has not started; measurement is ready to rerun.')
    if start is not None and now - start >= args.seconds:
        break
    if not socket.poll(100):
        continue
    raw = socket.recv()
    now = time.monotonic()
    for topic in (b'tracking', b'manager_state', b'pose', b'g1_debug', b'robot_config'):
        if raw.startswith(topic):
            break
    else:
        continue
    if topic == b'manager_state':
        _, fields = decode(raw, topic)
        mode = int(fields['stream_mode'][0])
    elif topic == b'tracking':
        frame = msgpack.unpackb(raw[len(topic):], raw=False)
        status = tuple(frame.get(k) is not None for k in ('head', 'left_controller', 'right_controller', 'body'))
        all_tracked = all(status)
    if start is None:
        if mode != 1 or not all_tracked:
            continue
        start = now
        print(f'Recording {args.seconds:g} seconds of live teleoperation.', flush=True)
    rows.append((now, raw))
    times[topic.decode()].append(now)
    if topic == b'manager_state':
        modes[str(mode)] += 1
    elif topic == b'tracking':
        tracking[str(status)] += 1
        relay_age.append((now - frame['sample_time_ns'] / 1e9) * 1000)
        for key in ('head', 'left_controller', 'right_controller', 'body'):
            value = frame.get(key)
            if value is not None and value != previous.get(key):
                changed[key].append(now)
            previous[key] = value
elapsed = time.monotonic() - start
percentiles = lambda x: dict(zip(('p50', 'p95', 'max'), np.percentile(x, [50, 95, 100]).round(3).tolist())) if len(x) else None
summary = {
    'seconds': elapsed,
    'rates_hz': {k: round(len(v)/elapsed, 2) for k, v in times.items()},
    'modes': dict(modes), 'tracking': dict(tracking),
    'changed_rates_hz': {k: round(len(v)/elapsed, 2) for k, v in changed.items()},
    'changed_gap_ms': {k: percentiles(np.diff(v)*1000) for k, v in changed.items()},
    'relay_age_ms': percentiles(relay_age),
}
a = [x for x in lowstate if x[0] >= start]
if len(a) > 1:
    summary['sim_realtime_factor'] = (a[-1][1]-a[0][1])/1000/(a[-1][0]-a[0][0])
with base.with_suffix('.msgpack').open('wb') as f:
    msgpack.pack(rows, f, use_bin_type=True)
# Match unique reference vectors, excluding duplicates and the initial boundary.
source = (root / 'gear_sonic_deploy/src/g1/g1_deploy_onnx_ref/include/policy_parameters.hpp').read_text()
mapping = list(map(int, re.search(r'isaaclab_to_mujoco\s*=\s*\{([^}]+)', source).group(1).replace(',', ' ').split()))
frames = {}
feedback = []
processing = []
latest = -1
for t, raw in rows:
    if raw.startswith(b'pose'):
        _, f = decode(raw, b'pose')
        latest = int(f['frame_index'][-1])
        receipt = float(f['timestamp_monotonic'][0])
        processing.append((t-receipt)*1000)
        for j, index in enumerate(f['frame_index']):
            stamp = f.get('frame_timestamp_ns')
            frames.setdefault(int(index), (t, f['joint_pos'][j][mapping].copy(), receipt,
                                           float(stamp[j])/1e9 if stamp is not None else None))
    elif raw.startswith(b'g1_debug'):
        f = msgpack.unpackb(raw[8:], raw=False)
        feedback.append((t, np.asarray(f['body_q_target']), latest))
counts = Counter(tuple(x[1]) for x in frames.values())
ids = [i for i, x in frames.items() if counts[tuple(x[1])] == 1]
if ids and feedback and set(modes) == {'1'}:
    distances, indices = cKDTree([frames[i][1] for i in ids]).query([x[1] for x in feedback])
    publication_age, receipt_age, reference_age, lag = [], [], [], []
    for row, distance, j in zip(feedback, distances, indices):
        index = ids[j]
        if distance > 1e-9 or index <= min(frames)+5:
            continue
        publication_age.append((row[0]-frames[index][0])*1000)
        receipt_age.append((row[0]-frames[index][2])*1000)
        if frames[index][3] is not None:
            reference_age.append((row[0]-frames[index][3])*1000)
        lag.append(row[2]-index)
    summary.update({
        'matched_feedback': len(lag), 'feedback_total': len(feedback),
        'manager_receipt_to_publication_ms': percentiles(processing),
        'pose_publication_to_current_reference_ms': percentiles(publication_age),
        'manager_receipt_to_current_reference_ms': percentiles(receipt_age),
        'resampled_reference_time_to_feedback_ms': percentiles(reference_age),
        'latest_published_minus_reference_frames': percentiles(lag),
        'note': 'Reference ages, not human-motion-to-visible-robot latency. Encoder also reads three later frames.',
    })
if set(modes) != {'1'}:
    summary['note'] = ('Capture includes mode changes. Reference indices can restart; '
                       'run analyze_pico_sim_latency.py to select the longest continuous POSE interval.')
base.with_suffix('.json').write_text(json.dumps(summary, indent=2)+'\n')
print(json.dumps(summary, indent=2), flush=True)
