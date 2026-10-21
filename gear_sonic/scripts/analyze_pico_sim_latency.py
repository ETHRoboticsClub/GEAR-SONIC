"""Match reference feedback to newest encoder input; report capture timings."""
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import re

import msgpack
import numpy as np
from scipy.spatial import cKDTree

from gear_sonic.tests.teleop.fakes import decode

parser = argparse.ArgumentParser()
parser.add_argument('capture', type=Path)
args = parser.parse_args()
root = Path(__file__).resolve().parents[2]
rows = msgpack.unpackb(args.capture.read_bytes(), raw=False)
if not rows:
    raise SystemExit('Capture is empty.')
capture_seconds = rows[-1][0] - rows[0][0]
# Frame indices restart on POSE entry: never match across separate sessions.
segments, mode, start = [], None, None
for index, (_, raw) in enumerate(rows):
    if raw.startswith(b'manager_state'):
        _, fields = decode(raw, b'manager_state')
        new_mode = int(fields['stream_mode'][0])
        if new_mode != mode:
            if mode == 1:
                segments.append((start, index))
            start = index if new_mode == 1 else None
            mode = new_mode
if mode == 1:
    segments.append((start, len(rows)))
if not segments:
    raise SystemExit('No POSE interval in capture.')
first, last = max(segments, key=lambda pair: rows[pair[1]-1][0] - rows[pair[0]][0])
rows = rows[first:last]
source = (root / 'gear_sonic_deploy/src/g1/g1_deploy_onnx_ref/include/policy_parameters.hpp').read_text()
mapping = list(map(int, re.search(r'isaaclab_to_mujoco\s*=\s*\{([^}]+)', source).group(1).replace(',', ' ').split()))
frames, feedback = {}, []
timings, arrivals = defaultdict(list), defaultdict(list)
modes = Counter()
last_body_sequence = None
for t, raw in rows:
    if raw.startswith(b'pose'):
        arrivals['pose'].append(t)
        _, f = decode(raw, b'pose')
        receipt = float(f['timestamp_monotonic'][0])
        for j, index in enumerate(f['frame_index']):
            frames.setdefault(int(index), (t, f['joint_pos'][j][mapping].copy(), receipt))
        if 'prep_timing_ms' in f:
            smpl, preparation, queue = map(float, f['prep_timing_ms'])
            timings['smpl_mapping_ms'].append(smpl)
            timings['pose_preparation_ms'].append(preparation)
            timings['other_pose_preparation_ms'].append(preparation-smpl)
            timings['reader_to_mapping_start_ms'].append(queue)
            timings['packing_transport_monitor_ms'].append((t-receipt)*1000-preparation-queue)
    elif raw.startswith(b'g1_debug'):
        arrivals['policy_feedback'].append(t)
        f = msgpack.unpackb(raw[8:], raw=False)
        feedback.append((t, np.asarray(f['body_q_target'])))
    elif raw.startswith(b'manager_state'):
        _, f = decode(raw, b'manager_state')
        modes[int(f['stream_mode'][0])] += 1
    elif raw.startswith(b'tracking'):
        f = msgpack.unpackb(raw[8:], raw=False)
        sequence = f.get('body_update_sequence')
        if f.get('body') is not None and sequence is not None and sequence != last_body_sequence:
            arrivals['fresh_body'].append(t)
            last_body_sequence = sequence
counts = Counter(tuple(x[1]) for x in frames.values())
ids = [i for i, x in frames.items() if counts[tuple(x[1])] == 1]
matched = 0
if ids and feedback and set(modes) == {1}:
    distances, indices = cKDTree([frames[i][1] for i in ids]).query([x[1] for x in feedback])
    for (t, _), distance, j in zip(feedback, distances, indices):
        i = ids[j]
        if distance > 1e-9 or i <= min(frames)+5 or i+3 not in frames:
            continue
        old, newest = frames[i], frames[i+3]
        if newest[0] > t:
            continue
        matched += 1
        timings['reference_publication_to_complete_window_ms'].append((newest[0]-old[0])*1000)
        timings['complete_window_publication_to_feedback_ms'].append((t-newest[0])*1000)
        timings['newest_encoder_sample_receipt_to_feedback_ms'].append((t-newest[2])*1000)
        timings['newest_encoder_sample_receipt_to_publication_ms'].append((newest[0]-newest[2])*1000)
        timings['oldest_reference_receipt_to_feedback_ms'].append((t-old[2])*1000)

def stats(values):
    return dict(zip(('p50','p95','max'), np.percentile(values,[50,95,100]).round(3).tolist())) if len(values) else None

summary = {'sample_count': matched, 'feedback_total': len(feedback), 'modes': dict(modes),
           'capture_seconds': capture_seconds, 'pose_segments': len(segments),
           'selected_pose_seconds': rows[-1][0] - rows[0][0],
           'timings': {k:stats(v) for k,v in timings.items()}, 'intervals': {}}
for key, times in arrivals.items():
    gaps = np.diff(times)*1000
    summary['intervals'][key] = {'count': len(times), 'gap_ms': stats(gaps),
        'percent_over_20ms': round(float(np.mean(gaps>20)*100),3),
        'percent_over_30ms': round(float(np.mean(gaps>30)*100),3)}
summary['note'] = ('Receipt is PC manager receipt, not headset capture. Feedback is received after action calculation; '
                   'does not measure DDS delivery or visible response. Pose preparation includes SMPL; do not add them. '
                   'SMPL statistics cover published packets. Modes other than POSE disable reference matching.')
output = args.capture.with_name(args.capture.stem+'-components.json')
output.write_text(json.dumps(summary,indent=2)+'\n')
print(json.dumps(summary,indent=2))
