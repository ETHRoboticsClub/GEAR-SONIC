#!/usr/bin/env bash
# Standalone, PC-local low-latency SONIC test with keyboard input.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$ROOT/.local/pico-sim/env.sh"
cd "$ROOT"
STATE="$ROOT/.local/sonic-keyboard"
SESSION=sonic-keyboard
TMUX=(tmux -L gear-sonic-keyboard)
export PYTHONUNBUFFERED=1

case "${1:-start}" in
    _simulator)
        exec python gear_sonic/scripts/run_sim_loop.py --interface lo --guard-policy-release \
            --readiness-status-path "$STATE/readiness.json"
        ;;
    _controller)
        cd gear_sonic_deploy
        exec ./target/release/g1_deploy_onnx_ref lo \
            policy/low_latency/model_decoder.onnx reference/example \
            --obs-config policy/low_latency/observation_config.yaml \
            --encoder-file policy/low_latency/model_encoder.onnx \
            --planner-file planner/target_vel/V2/planner_sonic.onnx \
            --input-type keyboard --output-type zmq --disable-crc-check
        ;;
    stop)
        if "${TMUX[@]}" has-session -t "$SESSION" 2>/dev/null; then
            "${TMUX[@]}" kill-session -t "$SESSION"
        fi
        rm -f "$STATE/readiness.json"
        exit
        ;;
    attach)
        exec "${TMUX[@]}" attach-session -t "$SESSION"
        ;;
    status)
        "${TMUX[@]}" list-panes -s -t "$SESSION" \
            -F '#{window_name}: #{?pane_dead,EXITED,RUNNING} #{pane_dead_status}'
        python - "$STATE/readiness.json" <<'PY'
import json
from pathlib import Path
import sys
import time

try:
    status = json.loads(Path(sys.argv[1]).read_text())
    if not 0 <= time.monotonic() - status['sample_time'] < 2:
        raise ValueError('stale simulator status')
except (OSError, ValueError, KeyError):
    raise SystemExit('Simulator is not reporting current readiness yet.')
print('Suspension:', 'ATTACHED' if status['band_enabled'] else 'RELEASED')
print('Pelvis height:', round(status['height'], 3), 'm')
print('Release:', 'READY' if status['ready'] else status['reason'])
PY
        exit
        ;;
    start) ;;
    *) echo "Usage: $0 [start|attach|status|stop]" >&2; exit 2 ;;
esac

if ! "${TMUX[@]}" has-session -t "$SESSION" 2>/dev/null; then
    python - <<'PY'
import socket

with socket.socket() as sock:
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    try:
        sock.bind(('0.0.0.0', 5557))
    except OSError as exc:
        raise SystemExit(f'Controller feedback port 5557 is occupied. Stop the other controller first: {exc}')
PY
    mkdir -p "$STATE/logs"
    archive="$STATE/logs/session-$(date +%Y%m%d-%H%M%S)-$$"
    mkdir -p "$archive"
    for component in simulator controller; do
        if [ -f "$STATE/logs/$component.log" ]; then
            mv "$STATE/logs/$component.log" "$archive/"
        fi
    done
    rm -f "$STATE/readiness.json"
    printf -v entry '%q' "$ROOT/install_scripts/run_sonic_keyboard_sim.sh"
    for component in simulator controller; do
        printf -v log '%q' "$STATE/logs/$component.log"
        printf -v launch '%q' "bash $entry _$component 2>&1 | tee $log"
        if [ "$component" = simulator ]; then
            "${TMUX[@]}" new-session -d -s "$SESSION" -n "$component" "bash -o pipefail -c $launch"
            "${TMUX[@]}" set-window-option -g remain-on-exit on
        else
            "${TMUX[@]}" new-window -t "$SESSION" -n "$component" "bash -o pipefail -c $launch"
        fi
    done
fi
"${TMUX[@]}" select-window -t "$SESSION:controller"
echo "SONIC low-latency keyboard simulation launched. Wait for Init Done."
echo "Controller terminal: ] starts policy; Enter toggles planner; W/S/A/D move; R stops movement; O exits."
echo "MuJoCo window: 9 toggles suspension (release requires active policy feedback)."
echo "Attach: bash install_scripts/run_sonic_keyboard_sim.sh attach"
echo "Stop:   bash install_scripts/run_sonic_keyboard_sim.sh stop"
if command -v gnome-terminal >/dev/null; then
    gnome-terminal --title='SONIC low-latency keyboard control' -- \
        tmux -L gear-sonic-keyboard attach-session -t "$SESSION"
fi
