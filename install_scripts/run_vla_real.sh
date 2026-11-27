#!/usr/bin/env bash
# A fine-tuned GR00T VLA driving the G1 through SONIC, the policy served elsewhere.
#
#   GR00T_HOST=<server> bash install_scripts/run_vla_real.sh check|start|attach|status|stop
#
# Onboard: the low-latency SONIC controller (run_pico_real.sh's, unchanged), the head camera
# as the recorder saw it (run_vla_camera.py), run_vla_inference.py sending JPEG frames to
# the GR00T server (serve_gr00t_jpeg.py) and publishing its motion tokens to the controller,
# and a keyboard pane for that client. Nothing here starts the policy: in the keyboard pane,
# `k` starts the control loop in PLANNER mode (standing) or stops it, `i` sends the initial
# pose and switches to POSE, `p` resumes the policy (it starts paused) or pauses it, and
# `t <text>` changes the prompt. In the controller pane, `o` is its emergency stop. Environment: GR00T_HOST (required), GR00T_PORT (5550), VLA_PROMPT (the task text),
# VLA_CAMERA (oakd, oakd:<mxid> or a ZED's /dev/videoN).
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
SESSION=vla-real
TMUX=(tmux -L gear-sonic-real)
LOG_DIR="$ROOT/.local/vla-real/logs"
TELEOP_PY="$ROOT/.venv_teleop/bin/python"
CAMERA_PY="${VLA_CAMERA_PYTHON:-$HOME/.venvs/g1-record/bin/python}"
GR00T_PORT="${GR00T_PORT:-5550}"
VLA_PROMPT="${VLA_PROMPT:-grasp the bottle}"
VLA_CAMERA="${VLA_CAMERA:-oakd}"
COMPONENTS=(controller camera vla keyboard)

pane_of() {
    "${TMUX[@]}" list-panes -s -t "$SESSION" -F '#{pane_id} #{window_name}' |
        awk -v name="$1" '$2 == name {print $1; exit}'
}

check() {
    [[ "$(uname -m)" == aarch64 ]] || { echo 'Run this launcher onboard the G1.' >&2; return 1; }
    [[ -n "${GR00T_HOST:-}" ]] || { echo 'Set GR00T_HOST to the GR00T server (or its tunnel).' >&2; return 1; }
    if [[ -n "$(docker ps -q --filter label=com.docker.compose.project=ethrc-robot)" ]]; then
        echo 'ETHRC containers are still running: cd ~/ETHRC-Humanoid-Controls && just down' >&2; return 1
    fi
    if "${TMUX[@]}" has-session -t pico-real 2>/dev/null; then
        echo 'PICO teleop (pico-real) is running: bash install_scripts/run_pico_real.sh stop' >&2; return 1
    fi
    for file in model_encoder.onnx model_decoder.onnx observation_config.yaml; do
        test -s "gear_sonic_deploy/policy/low_latency/$file" || { echo "missing low_latency/$file" >&2; return 1; }
    done
    test -s gear_sonic_deploy/planner/target_vel/V2/planner_sonic.onnx
    test -x gear_sonic_deploy/target/release/g1_deploy_onnx_ref || { echo 'Build the controller first.' >&2; return 1; }
    "$CAMERA_PY" -c 'import depthai, zmq, msgpack, cv2; import gear_sonic.data.record.oak_mjpeg' ||
        { echo "The camera needs $CAMERA_PY with depthai, pyzmq, msgpack and opencv (the recorder venv)." >&2; return 1; }
    "$TELEOP_PY" - <<'PY'
import os
import socket

import gear_sonic.scripts.run_vla_inference  # noqa: F401  # pinocchio, tyro, scipy, opencv, zmq
from gear_sonic.utils.inference.jpeg_policy import JpegPolicyClient

for port in (5555, 5556, 5557, 5580):
    with socket.socket() as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind(("0.0.0.0", port))
host, port = os.environ["GR00T_HOST"], int(os.environ.get("GR00T_PORT", "5550"))
client = JpegPolicyClient(host, port, timeout_ms=5000)
reachable = client.ping()
client.close()
assert reachable, f"no GR00T server answers at {host}:{port}"
print(f"Preflight passed: models, imports, ports, and the GR00T server at {host}:{port}.")
PY
}

case "${1:-help}" in
    _controller) exec bash "$ROOT/install_scripts/run_pico_real.sh" _controller ;;
    _camera) exec "$CAMERA_PY" gear_sonic/scripts/run_vla_camera.py --camera "$VLA_CAMERA" ;;
    _vla)
        exec "$TELEOP_PY" gear_sonic/scripts/run_vla_inference.py --host "$GR00T_HOST" \
            --port "$GR00T_PORT" --image-codec jpeg --prompt "$VLA_PROMPT"
        ;;
    _keyboard)
        exec "$TELEOP_PY" - <<'PY'
import time

import zmq

from gear_sonic.utils.data_collection.keyboard_subscriber import DEFAULT_ZMQ_KEYBOARD_PORT

socket = zmq.Context().socket(zmq.PUB)
socket.bind(f"tcp://127.0.0.1:{DEFAULT_ZMQ_KEYBOARD_PORT}")
time.sleep(0.5)
print("Keys, then Enter: k start (standing) / stop the control loop, i initial pose (POSE),")
print("p resume / pause the policy, [ or ] a hand open/closed in the initial pose, t <text> prompt.")
while True:
    key = input().strip()
    socket.send_string("prompt:" + key[2:] if key.startswith("t ") else key)
    print("sent", key)
PY
        ;;
    check) check; exit ;;
    stop)
        if "${TMUX[@]}" has-session -t "$SESSION" 2>/dev/null; then
            pane="$(pane_of controller)"
            if [[ -n "$pane" && "$("${TMUX[@]}" display-message -p -t "$pane" '#{pane_dead}')" == 0 ]]; then
                # The controller's own stop path, which publishes damping.
                "${TMUX[@]}" send-keys -t "$pane" o
                for ((attempt=0; attempt<100; attempt++)); do
                    [[ "$("${TMUX[@]}" display-message -p -t "$pane" '#{pane_dead}')" != 0 ]] && break
                    sleep 0.1
                done
            fi
            "${TMUX[@]}" kill-session -t "$SESSION"
        fi
        exit ;;
    attach) exec "${TMUX[@]}" attach-session -t "$SESSION" ;;
    status)
        "${TMUX[@]}" list-panes -s -t "$SESSION" -F '#{window_name}: #{?pane_dead,EXITED,RUNNING} #{pane_dead_status}'
        exit ;;
    start) ;;
    *) echo "Usage: GR00T_HOST=<server> $0 [check|start|attach|status|stop]"; exit ;;
esac

if "${TMUX[@]}" has-session -t "$SESSION" 2>/dev/null; then
    echo 'vla-real already exists. Use status/attach, or stop before restarting.' >&2
    exit 1
fi
check
mkdir -p "$LOG_DIR"
printf -v entry '%q' "$ROOT/install_scripts/run_vla_real.sh"
env_vars="GR00T_HOST=$(printf %q "$GR00T_HOST") GR00T_PORT=$GR00T_PORT VLA_PROMPT=$(printf %q "$VLA_PROMPT") VLA_CAMERA=$(printf %q "$VLA_CAMERA")"
"${TMUX[@]}" new-session -d -s "$SESSION" -n camera "$env_vars bash $entry _camera"
trap '"${TMUX[@]}" kill-session -t "$SESSION" 2>/dev/null || true' ERR
"${TMUX[@]}" set-option -g remain-on-exit on
for component in controller vla keyboard; do
    "${TMUX[@]}" new-window -t "$SESSION" -n "$component" "$env_vars bash $entry _$component"
done
for window in "${COMPONENTS[@]}"; do
    printf -v log '%q' "$LOG_DIR/$window.log"
    "${TMUX[@]}" pipe-pane -t "$SESSION:$window" "cat >> $log"
done
trap - ERR
"${TMUX[@]}" select-window -t "$SESSION:keyboard"
echo "Controller launched: it moves the robot toward the default standing pose. Wait for 'Init Done'."
echo "Then, in the keyboard window: k (stand), i (initial pose), p (the policy runs; p again pauses it)."
echo "Attach: bash install_scripts/run_vla_real.sh attach   (windows: camera controller vla keyboard)"
echo "Stop:   bash install_scripts/run_vla_real.sh stop"
echo "Logs:   $LOG_DIR"
