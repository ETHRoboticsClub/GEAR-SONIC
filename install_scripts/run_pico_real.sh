#!/usr/bin/env bash
# Standalone PICO teleoperation, entirely on the G1 onboard computer.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export PATH="$ROOT/.venv_teleop/bin:$PATH"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
export LD_LIBRARY_PATH="$HOME/TensorRT/lib:/opt/onnxruntime/lib:/usr/local/cuda/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
unset PICO_SIM_CONTROL_DIR PICO_SIM_TRACKING_FILE PICO_SIM_INTERFACE
SESSION=pico-real
TMUX=(tmux -L gear-sonic-real)
LOG_DIR="$ROOT/.local/pico-real/logs"
SERVICE_DIR=/opt/apps/roboticsservice

component_pane() {
    "${TMUX[@]}" list-panes -s -t "$SESSION" \
        -F '#{pane_id} #{?@pico_component,#{@pico_component},#{window_name}}' |
        awk -v component="$1" '$2 == component {print $1; exit}'
}

controller_state() {
    local pane
    pane="$(component_pane controller)"
    if [[ -n "$pane" ]]; then
        "${TMUX[@]}" display-message -p -t "$pane" '#{pane_dead}'
    fi
}

layout() {
    local controller pane component
    controller="$(component_pane controller)"
    [[ -n "$controller" ]] || return 0
    # Label panes before moving them; also upgrades existing separate windows.
    for component in controller bridge manager service; do
        pane="$(component_pane "$component")"
        if [[ -n "$pane" ]]; then
            "${TMUX[@]}" set-option -p -t "$pane" @pico_component "$component"
        fi
    done
    for component in bridge manager; do
        pane="$(component_pane "$component")"
        if [[ -n "$pane" ]] &&
            [[ "$("${TMUX[@]}" display-message -p -t "$pane" '#{window_id}')" != \
               "$("${TMUX[@]}" display-message -p -t "$controller" '#{window_id}')" ]]; then
            "${TMUX[@]}" join-pane -d -h -s "$pane" -t "$controller"
        fi
    done
    "${TMUX[@]}" rename-window -t "$controller" teleop
    "${TMUX[@]}" set-option -w -t "$controller" automatic-rename off
    "${TMUX[@]}" set-option -w -t "$controller" pane-border-status top
    "${TMUX[@]}" set-option -w -t "$controller" pane-border-format ' #{@pico_component} '
    "${TMUX[@]}" set-option -t "$SESSION" mouse on
    "${TMUX[@]}" select-layout -t "$controller" main-vertical >/dev/null
    "${TMUX[@]}" select-pane -t "$controller"
    "${TMUX[@]}" select-window -t "$controller"
}

check() {
    [[ "$(uname -m)" == aarch64 ]] || { echo 'Run this launcher onboard the G1.' >&2; return 1; }
    [[ -d /sys/class/net/enP8p1s0 ]] || { echo 'Robot Ethernet enP8p1s0 is missing.' >&2; return 1; }
    if systemctl is-active --quiet ethrc-controls.service; then
        echo 'Stop ethrc-controls.service before standalone teleop.' >&2; return 1
    fi
    if [[ -n "$(docker ps -q --filter label=com.docker.compose.project=ethrc-robot)" ]]; then
        echo 'ETHRC containers are still running.' >&2; return 1
    fi
    for file in model_encoder.onnx model_decoder.onnx observation_config.yaml; do
        test -s "gear_sonic_deploy/policy/low_latency/$file"
    done
    test -s gear_sonic_deploy/planner/target_vel/V2/planner_sonic.onnx
    test -x "$SERVICE_DIR/RoboticsServiceProcess"
    python - <<'PY'
import socket
from pathlib import Path
import xrobotoolkit_sdk as xrt
import gear_sonic.scripts.pico_manager_thread_server
assert hasattr(xrt, 'get_input_update_sequences'), 'Rebuild the XRoboToolkit Python binding'
binary = Path('gear_sonic_deploy/target/release/g1_deploy_onnx_ref').read_bytes()
for flag in (b'--stream-lookahead-frames', b'--immediate-stream-input'):
    assert flag in binary, f'Rebuild the controller: missing {flag.decode()}'
for port in (5556, 5557, 5558, 60061):
    with socket.socket() as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind(('0.0.0.0', port))
print('Preflight passed: models, imports, latency flags, and ports.')
PY
}

case "${1:-help}" in
    _controller)
        cd gear_sonic_deploy
        exec ./target/release/g1_deploy_onnx_ref enP8p1s0 \
            policy/low_latency/model_decoder.onnx reference/example \
            --obs-config policy/low_latency/observation_config.yaml \
            --encoder-file policy/low_latency/model_encoder.onnx \
            --planner-file planner/target_vel/V2/planner_sonic.onnx \
            --input-type zmq_manager --output-type zmq --zmq-host 127.0.0.1 \
            --stream-lookahead-frames 4 --immediate-stream-input
        ;;
    _bridge) exec python gear_sonic/scripts/pico_sim_bridge.py ;;
    _manager)
        exec python gear_sonic/scripts/pico_manager_thread_server.py \
            --manager --zmq_feedback_host 127.0.0.1 --zmq_feedback_port 5558 \
            --local_pico_timing
        ;;
    _service)
        cd "$SERVICE_DIR"
        export LD_LIBRARY_PATH="$PWD:$PWD/lib:$PWD/SDK/arm64:$LD_LIBRARY_PATH"
        export QT_PLUGIN_PATH="$PWD/plugins"
        export QT_QML_PATH="$PWD/qml"
        exec ./RoboticsServiceProcess
        ;;
    check) check; exit ;;
    stop|abort)
        if "${TMUX[@]}" has-session -t "$SESSION" 2>/dev/null; then
            state="$(controller_state)"
            if [[ "$state" == 0 ]]; then
                # Use the controller's normal stop path, which publishes damping.
                "${TMUX[@]}" send-keys -t "$(component_pane controller)" o
                for ((attempt=0; attempt<100; attempt++)); do
                    state="$(controller_state)"
                    [[ "$state" != 0 ]] && break
                    sleep 0.1
                done
                if [[ "$state" == 0 ]]; then
                    echo 'Controller did not exit after the stop request. Use the robot hardware stop; session retained for diagnosis.' >&2
                    exit 1
                fi
            fi
            "${TMUX[@]}" kill-session -t "$SESSION"
        fi
        exit ;;
    layout) layout; exit ;;
    attach) layout; exec "${TMUX[@]}" attach-session -t "$SESSION" ;;
    status)
        "${TMUX[@]}" list-panes -s -t "$SESSION" -F '#{?@pico_component,#{@pico_component},#{window_name}}: #{?pane_dead,EXITED,RUNNING} #{pane_dead_status}'
        exit ;;
    start|tracking) MODE="$1" ;;
    *) echo "Usage: $0 [check|tracking|start|attach|layout|status|stop|abort]"
       echo 'tracking: service and tracking bridge only; start: includes real robot control'
       exit ;;
esac

if "${TMUX[@]}" has-session -t "$SESSION" 2>/dev/null; then
    echo 'pico-real already exists. Use status/attach, or stop before restarting.' >&2
    exit 1
fi
check
mkdir -p "$LOG_DIR"
printf -v entry '%q' "$ROOT/install_scripts/run_pico_real.sh"
"${TMUX[@]}" new-session -d -s "$SESSION" -n service "bash $entry _service"
trap '"${TMUX[@]}" kill-session -t "$SESSION" 2>/dev/null || true' ERR
"${TMUX[@]}" set-option -g remain-on-exit on
python - <<'PY'
import socket, time
deadline = time.monotonic() + 15
while time.monotonic() < deadline:
    try:
        with socket.create_connection(('127.0.0.1', 60061), timeout=.2):
            break
    except OSError:
        time.sleep(.1)
else:
    raise SystemExit('XRoboToolkit service did not start on :60061')
PY
components=(bridge)
if [[ "$MODE" == start ]]; then components+=(manager controller); fi
for component in "${components[@]}"; do
    "${TMUX[@]}" new-window -t "$SESSION" -n "$component" "bash $entry _$component"
done
for window in service "${components[@]}"; do
    printf -v log '%q' "$LOG_DIR/$window.log"
    "${TMUX[@]}" pipe-pane -t "$SESSION:$window" "cat >> $log"
    "${TMUX[@]}" capture-pane -p -t "$SESSION:$window" >> "$LOG_DIR/$window.log"
done
trap - ERR
if [[ "$MODE" == start ]]; then
    layout
    echo "Real robot controller launched. Wait for 'Init Done' before enabling the policy."
else
    echo 'Tracking only: no robot controller or motor commands.'
fi
echo "Attach: bash install_scripts/run_pico_real.sh attach"
echo "Status: bash install_scripts/run_pico_real.sh status"
echo "Stop:   bash install_scripts/run_pico_real.sh stop"
echo "Logs:   $LOG_DIR"
