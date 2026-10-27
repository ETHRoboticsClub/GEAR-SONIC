#!/usr/bin/env bash
# PC-local MuJoCo teleoperation. All robot DDS traffic uses loopback.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$ROOT/.local/pico-sim/env.sh"
cd "$ROOT"
SESSION=pico-sim
TMUX=(tmux -L gear-sonic-pico)
export PYTHONUNBUFFERED=1
printf -v entry '%q' "$ROOT/install_scripts/run_pico_sim.sh"

launch_command() {
    local component="$1" log pipeline
    printf -v log '%q' "$PICO_SIM_DEPS/logs/$component.log"
    pipeline="bash $entry _$component 2>&1 | tee $log"
    printf -v launch '%q' "$pipeline"
}

case "${1:-start}" in
    _measurement)
        name="pico-latency-$(date +%Y%m%d-%H%M%S)-$$"
        python gear_sonic/scripts/measure_pico_sim_latency.py \
            --name "$name" --seconds 60 --warmup 5 --wait 0
        exec python gear_sonic/scripts/analyze_pico_sim_latency.py "$PICO_SIM_DEPS/logs/$name.msgpack"
        ;;
    _sim|_simulator)
        exec python gear_sonic/scripts/run_sim_loop.py --interface lo --viewer-state-only --guard-policy-release \
            --readiness-status-path "$PICO_SIM_DEPS/readiness.json"
        ;;
    _controller)
        cd gear_sonic_deploy
        exec ./target/release/g1_deploy_onnx_ref lo \
            policy/low_latency/model_decoder.onnx reference/example \
            --obs-config policy/low_latency/observation_config.yaml \
            --encoder-file policy/low_latency/model_encoder.onnx \
            --planner-file planner/target_vel/V2/planner_sonic.onnx \
            --input-type zmq_manager --output-type zmq --zmq-host 127.0.0.1 \
            --disable-crc-check --stream-lookahead-frames 4 --immediate-stream-input
        ;;
    _bridge)
        exec python gear_sonic/scripts/pico_sim_bridge.py
        ;;
    _manager)
        python gear_sonic/scripts/pico_sim_doctor.py wait-controller --log "$PICO_SIM_DEPS/logs/controller.log"
        exec python gear_sonic/scripts/pico_manager_thread_server.py \
            --manager --zmq_feedback_host 127.0.0.1 --zmq_feedback_port 5558 \
            --local_pico_timing
        ;;
    _service)
        cd /opt/apps/roboticsservice
        export LD_LIBRARY_PATH="$PWD:$PWD/lib:$PWD/SDK/x64:$LD_LIBRARY_PATH"
        export QT_PLUGIN_PATH="$PWD/plugins"
        export QT_QML_PATH="$PWD/qml"
        exec ./RoboticsServiceProcess
        ;;
    stop)
        if "${TMUX[@]}" has-session -t "$SESSION" 2>/dev/null; then
            "${TMUX[@]}" kill-session -t "$SESSION"
        fi
        rm -f "$PICO_SIM_DEPS/readiness.json"
        exit
        ;;
    attach)
        exec "${TMUX[@]}" attach-session -t "$SESSION"
        ;;
    measure)
        if ! "${TMUX[@]}" has-session -t "$SESSION" 2>/dev/null; then
            echo "Start the simulation first: bash install_scripts/run_pico_sim.sh" >&2
            exit 1
        fi
        if panes=$("${TMUX[@]}" list-panes -t "$SESSION:measurement" -F '#{pane_dead}' 2>/dev/null); then
            if [[ "$panes" == *0* ]]; then
                echo "Measurement is already armed/running. See the measurement window or status."
                exit
            fi
            "${TMUX[@]}" kill-window -t "$SESSION:measurement"
        fi
        launch_command measurement
        "${TMUX[@]}" new-window -d -t "$SESSION" -n measurement "bash -o pipefail -c $launch"
        echo "Measurement armed: 5 seconds of live POSE, then 60 seconds recorded automatically."
        exit
        ;;
    status)
        if "${TMUX[@]}" has-session -t "$SESSION" 2>/dev/null; then
            "${TMUX[@]}" list-panes -s -t "$SESSION" \
                -F '#{window_name}: #{?pane_dead,EXITED,RUNNING} #{pane_dead_status}'
        else
            echo "No pico-sim session running."
        fi
        python - <<'PY'
import json, os
from pathlib import Path
path = Path(os.environ['PICO_SIM_DEPS']) / 'logs/latest-latency-capture.json'
if path.is_file():
    data = json.loads(path.read_text())
    print(f"[MEASUREMENT] {data['name']}: {data['state']}; {data.get('elapsed_seconds', 0):g}s recorded")
    print(f"              {data['summary']}")
PY
        exec python gear_sonic/scripts/pico_sim_doctor.py status
        ;;
    network)
        exec python gear_sonic/scripts/pico_sim_doctor.py network "${2:-status}"
        ;;
    restart)
        bash "$0" stop
        exec bash "$0" start
        ;;
    start) ;;
    *) echo "Usage: $0 [start|attach|status|stop|restart|measure|network [status|up|restart]]" >&2; exit 2 ;;
esac

if ! "${TMUX[@]}" has-session -t "$SESSION" 2>/dev/null; then
    python gear_sonic/scripts/pico_sim_doctor.py network up
    # Fail before opening windows if another stack already owns these ports.
    python - <<'PY'
import socket
for port in (5556, 5557, 5558):
    with socket.socket() as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            sock.bind(('0.0.0.0', port))
        except OSError as exc:
            raise SystemExit(f'Port {port} is occupied; stop the existing teleop stack first: {exc}')
PY
    mkdir -p "$PICO_SIM_DEPS/logs"
    # Keep each run's component logs separate; readiness must never match an old Init Done.
    archive="$PICO_SIM_DEPS/logs/session-$(date +%Y%m%d-%H%M%S)-$$"
    mkdir -p "$archive"
    for component in simulator controller bridge manager service measurement; do
        if [ -f "$PICO_SIM_DEPS/logs/$component.log" ]; then
            mv "$PICO_SIM_DEPS/logs/$component.log" "$archive/"
        fi
    done
    rm -f "$PICO_SIM_DEPS/readiness.json"
    # tee is installed before the component starts, retaining early failures and Init Done.
    launch_command simulator
    "${TMUX[@]}" new-session -d -s "$SESSION" -n simulator "bash -o pipefail -c $launch"
    "${TMUX[@]}" set-window-option -g remain-on-exit on
    if ! pgrep -x RoboticsService >/dev/null; then
        launch_command service
        "${TMUX[@]}" new-window -t "$SESSION" -n service "bash -o pipefail -c $launch"
    fi
    if ! python - <<'PY'
import socket, time
deadline = time.monotonic() + 10
while time.monotonic() < deadline:
    try:
        with socket.create_connection(('127.0.0.1', 60061), timeout=.2):
            break
    except OSError:
        time.sleep(.1)
else:
    raise SystemExit('XRoboToolkit PC service did not start on port 60061')
PY
    then
        "${TMUX[@]}" kill-session -t "$SESSION"
        exit 1
    fi
    for component in controller bridge manager; do
        launch_command "$component"
        "${TMUX[@]}" new-window -t "$SESSION" -n "$component" "bash -o pipefail -c $launch"
    done
    bash "$0" measure
    "${TMUX[@]}" select-window -t "$SESSION:controller"
fi
echo "MuJoCo/PICO processes launched. Manager waits for controller initialization."
echo "Use PC Service 10.42.0.1 in XRoboToolkit on teleop-sim; confirm WORKING and tracking."
echo "Wait for manager controls, calibrate/start with A+B+X+Y, then run status before pressing 9."
echo "Status: bash install_scripts/run_pico_sim.sh status"
echo "Attach: bash install_scripts/run_pico_sim.sh attach"
echo "Stop:   bash install_scripts/run_pico_sim.sh stop"
echo "Capture: automatic 60s recording after 5s in live POSE; use 'measure' for another capture."
echo "Logs:   $PICO_SIM_DEPS/logs"
