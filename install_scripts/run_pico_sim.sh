#!/usr/bin/env bash
# PC-local MuJoCo teleoperation. All robot DDS traffic uses loopback.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$ROOT/.local/pico-sim/env.sh"
cd "$ROOT"
SESSION=pico-sim
TMUX=(tmux -L gear-sonic-pico)

case "${1:-start}" in
    _sim)
        exec python gear_sonic/scripts/run_sim_loop.py --interface lo
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
        "${TMUX[@]}" kill-session -t "$SESSION"
        exit
        ;;
    attach)
        exec "${TMUX[@]}" attach-session -t "$SESSION"
        ;;
    status)
        "${TMUX[@]}" list-panes -s -t "$SESSION" \
            -F '#{window_name}: #{?pane_dead,EXITED,RUNNING} #{pane_dead_status}'
        exit
        ;;
    start) ;;
    *) echo "Usage: $0 [start|attach|status|stop]" >&2; exit 2 ;;
esac

if ! "${TMUX[@]}" has-session -t "$SESSION" 2>/dev/null; then
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
    printf -v entry '%q' "$ROOT/install_scripts/run_pico_sim.sh"
    "${TMUX[@]}" new-session -d -s "$SESSION" -n simulator "bash $entry _sim"
    "${TMUX[@]}" set-option -t "$SESSION" remain-on-exit on
    if ! pgrep -x RoboticsService >/dev/null; then
        "${TMUX[@]}" new-window -t "$SESSION" -n service "bash $entry _service"
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
        "${TMUX[@]}" new-window -t "$SESSION" -n "$component" "bash $entry _$component"
    done
    for window in $("${TMUX[@]}" list-windows -t "$SESSION" -F '#{window_name}'); do
        printf -v log '%q' "$PICO_SIM_DEPS/logs/$window.log"
        "${TMUX[@]}" pipe-pane -t "$SESSION:$window" "cat >> $log"
    done
    "${TMUX[@]}" select-window -t "$SESSION:controller"
fi
echo "MuJoCo/PICO session started. Wait for 'Init Done' in the controller log."
echo "Attach: bash install_scripts/run_pico_sim.sh attach"
echo "Stop:   bash install_scripts/run_pico_sim.sh stop"
echo "Logs:   $PICO_SIM_DEPS/logs"
echo "PC IPv4 addresses: $(hostname -I)"
