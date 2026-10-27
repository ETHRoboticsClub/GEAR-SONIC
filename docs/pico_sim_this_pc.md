# PICO teleoperation on this PC

For the current step-by-step session procedure and connection recovery, use the
[operator guide](pico_sim_operator_guide.md). It also explains the startup and
suspension-release protections added on 2026-10-27.

For the measured before/after results of each optimization, see
[PICO → MuJoCo latency improvements](pico_sim_latency.md).

Configured on 2026-10-21 for Ubuntu 24.04, GTX 1650 (4 GB), and 64 GB RAM.
Use **MuJoCo**. The GTX 1650 does not meet [Isaac Sim's RTX GPU
requirements](https://docs.isaacsim.omniverse.nvidia.com/5.1.0/installation/requirements.html).

## Start and stop

From this repository:

```bash
bash install_scripts/run_pico_sim.sh
bash install_scripts/run_pico_sim.sh attach
```

The first command opens the MuJoCo window and starts a tmux session with
simulator, controller, tracking bridge, teleop manager, and XRoboToolkit service
windows. An already running XRoboToolkit service is reused. Wait for **`Init
Done`** in the controller window before engaging the policy. In tmux, use
**Ctrl+B, then N** to change windows, and **Ctrl+B, then D** to detach.

```bash
bash install_scripts/run_pico_sim.sh status
bash install_scripts/run_pico_sim.sh stop
```

`status` reports process state, hotspot configuration, live tracking, policy
feedback, and the simulator's release-readiness snapshot. A running process
alone is not a connected headset. Logs are in `.local/pico-sim/logs/`.
Each new session archives the previous component logs in a dated subdirectory.
Stopping the session also closes its simulator. A service reused from another
session is left running. After stopping with the PICO button combination,
stop and start the launcher to begin a fresh session.

## Headset steps

1. Install [XRoboToolkit PICO 1.1.1](https://github.com/XR-Robotics/XRoboToolkit-Unity-Client/releases/download/v1.1.1/XRoboToolkit-PICO-1.1.1.apk)
  on the headset if needed. Use the headset browser with Developer Mode enabled;
   the installed application appears under Unknown applications.
2. Connect the headset to the saved **`teleop-sim`** hotspot. The launcher
   activates the existing profile. Check `bash install_scripts/run_pico_sim.sh network`
   for its address and current band; the hotspot PC address is **`10.42.0.1`**.
   Avoid choosing among unrelated addresses from `hostname -I`.
3. In XRoboToolkit, set **PC Service** to that hotspot address, connect, and confirm
   **WORKING**. Enable **Head**, **Controller**, and **Send**.
4. For whole-body teleop, pair and calibrate the two ankle motion trackers,
   then select **Full body** in XRoboToolkit. See the existing
   [PICO hardware setup guide](source/getting_started/vr_teleop_setup.md).
5. The bridge window should report `head=True left=True right=True body=True`.
   Without ankle/body tracking, planner and head/controller three-point input
   are available; full-body POSE mode requires body tracking.
6. Wait for the manager's controls to appear: it now waits for controller
   `Init Done` before accepting calibration/start. Stand upright with upper arms at your sides, elbows bent 90 degrees forward,
   and palms inward. Press **A+B+X+Y** once to calibrate and engage the policy.
7. Run `bash install_scripts/run_pico_sim.sh status`; require live policy
   feedback and **RELEASE READY**. Click the **MuJoCo window** and press **9**.
   Release is blocked until policy feedback and powered DDS leg commands have
   been continuously fresh for 0.5 seconds. The band starts enabled on every fresh launch.
8. Match the robot's posture, then press **A+X** to enter full-body **POSE** mode.
   **A+X** returns to planner mode. With head and controller tracking, clicking
   the **left stick** from planner mode enters calibrated three-point control.
9. **A+B+X+Y** stops the session's controller and teleop manager.

The simulator is viewed on the PC monitor. This setup streams tracking from the
PICO; it does not stream stereoscopic simulator video into the headset.

## Installed configuration

- Repaired the existing `.venv_teleop` Python 3.10.22 interpreter link, which
  pointed into an obsolete VS Code snap revision. The project supports Python
  3.10 and newer; the upstream fresh-install script chooses 3.11.
- Installed this checkout's `gear_sonic[teleop,sim]` and Unitree SDK Python
  bindings as editable packages; reused its working XRoboToolkit SDK binding.
- Reused `/opt/apps/roboticsservice` and
  `~/TensorRT/TensorRT-10.13.0.35`.
- Rebuilt the XRoboToolkit Python binding with per-device packet receipt counters.
  These keep stationary controllers' buttons available and expire devices whose
  data stops arriving, independently of other devices. They also handle clients
  that omit the body timestamp. Rebuilding this checkout's SDK preserves the fix.
- Installed CMake 3.31.10, CUDA 12.8 runtime/development headers, ONNX Runtime
  1.16.3, and the Ubuntu C++ build dependencies under `.local/pico-sim/` and the
  virtual environment. No system-wide package installation was needed.
- Built `gear_sonic_deploy/target/release/g1_deploy_onnx_ref` and downloaded the
  low-latency encoder, decoder, matching observation configuration, and V2
  planner with `download_from_hf.py --low-latency`. TensorRT engines are cached
  beside their ONNX models for this GPU.
- Runtime paths: `.local/pico-sim/env.sh`. Python package inventory:
  `.local/pico-sim/installed-python.txt`. Build logs: `configure.log` and
  `build.log` in the logs directory.

The launcher explicitly uses DDS interface **`lo`** and the simulator's
`--disable-crc-check` option, matching `deploy.sh sim`. It directly launches the
built executable so a run does not trigger another dependency installation.

This fork's manager receives tracking through ETHRC's wire format rather than
reading the headset SDK. `gear_sonic/scripts/pico_sim_bridge.py` supplies that
format for this standalone simulation: it reads the local SDK, converts OpenXR
poses to the manager's operator frame, expires stale values, and forwards C++
robot feedback without modifying it. Connections are:

```text
XRoboToolkit SDK -> simulation bridge :5558 -> teleop manager
C++ feedback :5557 -> simulation bridge :5558 -> teleop manager
teleop manager :5556 -> C++ controller -> loopback DDS -> MuJoCo
```

The bridge binds only to localhost. It is specific to the standalone simulation
workflow; the ETHRC robot API workflow is unchanged.

## Verification

- MuJoCo loads the G1 scene and publishes finite simulated joint/IMU state over DDS.
- The controller reaches `Init Done` with all three TensorRT engines loaded.
- Synthetic head/controller input exercises calibration, planner start, policy
  inference, feedback, and stop: 598 feedback frames at approximately **50.1 Hz**.
  Decoder inference was approximately **1.3 ms** per control tick; observed GPU
  usage during initialization/control was under **2 GB**.
- Fixed a pre-existing zero-angle rotation division that caused the full-body
  streamer to crash on straight/neutral limbs. An isolated manager run produced
  **192 full-body pose packets** from synthetic 24-joint tracking after the fix.
- All **88** tests in `gear_sonic/tests/teleop` pass, including adapter coordinate
  conversion, invalid-data rejection, stationary-controller button holds and
  releases, and per-device stale-data expiry.

Synthetic checks do not validate a real PICO's pairing, network latency, tracking
quality, or the operator's calibration. The physical headset must still be
connected and calibrated to check those.

## Latency investigation (2026-10-21)

The simulator previously ran at about 168 Hz / 0.84 times real time. Its rate
limiter restarted the clock after every step, so occasional slow viewer updates
permanently slowed physics. The loop now uses cumulative monotonic deadlines
with bounded catch-up after long pauses. A five-second DDS measurement after
the change measured 200.0 Hz / 1.000 times real time. Regression checks cover
slow viewer updates and recovery from a half-second stall.

Existing controller logs show roughly 1–2 ms policy inference, while full-body
pose output varied around 31–42 Hz. Bridge logs also show tracking gaps exceeding
the 200 ms freshness limit. These observations do not establish full headset-to-
robot latency: a live connected PICO test is still needed to separate tracking/
network delays, pose buffering, and the robot's physical response.

### Live network comparison

During a subsequent 60-second full-body POSE test, the PICO connected directly
to the PC's existing 5 GHz `teleop-sim` hotspot (PC `10.42.0.1`, headset
`10.42.0.57`). Compared with the preceding 30-second `ETHRC_SL` test:

| Measurement | ETHRC_SL | teleop-sim |
| --- | ---: | ---: |
| Pose publications | 36.1 Hz | 49.9 Hz |
| Changed body samples received | 37.7 Hz | 72.0 Hz |
| Body update gap, 95th percentile | 99.2 ms | 18.5 ms |
| Longest body update gap | 276.8 ms | 56.7 ms |
| Mean ping round-trip time | 45.7 ms (20 pings) | 4.9 ms (60 pings) |

All 4,627 sampled hotspot tracking messages reported head, both controllers,
and body present; no stale-tracking episodes were observed during that minute.
Controller feedback held 50.0 Hz. Simulation advanced at 0.955 times wall-clock
speed under the full teleoperation/measurement load, so the earlier unloaded
1.000 result is not a guarantee under load. Ping measures round-trip network
time; changed-sample intervals and pose publication rates do not measure total
human-motion-to-robot-response latency.

Raw local captures and summaries are in `.local/pico-sim/logs/`:
`live-latency-20261021.{json,msgpack}` and
`hotspot-latency-20261021.{json,msgpack}`, with hotspot ping results in
`hotspot-ping-20261021.txt`. The earlier capture subscribed to feedback both
directly and via the relay, so its raw 100.1 Hz feedback count contains duplicates.
Use **teleop-sim / 10.42.0.1** for subsequent standalone simulation tests.

Further analysis matched unique streamed joint-reference vectors to the
controller's `body_q_target` feedback for 2,991 control ticks (excluding the
capture boundary and ambiguous/nonmatching frames):

- Manager sample receipt to pose publication: median 10.0 ms, p95 13.9 ms.
- First pose publication to use as the current playback reference: median
  138.5 ms, p95 150.9 ms.
- Manager sample receipt to that current-reference feedback: median 148.6 ms,
  p95 161.7 ms. The playback reference lagged the newest published frame by
  6 frames typically, 7 at p95.

These are **reference ages**, not a measured human-to-visible-robot response
time: the full-body encoder observes the current reference plus three later
frames, and the capture does not establish headset sensing latency or the
robot/display response. The substantial playback backlog is a stronger
optimization candidate than the approximately 1.4 ms decoder inference.
Analysis results: `.local/pico-sim/logs/hotspot-playback-latency-20261021.json`.

### Latest-window playback update

The simulation launcher now passes `--stream-lookahead-frames 4` to the rebuilt
controller. Before each policy inference, the controller selects the newest
complete four-frame window under the motion lock. Without a new packet, it
holds the same complete window. This option is explicit: other launchers keep
sequential playback. The controller checks the active observation horizon before
issuing an action and stops if the requested window is too short for the model.
The sequential playback boundary also now admits the last complete window.

The manager resamples tracking onto a fixed 20 ms grid and publishes overlapping
five-frame packets; the controller consumes the newest four. Delayed arrivals
generate at most the newest packet's worth of interpolated frames, without
replaying an accumulated queue or extrapolating future tracking. Packets now
include `frame_timestamp_ns` for measuring reference age. Duplicate sample times
are rejected before SMPL mapping, and three-point retargeting computes only the
four joint transforms it actually uses. Policy frequency remains 50 Hz.

Verification: 95 Python tests and all three new C++ `StreamPlayback` tests pass.
The existing FK test requires the absent `reference/bones_072925_test` fixture
and was not validated. The binary backup is
`.local/pico-sim/controller-before-latest-window`; build/test logs are under
`.local/pico-sim/logs/latest-window-*`.

To capture another live POSE run from the repository root:

```bash
source .local/pico-sim/env.sh
python gear_sonic/scripts/measure_pico_sim_latency.py --seconds 60 --wait 180 --name latest-window-repeat
```

This waits for POSE mode with all tracking present, then records rates, frame
spacing and reference ages. The result is a local `.json` summary and `.msgpack`
capture in the logs directory; it does not measure display or physical response.

### Live verification after the update

The subsequent 60-second PICO test completed in POSE mode throughout, with all
4,636 tracking messages reporting head, controllers and body present. The user
reported smooth motion. Controller output remained 50.0 Hz; pose publications
averaged 49.6 Hz. All 2,976 pose packets retained exact 20 ms frame spacing.

| Matched-reference measurement | Before update | After update |
| --- | ---: | ---: |
| Manager receipt to current-reference feedback, median | 148.6 ms | 86.7 ms |
| Same measurement, p95 | 161.7 ms | 107.8 ms |
| Pose publication to current-reference feedback, median | 138.5 ms | 78.0 ms |
| Latest published frame minus playback reference, median | 6 frames | 3 frames |

This reduces median PC-side reference age by about 62 ms (42%). The new
resampled-reference timestamp to feedback age was 92.1 ms median / 113.3 ms
p95, a distinct measurement without an equivalent timestamp in the old capture.
Reference matching succeeded for 2,990 of 3,000 feedback ticks, excluding
ambiguous vectors and capture boundaries. The simulator ran at 0.958 times
wall-clock speed under load. These results still do not measure headset sensing
or display latency. Capture and summary:
`.local/pico-sim/logs/latest-window-latency-20261021.{msgpack,json}`.

Matching the newest encoder frame (`reference_index + 3`) separately distinguishes
fresh-input processing delay from the oldest reference's age. Across the same
2,990 matched control ticks:

- Newest encoder sample's manager receipt to pose publication: median 9.1 ms,
  p95 11.5 ms.
- Complete window publication to controller feedback: median 20.1 ms,
  p95 40.8 ms. This includes controller polling/scheduling and inference.
- Newest encoder sample's manager receipt to that feedback: median 28.9 ms,
  p95 49.4 ms. Feedback is emitted after the tick computes its action; this is
  not a measurement of DDS command delivery or visible robot response.
- First publication of the oldest reference to first publication of its third
  later frame: median 56.8 ms, p95 67.7 ms (the resampled grid spans exactly 60 ms).

Thus the earlier 86.7 ms result describes **oldest-reference age**, not the time
before newly received tracking can participate in an action calculation. The
capture cannot determine how strongly each encoder frame influences the action.
Detailed summary: `latest-window-component-timing-20261021.json` in the logs
directory. Decoder runtime is approximately 1.4 ms, and observation preparation
plus encoder plus decoder is approximately 2 ms, included in the 20.1 ms stage.

### Body timing warnings

The simulation launcher enables `--local_pico_timing` on the manager. In POSE
mode, `[PICO timing] WARNING` appears in the **manager** tmux window and
`.local/pico-sim/logs/manager.log` when a measured interval exceeds 20 ms.
Warnings are limited to one per second and report the largest fresh-body arrival
gap and number of late gaps since the last report, current time without a fresh
body update, maximum PC bridge-to-reader delay, and latest fresh body's age from
PC bridge sampling to the manager loop. SDK body counters distinguish fresh
packets from heartbeats and controller-only updates, including stationary poses.
Silence is checked even when no new packets arrive; a blocked manager can report
only after it resumes. Warnings are suppressed outside POSE mode and until the
first valid body counter is received.

These are arrival-gap and PC-local timing measurements, **not headset-to-PC
one-way latency**. A missed 20 ms budget can cause stale references and a target
jump when tracking catches up; it does not prove a jump occurred. The warnings
do not change control or stop the policy. Only use the flag with the local
`pico_sim_bridge.py`, whose timestamps share the manager's monotonic clock.

After updating, both bridge and manager need restarting. When finished with an
active teleoperation run, restart the stack with:

```bash
bash install_scripts/run_pico_sim.sh stop
bash install_scripts/run_pico_sim.sh
```

Then use the usual calibration/start and MuJoCo release procedure. An already
running Python process does not pick up these changes automatically.

### Immediate stream preparation

The simulation launcher also passes `--immediate-stream-input` to the C++
controller. A received pose now wakes a dedicated input worker immediately,
instead of waiting for the next 100 Hz poll. That worker decodes and merges the
latest packet, then swaps the completed motion snapshot under the motion lock.
The next 50 Hz policy tick reads that snapshot and selects the newest complete
four-frame window. SMPL mapping and C++ packet decoding remain outside the policy
thread. A 10 ms idle timeout still services commands/keyboard input; it is not a
delay imposed on arriving pose packets. Wakeups coalesce and packets overwrite
the pending slot, so the worker does not build a separate FIFO backlog.

The Python manager was already event-driven. It now uses the bridge's SDK body
counter to avoid mapping cached body data again when only controllers or bridge
heartbeats update. Stationary but newly received body packets still advance the
reference. Reference spacing remains 20 ms and policy frequency remains 50 Hz.
Per-packet C++ debug output is disabled; use `--zmq-verbose` for receipt messages.

Timing is reported every five seconds while streaming:

- Manager: SMPL mapping median/p95/max, and all pose preparation before packing.
- Controller: `[Stream preparation]` receipt-to-worker and decode/publish mean/max.
- Pose packets: `prep_timing_ms` contains `[SMPL mapping, preparation before
  packing, reader-to-manager wait]` in milliseconds. The SMPL duration is included
  in the preparation duration, not additional to it. Packing, ZMQ transport and
  policy scheduling are outside these preparation timings.

A warm CPU replay benchmark of 500 recorded PICO samples measured SMPL mapping
(including conversion to the arrays used for streaming) at **1.94 ms median,
1.96 ms p95, 2.01 ms maximum** on this PC. This is a computation benchmark, not a
new live headset-to-action measurement. Result:
`.local/pico-sim/logs/immediate-input-smpl-benchmark.json`.

Validation: 100 Python teleop tests and six C++ InputWakeup/StreamPlayback tests
pass; the controller rebuild and launcher syntax check pass. The separate legacy
`gear_sonic/tests/test_input_readers.py` cannot collect because it imports the
already-absent `build_body_pose_sample`; it is not included in the 100 passing
tests. The existing FK fixture limitation noted above still applies.
Previous controller binary: `.local/pico-sim/controller-before-immediate-input`.
Build and test logs: `.local/pico-sim/logs/immediate-input-*`.

### Live measurement with immediate preparation

The next 60.0-second capture contained 7.1 seconds in POSE, a 17.2-second
PLANNER pause, then 35.7 seconds of uninterrupted POSE after the user restarted.
Use that final continuous interval for comparison: frame indices reset on mode
entry, so combining both POSE intervals would corrupt reference matching.
The analysis matched 1,778 of 1,788 policy feedback ticks in the selected interval.

| PC-side measurement | Previous median / p95 | Immediate preparation median / p95 |
| --- | --- | --- |
| Newest encoder input manager receipt → action feedback | 28.9 / 49.4 ms | 24.3 / 30.8 ms |
| Complete window publication → action feedback | 20.1 / 40.8 ms | 15.0 / 22.7 ms |
| Newest encoder input manager receipt → publication | 9.1 / 11.5 ms | 9.2 / 11.1 ms |

The live instrumented SMPL mapping was **4.38 ms median / 5.43 ms p95** on
published samples, higher than the tight-loop CPU replay benchmark. All pose
preparation before packing was 8.19 / 9.87 ms (including SMPL); reader-to-mapping
wait was 0.62 / 0.94 ms. Controller logs show receive-to-worker wakeup around
0.03 ms mean and decoding/publishing the motion snapshot around 2.9 ms mean.
Those controller log aggregates are not per-frame percentile measurements.

Policy feedback remained 50 Hz; pose publication was about 49.5 Hz. Fresh body
arrival gaps had median 13.8 ms, p95 18.4 ms and max 50.9 ms; 2.9% exceeded 20 ms.
Pose publication gaps had p95 29.4 ms and max 68.3 ms. Input-to-action feedback
max was 68.8 ms. Thus average latency and the p95 improved, but occasional gaps
remain. These are PC-side measurements, not headset sensing/network delay or
MuJoCo display/physical-response latency.

Raw capture: `.local/pico-sim/logs/immediate-input-live-20261021.msgpack`.
Selected continuous interval: the same prefix with `-pose.msgpack`.
Component summary: the same prefix with `-pose-components.json`.
Reproduce the component analysis with
`python gear_sonic/scripts/analyze_pico_sim_latency.py <capture.msgpack>`.
