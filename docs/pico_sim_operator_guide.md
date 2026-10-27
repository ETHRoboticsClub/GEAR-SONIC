# PICO → MuJoCo: operator procedure

Run commands from `/home/ethan/Humanoid/GEAR-SONIC`. This procedure is for
standalone simulation with DDS on loopback.

## Every session

1. Turn on the headset, both controllers, and both ankle trackers. Check their
   batteries. Wear the trackers on their correct ankles, visible to the cameras.
2. On the PC run:

   ```bash
   bash install_scripts/run_pico_sim.sh
   ```

   This activates the **existing** `teleop-sim` hotspot, starts the PC service
   and simulator/controller/bridge, and waits for the controller to initialize
   before starting the manager. TensorRT engine creation can take minutes on a
   cold start. Keep the robot suspended while it loads.
3. In PICO Wi-Fi settings connect to the saved **teleop-sim** network. Keep
   the same saved credentials. Wait for Wi-Fi connection and a **10.42.0.x**
   address. A local connection can work without Internet; if PICO asks whether
   to keep using this network, stay connected.
4. In XRoboToolkit set **PC Service = 10.42.0.1**, connect, and require
   **WORKING**. Enable **Head**, **Controller**, and **Send**. This is a
   separate connection from joining Wi-Fi. The PC's other Wi-Fi, Docker, and
   VPN addresses are not the hotspot service address.
5. In PICO's tracker settings/app, confirm both ankle trackers are paired and
   complete the tracker calibration when required, following its prompts. In
   XRoboToolkit choose **Full body** and confirm the tracked avatar is upright
   with correctly placed feet. This tracker calibration is separate from the
   teleoperation calibration below. Follow the [hardware guide](source/getting_started/vr_teleop_setup.md)
   for first-time pairing; repeated unpairing is not the normal session procedure.
6. Run:

   ```bash
   bash install_scripts/run_pico_sim.sh status
   ```

   Require `head=OK left=OK right=OK body=OK` for full-body operation. The
   policy is expected to be inactive before the next step. Wait for the manager
   controls in its log/window; use `attach` to inspect it if necessary.
7. Stand upright, arms at your sides, elbows bent forward 90°, palms inward.
   Keep sticks centered. Press **A+B+X+Y once**, then release all buttons.
   This captures teleop calibration and requests planner policy start. Pressing
   it again is a stop action, so do not keep retrying the combination blindly.
8. Run `status` again. Require **POLICY ACTIVE** and **RELEASE READY**.
   The simulator must receive active-policy feedback and powered body commands
   continuously for at least **0.5 seconds**, with neither older than **200 ms**.
   `Init Done`, `Calibration completed`, and `WORKING` alone do not establish this.
9. Click the **MuJoCo** window and press **9 once**. The simulator logs
   `[Suspension] RELEASED`. If blocked, it logs the missing requirement. Let
   the robot settle in planner IDLE before beginning whole-body movement.
10. Match the robot's posture and press **A+X** once to enter POSE mode. Start
    with a small arm movement, then small body movements. **A+X** returns to
    planner mode. Keep the headset awake and trackers visible.
11. Finish by reattaching suspension with **9** before stopping policy with
    **A+B+X+Y**. Then run `bash install_scripts/run_pico_sim.sh stop`.
    A new operator should start a fresh session and calibrate again.

The release guard checks command availability; it cannot guarantee balance,
correct calibration, accurate body tracking, or a correct model. It does not
automatically catch a robot when tracking drops after suspension is released.
After a fall, the guarded simulator resets with suspension attached and refreshed
MuJoCo transforms. Restart the stack and recalibrate before continuing.

## Measure the updated low-latency setup

The launcher on `feature-low-latency-sim` already selects the low-latency
encoder/decoder/configuration, four reference frames, immediate input preparation
and the updated controller. It also enables the measured state-only viewer
optimization. Keep these settings fixed for the first comparison.

The next launcher invocation also uses the
[further CPU optimization and packet-delay diagnostics](pico_latency_round2_20261027.md).
The PC recorder adds a `-transport.jsonl` file, retains source/SDK timestamps
when the headset supplies them, and reports relative delivery delay without
assuming the two clocks agree. No additional headset toggle is needed.

In XRoboToolkit use **Head + Controller + Full body + Send**, with **Remote
Vision** and the app's **Data Collection** off. The PC recorder saves the data
independently. Use the saved **teleop-sim** connection; the 19:03 repair on
27 October started it at **5 GHz/channel 36** using a temporary fallback driver
(see [network repair details](pico_tracking_gaps.md)). Keep this topology fixed
during the test; change it in a separate comparison. Keep the headset awake and
use the same lighting, tracker placement and operator position between trials.

Start normally and follow steps 1–10 above. A `measurement` tmux window is now
armed automatically. It waits without the old three-minute timeout, requires
fresh POSE, tracked devices and policy feedback for five seconds, then records
60 seconds. A mode/tracking change after recording starts is retained in the
data. Check progress using:

```bash
bash install_scripts/run_pico_sim.sh status
```

During the recorded minute, use a repeatable gentle sequence: 10 seconds still,
20 seconds raising/lowering and alternating the arms, 20 seconds of small knee
bends and weight shifts, then 10 seconds of arm motion. Keep POSE active and
avoid pressing mode buttons. If the simulation falls or tracking drops, let the
capture finish if practical so the fault is included; restart/recalibrate before
another normal movement trial. The recorder does not require a screen recording.

Completion produces timestamped raw `.msgpack`, a `.json` capture/configuration
summary, `-components.json` timing analysis, `-lowstate.msgpack` simulation-clock
samples when available, and a `-logs/` copy of the component logs in
`.local/pico-sim/logs/`. `latest-latency-capture.json` points to the newest
capture and its state. Old captures are preserved. An interrupted recorder
saves partial data when it receives a handled stop signal; do not rely on
preserving data after forced process termination or power loss.

For a second one-minute capture in the same running session:

```bash
bash install_scripts/run_pico_sim.sh measure
```

This refuses to create a duplicate live recorder. The offline analysis reports
pose/feedback rates, head/controller/body arrival gaps using SDK receipt
counters, local bridge delay, preparation timings, and matched newest-input
receipt → controller feedback. The capture summary reports simulation speed.
Counter jumps can reflect coalescing by the bridge or observer; they are not
proof of Wi-Fi packet loss. This instrumentation cannot measure headset
sensor-to-screen latency. An external high-frame-rate video showing human
movement and the monitor together is optional for that separate measurement.

Repeat the same sequence twice before changing one setting (for example the
network topology). Compare medians, p95, gap counts and simulation speed. The
October 27 pre-fix run has aggregate logs but no raw capture, so it cannot supply
a directly matched per-frame baseline for the new captures.

## Recover without deleting the Wi-Fi network

| Where it is stuck | What it means / next step |
| --- | --- |
| PICO Wi-Fi settings: Connecting; no 10.42.0.x address | Wi-Fi association/authentication or address assignment has not completed. XRoboToolkit and MuJoCo cannot fix that stage. Toggle PICO Wi-Fi off/on and reconnect to the saved network. Check the hotspot with the command below. |
| PICO Wi-Fi connected, XRoboToolkit: Connecting | Check PC Service **10.42.0.1**, PC service listening, and app reconnection. Close/reopen XRoboToolkit if needed. If all network traffic is failing, diagnose the Wi-Fi link as well. |
| WORKING but tracking MISSING | Check Head/Controller/Send, headset sleep, controller visibility and tracker calibration. Body MISSING alone requires investigating full-body tracking. |
| Calibration completed but no POLICY ACTIVE | Calibration happened in Python; the C++ controller has not confirmed active inference. Check controller.log for initialization, errors or termination. Restart the stack and calibrate once after manager readiness. |
| Pressing 9 is blocked | Read the simulator's `[Release blocked]` reason. Check live policy feedback and DDS motor commands; leave suspension attached while resolving it. |
| Robot falls after a permitted release | Availability checks passed but do not prove balance. Save logs and check tracking gaps, initial pose, joint targets/model compatibility and the first motion. Restart and recalibrate in planner IDLE. |

```bash
bash install_scripts/run_pico_sim.sh network
bash install_scripts/run_pico_sim.sh stop
bash install_scripts/run_pico_sim.sh network restart
bash install_scripts/run_pico_sim.sh
```

`network restart` restarts the existing hotspot profile and retains its password,
SSID and band. It refuses to disconnect the hotspot while a `pico-sim` session is
running. Reconnect on the headset, then reconnect XRoboToolkit and recalibrate.
Forget the PICO's saved network only if credentials/security changed, or the
saved profile remains unusable after these steps. Do not recreate the PC hotspot
on every run.

## What the 2026-10-27 evidence establishes

- The current `teleop-sim` profile is bound to USB adapter
  `wlxec750c689f4a`, **2.4 GHz / channel 6**, WPA-PSK with RSN, IPv4 shared,
  and Wi-Fi power saving disabled. Autoconnect is off; the updated launcher
  activates it explicitly. The 2026-10-21 latency results used **5 GHz**.
  Do not apply those performance figures to the current band without a new test.
- The saved restart metadata records a prior episode of stopped PICO traffic
  while the hotspot remained active, failed ping, and firmware TX-report errors
  from `rtw88_8822bu`. The later Wi-Fi watch recorded no associated stations
  despite the AP showing active. Bridge logs also show complete tracking losses.
  These support investigating the adapter/link; they do not identify one proven
  cause for every connection failure. Administrator kernel/NetworkManager logs
  were unavailable to this investigation.
- Today's saved `controller.log` ends at `Init Done`, with no transition to
  `CONTROL`. The manager log records calibration and OFF → PLANNER. Its start
  command was sent once on that transition. A start sent before the C++ ZMQ
  subscriber exists can be lost; the log also shows TensorRT engines being built.
  This is consistent with the reported unsupported fall after pressing 9,
  although the exact button timing was not recorded.
- The DDS domain warning came from repeated initialization in the same simulator
  process. The duplicate call is removed. It was not proof that live DDS failed.
- Previously, a fall reset left suspension released, causing repeated falls.
  Guarded resets now reattach it and call MuJoCo forward before reading transforms.

Retain a stable hotspot profile during testing. A controlled 5 GHz test should use
a locally permitted non-DFS channel and keep the SSID/password stable. If this
USB adapter continues losing its link, evaluate a dedicated access point with
the PC connected by Ethernet. Firmware errors alone do not establish that a
specific replacement or driver change will solve the problem.

Current logs: `.local/pico-sim/logs/{simulator,controller,bridge,manager,service}.log`.
Previous runs: dated `session-*` subdirectories. The simulator writes a readiness
snapshot to `.local/pico-sim/readiness.json`; status rejects stale snapshots.

## Verification of the workflow update

On 2026-10-27, all **110** tests in `gear_sonic/tests/teleop`,
`test_sim_timing.py`, and `test_sim_release_guard.py` passed. Launcher syntax,
patch whitespace checks and the virtual environment's package check passed.
The launcher was exercised on this PC: the manager waited for `Init Done`,
and an independent DDS subscriber received 399 simulator-state packets in two
seconds. The duplicate DDS domain error was absent.

A six-second synthetic planner run with the actual C++ controller and MuJoCo
kept suspension attached throughout. Release readiness was blocked before
policy activation, stayed ready throughout the final 4.5 seconds of stable
control, and was withdrawn after stop. Its local report is
`.local/pico-sim/logs/workflow-smoke-20261027.json`. These checks did not involve
the physical headset, release suspension, or validate Wi-Fi recovery, calibration
quality, free-standing balance, or end-to-end latency. A supervised PICO session
is still required to validate those parts.

Configuration references: [NetworkManager profile settings](https://networkmanager.pages.freedesktop.org/NetworkManager/NetworkManager/nm-settings-nmcli.html),
[upstream PICO setup](https://nvlabs.github.io/GR00T-WholeBodyControl/getting_started/vr_teleop_setup.html),
[Linux DFS behavior](https://wireless.docs.kernel.org/en/latest/en/developers/dfs.html).
