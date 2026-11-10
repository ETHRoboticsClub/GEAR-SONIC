# Low latency PICO teleoperation onboard G1

Run from `/home/unitree/GEAR-SONIC` on `unitree_g1`. XRoboToolkit service,
tracking bridge, manager, and TensorRT controller all run onboard. Connect the
PICO to `teleop-sim` and set its XRoboToolkit PC Service address to `10.42.0.1`.
Enable head/controller sending and calibrated full-body tracking.

```bash
ssh unitree_g1
cd ~/GEAR-SONIC
# Needed after a reboot if the ETHRC service has restarted:
sudo systemctl stop ethrc-controls.service
bash install_scripts/run_pico_real.sh check
bash install_scripts/run_pico_real.sh start
bash install_scripts/run_pico_real.sh attach
```

The `teleop` window shows three labeled terminals: controller in the large left
pane, tracking bridge and PICO manager on the right. Click a pane or use
Ctrl+b followed by an arrow key to focus it. Ctrl+b then z zooms the focused pane;
repeat to restore all three. The XRoboToolkit service has its own `service` window.
`attach` also rearranges an older session into this layout without restarting it.

`start` launches real motor control: initialization ramps the robot toward its
default standing pose before the PICO start combination. Have the robot supported
and ready before running it. Wait for `Init Done`, then stand in the calibration
pose (upper arms at sides, elbows bent forward 90 degrees, palms inward).
Press A+B+X+Y to calibrate and start the policy; A+X toggles full-body POSE mode.

## Abort and stop

In the attached **controller** pane, press **o** to request the controller's
stop path. While the policy is active, **A+B+X+Y** also sends a stop request.
These stop paths send damping commands; damping does not maintain standing balance.

From another terminal:

```bash
ssh unitree_g1 'cd ~/GEAR-SONIC && bash install_scripts/run_pico_real.sh abort'
```

`abort` (also named `stop`) sends `o` to the controller, waits up to ten seconds
for its exit, then closes the session. If the controller does not exit, it reports
failure and retains the session; use the robot hardware stop in that case.
Closing SSH or detaching tmux does not stop teleoperation.

```bash
bash install_scripts/run_pico_real.sh status
tail -f .local/pico-real/logs/controller.log
```

To test input without starting a robot controller:

```bash
bash install_scripts/run_pico_real.sh tracking
tail -f .local/pico-real/logs/bridge.log
# Expect head=True left=True right=True body=True.
bash install_scripts/run_pico_real.sh stop
```

Stop a tracking-only session before running `start`. Logs are stored under
`.local/pico-real/logs/`. The launcher refuses to start over an existing session,
running ETHRC containers, or occupied service ports.

## Motion fidelity

If the robot's motion does not match the operator's, see
[pico_fidelity.md](pico_fidelity.md) for the capture report that separates PICO
tracking, SMPL mapping and timing.

## Configuration and validation

- DDS uses `enP8p1s0`, domain 0. CRC and hardware velocity checks stay enabled.
- Low latency encoder, decoder, and observation config are copied byte-for-byte
  from the PC simulation's `policy/low_latency` bundle.
- Controller flags: `--stream-lookahead-frames 4 --immediate-stream-input`.
- Local connections: manager publishes :5556; controller feedback :5557;
  tracking/feedback bridge :5558; XRoboToolkit SDK service :60061.
- ARM SDK and controller rebuilt from `feature-low-latency-sim`; existing onboard
  TensorRT memory-mapping changes retained. TensorRT caches built onboard.
- 45 tracking/resampling Python tests and six C++ input/playback tests passed.
- Service, bridge and manager startup checked without motor control. Live PICO
  head/controllers/body data received, but the headset disconnected during the
  measurement; continuous streaming and real-robot latency remain to be checked.
