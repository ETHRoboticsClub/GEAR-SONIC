# G1 + Pico teleoperation setup

Session: **12 October 2026**.

## Setup used

```text
Pico headset/controllers ── Wi-Fi / XRoboToolkit ── PC ── Ethernet / ZMQ ── G1
```

| Item | Configuration |
|---|---|
| PC repository | `/home/ubuntu-alain-ethrc/Documents/eth_rc/GEAR-SONIC` |
| PC Python environment | `.venv_teleop` inside this repository; now installed |
| PC Wi-Fi IP | `172.20.10.8` — enter this in Pico's **PC Service** field |
| PC Ethernet IP used | `192.168.123.99` — use this for the G1's `--zmq-host` |
| PC Ethernet interface | `enp4s0` |
| G1 login | `unitree@192.168.123.164` |
| G1 hostname | `unitree-g1-nx` |
| G1 repository | `/home/unitree/GEAR-SONIC` |
| Controller | SONIC v1.1 |
| Planner | `planner/target_vel/V2/planner_sonic.onnx` |
| Teleop input / output | `zmq_manager` / `zmq` |
| PC command publisher | TCP `5556`: pose, planner, and commands |
| G1 feedback publisher | TCP `5557`: robot feedback |

Use the PC repository above. The earlier command pointing to
`.../eth_rc/g1_teleop/GEAR-SONIC` referred to a different checkout.

The PC's Wi-Fi IP was initially `10.5.11.190` and later changed to `172.20.10.8`.
Use the address for the Wi-Fi network currently shared with Pico. At the last
observation, Ethernet was disconnected; reconnect it before using the wired
PC/G1 setup above.

## G1 command

Open a robot terminal from the PC:

```bash
ssh unitree@192.168.123.164
```

Run on the G1:

```bash
cd /home/unitree/GEAR-SONIC/gear_sonic_deploy

bash deploy.sh \
    --cp policy/sonic_v1_1/model \
    --obs-config policy/sonic_v1_1/observation_config.yaml \
    --input-type zmq_manager \
    --output-type zmq \
    --zmq-host 192.168.123.99 \
    real
```

Confirm the deployment prompt and wait for `Init Done`. Keep this terminal
accessible for the **O/o** stop key. The launcher sources its environment setup
and builds the deployment automatically.

## PC command

```bash
cd /home/ubuntu-alain-ethrc/Documents/eth_rc/GEAR-SONIC
source .venv_teleop/bin/activate

python gear_sonic/scripts/pico_manager_thread_server.py \
    --manager \
    --input-source xrt \
    --zmq_feedback_host 192.168.123.164 \
    --vis_vr3pt --vis_smpl
```

Remove `--vis_vr3pt --vis_smpl` to run without visualization.
The feedback flag uses **underscores** and must point to the G1; otherwise it
uses `localhost` and looks for robot feedback on the PC.

The streamer automatically starts the installed XRoboToolkit service at
`/opt/apps/roboticsservice/runService.sh` and waits for Pico body-tracking data.
There is no need to launch the service separately for this workflow.

## Pico connection and start sequence

1. Start the G1 deployment.
2. Start the PC streamer.
3. Connect Pico to the same Wi-Fi network as the PC.
4. In Pico's XRoboToolkit app, enter **`172.20.10.8`** as **PC Service**, or the
   PC's updated Wi-Fi address. Wait for **WORKING**.
5. Enable **Head**, **Controller**, **Full body**, and **Send**. Calibrate the
   body trackers.
6. Assume the calibration pose: upright, feet together, upper arms at your
   sides, elbows bent 90 degrees, palms inward.
7. Press **A+B+X+Y** to engage/calibrate and enter PLANNER mode.
8. Align your body with the robot, then press **A+X** for whole-body POSE mode.
   Press **A+X** again to return to PLANNER.

The G1 can also start after Pico connects. The required order is that the
XRoboToolkit PC service runs before Pico connects; starting the streamer first
handles that.

## Stop deployment

- **G1:** press **`O` or `o`**, the letter O, in the terminal running deployment.
  This is a program shortcut, not a shell command. It requests shutdown and a
  damping-only motor command; it does not hold the robot standing.
- **Pico:** press **A+B+X+Y** while active to send stop. In this version, the
  streamer then exits. Relaunch deployment and the streamer for another session.
- After stopping the G1, press **Ctrl+C** in the PC terminal if the streamer is
  still running. Stopping only the PC process does not explicitly send the G1
  stop command.

## Issue encountered: missing default model

The first G1 launch failed with:

```text
Cannot find ONNX mode at path: policy/release/model_decoder.onnx
Failed to initialize control policy from: policy/release/model_decoder.onnx
Aborted (core dumped)
```

The original command selected the default `policy/release/model` prefix. The
G1 had the SONIC v1.1 encoder and decoder installed instead:

```text
gear_sonic_deploy/
├── policy/sonic_v1_1/
│   ├── model_encoder.onnx
│   ├── model_decoder.onnx
│   └── observation_config.yaml
└── planner/target_vel/V2/planner_sonic.onnx
```

The correction was to select both:

```bash
--cp policy/sonic_v1_1/model \
--obs-config policy/sonic_v1_1/observation_config.yaml
```

These flags are included in the G1 command above. `--cp` is a prefix; the
launcher appends `_encoder.onnx` and `_decoder.onnx`. No download was needed
for the installed v1.1 files. The failure occurred during model loading,
before teleoperation, and was not evidence of a ZMQ connection problem.

## Environment setup command

The correct PC checkout initially lacked `.venv_teleop`; it is now installed.
For a fresh installation, the command is:

```bash
cd /home/ubuntu-alain-ethrc/Documents/eth_rc/GEAR-SONIC
bash install_scripts/install_pico.sh
```

Run this only for installation: it deletes and recreates `.venv_teleop`.
Normal sessions only need `source .venv_teleop/bin/activate`.
