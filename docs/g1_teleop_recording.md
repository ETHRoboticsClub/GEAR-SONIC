# Recording teleop on the G1

The whole chain, in order: power on, start teleop, record episodes, stop, export,
then fine-tune GR00T on the result (section 7). Recording runs on the robot from one
folder: `/home/unitree/GEAR-SONIC`, checked out on `main`. Do not `cd` to `/home/unitree/worktrees/gear-sonic-record/iface`.
Do not leave uncommitted changes in `/home/unitree/GEAR-SONIC`: what is not on
`main` is lost the day the robot is reinstalled. If the robot itself is lost or
wiped, "Rebuilding the setup" at the end lists what to reinstall.

This procedure was run end to end on 2026-11-21: 5 episodes of
"grasp the bottle" exported to `ETHRC/grasp_bottle_g1` and
`ETHRC/grasp_bottle_g1_gr00t` (private).

The head camera is the **OAK-D W** in the crown mount. The recorder opens it with
`--camera oakd`. The ZED instructions are kept at the end for reference.

## What this is

The recorder follows the low-latency PICO teleop on the G1 without joining that
process. It subscribes to `g1_debug` on port 5557 and to tracking on port 5558,
and it reads the OAK color camera, JPEG-encoded on the camera itself (640x360,
30 fps). It does not bind the teleop ports and does not start the motors.

In the recorder terminal, a blank Enter starts an episode. The next blank Enter
stops it, and the terminal asks `success? [y/n]`. The PICO buttons do not mark
episodes. A+B+X+Y is not the episode marker.

## 0. Power on and checks

1. Power the robot on with the **hands plugged in**, the OAK on the crown USB
   cable, and the Wi-Fi dongle (antenna) plugged in. The robot must be supported
   (gantry or someone holding it) before teleop starts.
2. Wait about 2 minutes, then connect as `unitree` to the robot's onboard computer
   (hostname `unitree-g1-nx`), over the club VPN or the lab Wi-Fi. The club keeps
   the addresses; they are not in this public repository.
3. The hotspot `teleop-sim` (10.42.0.1, 5 GHz) starts by itself at boot on the
   Wi-Fi dongle. Check: `nmcli -t -f NAME,DEVICE c show --active` lists `teleop-sim`.
4. Hands: at boot both Dex3 hands move briefly and their LEDs blink green. Check
   that they answer:

   ```bash
   cd /home/unitree/GEAR-SONIC
   .venv_sim/bin/python gear_sonic/scripts/dex3_hand_check.py --seconds 5
   ```

   `hl` and `hr` (hand state messages per second) must be about 500-800. `0`
   means the hands are not talking to the robot even if their LEDs are on. On
   2026-11-21 that was the hand cable connector: the black hand cable (red,
   white, green, black wires) goes into a small clear module marked `DEVICE`
   behind the back cover, then into the board. Power off, reseat both connectors
   of that module, power on again. Arm motion pulls on that cable, so keep it
   strapped with slack before the module.
5. OAK: `lsusb -d 03e7:` must list `03e7:2485` (Movidius MyriadX). It shows on
   USB 2 until a program opens it; the recorder then reports `usb_speed: SUPER`.
6. Disk: `df -h /`. The recorder warns below 5 GiB free and refuses below 2 GiB.
   An OAK take uses about 95 MB per minute of raw log.

The club stack must not run at the same time. If ETHRC containers are up
(`docker ps`), stop them first:

```bash
cd /home/unitree/ETHRC-Humanoid-Controls && just down
```

## 1. Terminal A: start teleop

Support the robot. `start` moves it toward the default standing pose right away.

```bash
cd /home/unitree/GEAR-SONIC && bash install_scripts/run_pico_real.sh start
bash install_scripts/run_pico_real.sh attach
```

`attach` shows the `teleop` window: controller on the left, bridge and manager on
the right. Wait for a new `Init Done` in the controller pane (about 30 s). An older
line from a previous session does not count.

**Do not type in this window.** In the controller pane, the key `o` is the
controller's emergency stop: the controller exits at once and the robot goes limp.
Detach with Ctrl+b then d; teleop keeps running.

## 2. PICO

On the PICO, join the Wi-Fi network `teleop-sim`. In XRoboToolkit set PC Service
to `10.42.0.1`. Enable Head, Controller, Send, and Full body. The bridge pane
should show:

```text
Tracked head=True left=True right=True body=True
```

Stand in the calibration pose: upper arms at your sides, elbows bent forward 90
degrees, palms inward. Press A+B+X+Y only while the robot is supported. That
press calibrates and starts the policy, and the robot follows you.

- **A+X** toggles POSE and PLANNER. **The hands only move in POSE mode.** In POSE,
  the index trigger closes the hand, the grip brings the thumb in, release opens.
- **A+B+X+Y** again requests a stop. That stop damps the joints; it does not hold
  a stand.

## 3. Terminal B: record

Choose one dataset folder per task, for example `grasp_bottle_g1`. Each recorder
run writes a new session inside it. Use a second terminal, not the teleop window:

```bash
cd /home/unitree/GEAR-SONIC
/home/unitree/.venvs/g1-record/bin/python gear_sonic/scripts/run_pico_session_recorder.py \
  --task "grasp the bottle" --camera oakd \
  --root /home/unitree/datasets/grasp_bottle_g1/raw/$(date -u +%Y%m%dT%H%M%SZ)
```

- `--task` is the language instruction stored on every episode. Write the task.
- `--camera oakd` is required. The first line printed is the camera description;
  check `usb_speed: SUPER`.
- Without `--root`, sessions go to `/home/unitree/datasets/raw/<UTC timestamp>`,
  the default dataset of the publish step.

Press Enter on a blank line only while the robot is following you. Enter again
stops the episode, then answer `y` or `n`. You can mark many episodes in one
session. Keep one take under about 3 minutes (the export holds a whole take in
memory).

Once a second the terminal prints a health dictionary. `state`, `tracking`, and
`jpeg` must keep increasing (`jpeg` about 30 per second), and every `dropped`
count must stay 0. If `state` stops increasing, the controller is no longer
running: do not mark episodes, see section 5.

Ctrl+C quits the recorder. If an episode is still open, it is closed with
`success` left null.

## 4. Stop: recorder first, then teleop

```bash
# Terminal B: Ctrl+C in the recorder
# then, from /home/unitree/GEAR-SONIC:
bash install_scripts/run_pico_real.sh stop
```

`stop` asks the controller to take its stop path (damping, the robot goes limp:
support it), waits up to ten seconds, then closes the session. If the controller
does not exit, use the robot hardware stop.

## 5. The controller stopped (after `o`, a stop, or by itself)

```bash
bash install_scripts/run_pico_real.sh status
```

`controller: EXITED` means the controller program is gone. A+B+X+Y cannot bring it
back. Restart everything:

1. Ctrl+C the recorder.
2. Support the robot, then `bash install_scripts/run_pico_real.sh stop` and
   `bash install_scripts/run_pico_real.sh start`. Wait for the new `Init Done`.
3. Calibration pose, A+B+X+Y, then start a new recorder run (section 3).

## 6. Export (with teleop stopped)

Export builds a LeRobot v3 dataset and a GR00T v2.1 copy, then pushes both to
Hugging Face. For a task folder such as `grasp_bottle_g1`:

```bash
cd /home/unitree/GEAR-SONIC
B=/home/unitree/datasets/grasp_bottle_g1
/home/unitree/.venvs/g1-record/bin/python gear_sonic/scripts/publish_episodes.py \
  --raw-root $B/raw --v3-root $B/v3 --groot-root $B/groot \
  --repo-id ETHRC/grasp_bottle_g1 --groot-repo-id ETHRC/grasp_bottle_g1_gr00t --private
```

- Without `--private` the repos are **public**. The camera sees people in the lab.
- Without these flags, the command uses `/home/unitree/datasets/raw` and appends to
  the public `ETHRC/g1_pico_sonic` and `ETHRC/g1_pico_sonic_gr00t`, which already
  hold a ZED episode from 2026-11-17. Do not mix cameras in one dataset.
- To export only some sessions, list their directories at the end of the command.
- It takes about five times the recorded length on the Jetson. To keep it running
  after you close SSH, start it with `nohup ... > $B/publish.log 2>&1 &`.
- It refuses to encode while `g1_deploy_onnx` is running.
- The token is `~/.config/g1-record/hf_token`, mode `600`.
- Errors about `torchcodec` / `libnppicc.so` in the output are harmless: it falls
  back to PyAV.
- **Interrupted** (power cut, Ctrl+C)? Run the same command again. Episodes already
  in `$B/v3/meta/sources.jsonl` are not appended twice; the GR00T conversion and the
  upload resume. On 2026-11-21 the robot lost power during the GR00T conversion; the
  same command run again printed `no new episodes; uploaded episodes from an earlier
  run` and both repos arrived complete (5 episodes, 6757 frames each).

A window is skipped (`skip ...` line) when it has no robot state, or less than one
continuous second of valid rows. After both uploads succeed, a raw session is
deleted only if every Enter-marked window in it was uploaded. `--no-push` builds
locally and deletes nothing.

Success looks like:

```text
episodes added: 5
  20261121T113041Z episode 1 success=True rows=2122 valid_rows=2122
...
v3 repo: https://huggingface.co/datasets/ETHRC/grasp_bottle_g1
groot repo: https://huggingface.co/datasets/ETHRC/grasp_bottle_g1_gr00t
```

## 7. Fine-tune GR00T N1.7 on the task

This step runs on a GPU in the cloud, launched from your own computer, not from the
robot. It reads the `_gr00t` dataset from section 6 and writes a private model repository.
The full reference is `ethrc/groot_finetune/README.md`.

**Once per person:**

1. Hugging Face: sign in and accept the terms of
   [nvidia/Cosmos-Reason2-2B](https://huggingface.co/nvidia/Cosmos-Reason2-2B). GR00T
   N1.7 is built on it; until you accept, the run stops at its checks. Then run
   `hf auth login` with a token that reads the `ETHRC` datasets and can write the model
   repository.
2. SkyPilot, the club's cloud: `uv tool install 'skypilot[aws]'`, then
   `sky api login -e https://sky.ethrobotics.ch`. The launcher refuses to start
   without this login.
3. Clone this repository on your computer (`git clone https://github.com/ETHRoboticsClub/GEAR-SONIC.git`).

**Each fine-tune**, from the repository root on your computer:

```bash
ethrc/groot_finetune/cloud/train.sh ETHRC/grasp_bottle_g1_gr00t ETHRC/gr00t-g1-grasp-bottle
sky logs gr00t-grasp-bottle-g1-gr00t      # follow the run
```

- The first argument is the `_gr00t` dataset. The success labels are read from its
  twin without the suffix (`ETHRC/grasp_bottle_g1`).
- By default the takes answered `n` are left out. Add `--episodes successful` to keep
  only the `y` takes, or `--episodes all` to keep every take.
- The second argument is the model repository. It is created private.
- It runs on one L40S on AWS, or on an H100 if no L40S is available (about three times
  the price). The machine stops 30 minutes after the job ends. Its disk, with the
  checkpoints, is billed until `sky down -y <cluster>`. Agree on the budget before
  launching.
- Defaults are 10000 steps, global batch 32, learning rate 1e-4. Add flags after the
  two names, for example `--max-steps 20000`.
- Interrupted? Run the same command again: it resumes from the latest checkpoint.

**Check a dataset without a GPU.** On any machine with Isaac-GR00T installed
(`ethrc/groot_finetune/setup_groot.sh`), `--prepare-only` checks the dataset and selects
the episodes. It then computes the statistics and loads training steps with GR00T's
loader:

```bash
cd ~/Isaac-GR00T && uv run python <GEAR-SONIC>/ethrc/groot_finetune/train.py \
    --dataset ETHRC/grasp_bottle_g1_gr00t --prepare-only
```

On 2026-11-21 this check passed on `ETHRC/grasp_bottle_g1_gr00t`: 4 of 5 takes were
kept (the failed one was dropped) and it found 5057 training steps. No GPU fine-tune has
run yet. Five takes are enough to test the pipeline, not to train a useful policy; plan
on several dozen successful takes per task.

**Use the model.** Section 8 runs it on the G1.

## 8. Run the fine-tuned model on the G1 (GR00T on a DGX Spark)

The model runs off the robot: the G1's Orin NX shares its GPU with the SONIC controller
and would take most of a second per inference. GR00T serves it from one of the club's DGX
Sparks (about 120 ms per inference); on the robot, `run_vla_inference.py` sends it the head
camera and the robot state and plays the 40 motion tokens it answers through the same
low-latency SONIC controller as teleop. SONIC keeps balance; the model replaces the
operator. Frames travel as JPEG (`--image-codec jpeg`, about 40-90 KB instead of 690 KB),
which keeps a round trip within the 0.4 s between inferences on the lab's uplink.

**On your computer** (SkyPilot logged in, as in section 7):

```bash
ethrc/groot_finetune/cloud/serve.sh NAMESPACE/MODEL      # about 20 min the first time
sky logs gr00t-serve                                     # wait for "Server is ready"
ssh -N -L 0.0.0.0:5550:localhost:5550 gr00t-serve        # keep it open: the robot reaches GR00T here
```

The server listens inside the club's cluster only; that SSH forwards it to your computer,
which must be on the robot's network. `sky down -y gr00t-serve` frees the Spark afterwards.

**On the robot**, from `/home/unitree/GEAR-SONIC`, with the robot supported (gantry):

```bash
cd /home/unitree/ETHRC-Humanoid-Controls && just down      # the club stack must not run
cd /home/unitree/GEAR-SONIC
export GR00T_HOST=<your computer's address on the robot's network>
export VLA_PROMPT="grasp the bottle"       # the task text the model was trained with
bash install_scripts/run_vla_real.sh install               # once: the client's packages in .venv_teleop
bash install_scripts/run_vla_real.sh check                 # models, venvs, ports, the server
bash install_scripts/run_vla_real.sh start                 # the controller stands the robot up
bash install_scripts/run_vla_real.sh attach                # windows: camera controller vla keyboard
```

Wait for `Init Done` in the `controller` window. In the `keyboard` window, each key then
Enter: `k` starts the control loop (standing, PLANNER mode), `i` sends the initial pose and
switches to POSE, `p` lets the policy run (it starts paused); `p` again pauses it, `k`
stops the loop. `o` in the `controller` window is the emergency stop, as in teleop. Stop
everything with `bash install_scripts/run_vla_real.sh stop`.

- The camera is opened the recorder's way (`run_vla_camera.py`: the OAK's on-device JPEG
  at 640x360, quality 90; `VLA_CAMERA=/dev/video0` for a ZED's left eye at the same size),
  so the model sees what it was trained on. `composed_camera` opens the OAK at 640x480.
- The `vla` window prints each inference's latency; the image age is printed by the
  camera client. A round trip near or above 0.8 s (one action chunk) makes the motion
  jerky: check the computer's network first.
- Measured on 2026-11-28: 119 ms per inference on the Spark; 290 ms median (410 ms p90)
  round trip from a laptop through the SSH forward, with an 89 KB test frame.
- `install` adds the client's packages (extra `vla_client`: OpenCV, tyro, pinocchio, ...)
  to `.venv_teleop`; no Isaac-GR00T is installed on the robot.
- With the PICO hotspot instead of the lab network, the computer is on `teleop-sim` and
  `GR00T_HOST` is its `10.42.0.x` address. macOS may ask to accept incoming connections
  for `ssh` the first time.

## What is recorded

Joints, hand positions, the 64-float SONIC encoder output as
`action.motion_token`, raw PICO tracking, and the OAK color image as `ego_view`.
Not recorded: the 1247-float encoder input, wrist cameras, 26-joint XR hands.
The raw image log keeps the historical file names `zed_left.mjpeg` / `zed_left.idx`
whatever the camera; `session.json` says which camera wrote it (`camera_info`).

## Limits

- **Build memory.** The export holds a whole episode in RAM (3.8 GB for a 131 s
  take). Keep takes under about 3 minutes.
- **Timing.** Dataset grid 50 Hz; camera 30 fps, so rows share frames. A camera
  frame older than 100 ms, or a robot state gap over 60 ms, marks the row invalid.
  Action and motion token are never interpolated. Invalid rows are cut from the
  GR00T copy; segments under 1 s are dropped.
- **Controller exits.** The controller has exited during sessions (planner init
  timeout, a stop, the `o` key). Watch `state` in the health line.
- **Privacy.** Anything in the camera view is in the training video.
- **Tools.** ffmpeg 4.4.2 with libx264 is installed; encoding uses PyAV.

## Rebuilding the setup

What the working robot had on 2026-11-21, and where each piece comes from. Code is
in this repository. Model weights, virtual environments, and the built controller
are not; the commands below recreate them. Run everything on the G1 (Jetson Orin
NX, aarch64) as user `unitree`, from `/home/unitree/GEAR-SONIC` unless stated.
These steps are what the robot ran; they have not been replayed on a blank board.

| Piece | Where it came from | Recreate |
|---|---|---|
| This repository | `ETHRoboticsClub/GEAR-SONIC`, branch `main` | `git clone https://github.com/ETHRoboticsClub/GEAR-SONIC.git ~/GEAR-SONIC` |
| Low-latency policy (`model_encoder.onnx`, `model_decoder.onnx`) and planner (`planner_sonic.onnx`) | Hugging Face `nvidia/GEAR-SONIC`; the robot's files are byte-identical to it (sha256 checked 2026-11-21) | `python download_from_hf.py --low-latency`, then `git checkout -- gear_sonic_deploy/policy/low_latency/observation_config.yaml` (the download replaces the club's tracked config with NVIDIA's, which differs) |
| TensorRT engines (`*.trt`) | Built on the robot from the ONNX files on first start | Nothing to do; first start takes longer |
| Controller `gear_sonic_deploy/target/release/g1_deploy_onnx_ref` | Built on the robot | `docs/source/getting_started/installation_deploy.md` (`just build` in `gear_sonic_deploy`). `run_pico_real.sh check` refuses a binary without `--stream-lookahead-frames` and `--immediate-stream-input` |
| `.venv_teleop` (bridge, manager) | `install_scripts/install_pico.sh` | `bash install_scripts/install_pico.sh` |
| `.venv_sim` (Unitree SDK for `dex3_hand_check.py`) | `install_scripts/install_mujoco_sim.sh` | `bash install_scripts/install_mujoco_sim.sh` |
| XRoboToolkit PC service (`/opt/apps/roboticsservice`) | `gear_sonic_deploy/thirdparty/roboticsservice_1.0.0.0_arm64.deb` | `sudo dpkg -i gear_sonic_deploy/thirdparty/roboticsservice_1.0.0.0_arm64.deb` |
| Recorder venv `~/.venvs/g1-record` (Python 3.12) | `gear_sonic/data/record/requirements.txt` (includes `depthai` for the OAK) | `uv venv ~/.venvs/g1-record --python 3.12 && uv pip install --python ~/.venvs/g1-record/bin/python -r gear_sonic/data/record/requirements.txt` |
| GR00T converter venv `~/.venvs/groot-convert` (Python 3.11) | Used by the export (`gear_sonic/data/record/convert/segment.py`). The robot had lerobot 0.4.1, datasets 4.1.1, pandas 3.1.1, pyarrow 26.0.1, torch 2.7.1, av 15.1.0, jsonlines 4.0.0 | `uv venv ~/.venvs/groot-convert --python 3.11`, then install those versions |
| GR00T conversion script | `~/ETHRC-Humanoid-Isaac-GR00T` (club fork, robot at 376ba89), `scripts/lerobot_conversion/convert_v3_to_v2.py` | `git clone https://github.com/ETHRoboticsClub/ETHRC-Humanoid-Isaac-GR00T.git ~/ETHRC-Humanoid-Isaac-GR00T` |
| ffmpeg with libx264 | Ubuntu package (4.4.2 on the robot) | `sudo apt install ffmpeg` |
| Hugging Face token | Club robot token with write access to the `ETHRC` org | Put it in `~/.config/g1-record/hf_token`, `chmod 600`. Never commit it |
| Hotspot `teleop-sim` | NetworkManager profile on the Wi-Fi dongle: SSID `teleop-sim`, 5 GHz (band `a`, channel 36), `ipv4.method shared` (10.42.0.1/24), WPA-PSK, autoconnect | `sudo nmcli connection add type wifi ifname <dongle> con-name teleop-sim ssid teleop-sim mode ap 802-11-wireless.band a 802-11-wireless.channel 36 ipv4.method shared wifi-sec.key-mgmt wpa-psk wifi-sec.psk <password>`. Ask the club for the password the PICOs already know |
| OAK udev access | `/etc/udev/rules.d/80-cameras.rules` gives every user access to Luxonis devices (vendor `03e7`) | Without it, `--camera oakd` reports no camera. Luxonis' rule: `SUBSYSTEM=="usb", ATTRS{idVendor}=="03e7", MODE="0666"` |

Checks after a rebuild: `bash install_scripts/run_pico_real.sh check` prints
`Preflight passed`; `.venv_sim/bin/python gear_sonic/scripts/dex3_hand_check.py --seconds 5`
shows the hands; `~/.venvs/g1-record/bin/python -m pytest -q gear_sonic/tests/record`
passes; a recorder run with `--camera oakd` prints `usb_speed: SUPER`.

## ZED (former camera)

`--camera /dev/video0` instead of `--camera oakd`. The ZED needs USB id
`2b03:f682` at speed `5000` (`f681` at speed `12` means a USB 2 socket). It
records 1280x720 left-eye JPEG, about 300 MB per minute.
