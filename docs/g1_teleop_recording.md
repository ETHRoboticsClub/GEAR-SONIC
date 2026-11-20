# Recording teleop on the G1

This is the procedure for the person operating the robot. It records a PICO teleop session that is already running. It does not start that session, and it does not replace it.

## What this is

The recorder follows Alain's low-latency PICO teleop on the G1 without joining that process. It subscribes to `g1_debug` on port 5557 and to tracking on port 5558, and it opens the ZED left eye over V4L2 at `/dev/video0`. Those ports stay bound by the teleop process. This program does not bind them and does not start the motors.

Raw logs go to `/home/unitree/datasets/raw/<UTC timestamp>/`. The directory name is the UTC time when the recorder starts, for example `20261120T104200Z`. The large file in that directory is `zed_left.mjpeg`, the left-eye JPEG log. `session.json` stores the task text you pass on the command line.

In the recorder terminal, a blank Enter starts an episode. The next blank Enter stops it, and the terminal asks `success? [y/n]`. Answer `y` or `n` (`yes` and `no` are accepted). The PICO buttons do not mark episodes.

Later, with teleop stopped, `publish_episodes.py` appends every new Enter-marked episode into one LeRobot v3 dataset and one GR00T v2.1 dataset, then pushes two public Hugging Face repos: `ETHRC/g1_pico_sonic` and `ETHRC/g1_pico_sonic_gr00t`. The token file is `~/.config/g1-record/hf_token` and its mode should be `600`. Those two repos already contain one 6555-frame episode from the 131 s take on 2026-11-17. A new run appends to that dataset.

After both uploads succeed, publish deletes a raw session only when every Enter-marked window in that session was uploaded. A skipped window keeps the raw session. A window is skipped when it has no robot state, or when it is shorter than 1 second of valid rows. `--no-push` writes the local datasets and does not delete raw sessions. Run the normal publish command with no extra flags; that command does push, and it does delete under the rule above.

Two checkouts are involved:

- Live teleop stays in `/home/unitree/GEAR-SONIC` on branch `low-latency-real`. `install_scripts/run_pico_real.sh` is not executable. Run it with `bash` from that directory.
- Recorder commands run from `/home/unitree/worktrees/gear-sonic-record/iface` on branch `record/episode-iface`, with `/home/unitree/.venvs/g1-record/bin/python`. Draft pull request: https://github.com/ETHRoboticsClub/GEAR-SONIC/pull/1

## 1. Check the camera and stop the other stack

The ZED must show USB id `2b03:f682` at speed `5000`, and `/dev/video0` must exist. `2b03:f681` by itself at speed `12` is the HID interface on a USB 2 socket. Unplug the camera and put it in the USB 3 socket.

```bash
ls -l /dev/video0
lsusb -d 2b03:
for dev in /sys/bus/usb/devices/*; do
  [ -f "$dev/idVendor" ] || continue
  [ "$(cat "$dev/idVendor")" = "2b03" ] || continue
  printf '%s product=%s speed=%s\n' "$(basename "$dev")" "$(cat "$dev/idProduct")" "$(cat "$dev/speed")"
done
```

You want a line with `product=f682` and `speed=5000`.

ETHRC containers (compose project `ethrc-robot`) hold the camera. Stop them before GEAR-SONIC teleop. The two stacks cannot run together.

```bash
cd /home/unitree/ETHRC-Humanoid-Controls && just down
```

The teleop launcher also refuses to start while `ethrc-controls.service` is active. If `start` prints `Stop ethrc-controls.service before standalone teleop.`, stop that service and start again:

```bash
sudo systemctl stop ethrc-controls.service
```

## 2. Start teleop

Support the robot. Startup moves it toward the default standing pose before the PICO start buttons.

```bash
cd /home/unitree/GEAR-SONIC
bash install_scripts/run_pico_real.sh start
```

Wait until a new `Init Done` line appears in the controller log from this start. An older line from a previous session does not count.

```bash
tail -f .local/pico-real/logs/controller.log
```

On the PICO, join the Wi-Fi network `teleop-sim`. In XRoboToolkit set PC Service to `10.42.0.1`. Enable Head, Controller, Send, and Full body.

The bridge log should show tracking for all four sources. The line looks like this:

```text
Tracked head=True left=True right=True body=True
```

```bash
tail -f .local/pico-real/logs/bridge.log
```

Stand in the calibration pose: upper arms at your sides, elbows bent forward 90 degrees, palms inward. Press A+B+X+Y to calibrate and start the policy. Keep the robot supported. Pressing A+B+X+Y again requests a stop. That stop damps the joints. It does not hold a stand, and it does not mark a recording episode.

To see whether the controller pane is still running:

```bash
bash install_scripts/run_pico_real.sh status
```

## 3. Record

Use a second terminal. Leave this process in the foreground. Enter and the y/n answer are read from this terminal.

```bash
cd /home/unitree/worktrees/gear-sonic-record/iface
/home/unitree/.venvs/g1-record/bin/python gear_sonic/scripts/run_pico_session_recorder.py --task "describe what the robot does" --camera /dev/video0
```

`--task` is required. The string is stored as you typed it, in `session.json` and on the episode. `--camera /dev/video0` is required for this setup. The other arguments have defaults (state port 5557, tracking port 5558, raw directory under `/home/unitree/datasets/raw/`). Do not add flags.

Once a second the terminal prints a health dictionary. `appended` for `state`, `tracking`, and `jpeg` must all keep increasing. Every count under `dropped` must stay 0. `disk_warning` is true when free space was under 5 GiB at start. `writer_error` should stay null.

`jpeg` follows the camera, about 30 frames a second. `state` should keep climbing while the controller is publishing. If `state` stops increasing while `jpeg` and `tracking` still increase, the controller is no longer sending joints. Do not mark an episode after that.

Press Enter on a blank line only while the robot is following you. The terminal prints `episode_start` and a timestamp. Health lines keep printing during the episode. Press Enter again to stop. Answer `success? [y/n]`. The terminal then prints `episode_stop`, the timestamp, and `success=True` or `success=False`. You can mark more than one episode in the same session. Any line that is not empty is ignored until the success question.

If you press Ctrl+C while an episode is still open, including while `success? [y/n]` is waiting, the recorder closes that episode with `success` left null.

## 4. Stop the recorder, then stop teleop

Press Ctrl+C in the recorder terminal first. After that process has exited, stop teleop from the live checkout:

```bash
cd /home/unitree/GEAR-SONIC
bash install_scripts/run_pico_real.sh stop
```

`stop` asks the controller to take its normal stop path, waits up to ten seconds for it to exit, then closes the session. If the controller does not exit, the script keeps the session and tells you to use the robot hardware stop. Support the robot. The stop path damps the joints and does not hold a stand.

Stopping teleop first leaves the recorder writing camera and tracking with no new joints. Publish will then skip an episode marked in that stretch.

## 5. Publish

Run this only after teleop is stopped. From the recorder worktree, with no extra flags:

```bash
cd /home/unitree/worktrees/gear-sonic-record/iface
/home/unitree/.venvs/g1-record/bin/python gear_sonic/scripts/publish_episodes.py
```

With no session path, the command scans every subdirectory of `/home/unitree/datasets/raw`. It appends each new Enter-marked window into `/home/unitree/datasets/g1_pico_sonic` (LeRobot v3, 50 Hz, 640x360), cuts invalid frames into the GR00T v2.1 dataset at `/home/unitree/datasets/g1_pico_sonic_gr00t`, then pushes both public repos.

It refuses to encode while `g1_deploy_onnx` is actually running. A zombie left in `/proc` after the controller has exited does not count as running. If the command reports that `g1_deploy_onnx` is running, stop teleop and run the same command again.

The command reads the token from `~/.config/g1-record/hf_token`. Create that file with mode `600` before the first push. A mode wider than `600` prints a warning and the push still runs. A missing or empty file stops the command. The error does not print the token.

A window that cannot be used prints a `skip` line and is not uploaded. The usual reasons are `no robot state in the window`, and a window with less than one second of valid rows (50 frames at 50 Hz). A stretch of valid rows also has to be continuous for that full second. Rows that are valid in total, but broken into pieces shorter than one second, are skipped the same way.

After both Hugging Face uploads succeed, a raw session directory is deleted only if every Enter-marked window in it was uploaded. One skipped window keeps the whole raw session, including windows from that session that did upload. Those uploaded windows are not appended a second time on the next run. `--no-push` is not part of the normal command. If it is passed, raw sessions are kept.

## What success looks like

A new episode prints one line, then the two dataset URLs. The numbers and the session name come from the take you just recorded. The shape is:

```text
episodes added: 1
  20261120T104200Z episode 1 success=True rows=3000 valid_rows=2980
rows=3000 valid_rows=2980
v3 root: /home/unitree/datasets/g1_pico_sonic
groot root: /home/unitree/datasets/g1_pico_sonic_gr00t
v3 repo: https://huggingface.co/datasets/ETHRC/g1_pico_sonic
groot repo: https://huggingface.co/datasets/ETHRC/g1_pico_sonic_gr00t
```

`success=` is `True`, `False`, or `None`. `None` means the episode was closed without a y/n answer.

The two URLs are the public datasets:

- https://huggingface.co/datasets/ETHRC/g1_pico_sonic
- https://huggingface.co/datasets/ETHRC/g1_pico_sonic_gr00t

A skipped window prints before that summary, for example:

```text
skip 20261120T104200Z 123456789-123456789: no robot state in the window
```

If nothing new is waiting and both repos are already up to date, the command prints `nothing to publish`.

## Limits

**Disk.** The left-eye JPEG is about 300 MB per minute. On 2026-11-20 the board was at 3.5 GB free and 99% full, which is already under the warning line. The health line sets `disk_warning` to true when free space at start is below 5 GiB. The recorder refuses to start when free space is below 2 GiB and exits with an error that says how many bytes are free. A three-minute take is on the order of 1 GB of JPEG. Check free space before you start, and leave headroom for the encoded datasets as well.

**Build memory.** One episode is decoded fully into RAM before it is encoded. The 131 s take used 3.8 GB. Keep one take under about 3 minutes until this changes. Episodes are processed one at a time, so several short episodes do not stack in RAM.

**Publish time.** On this Orin, publish takes about five times the recording length. The video is re-encoded as all-intra H.264, so every frame is a keyframe. A two-minute take is on the order of ten minutes of encoding and upload.

**Picture.** Training video is 640x360 H.264, every frame a keyframe, about 1.5 Mbit/s. It is softer than the raw JPEG. Raw frames are the full left eye, 1280x720. The camera itself is opened at 2560x720 YUYV, 30 fps, and only the left half is kept. The dataset grid is 50 Hz, so several rows can share one camera frame. A camera frame older than 100 ms marks that row invalid.

**Timing.** Action and the motion token are never interpolated. They stay valid for up to 30 ms, which is 1.5 periods at 50 Hz, and only when that robot message includes a motion token. Robot state may be interpolated when the samples around a grid time are at most 60 ms apart. A longer gap marks the row invalid. A camera frame older than 100 ms marks the row invalid. PICO poses may be interpolated for up to 150 ms. A stale or interpolated PICO sample does not by itself make the row invalid.

Invalid rows are cut out when the GR00T copy is built. The LeRobot v3 episode still contains the full grid, including invalid rows, when the window is uploaded. A real dropped robot message still splits the take. A piece shorter than 1 second is dropped from the GR00T copy. If the marked window never contains one continuous second of valid rows, the window is not uploaded at all.

**Controller exits.** The controller has exited on its own during sessions, once on a planner init timeout and once on a later normal Stop. Camera and tracking can keep logging after that, with no new joints. An episode marked after the joints stop has no robot state and will not be uploaded. Watch `state` in the health line, and `Init Done` is not a guarantee that the controller is still up. `bash install_scripts/run_pico_real.sh status` shows `RUNNING` or `EXITED` for the controller pane.

**Labels.** The task text is whatever you passed to `--task`. The episode already on the Hub from the test take says `pick up the cup` even though that label was not a description of a verified task. `success` can be null when the recorder is interrupted before you answer y or n.

**Privacy.** Both repos are public. The camera sees people in the lab. Anything in the left-eye view is in the training video.

**What is uploaded.** There is no Hugging Face upload of the raw JPEG log. The converted datasets are what move. The raw directory is removed only after both uploads succeed and only when every Enter-marked window in it was uploaded.

**What this recorder leaves out.** It does not record encoder input, wrist cameras, or 26-joint XR hands. The live inference path copies the left index finger into the middle finger before it builds state. This dataset does not do that copy.

**Tools on this board.** ffmpeg must be installed. It is installed here: version 4.4.2, with libx264. torchcodec does not load on this board. Encoding uses PyAV.

**Other recorder.** Ramzy's Controls recorder is a different stack. It is not this pipeline.
