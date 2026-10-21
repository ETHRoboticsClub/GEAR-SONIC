# Pico tracking to G1 motor commands

When the Pico and G1 share Wi-Fi and the teleop processing runs on the G1,
Wi-Fi carries tracking from the headset to the G1. Motion processing, policy
inference and the Python-to-controller connection can all stay on the G1.
The final motor connection uses the G1's internal Unitree interface.

There are two controller arrangements: the standalone C++ controller launched
by `gear_sonic_deploy/deploy.sh`, and the ETHRC robot API and policy runtime.
The current Pico server reads operator tracking through ETHRC. It does not
read the Pico SDK directly, so starting only this server and `deploy.sh` does
not complete the tracking path.

This page describes source behavior checked at GEAR-SONIC `d04dd50` and
ETHRC-Humanoid-Controls `0b8ab4af`. It does not establish which revision,
checkpoint, services or achieved rates are running on a particular G1.

## How tracking passes through the robot API

The API **receives a stream pushed by the teleop module**. It does not poll
the Pico or fetch a new pose for every policy tick. The SONIC bridge opens a
subscription to that stream, then feeds the received frames to the Pico server.
After the server converts tracking into motion references, the bridge sends
those references back to the API on a separate stream.

For the XRoboToolkit backend, the path is:

```text
PICO HEADSET
  headset/controllers/body tracking
    -> XRoboToolkit app
         |
         | Wi-Fi: app connects to the G1's Wi-Fi IP
         v
G1
  XRoboToolkit PC Service
    -> xrobotoolkit_sdk, polled by ethrc_teleop's xrt backend
    -> backend stdout pipe: length-prefixed MessagePack samples
    -> ethrc_teleop: validate tracking and convert coordinate frames
         |
         | gRPC ProvideTeleop: push TrackingFrame messages
         v
  Robot API: validate and distribute tracking to subscribers
         |
         | gRPC WatchTracking: server-streaming subscription
         v
  ETHRC SONIC bridge (ethrc_sonic_teleop)
         |
         | ZMQ localhost:5557, topic tracking, MessagePack
         v
  pico_manager_thread_server.py --manager
    -> TrackingReader -> PoseStreamer or PlannerStreamer
         |
         | ZMQ localhost:5556, topics pose / planner / command
         v
  ETHRC SONIC bridge: convert messages to typed motion references
         |
         | gRPC InputService.StreamInput
         v
  Robot API -> assigned SONIC policy runtime
    -> encoder -> decoder -> joint-position targets
    -> ETHRC body control and Unitree hardware backend -> motors
```

The API appears twice because the messages have different meanings:

| API stream | Initiator and direction | Payload |
|---|---|---|
| `TeleopService.ProvideTeleop` | Teleop module opens a bidirectional stream; tracking flows module → API | Operator poses, controller inputs and tracking status; backend-selection requests flow back |
| `TeleopService.WatchTracking` | SONIC bridge opens a subscription; frames flow API → bridge | The module's tracked operator frames |
| `InputService.StreamInput` | SONIC bridge opens a stream; references flow bridge → API | `SmplMotion` or planner references for an assigned robot operation; receipts flow back |

The API's tracking distributor uses a bounded queue per subscriber (four
frames in the checked implementation), dropping the oldest on overflow.
This path forwards arrivals; it does not resample them to 50 Hz. The provider
also bounds its pending tracking queue to three frames.

The teleop module judges whether poses are present, finite, have unit
quaternions and are still changing. It converts tracking into operator space
(x forward, y left, z up). `TrackingReader` converts those frames back to the
OpenXR convention expected by the existing retargeting code. The XRoboToolkit
backend currently assumes its SDK poses use OpenXR stage coordinates; the
controls repository records vendor-frame verification as an open item.

Sources: [XRT backend][xrt], [provider][provider],
[API tracking distributor][api-tracking], [RPC handlers][rpc],
[SONIC bridge][bridge], [TrackingReader][reader].

## How the processes and connections are established

ETHRC's `containers/compose.common.yaml` defines separate `teleop` and
`sonic-teleop` services. The G1 overlay places them on housekeeping CPUs 0–4.
They use the robot-local API; the normal control endpoint is
`127.0.0.1:50051` and the data endpoint is `127.0.0.1:50053`.

1. `ethrc_teleop` opens `ProvideTeleop`. When the lease holder selects the
   `xrt` backend through `ControlService.SetTeleopBackend`, its supervisor
   starts the backend child. That child launches `$XRT_HOME/runService.sh`,
   initializes `xrobotoolkit_sdk`, and polls for samples. The default service
   directory is `/opt/apps/roboticsservice`.
2. `sonic-teleop` registers as an input source. It binds the feedback publisher
   on `127.0.0.1:5557` and connects its subscriber to `127.0.0.1:5556`.
   It does not acquire a lease or start a robot operation itself.
3. When an operator starts a SONIC policy with that input source, the bridge
   starts the Pico server as its child. Defaults are
   `/opt/sonic-teleop-venv/bin/python` and the checkout at
   `/opt/sonic-upstream`. The child receives `--manager`, `--port 5556`,
   `--zmq_feedback_host 127.0.0.1` and `--zmq_feedback_port 5557`.
4. The bridge relays `WatchTracking` frames as `tracking` messages. It also
   sends measured robot joints as `g1_debug` for calibration and robot-side
   mode requests as `mode_request` on the same feedback socket.
5. The server starts in `OFF`. The controller combination A+B+X+Y captures
   calibration and enters planner mode; A+X switches between planner and
   whole-body pose mode. The selected streamer publishes references. The
   bridge converts them and submits them to its assigned operation through
   `StreamInput`. When the operation ends, the bridge stops its server child.

The manager's mode selection determines what motion is requested:

| Server mode | Reference produced |
|---|---|
| `POSE` | Whole-body SMPL reference derived from tracked body joints |
| `PLANNER` | Locomotion mode, movement direction, facing, speed and height from controller inputs |
| `PLANNER_VR_3PT` | Planner commands plus calibrated left-wrist, right-wrist and neck/torso targets |
| `PLANNER_FROZEN_UPPER_BODY` | Planner commands plus saved upper-body targets; the ETHRC bridge reports a reduction to planner behavior because its reference contract does not carry these frozen joint targets |

Sources: [compose services][compose], [bridge entrypoint][bridge-main],
[child-process setup][upstream], [mode manager and streamers][server].

## Where standalone deploy.sh fits

`deploy.sh` builds and launches `g1_deploy_onnx_ref` with encoder, decoder,
planner and observation-config paths. The shell script does not run the
policy itself. The standalone runtime consumes Pico references as follows:

```text
Pico server PUB :5556
  -> C++ ZMQManager SUB localhost:5556
  -> pose window OR locomotion planner
  -> encoder + decoder at 50 Hz
  -> motor command buffer
  -> LowCommandWriter at 500 Hz
  -> Unitree DDS rt/lowcmd
  -> G1 motor controllers

G1 rt/lowstate + rt/secondary_imu -> C++ robot-state history -> next policy tick
C++ PUB :5557, g1_debug -> Pico server's calibration/robot feedback reader
```

The older standalone reader used `xrobotoolkit_sdk` directly. The current
`TrackingReader` instead subscribes to `tracking` on port 5557. The C++
`ZMQOutputHandler` publishes `g1_debug` and `robot_config`, but no `tracking`.
Thus its robot feedback does not supply the current reader's operator input.
The ETHRC bridge is an alternative integration with the ETHRC runtime; simply
starting it alongside the standalone controller also conflicts over the
default port 5557 publisher.

Two configuration details are independent of that missing input connection:

- The Pico manager's start/stop and planner/pose protocol needs C++ input type
  `zmq_manager`. `deploy.sh` defaults to `manager`, whose initially selected
  input is the keyboard.
- `--zmq-host localhost` identifies the local Python publisher. `real` selects
  the Unitree-facing interface, preferentially one with a `192.168.123.x`
  address. Neither setting is the Pico's Wi-Fi address. The address entered
  into the Pico app identifies the machine running the PC Service, which is
  the G1 in this arrangement.

Sources: [deploy.sh][deploy], [ZMQManager][zmq-manager],
[C++ feedback publisher][output], [C++ controller][controller].

## Pico frequency, SMPL frequency and policy frequency

**A 50 Hz reference does not imply 50 Hz tracking arrivals from Pico.** There
are separate acquisition, transport, resampling and control clocks.

| Stage | Rate in the checked code |
|---|---|
| Pico/XRoboToolkit tracking | Determined by the device, runtime and arrivals; no fixed 50 Hz requirement here |
| ETHRC XRT backend | Polls SDK timestamp; sleeps 1 ms between unchanged polls; emits on news, or after 50 ms without news so tracking can become stale |
| ETHRC API and bridge tracking relay | Forwards arrivals; no 50 Hz resampling |
| `PoseStreamer` SMPL timeline | `--target_fps 50` by default: nominal 20 ms spacing |
| `PoseStreamer` ZMQ publication | Targets approximately 50 messages/s with fresh input and sufficient processing time; each message contains an overlapping five-frame window by default |
| `PlannerStreamer` publication | On each new relayed input timestamp; not fixed at 50 Hz |
| Standalone C++ input thread | 100 Hz |
| Standalone C++ encoder and decoder | 50 Hz, one new action vector per 20 ms tick |
| Standalone C++ planner thread | 10 Hz; actual replanning depends on mode, command changes and timers |
| Standalone C++ motor writer | 500 Hz, republishing the latest command between policy ticks |

`PoseStreamer.run_once()` compares sample timestamps, converts global body
rotations to local SMPL rotations, and interpolates between the previous and
current sample at a target timestamp. Positions use linear interpolation,
rotations use the script's rotation interpolation helpers. It advances the
target timestamp by `1 / target_fps` and skips duplicate input timestamps.
It emits at most one resampled frame per invocation; it cannot guarantee 50
fresh frames/s when input or processing is slower, and gaps can skip target
times. Actual arrival cadence and reference spacing must therefore be
distinguished from the configured target.

The five-frame packet is a sliding window, not five newly acquired frames on
every send. The first send waits for that window to fill. The encoder's
reference window is a separate setting: the default and v1.1 configurations
request ten consecutive SMPL frames, spanning 180 ms from first to last.
The C++ stream merger accumulates overlapping packets. Its playback cursor
retains room for the observation window; gatherers repeat the final available
frame if a requested index exceeds the available motion. These windows add
buffering behavior; neither the 20 ms policy period nor the five-frame packet
size is a measurement of end-to-end latency.

The low-latency checkpoint uses a different SMPL lookahead. Use its matching
observation configuration rather than assuming all releases use ten frames.

`pico_fps` in a pose packet is computed from the body-sample timestamps seen by
`TrackingReader`. In the ETHRC XRT route those originate from the backend's
host polling clock. It is not a direct measurement of a headset sensor's
native frequency or end-to-end delay. The `[PoseLoop] FPS` log measures the
server's processed-frame cadence.

Sources: [XRT poll loop][xrt], [reader timing][reader],
[pose resampling and publication][server], [controller periods and playback][controller].

## Encoder and decoder: the exact division

Yes: the encoder produces a latent representation of the reference, and the
decoder combines that latent with robot observations to produce actions.
There are two refinements: the reference is a window, and some encoder
features also depend on the robot's current orientation.

```text
SMPL reference window + wrist references + encoder mode
                 + reference orientation relative to robot orientation/heading
                                      |
                                      v
                                  Encoder
                                      |
                         encoded_tokens: 64 floats
                                      |
                                      v
Robot state/action history -------> Decoder
                                      |
                            action: 29 floats
                                      |
                     joint mapping, scale, default pose
                                      |
                      29 desired joint angles in radians
```

In whole-body mode the C++ stream selects encoder mode `2` (`smpl`). For
`policy/release/observation_config.yaml`, its active features are:

- `encoder_mode_4`: mode ID followed by padding, four values.
- `smpl_joints_10frame_step1`: ten frames × 24 joints × xyz, 720 values.
- `smpl_anchor_orientation_10frame_step1`: ten relative orientations
  represented by six rotation-matrix values each, 60 values.
- `motion_joint_positions_wrists_10frame_step1`: ten frames × six wrist
  angles, 60 values.

These occupy slots in the full encoder input; slots for other reference modes
are zero-filled on the normal mode path. The transmitted `smpl_pose`
axis-angle array is not itself an enabled encoder feature in this config.
The Python server uses rotations to construct SMPL joint and wrist references.

The default config's anchor orientation uses the current full robot-base
orientation. SONIC v1.1 uses the robot's heading instead. Thus the encoder is
not, in these configurations, completely independent of measured robot state.
`GatherTokenState()` runs the encoder and copies its `encoded_tokens` into
the decoder observation buffer.

For both checked default and v1.1 configs, the decoder's input tensor is
`obs_dict`, with **994 float32 values** for one robot:

| Concatenation order | Values | Meaning |
|---|---:|---|
| Token | 64 | Current encoded reference |
| Base angular velocity history | 30 | 10 × 3 |
| Body joint position history | 290 | 10 × 29; default angles subtracted |
| Body joint velocity history | 290 | 10 × 29 |
| Previous action history | 290 | 10 × 29 raw policy actions |
| Gravity-direction history | 30 | 10 × 3 |
| **Total** | **994** | `64 + 10 × (3 + 29 + 29 + 29 + 3)` |

History is sampled at 20 ms intervals, oldest to newest within each feature
block. This gives the decoder recent dynamics as well as the latest measured
state. The stale `436` comment at the top of the default YAML is not the
dimension of its enabled ten-frame observations; initialization checks their
computed dimension against the loaded model.

Sources: [default observation config][obs], [v1.1 observation config][obs-v11],
[`GatherTokenState`, observation gatherers and `CreatePolicyCommand`][controller].

## What exactly comes out of the decoder?

The exported decoder has one output tensor named **`action`**, normally shaped
**`[1, 29]`** for batch size one, with float32 values. The C++ loader checks
that there is exactly one output named `action` and that its total element
count equals 29. These are dimensionless policy actions in IsaacLab joint
order. Numerical values depend on the loaded weights and that tick's inputs;
they are not a fixed pose. The deployment conversion does not impose a
`[-1, 1]` clamp on this output.

The output indices map to joints as follows. Paired indices correspond to
left, then right, in the stated order.

| Policy output indices | Joint(s) |
|---|---|
| 0, 1 | Left, right hip pitch |
| 2 | Waist yaw |
| 3, 4 | Left, right hip roll |
| 5 | Waist roll |
| 6, 7 | Left, right hip yaw |
| 8 | Waist pitch |
| 9, 10 | Left, right knee |
| 11, 12 | Left, right shoulder pitch |
| 13, 14 | Left, right ankle pitch |
| 15, 16 | Left, right shoulder roll |
| 17, 18 | Left, right ankle roll |
| 19, 20 | Left, right shoulder yaw |
| 21, 22 | Left, right elbow |
| 23, 24 | Left, right wrist roll |
| 25, 26 | Left, right wrist pitch |
| 27, 28 | Left, right wrist yaw |

The decoder produces one action vector for the current control tick. It does
not output a trajectory, base position/orientation, joint velocities,
motor gains, or Dex3 finger commands.

Sources: [decoder exporter][export], [PolicyEngine I/O checks][policy],
[joint mappings and action scales][parameters].

## From actions to motor effort

For hardware joint index `i`, standalone `CreatePolicyCommand()` computes:

```text
j = isaaclab_to_mujoco[i]              # policy index for hardware joint i
q_target[i] = default_angles[i] + g1_action_scale[i] * action[j]
dq_target[i] = 0
tau_feedforward[i] = 0
kp[i], kd[i] = configured gains, including any requested gain scaling
```

`q_target` is an absolute joint-position target in radians, expressed as an
offset from the configured default pose. It is **not an increment added to
the currently measured angle**, nor a prediction that the robot will reach
that position exactly 20 ms later. An action of zero selects that joint's
default angle; it does not mean zero motor effort.

The nominal spring-damper interpretation of these command fields is:

```text
torque demand = Kp * (q_target - q_measured)
              + Kd * (dq_target - dq_measured)
              + tau_feedforward
```

The drive/controller generates effort from the target and feedback; the
decoder is not outputting torque. This equation explains the interface,
not undocumented firmware limits, filtering or saturation.

The 500 Hz `LowCommandWriter()` reads the latest command buffer, fills the
Unitree `LowCmd` fields (`mode`, `q`, `dq`, `tau`, `kp`, `kd`), computes its
CRC and publishes `rt/lowcmd`. A fresh 50 Hz policy action is normally reused
for about ten writer ticks. State returns through `rt/lowstate` and
`rt/secondary_imu` to close the policy loop. Dex3 hands have a separate path
from trigger/grip-derived targets to their DDS command topics.

The ETHRC runtime uses the same network action interpretation through the
bundle's joint order, scales and defaults, then applies its own target guards
and body-control path. Do not assume its downstream behavior is identical to
the standalone C++ writer. See the controls repository's [SONIC runtime
contract][ethrc-sonic].

Sources: [`CreatePolicyCommand` and `LowCommandWriter`][controller],
[gains, scales and default angles][parameters].

[reader]: https://github.com/ETHRoboticsClub/GEAR-SONIC/blob/d04dd504cda3354f53b279fdbed9fa0cd819e792/gear_sonic/utils/teleop/input_readers.py
[server]: https://github.com/ETHRoboticsClub/GEAR-SONIC/blob/d04dd504cda3354f53b279fdbed9fa0cd819e792/gear_sonic/scripts/pico_manager_thread_server.py
[deploy]: https://github.com/ETHRoboticsClub/GEAR-SONIC/blob/d04dd504cda3354f53b279fdbed9fa0cd819e792/gear_sonic_deploy/deploy.sh
[controller]: https://github.com/ETHRoboticsClub/GEAR-SONIC/blob/d04dd504cda3354f53b279fdbed9fa0cd819e792/gear_sonic_deploy/src/g1/g1_deploy_onnx_ref/src/g1_deploy_onnx_ref.cpp
[zmq-manager]: https://github.com/ETHRoboticsClub/GEAR-SONIC/blob/d04dd504cda3354f53b279fdbed9fa0cd819e792/gear_sonic_deploy/src/g1/g1_deploy_onnx_ref/include/input_interface/zmq_manager.hpp
[output]: https://github.com/ETHRoboticsClub/GEAR-SONIC/blob/d04dd504cda3354f53b279fdbed9fa0cd819e792/gear_sonic_deploy/src/g1/g1_deploy_onnx_ref/include/output_interface/zmq_output_handler.hpp
[obs]: https://github.com/ETHRoboticsClub/GEAR-SONIC/blob/d04dd504cda3354f53b279fdbed9fa0cd819e792/gear_sonic_deploy/policy/release/observation_config.yaml
[obs-v11]: https://github.com/ETHRoboticsClub/GEAR-SONIC/blob/d04dd504cda3354f53b279fdbed9fa0cd819e792/gear_sonic_deploy/policy/release/observation_config_sonic_v1_1.yaml
[export]: https://github.com/ETHRoboticsClub/GEAR-SONIC/blob/d04dd504cda3354f53b279fdbed9fa0cd819e792/gear_sonic/utils/inference_helpers.py
[policy]: https://github.com/ETHRoboticsClub/GEAR-SONIC/blob/d04dd504cda3354f53b279fdbed9fa0cd819e792/gear_sonic_deploy/src/g1/g1_deploy_onnx_ref/include/control_policy.hpp
[parameters]: https://github.com/ETHRoboticsClub/GEAR-SONIC/blob/d04dd504cda3354f53b279fdbed9fa0cd819e792/gear_sonic_deploy/src/g1/g1_deploy_onnx_ref/include/policy_parameters.hpp
[xrt]: https://github.com/ETHRoboticsClub/ETHRC-Humanoid-Controls/blob/0b8ab4af2caf945a8483e447c0be4e7d3d366b50/ros2_ws/src/app/ethrc_teleop/ethrc_teleop/backends/xrt.py
[provider]: https://github.com/ETHRoboticsClub/ETHRC-Humanoid-Controls/blob/0b8ab4af2caf945a8483e447c0be4e7d3d366b50/ros2_ws/src/app/ethrc_teleop/ethrc_teleop/provider.py
[api-tracking]: https://github.com/ETHRoboticsClub/ETHRC-Humanoid-Controls/blob/0b8ab4af2caf945a8483e447c0be4e7d3d366b50/ros2_ws/src/app/ethrc_robot_api/ethrc_robot_api/teleop.py
[rpc]: https://github.com/ETHRoboticsClub/ETHRC-Humanoid-Controls/blob/0b8ab4af2caf945a8483e447c0be4e7d3d366b50/ros2_ws/src/app/ethrc_robot_api/ethrc_robot_api/gateway/servicers.py
[bridge]: https://github.com/ETHRoboticsClub/ETHRC-Humanoid-Controls/blob/0b8ab4af2caf945a8483e447c0be4e7d3d366b50/ros2_ws/src/app/ethrc_sonic_teleop/ethrc_sonic_teleop/bridge.py
[bridge-main]: https://github.com/ETHRoboticsClub/ETHRC-Humanoid-Controls/blob/0b8ab4af2caf945a8483e447c0be4e7d3d366b50/ros2_ws/src/app/ethrc_sonic_teleop/ethrc_sonic_teleop/main.py
[upstream]: https://github.com/ETHRoboticsClub/ETHRC-Humanoid-Controls/blob/0b8ab4af2caf945a8483e447c0be4e7d3d366b50/ros2_ws/src/app/ethrc_sonic_teleop/ethrc_sonic_teleop/upstream_process.py
[compose]: https://github.com/ETHRoboticsClub/ETHRC-Humanoid-Controls/blob/0b8ab4af2caf945a8483e447c0be4e7d3d366b50/containers/compose.common.yaml
[ethrc-sonic]: https://github.com/ETHRoboticsClub/ETHRC-Humanoid-Controls/blob/0b8ab4af2caf945a8483e447c0be4e7d3d366b50/docs/sonic.md
