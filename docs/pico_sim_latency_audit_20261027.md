# PICO → SONIC → MuJoCo latency audit, 27 October 2026

The latest full-body run contains useful timing data. Its largest measured
interruptions are gaps in fresh body input, upstream of the Python reader.
The present evidence cannot divide those gaps between headset tracking, Wi-Fi,
and the PC tracking service/SDK. Neural-network execution alone does not explain
the 156.8 ms gap observed during active streaming.

The initial audit did not start another simulation or change performance
settings. The subsequent authorized fixes are recorded below. The keyboard
test and PICO simulation remain stopped.

## Evidence from the latest run

The session started at 17:43:30. The bridge reported body tracking from
18:00:31.886 to 18:04:09.259, with one 24 ms tracking-availability interruption.
The manager entered `PLANNER`, then `POSE`; there are approximately 190 seconds
of streamed POSE data and 8,547 incoming motion-window reports.

Sources: [derived log summary](measurements/pico_sim_20261027/latest_run_log_summary.json).
Raw logs and SHA-256 manifest are preserved locally in
`.local/pico-sim/logs/latency-audit-20261027-1800/`.

| Stage or symptom | Latest-run observation | Meaning and limit |
| --- | --- | --- |
| Prepared pose output | 42.48–46.80 updates/s; median reporting window 45.14 | Below the configured 50 Hz target; excludes the first window containing idle time. |
| Fresh body input | Largest reported active gap 156.8 ms; 1,967 gaps exceeding 20 ms across the selected warning reports | Input can be unchanged across several policy ticks, then catch up. These are arrival gaps, not capture-to-PC latency. |
| Local bridge → Python reader | Reported maxima 0.6–2.5 ms | Little measured delay in this local relay; earlier capture/transport/service stages remain unmeasured. |
| Warm SMPL retargeting | Per-window medians 2.68–2.80 ms; per-window p95 values 3.50–3.90 ms | A useful optimization target, but smaller than the input gaps. |
| Warm pose preparation | Per-window medians 4.60–5.00 ms; per-window p95 values 5.81–6.39 ms | Includes SMPL; do not add the two rows. Excludes packing and subsequent stages. |
| C++ receive → input worker | Reported means about 0.018–0.023 ms after warm-up; largest reported maximum 1.011 ms | Immediate input preparation is working; no fixed polling delay here. |
| C++ stream decode/publish | Warm reporting-window means about 1.95–2.30 ms | Separate from the policy inference row. |
| Policy observation → motor command | Sampled once-per-second reports: median 1.506 ms, observed maximum 4.383 ms | Fast in these samples; not an exhaustive per-tick maximum or end-to-end timing. |
| Simulation state age at policy | Sampled-report median 4.476 ms; maximum 5.310 ms | DDS/state timing contributes a few milliseconds in the sampled data. |
| Cold start | First SMPL mapping 57.47 ms; first C++ decode/publish 10.886 ms | Prewarming could improve mode entry; these are not steady-state costs. |

The median of the warning windows' largest fresh-body gaps is 41.5 ms, and the
95th percentile of those maxima is 84.5 ms. **Those are not per-packet
percentiles.** Similarly, the table's ranges of window medians/p95s are not
global per-frame distributions. Warning reports are threshold-triggered.

The final tracking-silence period reaches 57 seconds. It is excluded from the
active-stream analysis; it must not be presented as normal motion latency.
Controller statistics use the interval between first and last incoming motion
logs, excluding the stale-input tail.

The controller's displayed `Streaming data mean delay` is also not end-to-end
latency. Its reported rolling means during active input span 11.444–15.273 ms.
The launcher supplies `127.0.0.1`, but the endpoint recognizes only the literal
`localhost` for source timestamp comparisons. In this configuration it measures
time since the C++ receive callback, excluding earlier pipeline stages. See
`zmq_endpoint_interface.hpp` (`GetLastUpdateTime`, `is_localhost_`,
`last_receive_time_`) and `g1_deploy_onnx_ref.cpp` (streaming delay calculation).

There is no timestamped raw ZMQ capture for this latest run. The older recorder
had already timed out before this POSE interval. Therefore today's exact
newest-input → action-feedback distribution, human motion → visible robot
latency, and simulation real-time factor cannot be reconstructed from these
logs. The earlier 24.3 ms median / 30.8 ms p95 measurement belongs to
**21 October**, not this run; see [historical measurements](pico_sim_latency.md).

## Causes and improvements, in priority order

| Priority | Cause and current evidence | Improvement or discriminating measurement |
| --- | --- | --- |
| 1 | Irregular tracking delivery is directly observed. Body fusion, headset application scheduling, tracker occlusion/smoothing, Wi-Fi retransmissions, TCP stalls, and the PC service/SDK can all be upstream of the reader. | Record independent head/controller/body SDK update counters and packet arrivals. Body-only pauses implicate tracking/fusion; simultaneous pauses across devices point toward shared transport/service, though further markers are needed to separate them. |
| 1 | The USB Wi-Fi path has demonstrated real faults: kernel USB timeouts, failed USB3 enumeration, and blocked NetworkManager/driver operations. This establishes a reliability problem, not the cause of every active body gap. | Compare a stable dedicated 5 GHz access point with the PC on Ethernet, changing only the network topology. This bypasses the current USB hotspot driver. The USB auto-switch workaround recovered enumeration but has not been shown to cure motion jitter. |
| 1 | This adapter currently cannot start the requested 5 GHz hotspot: channels are marked `no IR`, and AP initialization fails. The current working fallback is 2.4 GHz/channel 6. | Use hardware/driver/regulatory support that permits the intended AP operation, or a dedicated AP. Check channel occupancy, RSSI, retransmissions and USB errors during an actual run. Repeated restarts do not establish a latency fix. |
| 2 | C++ source contains `sleep(0.02)` in the main waiting loops. The running build was checked and calls POSIX `sleep` with zero seconds. Fractional seconds are truncated. | Replace with `std::this_thread::sleep_for(std::chrono::milliseconds(20))`, rebuild, then compare scheduler load and tail latency. The current calls fail to provide the intended wait; no millisecond saving has yet been measured. |
| 2 | The controller log contains 34,185 merger-prefixed lines and 8,546 endpoint-prefixed lines; merger debug logging is enabled. Total controller log is about 4.77 MB. | Gate per-packet logs behind a verbose option; keep aggregate timing and fault logs. Terminal/tee/logging work may cause jitter, but this file size does not prove disk saturation. |
| 2 | Rendering synchronization shares the Python physics loop. `base_sim.py` uses full `viewer.sync()`; current simulation speed and render stalls were not recorded. | Time physics, viewer sync, missed deadlines and simulated/wall time. Benchmark `sync(state_only=True)` where model edits allow it; use full sync when needed. Consider separating render cadence from physics while respecting viewer locking. |
| 3 | Python's tracking SUB socket does not explicitly limit or conflate its receive queue. Local relay timing was small in this run, so a large backlog is not demonstrated. | Make pose delivery favor the latest state and reject stale timestamps. Preserve button edges and stop commands separately; blindly conflating combined state/events can lose controls. |
| 3 | Warm Python preparation costs about 4.8 ms, including about 2.7 ms retargeting. It processes more body samples than ultimately produce 50 Hz grid output. | Profile before optimizing; avoid unused transforms, reuse allocations, cache unchanged finger work, and evaluate resampling raw poses before retargeting. Nonlinear retargeting/interpolation changes need motion-quality validation. |
| 3 | The policy runs at 50 Hz, producing up to roughly one 20 ms scheduling period between newly prepared input and consumption. | Measure arrival-to-tick phase, then align scheduling where practical. Raising rates without checking training assumptions is not a validated fix. |
| 3 | State publication/physics/control are asynchronous (DDS writer 500 Hz, physics 200 Hz, policy 50 Hz). | Correlate command generation, DDS receipt and the physics step that applies it. Sampled state age is already about 4.5 ms, making a DDS frequency increase a lower priority than 157 ms input gaps. |
| 3 | A fresh target can still produce slow visible joint motion because of policy behavior, actuator gains, torque limits and simulated inertia. | Compare generated motor commands and actual joint state. Do not treat the stream's partial body-joint target fields as a complete direct target for all robot joints. |
| 3 | The display adds viewer-frame phase, compositor/VSync and monitor response. None was measured today. | Use an external high-frame-rate camera showing human motion and the monitor together; correlate with application timing. This measures the result the operator actually feels. |
| 4 | CPU scheduling, GIL contention, thread oversubscription, thermal throttling and GPU contention could add tails, but no live profile establishes these as the main cause. CPU governor is already `performance`; memory pressure/swap was not apparent. | Capture per-process/thread CPU, run-queue delay, GPU load/clocks and temperatures alongside timing. Avoid assigning real-time priorities or moving small computations to the GPU without evidence. |
| 4 | TCP can delay later data behind lost bytes; serialization and service locks may add delay. The published XRoboToolkit v1.1.1 client already enables `NoDelay`, and its tracking queue is bounded to two queued messages. | Verify the installed APK build and instrument sender/receiver timing. Do not claim a missing TCP_NODELAY setting or an unbounded headset queue. Bounded queues and disabled Nagle do not remove TCP retransmission delays. |
| Reliability | Cached body state and held motion references can survive a tracking pause. A freeze followed by a jump can look like ordinary lag; release/calibration/startup failures are separate functional faults. | Use explicit stale-input status and a tested transition to a stable hold/idle, with deliberate resume. Keep policy-readiness checks and calibration status visible. |
| Handoff | Cold initialization plus initial reference-buffer filling can make the first response slower than subsequent motion. Planner transitions can also blend references. | Prewarm retargeting/decoding and measure mode-switch latency separately. Planner-specific blending is not a permanent POSE delay. |

MuJoCo documents state-only synchronization as faster, with restrictions on
which model/data changes it transfers: [official Python viewer documentation](https://mujoco.readthedocs.io/en/stable/python.html).
ZeroMQ conflation retains the latest message and does not support multipart
messages: [official socket options](https://libzmq.readthedocs.io/en/latest/zmq_setsockopt.html).
The published client behavior is visible in
[XRoboToolkit v1.1.1 TcpHandler.cs](https://github.com/XR-Robotics/XRoboToolkit-Unity-Client/blob/v1.1.1/Assets/Scripts/Network/TcpHandler.cs).
These support candidate changes; they do not demonstrate a measured improvement
on this installation.

## Model and configuration checks

The PICO launcher already selects `policy/low_latency` encoder, decoder and
observation configuration, four reference frames, and immediate stream input.
All three files and the V2 planner match the upstream release manifest at
revision `6733128a3d8a523b1418b06bca3cdf61c8b0987f`. The verification record is
`.local/pico-sim/sonic-low-latency-release-verification.json`; no additional
model-weight download was needed.

The official low-latency checkpoint uses four SMPL reference frames rather than
ten, described as approximately 80 ms instead of 200 ms reference lookahead
at 50 Hz. This is not a sensor-to-screen latency measurement. Four samples
spaced 20 ms apart span 60 ms from first to last; the newest frame can influence
the current action. Do not add the whole lookahead number to measured
newest-input latency, or shorten the trained horizon arbitrarily. The encoder,
decoder and observation configuration must stay matched.
[Official checkpoint documentation](https://nvlabs.github.io/GR00T-WholeBodyControl/getting_started/download_models.html).

Other misleading configuration interpretations to avoid:

- `--buffer_size 15` is unused inside the active `run_pico_manager` path; it is
  not evidence of a fixed 300 ms buffer. The actual pose sender has a separate
  five-frame buffer, with a one-time fill on mode entry.
- A socket `poll(timeout=100)` returns when input arrives; it does not impose a
  fixed 100 ms delay on every packet.
- A 200 ms freshness limit is an expiration threshold, not a mandatory buffer.
- Remote vision was not part of this path. Video encoding/transport/decoding
  becomes relevant if that feature is enabled later.
- Keyboard control bypasses the headset path. It can be a comparison baseline
  but cannot alone diagnose the current body-input gaps.

## Next measurement

Use one 60-second continuous POSE recording with synchronized PC-monotonic
timestamps and sequence IDs at bridge SDK observation, Python receipt,
retargeting completion, motion publication, C++ receipt/preparation, policy
consumption, DDS command receipt and physics application. Save head/controller/
body update counters, mode changes, physics real-time factor and viewer timing
alongside it. Device capture timestamps need clock synchronization before
subtracting them from PC timestamps; ping RTT is not a substitute.

Record a repeatable motion and, if visible response matters, an external
high-frame-rate video. Compare the current 2.4 GHz hotspot with a stable AP
topology first, then evaluate the C++ wait/logging fixes and render optimization
individually. Report per-frame medians/p95/max, missing-input counts and motion
quality. This separates a smoother stream from a faster robot response and
gives each proposed change a measured result.

## Fixes applied after the audit

At the operator's request, the controller was updated and rebuilt on 27 October:

- The main waiting loops now use a real 20 ms `std::this_thread::sleep_for`
  instead of `sleep(0.02)`, which truncates to zero seconds. The intended
  500 ms cleanup wait was corrected in the same way. Policy/control rates are
  unchanged; this is the supervisory thread's wait.
- The motion merger is quiet by default and follows the endpoint's existing
  `--zmq-verbose` option. The separate unconditional protocol-v3 packet print
  now follows that option too. Validation errors and aggregate timing remain.
- The source-timestamp check now recognizes `127.0.0.1` as local, matching
  `localhost`. Thus future local `Streaming data mean delay` reports include
  the Python source timestamp age rather than only time since the C++ receive
  callback. Old and new readings of this metric are not directly comparable;
  the change does not itself reduce latency or measure headset capture age.

Validation: the deployment controller and C++ tests rebuilt successfully;
six existing stream-playback/input-wakeup checks and 110 Python
teleoperation/simulation checks passed. Disassembly confirms the rebuilt
`main` calls `nanosleep` and no longer calls POSIX `sleep`. Build and Python
test logs are `.local/pico-sim/latency-fixes-build.log` and
`.local/pico-sim/latency-fixes-python-tests.log`.

No new live latency result is available. The recorded 156.8 ms fresh-body gap
has not been demonstrated to be fixed. Network/driver changes, render-sync
changes, model changes, and pose-dropping changes were not applied in this
patch. The next normal launcher invocation uses the rebuilt controller.
