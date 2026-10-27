# PICO → MuJoCo latency improvements

Live comparisons measured on 2026-10-21 on the Ubuntu 24.04 / GTX 1650 PC described in
[the setup guide](pico_sim_this_pc.md); section 5 adds an isolated viewer benchmark
from 2026-10-27. All live comparisons use the same
four-frame low-latency SONIC checkpoint and a 50 Hz policy. These are successive
run comparisons, not controlled estimates of each individual code change.

## 1. Correct MuJoCo pacing

Use cumulative monotonic deadlines instead of restarting the sleep period after
every physics step. This prevents occasional slow rendering steps from causing
permanent simulation slowdown.

| Measurement | Before | After |
| --- | ---: | ---: |
| Physics rate, standalone check | ~168 Hz | 200 Hz |
| Simulation / wall-clock speed, standalone check | ~0.84× | 1.00× |

This measures simulation pacing, not input latency. Later full-stack live runs
remained slightly below real time (~0.96–0.97×).

## 2. Use the PC's direct 5 GHz hotspot

Switch the PICO connection from ETHRC_SL to teleop-sim.

| Measurement | Shared Wi-Fi | Direct hotspot |
| --- | ---: | ---: |
| Ping round-trip average | 45.7 ms | 4.9 ms |
| Changed-body arrival gap, p95 | 99.2 ms | 18.5 ms |
| Changed-body arrival gap, maximum | 276.8 ms | 56.7 ms |
| Changed-body update rate | 37.7 Hz | 72.0 Hz |
| Prepared pose publication rate | 36.1 Hz | 49.9 Hz |

Sources: [shared-network capture summary](measurements/pico_sim_20261021/live-latency-20261021.json),
[hotspot capture summary](measurements/pico_sim_20261021/hotspot-latency-20261021.json).
The runs lasted 30 and 60 seconds respectively. Changed-body rate was measured
from changing pose values; it is not an SDK packet-counter measurement. Ping
is round-trip time, not one-way PICO transport latency. The shared-network
summary's 100.1 Hz feedback includes duplicate relay subscriptions; the actual
policy remained 50 Hz.

## 3. Follow the latest complete reference window

Select the newest complete four-frame window on each policy tick, correct the
last-valid-window boundary, resample onto a 20 ms reference grid, skip stale
backlog, avoid duplicate SMPL work, and compute only the four three-point
transforms that are used.

| Measurement | Sequential playback | Latest complete window |
| --- | ---: | ---: |
| Oldest reference receipt → action feedback, median | 148.6 ms | 86.7 ms |
| Oldest reference receipt → action feedback, p95 | 161.7 ms | 107.8 ms |
| Published reference lag, median | 6 frames | 3 frames |
| Manager receipt → pose publication, median | 10.0 ms | 9.1 ms |
| Three-point retargeting, isolated CPU benchmark | 4.76 ms | 1.05 ms |

Median oldest-reference age fell by **42%**. Sources:
[sequential playback summary](measurements/pico_sim_20261021/hotspot-playback-latency-20261021.json),
[latest-window summary](measurements/pico_sim_20261021/latest-window-latency-20261021.json).
Both live runs lasted 60 seconds. The retargeting benchmark is a separate
computation measurement, not an end-to-end latency measurement.

**Oldest-reference age is not newest-input latency.** The encoder reads that
reference plus three later frames. The remaining three-frame difference is the
model's 60 ms reference horizon; the newest frame can already influence the
current policy action.

## 4. Prepare on arrival; consume at the next policy tick

Wake the C++ input worker on pose arrival, publish the prepared motion snapshot,
and consume it on the next 50 Hz policy tick. Keep decoding outside the policy
thread, coalesce notifications, suppress per-packet debug output, and identify
fresh body samples using SDK receipt counters.

| Measurement | Latest window with polling | Immediate preparation |
| --- | ---: | ---: |
| Newest input receipt → action feedback, median | 28.9 ms | 24.3 ms |
| Newest input receipt → action feedback, p95 | 49.4 ms | 30.8 ms |
| Newest input receipt → action feedback, maximum | 86.3 ms | 68.8 ms |
| Complete window publication → action feedback, median | 20.1 ms | 15.0 ms |
| Complete window publication → action feedback, p95 | 40.8 ms | 22.7 ms |
| Newest input receipt → publication, median | 9.1 ms | 9.2 ms |

Newest-input latency fell by **16% at the median and 38% at p95**. Sources:
[polling component summary](measurements/pico_sim_20261021/latest-window-component-timing-20261021.json),
[immediate-preparation component summary](measurements/pico_sim_20261021/immediate-input-live-20261021-pose-components.json).
The last capture lasted 60 seconds but included a PLANNER pause and restarted
frame indices. The comparison uses its final **35.7-second uninterrupted POSE
interval**, matching 1,778 of 1,788 policy ticks. The unfiltered archived summary
is retained for rate/tracking context; its combined reference matches should not
be used across mode changes.

Live SMPL mapping was **4.38 ms median / 5.43 ms p95**, and total preparation
before packing was **8.19 / 9.87 ms**, including SMPL. A separate warm tight-loop
CPU benchmark measured ~1.94 ms; it should not replace the live figure. C++ logs
show receive-to-worker wakeup around **0.03 ms mean**, and decode/publish around
**2.9 ms mean**. Those means are distinct from the per-frame percentile metrics.

The policy still outputs at 50 Hz. Fresh-body gaps occasionally exceeded 20 ms
(2.9% of gaps in the selected interval); regular average throughput does not
guarantee a fresh input at every tick. Timing warnings identify these gaps but
do not themselves change robot control.

## 5. Reduce blocking viewer synchronization (2026-10-27)

The physics thread called the full `viewer.sync()` every 20 ms. On this scene,
copying all model/data arrays took longer than one 5 ms physics period.
The PICO launcher now passes `--viewer-state-only`, using MuJoCo's existing
`viewer.sync(state_only=True)` after one initial full synchronization.
[MuJoCo documents this faster path](https://mujoco.readthedocs.io/en/stable/python.html#passive-viewer)
for applications that do not edit model arrays at runtime. The PICO simulation
updates physics state, commands and applied forces; GUI controls still synchronize.
Other launches retain full synchronization by default, and versions without
the API fall back to full synchronization with a startup message.

A full / state-only / state-only / full comparison used the actual 43-DOF G1
scene, MuJoCo 3.17.0, 200 Hz physics and 50 Hz viewer updates. Each run lasted
three simulated seconds (600 physics steps and 150 timed sync calls).

| Measurement | Full sync, two runs | State-only, two runs |
| --- | ---: | ---: |
| Viewer sync median | 5.806 / 5.824 ms | 0.318 / 0.289 ms |
| Viewer sync p95 | 6.000 / 6.064 ms | 0.488 / 0.395 ms |
| Physics + sync work p95 | 6.154 / 6.160 ms | 0.773 / 0.619 ms |
| Steps whose work exceeded 5 ms | 150 / 150 | 0 / 0 |

This removes about **95% of the blocking viewer synchronization time**. All four
physics trajectories matched exactly in joint positions and velocities, and
viewer joint positions matched current simulation positions. All runs remained
approximately 1.00 times real time; the improvement is reduced blocking/jitter
and CPU work, not an increase in average physics frequency.

This benchmark uses contact-stabilized physics without PICO, policy inference
or DDS. It does not measure total teleoperation latency, display scanout, or
fix upstream tracking gaps. No operator interacted with the viewer. The normal
PICO simulator entry point was also smoke-tested with the option enabled and
the policy-release guard active. Ten simulation pacing/release tests passed.

Source: [complete benchmark results](measurements/pico_sim_20261027/viewer_state_only_benchmark.json).
Reproduce without starting a teleop/controller session:

```bash
source .local/pico-sim/env.sh
python gear_sonic/scripts/benchmark_mujoco_viewer.py --seconds 3 \
  --output .local/pico-sim/logs/viewer-benchmark.json
```

Keep the viewer untouched while the comparison runs. For manual full-sync
operation, run `run_sim_loop.py` without `--viewer-state-only`; runtime changes
to model arrays such as geometry colors require that full-sync mode.

## Measurement boundaries and reproduction

"Input receipt" means the PC manager's receive timestamp. Feedback is emitted
after an action is calculated. None of these figures includes headset sensing,
headset-to-PC one-way network delay, DDS command delivery, or visible/physical
robot response. The different timing definitions in steps 3 and 4 must not be
combined into an apparent 149 → 24 ms end-to-end improvement.

Run from the checkout with the simulation environment installed:

```bash
source .local/pico-sim/env.sh
python gear_sonic/scripts/measure_pico_sim_latency.py --seconds 60 --wait 180 --name repeat
python gear_sonic/scripts/analyze_pico_sim_latency.py .local/pico-sim/logs/repeat.msgpack
```

The recorder waits for POSE with tracking present, then records a wall-clock
window. The analyzer selects the longest continuous POSE interval, preventing
frame-index resets from contaminating matching. Captures remain local; compact
JSON summaries are stored beside this report. Build/test limitations and exact
controls are documented in the setup guide.
