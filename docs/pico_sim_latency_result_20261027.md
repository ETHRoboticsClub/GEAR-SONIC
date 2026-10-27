# Stopped 5 GHz run: latency results, 27 October 2026

For a comparison against today's preceding session, see
[before-and-after results](pico_sim_before_after_20261027.md).

The simulator, controller, PC service, tracking bridge and recorder are stopped.
The `teleop-sim` hotspot is also stopped; the PC's other Wi-Fi connection remains
connected. The fallback driver remains loaded until unloaded or rebooted.

The saved raw measurement contains one 60-second continuous POSE interval,
after warm-up. Analysis matched 2,994 of 3,000 controller feedback samples
(99.8%) to unique motion references. The rest of the session has component logs,
but no additional raw packet capture. Packet/matching checks passed; these checks
do not establish balance or absence of robot falls.

| Measured quantity | Median | p95 | Maximum |
| --- | ---: | ---: | ---: |
| Newest encoder input received at PC manager → controller feedback | 14.265 ms | 32.924 ms | 56.194 ms |
| Pose preparation, including SMPL mapping | 4.288 ms | 4.803 ms | 5.694 ms |
| Complete motion window publication → controller feedback | 9.461 ms | 28.118 ms | 51.063 ms |
| Fresh body arrival gaps during the captured minute | 11.071 ms | 15.351 ms | 39.957 ms |
| Local bridge → capture observer | 0.270 ms | 0.406 ms | 0.829 ms |

Median/p95/maximum describe the observed samples, not guaranteed bounds. Pose
preparation includes SMPL mapping. Publication-to-feedback includes stream
preparation, scheduling phase and policy/feedback processing; these are not
three independent delays to add. A sum of stage medians need not equal the
median of the overall measurement.

Policy feedback was 50.00 Hz and pose publication 49.85 Hz. Simulation speed was
1.000012 times real time. No tracking-missing frames occurred in the raw
measurement window. Of fresh body arrival gaps, 0.688% exceeded 20 ms and
0.112% exceeded 30 ms.

**These are PC-side timings.** They exclude headset sensor/fusion and network
delay before PC receipt, DDS command application, the robot's physical/simulated
joint response, and display rendering/scanout. They do not measure complete
human-motion-to-visible-robot latency. An external synchronized observation or
video is needed for that quantity.

## What the complete session adds

Across the longer session, active timing warnings report a largest completed
fresh-body arrival gap of **88.3 ms**, versus **39.957 ms** in the recorded minute.
The largest reported local bridge-to-reader delay was 2.4 ms. Consequently, the
40 ms figure must not be described as the worst case of the entire session.
The warnings are aggregates and cannot supply full-session per-packet p95s.

The simulator logs also record fall/reset events. A fall was already present in
the log snapshot copied at the end of the capture; these event lines have no
timestamps, so they do not establish whether that fall preceded or occurred
within the measured minute. Complete logs were preserved at shutdown. The
timing measurements remain useful for the software path, but this was not a
demonstration of uninterrupted balanced motion.

The earlier 2.4 GHz session's reported largest gap was 156.8 ms. The lower
observed maxima in this session are encouraging, but durations, operator motion,
software and the driver/network setup differ. They do not isolate a causal
latency saving from 5 GHz alone. The old 157 ms figure is a tracking-arrival
gap, not a previous end-to-end latency value to subtract from the 14 ms median.

Sources: [raw-window component analysis](measurements/pico_sim_20261027/5ghz-live-20261027-components.json),
[capture metadata and simulation speed](measurements/pico_sim_20261027/5ghz-live-20261027-capture.json),
[complete-session findings and raw hash](measurements/pico_sim_20261027/5ghz-live-20261027-final-session.json).
Raw packets and full-session logs remain under
`.local/pico-sim/logs/pico-latency-20261027-190348-100649*`.
