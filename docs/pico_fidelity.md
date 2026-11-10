# PICO teleoperation fidelity

Low-latency teleoperation follows the operator quickly but not always faithfully.
This page covers two causes fixed in the reference path and a capture report that
separates the remaining ones: PICO tracking and calibration, SMPL mapping, timing.
It applies to the standalone setups in [pico_real_g1.md](pico_real_g1.md) and
[pico_sim_this_pc.md](pico_sim_this_pc.md).

## Fixed in the reference path

**Body samples are dated by capture, not by Wi-Fi arrival.** The bridge stamped
each body with the time its packet reached the PC. Wi-Fi delays packets unevenly
(72 Hz bodies, arrival gaps of 18.5 ms at p95 and 56.7 ms at most on the direct
hotspot), so a motion sampled evenly by the headset was resampled to 50 Hz as if
it sped up and slowed down. The bridge now forwards the headset's timestamps
(`device_time_ns`) and the manager maps them onto the PC clock, dating every
sample as if it had the fastest transport seen in the last two seconds. On a
modelled stream with those arrival statistics, a 1.2 Hz reach peaking at 2.6 m/s
had a reference velocity error of 0.36 m/s RMS and 2.3 times the true
acceleration; with capture stamps the error is 0.01 m/s. This is a simulation of
the timing only, not a robot measurement.

No latency is added beyond the delay a packet really had. A client that sends no
usable timestamp, and the robot API's relay, keep arrival stamps. The manager logs
`Body time base: body`, `packet` or `arrival`. `--arrival_stamps` restores the
previous behavior for comparison.

**A short tracking gap no longer re-aligns the heading.** Frame indices follow
time, so a stall of about 120 ms or more skips indices. The controller treated
that as a new stream and re-aligned the reference heading with the robot's
current heading, discarding whatever the operator had turned so far that the
robot had not yet followed. Gaps up to one second (50 frames) are now closed up:
playback and the heading alignment continue. A longer dropout still restarts the
stream. The controller prints `Tracking gap of N frames closed up` when it happens.
This change is in the C++ controller: rebuild it.

## Find what remains

Record while teleoperating in POSE mode, moving every limb through its range,
then report. The recorder only listens; it sends nothing to the robot.

```bash
python gear_sonic/scripts/analyze_pico_fidelity.py record .local/fidelity/run1.msgpack --seconds 30
python gear_sonic/scripts/analyze_pico_fidelity.py report .local/fidelity/run1.msgpack
```

Captures from `measure_pico_sim_latency.py` can be reported as well.

| Section | What it compares | Reading |
| --- | --- | --- |
| Skeleton agreement | Each bone of the SMPL skeleton sent to the policy against the same bone in PICO's own joint positions | Small errors on every limb bone: the SMPL mapping reproduces what PICO measured, so look at PICO calibration or the policy. One limb bone far above the rest: that joint is mapped wrongly. |
| Body timing | Arrival gaps of fresh bodies; whether a headset timestamp is present, advancing and running at the PC clock's rate | `arrival_jitter_ms` is what capture stamps remove. `present: false` for both fields means the client sends no timestamp and arrival stamps stay in use. |
| Wrist following | Robot wrist joints against their reference, and the delay that best aligns them | A large error that vanishes at the best delay is lag; one that remains is tracking error. |

The policy receives PICO's joint rotations applied to a standard SMPL skeleton,
never PICO's joint positions. Torso and collar bones can differ by a few degrees
because the two skeletons have different proportions; `length_pico_over_smpl`
shows the operator's size relative to that skeleton.

The report has been checked on synthetic bodies of known truth only. It has not
yet been run on a real capture.
