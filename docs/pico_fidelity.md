# PICO teleoperation fidelity

Low-latency teleoperation follows the operator quickly but not always faithfully.
This page covers three causes fixed in the reference path and a capture report that
separates the remaining ones: PICO tracking and calibration, SMPL mapping, timing.
It applies to the standalone setups in [pico_real_g1.md](pico_real_g1.md) and
[pico_sim_this_pc.md](pico_sim_this_pc.md).

## Fixed in the reference path

**SMPL local joints use the pelvis as their origin.** The FK model's rest pelvis
has a nonzero offset. Removing only the root rotation turns that fixed offset
into a heading-dependent translation of the entire reference skeleton. The
manager now subtracts each frame's pelvis before rotating into the root frame.
The local pelvis is zero and the local skeleton stays unchanged when only the
global orientation changes; bone lengths, body rotations and root orientation
are preserved. Both the CPU and generic torch paths follow this convention.
Old processed pose recordings retain their original coordinates: regenerate
them from raw tracking for a corrected comparison. This geometric correction
does not establish live calibration accuracy or stable unsupported motion.

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
Controls observer captures are also accepted:

```bash
python gear_sonic/scripts/analyze_pico_fidelity.py report tracking.jsonl --json fidelity.json
```

A tracking-only file contains no robot target/feedback pairs. Older Controls
captures also lack device timestamps and body receipt counters: their poll rate
is not the headset's capture rate, and missing fields do not prove the headset
sent none. With counters, identities include the backend run so a restart is
not mistaken for a duplicate body.

| Section | What it compares | Reading |
| --- | --- | --- |
| Skeleton agreement | Each bone of the SMPL skeleton against PICO's joint positions | A discrepancy can reflect mapping, rest axes, proportions or inconsistent source geometry. It does not alone identify a wrong joint. |
| Bone consistency | PICO lengths and bone directions expressed in the parent joint's frame, before SMPL | Variable lengths cannot be reproduced by one fixed-shape skeleton. Direction changes can also reflect vendor joint-frame conventions. |
| Body timing | Arrival gaps and the timestamp fields carried by the recording | Receipt counters identify new body messages when available; otherwise only the poll rate is reported. `arrival_jitter_ms` describes timing variation, not absolute sensor latency. |
| Wrist following | Robot wrist joints against their reference, and the delay that best aligns them | A large error that vanishes at the best delay is lag; one that remains is tracking error. |

The policy receives PICO's joint rotations applied to a standard SMPL skeleton,
never PICO's joint positions. Torso and collar bones can differ by a few degrees
because the two skeletons have different proportions; `length_pico_over_smpl`
shows the operator's size relative to that skeleton.

The report's synthetic checks establish that it detects an injected rotation
error. Small errors against PICO positions do not establish accuracy against
the operator: PICO rotations and positions can share the same calibration error.
Large errors can also reflect different rest-bone axes, body proportions or
unrepresented hand rotations; they do not alone prove an incorrect joint map.

## Controls and Isaac Lab

The branch also retains Isaac Lab's batched CPU SMPL/three-point conversion and
optional `--three_point_lead_ms` prediction. Prediction defaults to zero in the
server. Set `SONIC_THREE_POINT_LEAD_MS=0` in Controls for a spatial-fidelity
comparison; preserve the configured deployment value in a separate latency run.
Prediction changes the reference, so improved apparent following is not evidence
that tracking or transport improved.

Controls forwards `body_update_sequence` and an already mapped
`body_sample_time_ns` on its tracking socket. The reader consumes the latter on
the provider's clock, without estimating another offset on the remote host.
Older recordings without that metadata retain their poll timestamps and cannot
establish headset capture latency. Rebuild Controls and the WebUI together when
updating the tracking protobuf contract.

A replay of 300 distinct bodies from the saved November 9 Controls simulation
session showed median PICO/SMPL forearm-direction differences of 11.6 degrees
(left) and 25.5 degrees (right); length ratios were about 1.25 and 1.27. These
are observations from one 43.35-second excerpt, not an isolated cause. Head and
hand bones also differ substantially; do not automatically apply angular offsets
from this report to the policy's trained reference.

Over the longer recording, 29,073 distinct poses show nearly constant
shoulder-to-elbow lengths, while elbow-to-wrist distances vary: their absolute
deviation from the median reaches 22.0% (left) and 16.7% (right) at p95. That
variation exists in the received PICO positions before SMPL. A rotation-only
mapping onto a fixed-shape SMPL skeleton cannot exactly reproduce it. The joint
indices match the vendor's 24-role enumeration; this does not establish that
the operator's calibration or every vendor joint frame is correct.

To isolate the source, record the PICO calibration procedure, operator posture,
tracker locations, body-scale settings and floor/heading reference. Capture
static neutral and symmetric poses, then repeated isolated elbow bends, arm
raises and yaw turns. Compare those same frames at raw PICO, bridge, SMPL,
retargeted reference and measured simulated-robot boundaries. Examine spatial
errors after time alignment, and report dropouts separately. A changed calibration
or mapping needs before/after recordings of the same motions; synthetic checks
alone cannot validate it.

## Headset-client checks

SDK receipt counters establish packet receipt, not independent sensor validity.
The public XRoboToolkit Unity client can retain an old `Body` value when
calibration or tracking mode becomes invalid, and does not check the body-read
return code before serialization. Such repeated bodies can advance transport
counters. Include this case in a live tracking-loss test; counter freshness
alone does not rule it out.

The consulted client also scales predicted display time by 1,000 for its
microsecond recording field before passing it to controller APIs documented to
take milliseconds. Keep those two units separate when fixing that client.
These are source findings, not proof that the installed headset APK contains
the same code or that either issue caused the recorded spatial discrepancy.
Identify the APK/build, then compile and compare before/after captures before
using a client correction in teleoperation. No headset update is required by
the report command itself.

Source references: [client tracking serialization](https://github.com/XR-Robotics/XRoboToolkit-Unity-Client/blob/cdc53166b0bf412efae71046c6a225eb5091605f/Assets/Scripts/TrackingData.cs),
[controller prediction API](https://github.com/XR-Robotics/XRoboToolkit-Unity-Client/blob/cdc53166b0bf412efae71046c6a225eb5091605f/PICO%20Unity%20Integration%20SDK/Runtime/Scripts/Features/PXR_Input.cs),
[body joint enumeration](https://github.com/XR-Robotics/XRoboToolkit-Unity-Client/blob/cdc53166b0bf412efae71046c6a225eb5091605f/PICO%20Unity%20Integration%20SDK/Runtime/Scripts/PXR_Plugin.cs).
