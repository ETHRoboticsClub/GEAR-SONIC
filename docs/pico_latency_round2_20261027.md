# Further latency work after the first 5 GHz run

The simulator and hotspot remain stopped. These changes are ready for the next
normal launcher invocation on `feature-low-latency-sim`.

## Measured CPU improvement

The saved 5 GHz recording was replayed through the pre-change and updated pose
preparation code in before/after/after/before order. Each pass consumed the same
600 tracked frames and published the same 320 motion packets. Three-point
calibration and finger targets were included; no simulation or headset was run.

| Pose preparation | Before, two passes | After, two passes |
| --- | ---: | ---: |
| Median | 3.706 / 3.709 ms | 1.796 / 1.804 ms |
| p95 | 3.923 / 3.903 ms | 1.896 / 2.026 ms |

The mean of the two median values fell by **1.907 ms (51.4%)**. This is an
offline CPU preparation saving, not a new measurement of the complete 14.3 ms
live PC-input-to-feedback median. Scheduling phase means a CPU saving does not
translate automatically into an equal reduction in live latency.

The CPU path now batches local-rotation conversion, computes the same 55-joint
rest-skeleton FK by tree level, and batches the four consumed three-point
transforms. Quaternion interpolation is also vectorized. Skeleton data is
preloaded before POSE entry. CUDA/autograd callers retain the original torch FK.
The policy rate remains 50 Hz and its trained reference context remains four
frames.

Across the replay, frame indices/timestamps, published joint targets, hand
targets, three-point targets, root quaternions and SMPL angle-axis poses were
unchanged. The largest SMPL joint-position difference was approximately
1.93e-6 m. Batched random/near-zero/large-rotation FK tests also pass against
the original torch implementation.

Source: [complete bracketed benchmark](measurements/pico_sim_20261027/pose_preparation_round2_final_benchmark.json).
Reproduce with the saved trusted pre-change script:

```bash
source .local/pico-sim/env.sh
python gear_sonic/scripts/benchmark_pico_pose_preparation.py \
  --capture .local/pico-sim/logs/pico-latency-20261027-190348-100649.msgpack \
  --baseline-script .local/pico-sim/latency-round2-baseline/pico_manager_thread_server.py \
  --output .local/pico-sim/another-preparation-benchmark.json
```

## Packet delay that arrival-gap checks can miss

Packets can arrive regularly while carrying increasingly old poses. The bridge
previously stamped its polling time; that alone could not reveal time spent in
headset, network or PC-service queues before the SDK callback.

The SDK binding now retains the headset message timestamp and PC callback
monotonic/wall timestamps alongside the body receipt sequence, atomically under
the body mutex. The bridge includes them only when their sequence matches its
body sample. Missing or mismatched timing information is omitted. The updated
extension was built, tested and installed for this checkout. Its changes are
committed directly in the
[SDK bindings source](../external_dependencies/XRoboToolkit-PC-Service-Pybind_X86_and_ARM64/bindings/py_bindings.cpp),
alongside the existing receipt-counter bindings.

The published v1.1.1 client stamps tracking-message preparation with a UTC
timestamp; it does not timestamp the original sensor measurement. See
[TrackingData.cs](https://github.com/XR-Robotics/XRoboToolkit-Unity-Client/blob/v1.1.1/Assets/Scripts/TrackingData.cs)
and [Utils.cs](https://github.com/XR-Robotics/XRoboToolkit-Unity-Client/blob/v1.1.1/Assets/Scripts/Utils.cs).
The installed headset build's actual timestamp availability will be checked
from the next recording rather than assumed.

Without synchronized clocks, the analysis does **not** label the raw wall-clock
difference as one-way latency. It compares source-message intervals with SDK
receipt intervals, and reports additional transit time relative to the fastest
observed delivery. Clock reversals and large discontinuities suppress/reset that
estimate. Constant latency remains unknown. A test demonstrates regular 12 ms
arrivals whose source messages were generated every 10 ms: queue delay grows
by 38 ms while every arrival gap stays below the old 20 ms warning threshold.
The live timing monitor now warns for that relative delay as well.

The recorder also runs a passive TCP observer off its packet-receive thread.
Once per second during capture it records matching hotspot/local connection
RTT, RTT variation, retransmission counters, and receive/send queue sizes to
`-transport.jsonl`, with a summary in the main capture JSON. TCP RTT is a round
trip measurement; sampled counters/queues can miss short stalls. No network or
socket tuning is applied by this observer.
[Linux ss documentation](https://man7.org/linux/man-pages/man8/ss.8.html).

## Stale body data

Body freshness now expires independently of continuing head/controller traffic.
When available on this PC, SDK callback time also prevents a fresh bridge stamp
from concealing an already old body sample. The existing 200 ms expiration
limit is retained; it is not a new buffering delay. Expired body state is
unavailable to full-body/three-point consumers, while fresh controller data
remains available for operator controls. Pose-window resampling already skips
obsolete windows; combined tracking/control packets are not blindly conflated.

## Verification and next live comparison

- 135 Python teleoperation, timing, capture, FK and release checks passed.
- The SDK extension rebuilt and its real JSON callback passed a native test
  for advancing counters, stationary poses and missing source timestamps.
- A passive TCP test detected a real 4,096-byte receive backlog and RTT on
  isolated local test sockets.
- Synthetic capture exercised raw packet saving, background TCP sampling,
  source/SDK timestamp retention and offline relative-transit analysis.

The next normal 60-second POSE capture automatically includes these diagnostics.
Keep the working 5 GHz topology and motion sequence unchanged. Compare PC
receipt-to-feedback timing against 14.265 ms median / 32.924 ms p95, inspect
source-message gaps vs SDK gaps and relative transit excess, and correlate
with TCP retransmissions/queue snapshots. No reduction in the 88.3 ms
full-session tracking gap has yet been demonstrated by this round of work.
The previous fall/reset events also remain a separate unresolved issue.

Backups of the previous Python server, reader and installed SDK extension are
under `.local/pico-sim/latency-round2-baseline/`. Build/test logs are
`.local/pico-sim/{sdk-timing-build,sdk-timing-callback-test,latency-round2-final-tests}.log`.
