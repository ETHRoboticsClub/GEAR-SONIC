# Investigating long PICO tracking gaps

**Update, 27 October 19:03:** `teleop-sim` is now running on **5 GHz,
channel 36 (5180 MHz), 20 MHz width**, at `10.42.0.1`. Reapplying Switzerland's
country setting and reinitializing the native driver did not lift its `no IR`
restrictions; both activation attempts timed out. A temporary `88x2bu` fallback
driver successfully started the same saved hotspot profile. The other Wi-Fi
adapter remains on the native driver and connected to `ETHRC_SL`. Its scan
independently detected the hotspot beacon at 5180 MHz. The PC checks below
describe the earlier state, before this repair. No reduction in tracking gaps
was claimed from hotspot activation alone. The subsequent live capture is
summarized below.

The fallback was built from
[RinCat/RTL88x2BU-Linux-Driver](https://github.com/RinCat/RTL88x2BU-Linux-Driver),
revision `465ae7c`, for kernel `6.17.0-23-generic`. It was loaded for the hotspot
adapter only, with `rtw_country_code=CH`, USB mode switching disabled, and
power saving disabled. No global native-driver blacklist, DKMS installation,
or permanent module replacement was made. The module remains loaded until
unloaded or rebooted. To restore it after reboot, with the adapter in the same
USB port and the same kernel, run:

```bash
pkexec --disable-internal-agent bash \
  /home/ethan/Humanoid/GEAR-SONIC/install_scripts/recover_pico_hotspot_5ghz.sh
```

This may require an Ubuntu administrator authentication dialog. A kernel update
requires rebuilding the module. Diagnostic/build logs are in
`.local/pico-sim/{rebind-hotspot-5ghz,hotspot-fallback-driver,rtl88x2bu-recovery-build}.log`.

The recovery helper is deliberately workstation-specific: it verifies this
adapter's USB identity/port and leaves the separate Internet adapter bound to
its existing driver. It requires the locally built module and the local `iw`
binary; neither is downloaded or installed by the helper. To rebuild the
fallback source already at `~/.local/share/gear-sonic/rtl88x2bu-5ghz-recovery`,
run `make -j 4 CONFIG_RTW_DEBUG=n CONFIG_MP_INCLUDED=n` there with headers for
the running kernel installed. An optional first argument selects the compiled
module path. The normal simulator launcher never performs driver replacement.

## First live capture after the 5 GHz repair

The updated simulation and recorder ran successfully. The headset joined the
hotspot, all tracking devices were present, calibration/policy start completed,
and suspension was released. The recorder then saved a 60-second continuous
POSE interval, matching 2,994 of 3,000 feedback samples to unique references.
Its quality checks reported no issues.

After shutdown, the full-session logs reported an 88.3 ms completed body gap
and fall/reset events; the 39.957 ms maximum below applies only to the saved
minute. Packet-quality checks do not detect falls. See the
[final latency report](pico_sim_latency_result_20261027.md) for these limits and
the complete-session findings.

| Measurement | New 5 GHz run |
| --- | ---: |
| PC newest encoder-input receipt → controller feedback, median / p95 / maximum | 14.265 / 32.924 / 56.194 ms |
| Fresh body arrival gap, median / p95 / maximum | 11.071 / 15.351 / 39.957 ms |
| Fresh body gaps above 20 ms / 30 ms | 0.688% / 0.112% |
| Pose preparation, median / p95 | 4.288 / 4.803 ms |
| Bridge → capture, median / p95 / maximum | 0.270 / 0.406 / 0.829 ms |
| Largest bridge polling / SDK counter-read diagnostic maxima | 1.729 / 0.137 ms |

Head and both controller streams had nearly identical arrival gaps to the body
stream. The largest remaining pauses were shared across those streams while
the PC bridge's polling/SDK read maxima stayed small. This supports investigating
the shared upstream path if further tail reduction is needed; it does not
separate headset scheduling, TCP/Wi-Fi and PC service processing.

The earlier 2.4 GHz run reported a largest completed body gap of 156.8 ms. This
capture's largest observed gap is 39.957 ms. The earlier interval was longer and
the software/driver/network setup changed together, so the comparison does not
establish a causal improvement from the frequency band alone. PC receipt to
feedback excludes headset capture/network delay before PC receipt, DDS command
application and visible display response. It is not sensor-to-screen latency.

Sources: [component timings](measurements/pico_sim_20261027/5ghz-live-20261027-components.json),
[capture configuration and simulation speed](measurements/pico_sim_20261027/5ghz-live-20261027-capture.json),
[verified network startup](measurements/pico_sim_20261027/hotspot_5ghz_startup.json).
The raw capture and matching component logs remain in `.local/pico-sim/logs/`
under prefix `pico-latency-20261027-190348-100649`.

The 156.8 ms value in the 27 October session is the largest reported completed
gap between fresh body samples at the Python reader. It is not constant latency,
nor a capture-to-screen measurement. No live headset stream was available during
the subsequent PC checks, so no reduction in that gap has been demonstrated.

## What was checked on this PC

[Saved PC checks](measurements/pico_sim_20261027/tracking_gap_pc_checks.json):

- The saved `teleop-sim` profile is 2.4 GHz, channel 6; its adapter is currently
  disconnected. The PC's Internet connection uses the other adapter.
- The hotspot adapter marks 5 GHz channel 36 `no IR`, preventing the intended
  AP operation in the present driver/regulatory state.
- Profile Wi-Fi power saving is disabled, driver deep sleep is disabled, and
  USB runtime suspension is disabled (`power/control=on`). These are already
  configured; disabling them again is not a new optimization.
- A subsequent read-only `iw phy ... info` query timed out after five seconds.
  Together with the earlier USB/driver faults in the
  [latency audit](pico_sim_latency_audit_20261027.md), this makes a stable network
  path comparison valuable. It does not prove the cause of each tracking gap.

No network settings were changed by these checks. For the network comparison,
use an existing stable 5 GHz access point with the PC connected over Ethernet
where available, bypassing the USB hotspot. Record the same motions before and
after changing only this topology. The earlier 21 October 5 GHz results are
historical comparisons, not predictions for this adapter's current state.

## Probe SDK updates without sending control commands

Start the normal simulation launcher, connect the PICO in XRoboToolkit, and
enable calibrated body tracking using the [operator guide](pico_sim_operator_guide.md).
Then run:

```bash
source .local/pico-sim/env.sh
python gear_sonic/scripts/probe_pico_tracking.py --seconds 60 --wait 120
```

The probe opens its own SDK client and polls receipt counters, independently
of the simulation bridge, retargeting, policy and viewer. It issues no control
commands. It waits for body data, records a 60-second window, and writes a
timestamped JSON report under `.local/pico-sim/logs/`. Existing reports are never
overwritten. It checks that the local PC service is running before initializing
the SDK. An expired wait is reported as `no_body_tracking`, with any observed
head/controller traffic retained. It does not count as a successful live test.

For an input-only baseline, the PC service and streaming headset suffice; no
simulator/controller process is required. Compare that baseline with a probe
under the full simulation workload on the same network.

For each long body gap, inspect:

| Observation | Interpretation |
| --- | --- |
| Head/controllers advance during the body pause and PC polling remains regular | Body-section delivery paused independently. Investigate body tracking, tracker visibility, and body-specific sender/service processing. |
| No peer updates while polling remains regular | A shared upstream pause is possible. Check which peer devices were actually enabled before attributing this to networking/service. |
| Large `max_poll_interval_ms` or `max_sdk_read_ms` | The observer itself stalled or blocked; it may have missed intermediate receipts. Separate PC scheduling/SDK contention from upstream pauses. |
| Counters jump between polls | Updates occurred between observations. This alone is not evidence of network packet loss. |

Device gaps use PC observation times. They cannot separate headset capture,
sender queues, Wi-Fi/TCP and PC-service delay without further markers. The report
keeps unfinished end-of-capture silence separate from completed gaps and resets
the timing baseline when SDK counters restart.

## Diagnostics retained in normal captures

The simulation bridge now adds `input_update_sequences`,
`bridge_poll_interval_max_ms`, and `sdk_counter_read_max_ms` to tracking messages.
The two maxima cover the interval since the preceding publication, preserving a
stall even when it ends between messages. Existing tracking readers ignore these
additional fields; the normal latency recorder saves them with the raw capture.

Twenty-one targeted tests pass, covering a 157 ms body-only pause, a shared
pause, a delayed PC/SDK observation, counter resets, absent tracking, waiting
timeouts, and retention of an injected SDK stall through bridge publication.
These are diagnostic correctness checks, not measured network improvements.
