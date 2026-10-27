# Before and after the 27 October updates

This comparison uses today's approximately 18:00 full-body run before the
updates and the approximately 19:05 run just completed. It does not substitute
the older 21 October capture for today's before session.

Both sessions used the four-frame low-latency checkpoint. The earlier session
used the native-driver 2.4 GHz hotspot and full viewer synchronization. The
later session used the wait/logging fixes, state-only viewer synchronization,
additional diagnostics, and the fallback-driver 5 GHz hotspot. These changed
together, so their individual contributions cannot be isolated from these runs.

| Shared measurement | Before | After | Observed change |
| --- | ---: | ---: | ---: |
| Largest reported completed fresh-body arrival gap | 156.8 ms | 88.3 ms | 68.5 ms lower (43.7%) |
| Typical pose output rate | 45.14 updates/s | 49.78 updates/s | 4.64 updates/s higher (10.3%) |
| Typical pose preparation time, including SMPL | 4.76 ms | 4.26 ms | 0.50 ms lower (10.5%) |
| Typical reporting-window p95 of pose preparation | 6.12 ms | 4.74 ms | 1.38 ms lower (22.5%) |
| Typical SMPL mapping time | 2.74 ms | 2.43 ms | 0.31 ms lower (11.3%) |
| Largest reported local bridge → reader delay | 2.5 ms | 2.4 ms | Little change |

For a consistent comparison, output rate is the median of periodic log reports,
and typical preparation/mapping time is the median of reporting-window medians.
The preparation p95 row is the median of reporting-window p95 values; it is not
a global per-frame percentile. The first FPS report and the single-sample cold
mapping report are excluded. There are 37 usable rate windows before and 49
after. The 60-second raw capture's preparation median of 4.288 ms is a separate
per-packet statistic, which is why it differs slightly from the 4.26 ms value
used for the full-session log comparison.

Body gap maxima use active warning reports with positive completed gaps and
less than 200 ms current silence, consistently across both sessions. The
before result is based on 189 such reports and the after result on 159. These
are threshold-triggered reports, not a regular sampling denominator. Startup,
waiting and end-of-test silence are not counted as normal movement latency.
The after run lasted longer; this is an observational comparison with different
operator motions and durations, not a controlled A/B trial.

## What cannot be compared directly

| PC manager receipt of newest encoder sample → controller feedback | Before | After, saved 60-second window |
| --- | ---: | ---: |
| Median | Not captured | 14.265 ms |
| p95 | Not captured | 32.924 ms |
| Maximum | Not captured | 56.194 ms |

The before run has no timestamped raw packet recording, so no numerical saving
for this overall PC-side quantity can be established against that session.
The older controller's `Streaming data mean delay` reports used a different
timestamp basis and are also not a compatible baseline after the localhost
timestamp fix. A body-arrival gap is a different quantity from PC receipt to
feedback; the old 156.8 ms must not be subtracted from the new 14.265 ms median.
Neither quantity measures the full sensor-to-screen delay.

The strongest demonstrated change is a steadier input/output pipeline, with
pose output near the 50 Hz target and a smaller observed maximum tracking gap.
The current session still contains an 88.3 ms input gap and two logged robot
falls/resets. The changes have not established uninterrupted tracking or
balanced motion throughout the session.

Sources: [before-run log summary](measurements/pico_sim_20261027/latest_run_log_summary.json),
[after-run full-session findings](measurements/pico_sim_20261027/5ghz-live-20261027-final-session.json),
[after-run raw-window timings](measurements/pico_sim_20261027/5ghz-live-20261027-components.json).
Shared metrics were recalculated from the preserved `manager.log` files at
`.local/pico-sim/logs/latency-audit-20261027-1800/` and
`.local/pico-sim/logs/pico-latency-20261027-190348-100649-full-session-logs/`.
