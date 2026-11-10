"""PC-local PICO body timing: diagnostics and capture-time stamps, on this PC's monotonic clock."""

from collections import deque

# The fastest transport seen within this window defines the clock offset. Two
# seconds spans ~140 packets at 72 Hz and lets the offset follow clock drift.
OFFSET_WINDOW_NS = 2_000_000_000
# A headset field is used once it has advanced like a clock for this long.
RATE_CHECK_NS = 1_000_000_000
RATE_TOLERANCE = 0.1
# Headset time and arrival time advancing this differently is a clock restart
# or a paused application, not transport jitter.
DISCONTINUITY_NS = 150_000_000


class PicoTimingMonitor:
    """Called under the reader lock; reporting runs in the manager, never per packet.

    SDK counters distinguish new body data from bridge heartbeats or controller
    updates. These measurements cannot establish headset capture/network latency.
    """

    def __init__(self, threshold_ms=20.0):
        self.threshold_ns = int(threshold_ms * 1_000_000)
        self.sequence = None
        self.received_ns = None
        self.sent_ns = None
        self.last_warning_ns = None
        self._clear()

    def _clear(self):
        self.max_gap_ns = 0
        self.max_relay_ns = 0
        self.late_gaps = 0

    def observe(self, frame, received_ns):
        sequence = frame.get("body_update_sequence")
        if frame.get("body") is None or not isinstance(sequence, int) or sequence <= 0:
            return
        if sequence == self.sequence:
            return
        if self.received_ns is not None:
            gap = received_ns - self.received_ns
            self.max_gap_ns = max(self.max_gap_ns, gap)
            self.late_gaps += int(gap > self.threshold_ns)
        self.sequence = sequence
        self.received_ns = received_ns
        self.sent_ns = int(frame["sample_time_ns"])
        self.max_relay_ns = max(self.max_relay_ns, received_ns - self.sent_ns)

    def warning(self, now_ns, enabled=True):
        if not enabled:
            self._clear()
            self.last_warning_ns = None
            return None
        if self.received_ns is None:
            return None
        silence = max(0, now_ns - self.received_ns)
        age = max(0, now_ns - self.sent_ns)
        if max(self.max_gap_ns, self.max_relay_ns, silence, age) <= self.threshold_ns:
            return None
        if self.last_warning_ns is not None and now_ns - self.last_warning_ns < 1_000_000_000:
            return None
        message = (
            f"[PICO timing] WARNING: exceeded {self.threshold_ns / 1e6:.1f} ms: "
            f"fresh-body arrival gap max={self.max_gap_ns / 1e6:.1f} ms "
            f"({self.late_gaps} late gaps since last report); "
            f"no fresh body for={silence / 1e6:.1f} ms; "
            f"PC bridge->reader max={self.max_relay_ns / 1e6:.1f} ms; "
            f"latest fresh body PC bridge->manager age={age / 1e6:.1f} ms. "
            "A policy tick may use stale input; catch-up can cause a target jump. "
            "Headset-to-PC latency is not measured."
        )
        self.last_warning_ns = now_ns
        self._clear()
        return message


class _SlidingMinimum:
    """Smallest value pushed during the last ``window_ns``."""

    def __init__(self, window_ns):
        self.window_ns = window_ns
        self._items = deque()

    def clear(self):
        self._items.clear()

    def push(self, time_ns, value):
        while self._items and self._items[-1][1] >= value:
            self._items.pop()
        self._items.append((time_ns, value))
        while self._items[0][0] < time_ns - self.window_ns:
            self._items.popleft()
        return self._items[0][1]


class _HeadsetClock:
    """One headset timestamp field, mapped onto this PC's monotonic clock.

    ``arrival - headset_time`` is the unknown clock offset plus that packet's
    transport delay. Its minimum over a short window is the offset plus the
    fastest transport, so ``headset_time + minimum`` spaces samples exactly as
    the headset captured them and never dates one after it arrived.
    """

    def __init__(self):
        self._offsets = _SlidingMinimum(OFFSET_WINDOW_NS)
        self.reset()

    def reset(self):
        self._base = None  # (arrival, headset time) the rate check started from
        self._last = None
        self._trusted = False
        self._offsets.clear()

    def stamp(self, arrival_ns, headset_ns):
        """Capture time on the PC clock, or None while this field cannot give one."""
        if headset_ns is None or headset_ns <= 0:
            return None
        if self._last is not None:
            advanced = headset_ns - self._last[1]
            if advanced == 0:
                return None  # The packet carried no new time for this field.
            # A restarted or paused headset clock invalidates the offset.
            if advanced < 0 or abs(advanced - (arrival_ns - self._last[0])) > DISCONTINUITY_NS:
                self.reset()
        self._last = (arrival_ns, headset_ns)
        if self._base is None:
            self._base = self._last
            return None
        offset = self._offsets.push(arrival_ns, arrival_ns - headset_ns)
        if not self._trusted:
            elapsed = arrival_ns - self._base[0]
            if elapsed < RATE_CHECK_NS:
                return None
            # A field in other units, or one that is not a clock, runs at another rate.
            if abs((headset_ns - self._base[1]) / elapsed - 1.0) > RATE_TOLERANCE:
                self.reset()
                return None
            self._trusted = True
        return headset_ns + offset


class CaptureClock:
    """When each body sample was captured, on this PC's monotonic clock.

    The bridge stamps a body with the time its packet arrived. Wi-Fi delays
    packets unevenly, so those stamps bunch and spread a motion the headset
    sampled evenly, and the 50 Hz reference resampled from them speeds up and
    slows down although the operator did not. The headset's own timestamps
    remove that. ``headset_times`` lists the candidate fields in order of
    preference; the first that behaves like a clock is used. Without one the
    arrival time is returned unchanged.
    """

    ARRIVAL = "arrival"

    def __init__(self, names=("body", "packet")):
        self._names = names
        self._clocks = [_HeadsetClock() for _ in names]
        self._last_ns = None
        self.source = self.ARRIVAL

    def reset(self):
        for clock in self._clocks:
            clock.reset()
        self._last_ns = None
        self.source = self.ARRIVAL

    def stamp(self, arrival_ns, headset_times=()):
        capture_ns, source = arrival_ns, self.ARRIVAL
        # Every field is followed on every sample so a fallback is ready at once.
        for name, clock, headset_ns in reversed(list(zip(self._names, self._clocks, headset_times))):
            candidate = clock.stamp(arrival_ns, headset_ns)
            if candidate is not None:
                capture_ns, source = min(candidate, arrival_ns), name
        if self._last_ns is not None and capture_ns <= self._last_ns:
            capture_ns = self._last_ns + 1  # Consumers require strictly increasing stamps.
        self._last_ns = capture_ns
        self.source = source
        return capture_ns
