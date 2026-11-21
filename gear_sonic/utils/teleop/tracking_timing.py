"""PC-local PICO body timing diagnostics; all times use this PC's monotonic clock."""


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
