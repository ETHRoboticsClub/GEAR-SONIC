"""PC-local PICO body timing diagnostics; all times use this PC's monotonic clock."""

from gear_sonic.utils.teleop.transport_timing import MessageTransitTracker


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
        self.message_transit = MessageTransitTracker()
        self._clear()

    def _clear(self):
        self.max_gap_ns = 0
        self.max_relay_ns = 0
        self.late_gaps = 0
        self.max_message_excess_ns = None

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
        timing = frame.get('body_input_timing', {})
        if isinstance(timing, dict) and timing.get('body_sequence') == sequence:
            excess = self.message_transit.observe(timing)
            if excess is not None:
                self.max_message_excess_ns = max(self.max_message_excess_ns or 0, excess)

    def warning(self, now_ns, enabled=True):
        if not enabled:
            self._clear()
            self.last_warning_ns = None
            return None
        if self.received_ns is None:
            return None
        silence = max(0, now_ns - self.received_ns)
        age = max(0, now_ns - self.sent_ns)
        if max(self.max_gap_ns, self.max_relay_ns, silence, age,
               self.max_message_excess_ns or 0) <= self.threshold_ns:
            return None
        if self.last_warning_ns is not None and now_ns - self.last_warning_ns < 1_000_000_000:
            return None
        transit = ('' if self.max_message_excess_ns is None else
                   f'message excess transit max={self.max_message_excess_ns / 1e6:.1f} ms '
                   '(relative to fastest observed, clocks not synchronized); ')
        message = (
            f"[PICO timing] WARNING: exceeded {self.threshold_ns / 1e6:.1f} ms: "
            f"fresh-body arrival gap max={self.max_gap_ns / 1e6:.1f} ms "
            f"({self.late_gaps} late gaps since last report); "
            f"no fresh body for={silence / 1e6:.1f} ms; "
            f"PC bridge->reader max={self.max_relay_ns / 1e6:.1f} ms; "
            f"latest fresh body PC bridge->manager age={age / 1e6:.1f} ms. "
            f"{transit}"
            "A policy tick may use stale input; catch-up can cause a target jump. "
            "Headset-to-PC latency is not measured."
        )
        self.last_warning_ns = now_ns
        self._clear()
        return message
