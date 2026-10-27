"""Readiness for releasing a simulated robot's suspension, never a robot control input."""

import json
from pathlib import Path
import time

import zmq


class PolicyReleaseGuard:
    """Require sustained policy feedback and live, powered body commands."""

    def __init__(self, stale_after=0.2, settle_time=0.5):
        self.stale_after = stale_after
        self.settle_time = settle_time
        self.feedback_at = None
        self.command_at = None
        self.powered = False
        self.healthy_since = None

    def update(self, now, feedback_at, command_at, powered):
        # A gap must reset the settling period even if packets just resumed.
        if self._missing(now):
            self.healthy_since = None
        self.feedback_at, self.command_at, self.powered = feedback_at, command_at, powered
        if self._missing(now):
            self.healthy_since = None
        elif self.healthy_since is None:
            self.healthy_since = now

    def _missing(self, now):
        if self.feedback_at is None or not 0 <= now - self.feedback_at <= self.stale_after:
            return "No live policy feedback. Wait for controller initialization, then start its policy."
        if self.command_at is None or not 0 <= now - self.command_at <= self.stale_after:
            return "No fresh DDS body commands. Check the controller and loopback DDS connection."
        if not self.powered:
            return "Leg motor commands have zero gains or invalid values. The robot is not ready."
        return ""

    def reason(self, now):
        missing = self._missing(now)
        if missing:
            return missing
        if self.healthy_since is None or now - self.healthy_since < self.settle_time:
            return "Waiting for 0.5 seconds of continuous policy feedback and motor commands."
        return ""

    def reset(self):
        self.healthy_since = None


class PolicyFeedbackMonitor:
    """Poll only the latest active-policy feedback, without blocking physics."""

    def __init__(self, status_path=""):
        self.guard = PolicyReleaseGuard()
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.SUB)
        self.socket.setsockopt(zmq.SUBSCRIBE, b"g1_debug")
        self.socket.setsockopt(zmq.CONFLATE, 1)
        self.socket.connect("tcp://127.0.0.1:5557")
        self.feedback_at = None
        self.status_path = Path(status_path) if status_path else None
        self.last_report = 0.0
        self.last_reason = None

    def update(self, bridge, band_enabled, height):
        try:
            self.socket.recv(zmq.NOBLOCK)
            self.feedback_at = time.monotonic()
        except zmq.Again:
            pass
        command_at, powered = bridge.body_command_status()
        # DDS callbacks run concurrently; sample the comparison clock after
        # reading their timestamp so a just-arrived command cannot look future-dated.
        now = time.monotonic()
        self.guard.update(now, self.feedback_at, command_at, powered)
        reason = self.guard.reason(now)
        if reason != self.last_reason:
            print(f"[Release guard] {reason or 'READY: click MuJoCo and press 9 to release suspension.'}", flush=True)
            self.last_reason = reason
        if self.status_path and now - self.last_report >= 0.5:
            self.status_path.parent.mkdir(parents=True, exist_ok=True)
            temporary = self.status_path.with_suffix(".tmp")
            temporary.write_text(json.dumps({
                "sample_time": now, "ready": not reason, "reason": reason,
                "band_enabled": bool(band_enabled), "height": float(height),
                "policy_age": None if self.feedback_at is None else now - self.feedback_at,
                "command_age": None if command_at is None else now - command_at,
            }) + "\n")
            temporary.replace(self.status_path)
            self.last_report = now

    def close(self):
        self.socket.close(linger=0)
        self.context.term()
