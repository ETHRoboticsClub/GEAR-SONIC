"""Discriminate SDK body pauses from shared pauses and delayed PC observation."""

from gear_sonic.scripts import probe_pico_tracking as probe_script
from gear_sonic.scripts.probe_pico_tracking import TrackingGapProbe


MS = 1_000_000


def observe(probe, time_ms, body, peers=None, read_ms=0):
    peers = body if peers is None else peers
    probe.observe(int((time_ms - read_ms) * MS), int(time_ms * MS),
                  dict(head=peers, left=peers, right=peers, body=body))


def test_body_pause_with_peers_progressing_and_regular_polling():
    probe = TrackingGapProbe()
    for index in range(201):
        observe(probe, index, body=1 if index < 157 else 2, peers=1 + index // 14)
    report = probe.summary()
    gap, = report["long_body_gaps"]
    assert gap["gap_ms"] == 157
    assert gap["max_poll_interval_ms"] == 1
    assert gap["peer_updates_observed_while_body_unchanged"]["head"] == 11
    assert report["devices"]["body"]["ending_silence_ms"] == 43
    assert report["devices"]["head"]["update_gap_ms"]["max"] == 14


def test_shared_pause_is_not_attributed_to_body_tracking_alone():
    probe = TrackingGapProbe()
    for index in range(158):
        observe(probe, index, body=1 if index < 157 else 2)
    gap, = probe.summary()["long_body_gaps"]
    assert gap["max_poll_interval_ms"] == 1
    assert gap["peer_updates_observed_while_body_unchanged"] == dict(head=0, left=0, right=0)


def test_pc_stall_and_counter_jumps_do_not_look_like_packet_loss():
    probe = TrackingGapProbe()
    observe(probe, 0, body=1)
    observe(probe, 157, body=12, read_ms=150)
    report = probe.summary()
    gap, = report["long_body_gaps"]
    assert gap["max_poll_interval_ms"] == 157
    assert gap["max_sdk_read_ms"] == 150
    assert gap["body_counter_advance"] == 11
    assert report["devices"]["body"]["updates_between_polls"] == 10
    assert report["sdk_counter_read_ms"]["max"] == 150


def test_reconnect_resets_timeline_and_does_not_invent_a_body_gap():
    probe = TrackingGapProbe()
    observe(probe, 0, body=100)
    observe(probe, 14, body=101)
    observe(probe, 1000, body=1)
    observe(probe, 1014, body=2)
    report = probe.summary()
    assert report["counter_resets"] == 1
    assert not report["long_body_gaps"]
    assert report["devices"]["body"]["update_gap_ms"]["max"] == 14


def test_untracked_body_and_end_of_capture_silence_are_explicit():
    probe = TrackingGapProbe()
    observe(probe, 0, body=0, peers=1)
    observe(probe, 50, body=0, peers=2)
    report = probe.summary()
    assert report["devices"]["body"]["ending_silence_ms"] is None
    assert report["devices"]["body"]["observed_updates"] == 0
    observe(probe, 100, body=1, peers=3)
    observe(probe, 300, body=1, peers=4)
    report = probe.summary()
    assert report["devices"]["body"]["ending_silence_ms"] == 200
    assert not report["long_body_gaps"]  # An unfinished silence is not a completed gap.


def test_wait_timeout_still_reports_other_devices(monkeypatch):
    from types import SimpleNamespace

    now = [0.0]
    monkeypatch.setattr(probe_script, "time", SimpleNamespace(
        monotonic=lambda: now[0], monotonic_ns=lambda: round(now[0] * 1e9),
        sleep=lambda duration: now.__setitem__(0, now[0] + duration),
    ))
    sdk = SimpleNamespace(get_input_update_sequences=lambda:
                          dict(head=1 + round(now[0] * 1000), left=0, right=0, body=0))
    result = probe_script.capture(sdk, seconds=1, wait=.003, poll_s=.001)
    assert result["status"] == "no_body_tracking"
    assert result["devices"]["head"]["observed_updates"] == 4
    assert result["devices"]["body"]["observed_updates"] == 0
