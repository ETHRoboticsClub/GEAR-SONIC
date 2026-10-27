"""Exercise diagnostics through the reader, including heartbeat and stall cases."""

import numpy as np

from gear_sonic.tests.teleop import fakes
from gear_sonic.utils.teleop.input_readers import TrackingReader

MS = 1_000_000


def timing_warnings(caplog):
    return [r for r in caplog.records if "[PICO timing]" in r.getMessage()]


def reader_and_feed(**kwargs):
    clock = fakes.Clock()
    reader = TrackingReader(clock=clock, **kwargs)

    def feed(received_ms, sequence, sent_ms=None, body=True):
        clock.now_ns = int(received_ms * MS)
        reader._update({
            **fakes.raw(body=fakes.body(np.zeros((24, 3))) if body else None),
            "sample_time_ns": int((received_ms if sent_ms is None else sent_ms) * MS),
            "body_update_sequence": sequence,
        }, clock.now_ns)

    return reader, clock, feed


def test_boundary_and_continuous_stationary_updates(caplog):
    reader, clock, feed = reader_and_feed(local_pico_timing=True)
    for i in range(10):
        feed(1000 + 20 * i, i + 1)
        reader.warn_if_tracking_delayed()
    assert not timing_warnings(caplog)  # Identical positions are still fresh packets.
    clock.now_ns += 20 * MS + 1
    reader.warn_if_tracking_delayed()
    assert "no fresh body for=20.0 ms" in caplog.text


def test_heartbeats_and_tracking_loss_do_not_hide_body_stall(caplog):
    reader, clock, feed = reader_and_feed(local_pico_timing=True)
    feed(1000, 1)
    feed(1014, 1)  # Controller traffic/heartbeat with the same cached body.
    feed(1028, 1, body=False)
    reader.warn_if_tracking_delayed()
    assert "no fresh body for=28.0 ms" in caplog.text
    assert "bridge->manager age=28.0 ms" in caplog.text


def test_gap_is_retained_even_when_new_data_arrives_before_manager_runs(caplog):
    reader, clock, feed = reader_and_feed(local_pico_timing=True)
    feed(1000, 1)
    feed(1045, 2)
    feed(1059, 3)
    reader.warn_if_tracking_delayed()
    assert "arrival gap max=45.0 ms (1 late gaps" in caplog.text
    assert "no fresh body for=0.0 ms" in caplog.text


def test_relay_delay_and_manager_stall_are_reported_separately(caplog):
    reader, clock, feed = reader_and_feed(local_pico_timing=True)
    feed(1030, 1, sent_ms=1000)
    clock.now_ns = 1040 * MS
    reader.warn_if_tracking_delayed()
    assert "bridge->reader max=30.0 ms" in caplog.text
    assert "bridge->manager age=40.0 ms" in caplog.text
    assert "Headset-to-PC latency is not measured" in caplog.text


def test_warning_rate_limit_and_off_mode_reset(caplog):
    reader, clock, feed = reader_and_feed(local_pico_timing=True)
    feed(1000, 1)
    for now in (1030, 1040, 1200, 2029):
        clock.now_ns = now * MS
        reader.warn_if_tracking_delayed()
    assert len(timing_warnings(caplog)) == 1
    clock.now_ns = 2030 * MS
    reader.warn_if_tracking_delayed()
    assert len(timing_warnings(caplog)) == 2
    reader.warn_if_tracking_delayed(enabled=False)
    feed(2040, 2)
    reader.warn_if_tracking_delayed(enabled=False)
    feed(2054, 3)
    reader.warn_if_tracking_delayed()
    assert len(timing_warnings(caplog)) == 2


def test_remote_readers_and_startup_are_quiet(caplog):
    for kwargs in ({}, {"local_pico_timing": True}):
        reader, clock, feed = reader_and_feed(**kwargs)
        reader.warn_if_tracking_delayed()
        feed(1000, 0)  # No valid SDK body counter yet.
        clock.now_ns = 2000 * MS
        reader.warn_if_tracking_delayed()
    reader, clock, feed = reader_and_feed()
    feed(1000, 1, sent_ms=-100000)  # Arbitrary remote monotonic clock.
    reader.warn_if_tracking_delayed()
    assert not timing_warnings(caplog)


def test_regular_packets_with_increasing_queue_delay_trigger_a_warning():
    from gear_sonic.utils.teleop.tracking_timing import PicoTimingMonitor
    monitor = PicoTimingMonitor()
    for i in range(20):
        receipt = (1000 + i * 12) * MS
        monitor.observe({'body': {}, 'body_update_sequence': i+1, 'sample_time_ns': receipt,
                         'body_input_timing': {'body_sequence': i+1,
                                              'headset_frame_unix_ns': 1_700_000_000_000_000_000 + i*10*MS,
                                              'sdk_receipt_monotonic_ns': receipt}}, receipt)
    message = monitor.warning(receipt)
    assert 'arrival gap max=12.0 ms' in message
    assert 'message excess transit max=38.0 ms' in message
    assert 'clocks not synchronized' in message
