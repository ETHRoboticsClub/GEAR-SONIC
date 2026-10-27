from gear_sonic.utils.teleop.transport_timing import parse_tcp_info, summarize_message_timing


def test_tcp_info_reports_only_the_observed_connection_counters():
    rows = parse_tcp_info('''ESTAB 2400 128 10.42.0.1:12345 10.42.0.57:45721
 cubic rtt:4.25/1.1 retrans:0/3 bytes_retrans:1448 notsent:64
ESTAB 0 0 127.0.0.1:60061 127.0.0.1:45123
 cubic rtt:0.02/0.01
''')
    assert len(rows) == 2
    assert rows[0]['scope'] == 'hotspot' and rows[1]['scope'] == 'loopback'
    assert rows[0]['rtt_ms'] == 4.25
    assert rows[0]['recv_queue_bytes'] == 2400
    assert rows[0]['retransmissions_total'] == 3
    assert 'retransmissions_total' not in rows[1]


def samples(offset=0):
    # Devices generate regularly; the middle message spends 30 ms extra upstream.
    return [{'headset_frame_unix_ns': 1_700_000_000_000_000_000 + i * 20_000_000 + offset,
             'sdk_receipt_monotonic_ns': 10_000_000_000 + i * 20_000_000 + delay}
            for i, delay in enumerate((0, 30_000_000, 20_000_000))]


def test_constant_clock_offset_does_not_become_latency():
    first = summarize_message_timing(samples())
    shifted = summarize_message_timing(samples(100_000_000_000))
    assert first['relative_transit_excess_ms']['max'] == 30
    assert shifted['relative_transit_excess_ms'] == first['relative_transit_excess_ms']
    assert first['headset_message_interval_ms']['p50'] == 20


def test_clock_steps_are_flagged_instead_of_reported_as_network_delay():
    rows = samples()
    rows[-1]['headset_frame_unix_ns'] -= 2_000_000_000
    result = summarize_message_timing(rows)
    assert result['clock_discontinuity_or_long_pause']
    assert 'relative_transit_excess_ms' not in result


def test_missing_source_timestamps_cannot_establish_one_way_latency():
    assert summarize_message_timing([])['samples'] == 0
    assert parse_tcp_info('') == []


def test_growing_queue_is_visible_even_when_arrival_gaps_are_small():
    rows = [{'headset_frame_unix_ns': 1_700_000_000_000_000_000 + i * 10_000_000,
             'sdk_receipt_monotonic_ns': 10_000_000_000 + i * 12_000_000}
            for i in range(20)]
    result = summarize_message_timing(rows)
    assert result['sdk_receipt_interval_ms']['max'] == 12
    assert result['relative_transit_excess_ms']['max'] == 38
