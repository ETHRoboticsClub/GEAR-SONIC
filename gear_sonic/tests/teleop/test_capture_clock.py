"""Body samples are dated by their headset capture, not by their Wi-Fi arrival."""

import numpy as np

from gear_sonic.scripts.pico_manager_thread_server import UniformPoseResampler
from gear_sonic.tests.teleop import fakes
from gear_sonic.utils.teleop.input_readers import TrackingReader
from gear_sonic.utils.teleop.tracking_timing import CaptureClock

MS = 1_000_000
PERIOD_NS = round(1e9 / 72)
HEADSET_EPOCH_NS = 5_000_000_000_000  # The headset's clock is unrelated to the PC's.
FASTEST_NS = 4 * MS
# The offset is re-estimated as the fastest packet leaves its window; the spacing
# moves by far less than the milliseconds of arrival jitter it replaces.
SPACING_TOLERANCE_NS = 0.2 * MS


def assert_even(stamps):
    np.testing.assert_allclose(np.diff(stamps), PERIOD_NS, atol=SPACING_TOLERANCE_NS)


def jittery_stream(count, seed=0):
    """(capture, arrival) on the PC clock: even captures, uneven in-order arrivals."""
    rng = np.random.default_rng(seed)
    capture = 10_000 * MS + np.arange(count) * PERIOD_NS
    delay = FASTEST_NS + rng.gamma(1.2, 2.2 * MS, count)
    delay[::97] += 30 * MS  # Occasional Wi-Fi stalls.
    delay[5] = FASTEST_NS  # The fastest transport occurs at least once.
    arrival = (capture + delay).astype(np.int64)
    for i in range(1, count):  # In order; the bridge polls a delayed burst 1 ms apart.
        arrival[i] = max(arrival[i], arrival[i - 1] + MS)
    return capture, arrival


def headset(capture):
    return [int(c) + HEADSET_EPOCH_NS for c in capture]


def test_even_captures_are_recovered_from_uneven_arrivals():
    capture, arrival = jittery_stream(72 * 20)
    clock = CaptureClock()
    stamps = np.array(
        [clock.stamp(int(a), [h, 0]) for a, h in zip(arrival, headset(capture))]
    )
    assert clock.source == "body"
    settled = slice(72 * 4, None)
    assert np.ptp(np.diff(arrival[settled])) > 20 * MS  # The input really is uneven.
    assert_even(stamps[settled])
    assert (stamps <= arrival).all()  # Never dated after it arrived.
    # Dated as if every packet had the fastest transport: no latency is added
    # beyond the delay a packet actually suffered.
    np.testing.assert_allclose(stamps[settled], capture[settled] + FASTEST_NS, atol=0.2 * MS)


def test_arrival_time_is_kept_without_a_headset_clock():
    _, arrival = jittery_stream(200)
    for headset_times in ((), [0, 0], [None, None]):
        clock = CaptureClock()
        assert [clock.stamp(int(a), headset_times) for a in arrival] == arrival.tolist()
        assert clock.source == CaptureClock.ARRIVAL


def test_packet_clock_serves_a_client_without_body_timestamps():
    capture, arrival = jittery_stream(72 * 4)
    clock = CaptureClock()
    stamps = [clock.stamp(int(a), [0, h]) for a, h in zip(arrival, headset(capture))]
    assert clock.source == "packet"
    assert_even(stamps[-72:])


def test_field_that_is_not_a_nanosecond_clock_is_ignored():
    capture, arrival = jittery_stream(72 * 4)
    clock = CaptureClock()
    milliseconds = [h // MS for h in headset(capture)]
    stamps = [clock.stamp(int(a), [m, 0]) for a, m in zip(arrival, milliseconds)]
    assert clock.source == CaptureClock.ARRIVAL
    assert stamps == arrival.tolist()


def test_restarted_headset_clock_is_anchored_again():
    capture, arrival = jittery_stream(72 * 8)
    times = headset(capture)
    restart = 72 * 4
    for i in range(restart, len(times)):
        times[i] -= HEADSET_EPOCH_NS // 2
    clock = CaptureClock()
    stamps, sources = [], []
    for a, h in zip(arrival, times):
        stamps.append(clock.stamp(int(a), [h, 0]))
        sources.append(clock.source)
    assert (np.diff(stamps) > 0).all()
    assert sources[restart - 1] == "body"
    assert sources[restart] == CaptureClock.ARRIVAL  # Its arrival, until trusted again.
    assert abs(stamps[restart] - arrival[restart]) <= 1
    assert sources[-1] == "body"
    assert_even(stamps[-72:])


def body_frame(arrival_ns, sequence, device_time_ns=None):
    frame = {
        **fakes.raw(body=fakes.body(np.zeros((24, 3)))),
        "sample_time_ns": int(arrival_ns),
        "body_update_sequence": sequence,
    }
    if device_time_ns is not None:
        frame["device_time_ns"] = device_time_ns
    return frame


def test_reader_dates_bodies_by_capture_and_wakes_on_arrival():
    capture, arrival = jittery_stream(72 * 3)
    reader = TrackingReader(clock=fakes.Clock())
    stamps = []
    for i, (a, h) in enumerate(zip(arrival, headset(capture))):
        reader._update(body_frame(a, i + 1, [h, h]), int(a))
        stamps.append(reader.get_latest()["timestamp_ns"])
        assert reader.get_input_timestamp_ns() == a
    assert_even(stamps[-72:])
    assert abs(reader.get_latest()["fps"] - 72) < 0.5


def test_reader_keeps_arrival_stamps_for_frames_without_headset_time():
    _, arrival = jittery_stream(50)
    relayed, disabled = TrackingReader(clock=fakes.Clock()), TrackingReader(
        clock=fakes.Clock(), capture_clock=False
    )
    capture = headset(arrival)
    for i, a in enumerate(arrival):
        relayed._update(body_frame(a, i + 1), int(a))  # The robot API's relay sends none.
        disabled._update(body_frame(a, i + 1, [capture[i], capture[i]]), int(a))
        assert relayed.get_latest()["timestamp_ns"] == a
        assert disabled.get_latest()["timestamp_ns"] == a
    relayed._update(body_frame(arrival[-1] + PERIOD_NS, 51, "not a list"), 0)
    assert relayed.get_latest()["timestamp_ns"] == arrival[-1] + PERIOD_NS


def test_reference_follows_the_motion_instead_of_the_network():
    """A 1.2 Hz reach resampled to 50 Hz: arrival stamps distort its velocity."""
    capture, arrival = jittery_stream(72 * 30, seed=1)

    def hand(t_ns):
        return 0.35 * np.sin(2 * np.pi * 1.2 * np.asarray(t_ns) * 1e-9)

    clock = CaptureClock()
    captured = [clock.stamp(int(a), [h, 0]) for a, h in zip(arrival, headset(capture))]

    def velocity_error(stamps):
        resampler = UniformPoseResampler(50, 5)
        pose, quat = np.zeros((21, 3), np.float32), np.array([0, 0, 0, 1], np.float32)
        frames = []
        for stamp, c in zip(stamps[72 * 4 :], capture[72 * 4 :]):
            joints = np.full((24, 3), hand(c), dtype=np.float64)
            frames += [(f[1], f[3][0, 0]) for f in resampler.push(int(stamp), pose, joints, quat)]
        times, positions = np.array(frames).T
        truth = hand(times - np.median(np.asarray(stamps) - capture))
        return np.sqrt(np.mean((np.diff(positions) - np.diff(truth)) ** 2)) / 0.02

    by_arrival, by_capture = velocity_error(arrival), velocity_error(captured)
    assert by_arrival > 0.15  # m/s RMS on a motion peaking at 2.6 m/s.
    assert by_capture < 0.1 * by_arrival
