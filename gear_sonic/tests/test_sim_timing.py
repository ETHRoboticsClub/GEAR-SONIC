"""Physics should keep wall-clock pace despite intermittent rendering work."""

from types import SimpleNamespace

import pytest

from gear_sonic.utils.mujoco_sim import base_sim


@pytest.mark.parametrize("long_pause", [False, True])
def test_simulation_pacing_with_slow_viewer(monkeypatch, long_pause):
    clock = [0.0]
    steps = []

    def sleep(duration):
        assert duration >= 0
        clock[0] += duration

    monkeypatch.setattr(base_sim, "time", SimpleNamespace(
        time=lambda: clock[0], monotonic=lambda: clock[0], sleep=sleep,
    ))
    sim = base_sim.BaseSimulator.__new__(base_sim.BaseSimulator)
    sim.sim_dt = .005
    sim.viewer_dt = sim.reward_dt = sim.image_dt = .02
    sim.redis_client = None
    sim._running = True

    def step():
        steps.append(clock[0])
        clock[0] += .001
        if len(steps) == 200:
            sim._running = False

    def viewer():
        if long_pause:
            if len(steps) == 1:
                clock[0] += .5
        else:
            clock[0] += .007

    sim.sim_env = SimpleNamespace(
        viewer=None, image_publish_process=None, sim_step=step,
        update_viewer=viewer, update_reward=lambda: None,
        update_render_caches=lambda: None,
    )
    sim.start()
    assert len(steps) == 200
    if long_pause:
        # A long external stall must not cause hundreds of catch-up steps.
        assert 1.48 < clock[0] < 1.51
    else:
        # 200 physics steps at 5 ms must take one wall-clock second even
        # when every fourth viewer update exceeds the physics period.
        assert clock[0] == pytest.approx(1.0, abs=.001)
