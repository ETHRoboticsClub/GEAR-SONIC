"""Release must reflect live control, rather than calibration or model loading."""

from types import SimpleNamespace
import threading

import mujoco
import numpy as np
import pytest

from gear_sonic.utils.mujoco_sim.base_sim import DefaultEnv
from gear_sonic.utils.mujoco_sim import policy_release_guard
from gear_sonic.utils.mujoco_sim.policy_release_guard import PolicyFeedbackMonitor, PolicyReleaseGuard
from gear_sonic.utils.mujoco_sim.unitree_sdk2py_bridge import ElasticBand, UnitreeSdk2Bridge


def test_initialized_motor_commands_without_policy_cannot_release():
    guard = PolicyReleaseGuard()
    band = ElasticBand(release_check=lambda: guard.reason(10.0))
    guard.update(10.0, None, 10.0, True)
    band.handle_keyboard_button("9")
    assert band.enable
    assert "policy feedback" in guard.reason(10.0)


@pytest.mark.parametrize("lost", ["policy", "dds", "gains"])
def test_loss_blocks_release_until_continuous_control_recovers(lost):
    guard = PolicyReleaseGuard()
    for now in np.arange(1.0, 1.7, 0.02):
        guard.update(now, now, now, True)
    assert guard.reason(1.7) == ""
    band = ElasticBand(release_check=lambda: guard.reason(1.7))
    band.handle_keyboard_button("9")
    assert not band.enable
    # Attaching must always remain available, including after a controller failure.
    guard.update(1.7, 1.0 if lost == "policy" else 1.7,
                 1.0 if lost == "dds" else 1.7, lost != "gains")
    band.handle_keyboard_button("9")
    assert band.enable
    band.handle_keyboard_button("9")
    assert band.enable
    for now in np.arange(1.8, 2.6, 0.02):
        guard.update(now, now, now, True)
    assert guard.reason(2.6) == ""


def test_stall_resets_settling_even_when_both_streams_resume_together():
    guard = PolicyReleaseGuard()
    for now in np.arange(1.0, 1.7, 0.02):
        guard.update(now, now, now, True)
    guard.update(2.0, 2.0, 2.0, True)
    assert guard.reason(2.0)
    assert guard.reason(2.21)  # callback must also reject packets that became stale


def test_concurrent_dds_arrival_is_compared_against_a_later_clock(monkeypatch):
    clock = iter([1.0, 1.05])
    monkeypatch.setattr(policy_release_guard.time, "monotonic", lambda: next(clock))
    monitor = PolicyFeedbackMonitor.__new__(PolicyFeedbackMonitor)
    monitor.socket = SimpleNamespace(recv=lambda flags: b"g1_debug")
    monitor.guard = PolicyReleaseGuard()
    monitor.feedback_at = None
    monitor.status_path = None
    monitor.last_reason = None
    # Callback updates after feedback receipt but before body_command_status returns.
    bridge = SimpleNamespace(body_command_status=lambda: (1.04, True))
    monitor.update(bridge, True, 1.0)
    assert monitor.guard.healthy_since == 1.05
    assert "continuous" in monitor.guard.reason(1.05)


def test_initial_zero_gains_and_invalid_motor_commands_are_not_powered():
    bridge = UnitreeSdk2Bridge.__new__(UnitreeSdk2Bridge)
    bridge.low_cmd_lock = threading.Lock()
    motors = [SimpleNamespace(q=0.0, dq=0.0, kp=0.0, kd=0.0, tau=0.0) for _ in range(29)]
    bridge.low_cmd = SimpleNamespace(motor_cmd=motors)
    bridge.last_low_cmd_at = None
    assert bridge.body_command_status() == (None, False)
    for motor in motors:
        motor.kp, motor.kd = 40.0, 2.0
    bridge.LowCmdHandler(bridge.low_cmd)
    stamp, powered = bridge.body_command_status()
    assert stamp is not None and powered
    motors[3].q = float("nan")
    assert not bridge.body_command_status()[1]


def test_fall_reset_reattaches_band_and_refreshes_mujoco_transforms():
    env = DefaultEnv.__new__(DefaultEnv)
    env.mj_model = mujoco.MjModel.from_xml_string('''
        <mujoco><worldbody><body name="pelvis" pos="0 0 1">
        <freejoint/><geom type="sphere" size="0.1" mass="1"/>
        </body></worldbody></mujoco>''')
    env.mj_data = mujoco.MjData(env.mj_model)
    env.elastic_band = ElasticBand()
    env.elastic_band.enable = False
    guard = PolicyReleaseGuard()
    env.policy_monitor = SimpleNamespace(guard=guard)
    env.mj_data.qpos[2] = 0.1
    env.check_fall()
    assert env.elastic_band.enable
    assert env.mj_data.qpos[2] == pytest.approx(1.0)
    assert np.linalg.norm(env.mj_data.xquat[1]) == pytest.approx(1.0)
