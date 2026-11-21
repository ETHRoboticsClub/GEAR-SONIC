"""Dex3 hand amounts from the Pico trigger and grip."""

import numpy as np

from gear_sonic.scripts.pico_manager_thread_server import compute_hand_joints_from_inputs
from gear_sonic.utils.teleop.solver.hand.g1_gripper_ik_solver import (
    G1GripperInverseKinematicsSolver,
)


def _pair():
    return (
        G1GripperInverseKinematicsSolver(side="left"),
        G1GripperInverseKinematicsSolver(side="right"),
    )


def _joints(trigger, grip):
    left, right = compute_hand_joints_from_inputs(*_pair(), trigger, grip, trigger, grip)
    return left.reshape(-1), right.reshape(-1)


def test_release_opens_and_a_light_touch_stays_open():
    left, right = _joints(0.0, 0.0)
    assert np.all(left == 0.0)
    assert np.all(right == 0.0)
    left, right = _joints(0.05, 0.04)
    assert np.all(left == 0.0)
    assert np.all(right == 0.0)


def test_trigger_closes_fingers_and_thumb_together():
    left_solver = G1GripperInverseKinematicsSolver(side="left")
    closed = left_solver.partial_close(1.0, 1.0)
    left, _ = _joints(1.0, 0.0)
    np.testing.assert_allclose(left, closed, atol=1e-6)
    assert np.any(closed[:3] != 0.0)
    assert np.any(closed[3:] != 0.0)


def test_grip_closes_only_the_thumb():
    left_solver = G1GripperInverseKinematicsSolver(side="left")
    thumb = left_solver.partial_close(0.0, 1.0)
    left, _ = _joints(0.0, 1.0)
    np.testing.assert_allclose(left, thumb, atol=1e-6)
    assert np.all(left[3:] == 0.0)
    assert np.any(left[:3] != 0.0)


def test_grip_can_lead_the_thumb_ahead_of_the_fingers():
    left, _ = _joints(0.0, 1.0)
    led, _ = _joints(0.5, 1.0)
    assert np.all(led[3:] != 0.0)
    np.testing.assert_allclose(led[:3], left[:3], atol=1e-6)
    assert np.max(np.abs(led[3:])) < np.max(np.abs(left_full_fingers()))


def left_full_fingers():
    return G1GripperInverseKinematicsSolver(side="left").partial_close(1.0, 0.0)[3:]


def test_half_pull_is_half_of_the_pinch_and_the_right_hand_mirrors():
    left_solver = G1GripperInverseKinematicsSolver(side="left")
    right_solver = G1GripperInverseKinematicsSolver(side="right")
    # 0.525 sits halfway after the 0.05 deadzone: (0.525 - 0.05) / 0.95 = 0.5
    left, right = _joints(0.525, 0.0)
    np.testing.assert_allclose(left, 0.5 * left_solver.partial_close(1.0, 1.0), atol=1e-5)
    np.testing.assert_allclose(right, 0.5 * right_solver.partial_close(1.0, 1.0), atol=1e-5)
    # Fingers mirror. Thumb opposition uses the pose's own sign on each hand.
    np.testing.assert_allclose(right[1:], -left[1:], atol=1e-5)


def test_missing_solver_reports_an_open_hand():
    left, right = compute_hand_joints_from_inputs(None, None, 1.0, 1.0, 1.0, 1.0)
    assert left.shape == (1, 7)
    assert right.shape == (1, 7)
    assert np.all(left == 0.0)
    assert np.all(right == 0.0)
