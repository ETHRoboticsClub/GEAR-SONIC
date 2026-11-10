import numpy as np
import pytest
import torch

from gear_sonic.scripts import pico_manager_thread_server as server
from gear_sonic.trl.utils.torch_transform import compute_human_joints
from gear_sonic.utils.teleop.smpl_cpu import human_joints_cpu


@pytest.mark.parametrize('scale', [0., 1e-7, .2, 3.14])
def test_cpu_skeleton_matches_generic_fk_for_batched_poses(scale):
    rng = np.random.default_rng(42)
    body = torch.from_numpy((rng.uniform(-1, 1, (16, 63)) * scale).astype(np.float32))
    root = torch.from_numpy((rng.uniform(-1, 1, (16, 3)) * scale).astype(np.float32))
    expected = compute_human_joints(body, root).numpy()
    actual = human_joints_cpu(body, root)
    assert actual.shape == (16, 24, 3)
    np.testing.assert_allclose(actual, expected, atol=5e-6, rtol=1e-5)


def test_gradient_callers_keep_the_generic_torch_path():
    body = torch.full((1, 69), .1, requires_grad=True)
    root = torch.full((1, 3), .2, requires_grad=True)
    result = server.process_smpl_joints(body, root, torch.zeros((1, 3)))
    result['joints'].sum().backward()
    assert body.grad is not None and torch.isfinite(body.grad).all()
    assert root.grad is not None and torch.isfinite(root.grad).all()


@pytest.mark.parametrize('gradient', [False, True])
@pytest.mark.parametrize('body_angle', [0., .2])
def test_local_body_has_pelvis_origin_and_is_independent_of_global_orientation(gradient, body_angle):
    body = torch.full((1, 69), body_angle, requires_grad=gradient)
    translation = torch.zeros((1, 3))
    reference = server.process_smpl_joints(body, torch.zeros((1, 3)), translation)
    local = reference['smpl_joints_local'].detach().numpy()
    np.testing.assert_allclose(local[:, 0], 0., atol=5e-6)
    for axis in range(3):
        root = torch.zeros((1, 3))
        root[0, axis] = 1.2
        turned = server.process_smpl_joints(body, root, translation)
        np.testing.assert_allclose(turned['smpl_joints_local'].detach().numpy(), local, atol=5e-6)
        # A change of origin/rotation cannot change a skeleton's bone lengths.
        raw_lengths = torch.linalg.vector_norm(turned['joints'][:, 1:] - turned['joints'][:, :1], dim=-1)
        local_lengths = torch.linalg.vector_norm(turned['smpl_joints_local'][:, 1:], dim=-1)
        torch.testing.assert_close(local_lengths, raw_lengths, atol=5e-6, rtol=1e-5)
