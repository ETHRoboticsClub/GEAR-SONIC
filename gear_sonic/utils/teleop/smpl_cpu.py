"""Batched CPU forward kinematics for inference-only SMPL teleoperation.

The generic torch implementation remains available for CUDA/autograd callers.
Rest joints, output indices and pose conventions match compute_human_joints.
"""

from functools import lru_cache
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation
import torch

_JOINTS_PATH = Path(__file__).resolve().parents[2] / 'data/human/human_joints_info.pkl'
_OUTPUT = np.r_[np.arange(22), 39, 54]


def model_local_joints(pelvis_local_joints, root_quats_wxyz):
    """SONIC encoder coordinates, including the training FK's fixed rest origin."""
    origin = prepare_cpu_kinematics()[0][0]
    offset = Rotation.from_quat(root_quats_wxyz, scalar_first=True).inv().apply(origin)
    return (pelvis_local_joints + offset[..., None, :]).astype(np.float32)


@lru_cache(maxsize=1)
def prepare_cpu_kinematics():
    # This is the same trusted local model asset loaded by the torch path.
    info = torch.load(_JOINTS_PATH, map_location='cpu', weights_only=False)
    rest = info['J'].detach().numpy().astype(np.float32)
    parents = np.asarray(info['parents_list'], dtype=np.intp)
    if rest.shape != (55, 3) or parents.shape != (55,):
        raise ValueError('Expected the 55-joint SMPL rest skeleton')
    offsets = rest.copy()
    offsets[1:] -= rest[parents[1:]]
    depth = np.zeros(55, dtype=int)
    for joint in range(1, 55):
        if not 0 <= parents[joint] < joint:
            raise ValueError('SMPL parents must precede children')
        depth[joint] = depth[parents[joint]] + 1
    levels = tuple(np.flatnonzero(depth == level) for level in range(1, depth.max() + 1))
    _pelvis_local_basis()
    return rest, offsets, parents, levels


def human_joints_cpu(body_pose, global_orient):
    """Return float32 (*batch, 24, 3) joints for CPU inference tensors."""
    return _human_joints_numpy(body_pose.detach().numpy(), global_orient.detach().numpy())


@lru_cache(maxsize=1)
def _pelvis_local_basis():
    from gear_sonic.isaac_utils.rotations import remove_smpl_base_rot

    unbased = remove_smpl_base_rot(torch.tensor([[1., 0., 0., 0.]]), w_last=False)
    return Rotation.from_quat(unbased.numpy()[0], scalar_first=True).inv().as_matrix().astype(np.float32)


def pelvis_local_joints_cpu(body_pose: np.ndarray) -> np.ndarray:
    """SMPL (*batch, 21, 3) rotvecs in rad → pelvis-relative, base-removed joints in m."""
    pose = np.asarray(body_pose, dtype=np.float32)
    if pose.shape[-2:] != (21, 3) or not np.isfinite(pose).all():
        raise ValueError('Expected finite SMPL body rotations with shape (*batch, 21, 3)')
    batch = pose.shape[:-2]
    joints = _human_joints_numpy(pose.reshape(*batch, 63), np.zeros((*batch, 3), np.float32))
    return (joints - joints[..., :1, :]) @ _pelvis_local_basis().T


def _human_joints_numpy(body: np.ndarray, root: np.ndarray) -> np.ndarray:
    rest, offsets, parents, levels = prepare_cpu_kinematics()
    batch = body.shape[:-1]
    full_pose = np.zeros((*batch, 55, 3), dtype=np.float32)
    full_pose[..., 0, :] = root
    full_pose[..., 1:22, :] = body[..., :63].reshape(*batch, 21, 3)
    local = Rotation.from_rotvec(full_pose.reshape(-1, 3)).as_matrix().astype(np.float32)
    local = local.reshape(*batch, 55, 3, 3)
    rotations = np.empty_like(local)
    positions = np.empty((*batch, 55, 3), dtype=np.float32)
    rotations[..., 0, :, :] = local[..., 0, :, :]
    positions[..., 0, :] = rest[0]
    for level in levels:
        parent_rotations = rotations[..., parents[level], :, :]
        positions[..., level, :] = (positions[..., parents[level], :]
                                    + np.einsum('...nij,nj->...ni', parent_rotations, offsets[level]))
        rotations[..., level, :, :] = parent_rotations @ local[..., level, :, :]
    return positions[..., _OUTPUT, :]
