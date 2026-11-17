"""Offline short-stride references preserving foot height and foot orientation.

By default only pelvis-relative ankle X excursion is scaled; Y/Z and ankle
orientation remain from the paired walking clip. The optional height scale
and bend-plane/centering choices are separate diagnostic variants. Two-link sphere intersection preserves
bone lengths and the nearest original bend-plane direction. Minimal changes
to each segment's global rotation realize the targets, preserving foot frames.
These are diagnostic references, not a live gain or certified feasible motion.
"""

from pathlib import Path
import argparse
import hashlib
import json
import numpy as np
from scipy.spatial.transform import Rotation
from gear_sonic.utils.teleop.smpl_cpu import pelvis_local_joints_cpu, prepare_cpu_kinematics

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--height-scale", type=float, default=1.0)
parser.add_argument("--center", choices=("initial", "hip"), default="initial")
parser.add_argument("--amplitudes", type=float, nargs="+", default=[0.25, 0.5, 1.0])
parser.add_argument("--bend-plane", choices=("original", "forward"), default="original")
parser.add_argument("--reference", type=Path, required=True)
parser.add_argument("--output-dir", type=Path, required=True)
args = parser.parse_args()
if not 0 < args.height_scale <= 1 or any(not 0 < value <= 1 for value in args.amplitudes):
    parser.error("Height and stride scales must be in (0, 1]")
out = args.output_dir
out.mkdir(parents=True, exist_ok=True)
with np.load(args.reference) as f:
    source = dict(f)
indices, first = np.unique(source["indices"], return_index=True)
assert np.array_equal(indices, np.arange(len(indices)))
original = source["pose"].reshape(-1, 21, 3)[first]
positions = pelvis_local_joints_cpu(original)
basis = Rotation.from_quat([0.5, 0.5, 0.5, 0.5])


def direction_change(old: np.ndarray, new: np.ndarray) -> Rotation:
    old = old / np.linalg.norm(old, axis=1)[:, None]
    new = new / np.linalg.norm(new, axis=1)[:, None]
    cross = np.cross(old, new)
    length = np.linalg.norm(cross, axis=1)
    dot = np.sum(old * new, axis=1)
    assert np.all(dot > -0.999), "Antipodal segment has no unique minimal correction"
    angle = np.arctan2(length, dot)
    return Rotation.from_rotvec(cross * (angle / np.where(length > 0, length, 1))[:, None])


def independent_joints(pose: np.ndarray) -> np.ndarray:
    rest, _, parents, _ = prepare_cpu_kinematics()
    rotations = np.empty((55, 3, 3))
    positions = np.empty((55, 3))
    rotations[0] = basis.as_matrix()
    positions[0] = rest[0]
    local = Rotation.from_rotvec(pose).as_matrix()
    for joint in range(1, 55):
        parent = parents[joint]
        positions[joint] = positions[parent] + rotations[parent] @ (rest[joint] - rest[parent])
        rotations[joint] = rotations[parent] @ (local[joint - 1] if joint <= 21 else np.eye(3))
    return positions[np.r_[np.arange(22), 39, 54]] - rest[:1]


report = []
for amplitude in args.amplitudes:
    case = f"short-stride-{int(amplitude * 100):03d}"
    if args.bend_plane == "forward":
        case += "-forward"
    if args.center == "hip":
        case += "-hip"
    if args.height_scale != 1:
        case += f"-height{round(args.height_scale * 100):03d}"
    target = positions.copy()
    pose = original.copy()
    orientation_errors = []
    for h, k, a in ((1, 4, 7), (2, 5, 8)):
        hip = positions[:, h]
        knee = positions[:, k]
        ankle = positions[:, a]
        desired = ankle.copy()
        center = ankle[0, 0] if args.center == "initial" else hip[:, 0]
        desired[:, 0] = center + amplitude * (ankle[:, 0] - center)
        desired[:, 2] = ankle[0, 2] + args.height_scale * (ankle[:, 2] - ankle[0, 2])
        delta = desired - hip
        distance = np.linalg.norm(delta, axis=1)
        l1 = np.linalg.norm(knee - hip, axis=1)
        l2 = np.linalg.norm(ankle - knee, axis=1)
        assert np.all(distance <= l1 + l2 + 1e-6) and np.all(distance >= np.abs(l1 - l2)), "Unreachable target"
        along = delta / distance[:, None]
        projection = (l1 * l1 - l2 * l2 + distance * distance) / (2 * distance)
        radius = np.sqrt(np.maximum(0, l1 * l1 - projection * projection))
        normal = knee - hip - np.sum((knee - hip) * along, axis=1)[:, None] * along
        if args.bend_plane == "forward":
            forward = np.tile([1.0, 0.0, 0.0], (len(pose), 1))
            normal = forward - np.sum(forward * along, axis=1)[:, None] * along
        length = np.linalg.norm(normal, axis=1)
        assert np.all(length > 1e-7), "Ambiguous straight-leg bend plane"
        new_knee = hip + projection[:, None] * along + radius[:, None] * normal / length[:, None]
        old_hip = basis * Rotation.from_rotvec(original[:, h - 1])
        old_knee = old_hip * Rotation.from_rotvec(original[:, k - 1])
        foot = old_knee * Rotation.from_rotvec(original[:, a - 1])
        new_hip = direction_change(knee - hip, new_knee - hip) * old_hip
        new_knee_rotation = direction_change(ankle - knee, desired - new_knee) * old_knee
        pose[:, h - 1] = (basis.inv() * new_hip).as_rotvec()
        pose[:, k - 1] = (new_hip.inv() * new_knee_rotation).as_rotvec()
        pose[:, a - 1] = (new_knee_rotation.inv() * foot).as_rotvec()
        actual_foot = (
            basis
            * Rotation.from_rotvec(pose[:, h - 1])
            * Rotation.from_rotvec(pose[:, k - 1])
            * Rotation.from_rotvec(pose[:, a - 1])
        )
        orientation_errors.append(np.degrees((foot.inv() * actual_foot).magnitude()).max())
        target[:, a] = desired
    actual = pelvis_local_joints_cpu(pose)
    error = np.linalg.norm(actual[:, [7, 8]] - target[:, [7, 8]], axis=-1)
    assert error.max() < 2e-6
    independent = np.array([independent_joints(p) for p in pose])
    assert np.linalg.norm(independent - actual, axis=-1).max() < 2e-6
    generated = dict(source, pose=pose[source["indices"]], joints=actual[source["indices"]])
    dest = out / (case + "-reference.npz")
    if dest.exists():
        raise FileExistsError(dest)
    np.savez_compressed(dest, **generated)
    report.append(
        {
            "case": case,
            "reference_sha256": hashlib.sha256(args.reference.read_bytes()).hexdigest(),
            "output_sha256": hashlib.sha256(dest.read_bytes()).hexdigest(),
            "x_scale": amplitude,
            "z_scale": args.height_scale,
            "horizontal_center": args.center,
            "bend_plane": args.bend_plane,
            "max_ankle_target_error_m": float(error.max()),
            "max_foot_orientation_change_deg": float(max(orientation_errors)),
            "max_adjacent_local_rotation_deg": float(
                np.degrees(
                    (
                        Rotation.from_rotvec(pose[:-1].reshape(-1, 3)).inv()
                        * Rotation.from_rotvec(pose[1:].reshape(-1, 3))
                    ).magnitude()
                ).max()
            ),
            "max_independent_fk_error_m": float(np.linalg.norm(independent - actual, axis=-1).max()),
        }
    )
(out / (f"{args.bend_plane}-{args.center}-height{round(args.height_scale * 100):03d}-generation.json")).write_text(
    json.dumps(report, indent=2) + "\n"
)
print(json.dumps(report, indent=2))
