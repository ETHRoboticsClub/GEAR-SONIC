"""The batched frame adapter preserves rigid transforms and the reader's sample."""

import numpy as np
from scipy.spatial.transform import Rotation

from gear_sonic.scripts import pico_manager_thread_server as server


def test_batch_relative_transform_matches_homogeneous_coordinates_and_does_not_mutate():
    rng = np.random.default_rng(12)
    poses = np.column_stack([rng.normal(size=(40, 3)), Rotation.random(40, random_state=rng).as_quat()])
    world = np.r_[rng.normal(size=3), Rotation.random(random_state=rng).as_quat()]
    saved_poses, saved_world = poses.copy(), world.copy()
    positions, quats = server._compute_rel_transform(poses, world, scalar_first=False)
    # The adapter maps Unity x,y,z onto robot -x,z,y. Solve reference-frame
    # coordinates independently, then change basis for both position and axes.
    basis = np.array([[-1.0, 0, 0], [0, 0, 1.0], [0, 1.0, 0]])
    world_rotation = Rotation.from_quat(world[3:]).as_matrix()
    for i, pose in enumerate(poses):
        expected_position = basis @ np.linalg.solve(world_rotation, pose[:3] - world[:3])
        expected_rotation = (
            basis @ np.linalg.solve(world_rotation, Rotation.from_quat(pose[3:]).as_matrix()) @ basis.T
        )
        np.testing.assert_allclose(positions[i], expected_position, atol=1e-12)
        np.testing.assert_allclose(
            Rotation.from_quat(quats[i], scalar_first=True).as_matrix(), expected_rotation, atol=1e-12
        )
    np.testing.assert_array_equal(poses, saved_poses)
    np.testing.assert_array_equal(world, saved_world)


def test_three_point_output_uses_only_its_keypoints_and_leaves_the_input_untouched():
    rng = np.random.default_rng(23)
    poses = np.column_stack([rng.normal(size=(24, 3)), Rotation.random(24, random_state=rng).as_quat()]).astype(
        np.float32
    )
    original = poses.copy()
    expected = server._process_3pt_pose(poses)
    np.testing.assert_array_equal(poses, original)
    unused = np.ones(24, dtype=bool)
    unused[[0, 22, 23, 12]] = False
    poses[unused] = 0  # even an untracked/invalid unused joint must not enter the conversion
    np.testing.assert_array_equal(server._process_3pt_pose(poses), expected)
