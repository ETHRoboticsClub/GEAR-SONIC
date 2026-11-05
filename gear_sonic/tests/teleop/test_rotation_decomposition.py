import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from gear_sonic.trl.utils.rotation_conversion import decompose_rotation_aa


@pytest.mark.parametrize("rotvec, twist, swing", [
    ([0, 0, 0], [1, 0, 0, 0], [1, 0, 0, 0]),
    ([0, np.pi / 2, 0], [2**-.5, 0, 2**-.5, 0], [1, 0, 0, 0]),
    ([np.pi / 2, 0, 0], [1, 0, 0, 0], [2**-.5, 2**-.5, 0, 0]),
    ([np.pi, 0, 0], [1, 0, 0, 0], [0, 1, 0, 0]),
])
def test_neutral_and_axis_aligned_limbs(rotvec, twist, swing):
    actual_twist, actual_swing = decompose_rotation_aa(np.array([rotvec]), np.array([0, 1, 0]))
    np.testing.assert_allclose(actual_twist[0], twist, atol=1e-12)
    np.testing.assert_allclose(actual_swing[0], swing, atol=1e-12)


def test_mixed_rotation_reconstructs_original():
    rotvec = np.array([[.2, -.7, .4], [1e-15, 0, 0]])
    twist, swing = decompose_rotation_aa(rotvec, np.array([0, 1, 0]))
    actual = Rotation.from_quat(twist, scalar_first=True) * Rotation.from_quat(swing, scalar_first=True)
    np.testing.assert_allclose(actual.as_matrix(), Rotation.from_rotvec(rotvec).as_matrix(), atol=1e-12)
