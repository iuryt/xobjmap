"""Basic tests for vectorial objective analysis."""

import numpy as np

from xobjmap import vectoa


def test_vectoa_returns_correct_shape():
    """vectoa output shape should match the target grid."""
    # Simple target grid
    gx, gy = np.meshgrid(np.linspace(0, 10, 8), np.linspace(0, 10, 6))

    # Scattered velocity observations
    rng = np.random.default_rng(42)
    x = rng.uniform(1, 9, 20)
    y = rng.uniform(1, 9, 20)
    u = rng.standard_normal(20)
    v = rng.standard_normal(20)

    psi = vectoa(gx, gy, x, y, u, v, corrlenx=3.0, corrleny=3.0, err=0.1)

    assert psi.shape == gx.shape


def test_vectoa_does_not_modify_inputs():
    """vectoa should not modify the input arrays."""
    gx, gy = np.meshgrid(np.linspace(0, 10, 5), np.linspace(0, 10, 5))
    x = np.array([2.0, 5.0, 8.0])
    y = np.array([2.0, 5.0, 8.0])
    u = np.array([1.0, 0.0, -1.0])
    v = np.array([0.0, 1.0, 0.0])

    gx_orig = gx.copy()
    x_orig = x.copy()
    u_orig = u.copy()

    vectoa(gx, gy, x, y, u, v, corrlenx=3.0, corrleny=3.0, err=0.1)

    np.testing.assert_array_equal(gx, gx_orig)
    np.testing.assert_array_equal(x, x_orig)
    np.testing.assert_array_equal(u, u_orig)
