"""Basic tests for scalar objective analysis."""

import numpy as np
import pytest

from xobjmap import error, scalar


def test_scalar_recovers_linear_field(backend):
    """scalar should exactly recover a linear field with small error."""
    # Create scattered observations of a linear field: t = 2*x + 3*y
    rng = np.random.default_rng(42)
    x = rng.uniform(0, 10, 50)
    y = rng.uniform(0, 10, 50)
    t = 2 * x + 3 * y

    # Target points
    xc = np.array([2.0, 5.0, 8.0])
    yc = np.array([3.0, 5.0, 7.0])
    t_true = 2 * xc + 3 * yc

    tp = scalar(xc, yc, x, y, t, corrlenx=5.0, corrleny=5.0, err=0.01, backend=backend)

    np.testing.assert_allclose(np.asarray(tp).ravel(), t_true, atol=0.5)


def test_error_field(backend):
    """error should return values between 0 and 1."""
    x = np.array([0.0, 1.0, 2.0])
    y = np.array([0.0, 1.0, 2.0])
    xc = np.array([0.5, 1.5])
    yc = np.array([0.5, 1.5])

    ep = error(xc, yc, x, y, corrlenx=2.0, corrleny=2.0, err=0.1, backend=backend)

    ep = np.asarray(ep)
    assert ep.shape == (2,)
    assert np.all(ep >= 0) and np.all(ep <= 1)


def test_error_near_observations_is_small(backend):
    """Error should be small near observation points."""
    x = np.array([5.0])
    y = np.array([5.0])
    xc = np.array([5.0])  # exactly at the observation
    yc = np.array([5.0])

    ep = error(xc, yc, x, y, corrlenx=2.0, corrleny=2.0, err=0.05, backend=backend)

    assert float(np.asarray(ep)[0]) < 0.1


def test_scalar_does_not_modify_inputs(backend):
    """scalar should not modify the input arrays."""
    x = np.array([0.0, 1.0, 2.0])
    y = np.array([0.0, 1.0, 2.0])
    t = np.array([10.0, 20.0, 30.0])
    xc = np.array([0.5, 1.5])
    yc = np.array([0.5, 1.5])

    x_orig = x.copy()
    xc_orig = xc.copy()
    t_orig = t.copy()

    scalar(xc, yc, x, y, t, corrlenx=3.0, corrleny=1.5, err=0.1, backend=backend)

    np.testing.assert_array_equal(x, x_orig)
    np.testing.assert_array_equal(xc, xc_orig)
    np.testing.assert_array_equal(t, t_orig)


def test_scalar_invalid_backend():
    """scalar should raise ValueError for unknown backends."""
    x = np.array([0.0, 1.0])
    y = np.array([0.0, 1.0])
    t = np.array([1.0, 2.0])
    with pytest.raises(ValueError, match="Unknown backend"):
        scalar([0.5], [0.5], x, y, t, corrlenx=1.0, corrleny=1.0, err=0.1, backend="cupy")
