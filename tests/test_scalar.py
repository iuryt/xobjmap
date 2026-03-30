"""Basic tests for scalar objective analysis."""

import numpy as np
import pytest

from xobjmap import scalar


def test_scalar_recovers_linear_field():
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

    tp, ep = scalar(xc, yc, x, y, t, corrlenx=5.0, corrleny=5.0, err=0.01)

    np.testing.assert_allclose(tp.ravel(), t_true, atol=0.5)


def test_scalar_error_only():
    """scalar without data should return only the error map."""
    x = np.array([0.0, 1.0, 2.0])
    y = np.array([0.0, 1.0, 2.0])
    xc = np.array([0.5, 1.5])
    yc = np.array([0.5, 1.5])

    ep = scalar(xc, yc, x, y, corrlenx=2.0, corrleny=2.0, err=0.1)

    assert ep.shape == (2,)
    assert np.all(ep >= 0) and np.all(ep <= 1)


def test_scalar_error_near_observations_is_small():
    """Error should be small near observation points."""
    x = np.array([5.0])
    y = np.array([5.0])
    xc = np.array([5.0])  # exactly at the observation
    yc = np.array([5.0])

    ep = scalar(xc, yc, x, y, corrlenx=2.0, corrleny=2.0, err=0.05)

    assert ep[0] < 0.1


def test_scalar_does_not_modify_inputs():
    """scalar should not modify the input arrays."""
    x = np.array([0.0, 1.0, 2.0])
    y = np.array([0.0, 1.0, 2.0])
    t = np.array([10.0, 20.0, 30.0])
    xc = np.array([0.5, 1.5])
    yc = np.array([0.5, 1.5])

    x_orig = x.copy()
    xc_orig = xc.copy()
    t_orig = t.copy()

    scalar(xc, yc, x, y, t, corrlenx=3.0, corrleny=1.5, err=0.1)

    np.testing.assert_array_equal(x, x_orig)
    np.testing.assert_array_equal(xc, xc_orig)
    np.testing.assert_array_equal(t, t_orig)
