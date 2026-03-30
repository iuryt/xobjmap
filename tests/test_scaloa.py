"""Basic tests for scalar objective analysis."""

import numpy as np
import pytest

from xobjmap import scaloa


def test_scaloa_recovers_linear_field():
    """scaloa should exactly recover a linear field with small error."""
    # Create scattered observations of a linear field: t = 2*x + 3*y
    rng = np.random.default_rng(42)
    x = rng.uniform(0, 10, 50)
    y = rng.uniform(0, 10, 50)
    t = 2 * x + 3 * y

    # Target points
    xc = np.array([2.0, 5.0, 8.0])
    yc = np.array([3.0, 5.0, 7.0])
    t_true = 2 * xc + 3 * yc

    tp, ep = scaloa(xc, yc, x, y, t, corrlenx=5.0, corrleny=5.0, err=0.01)

    np.testing.assert_allclose(tp.ravel(), t_true, atol=0.5)


def test_scaloa_error_only():
    """scaloa without data should return only the error map."""
    x = np.array([0.0, 1.0, 2.0])
    y = np.array([0.0, 1.0, 2.0])
    xc = np.array([0.5, 1.5])
    yc = np.array([0.5, 1.5])

    ep = scaloa(xc, yc, x, y, corrlenx=2.0, corrleny=2.0, err=0.1)

    assert ep.shape == (2,)
    assert np.all(ep >= 0) and np.all(ep <= 1)


def test_scaloa_error_near_observations_is_small():
    """Error should be small near observation points."""
    x = np.array([5.0])
    y = np.array([5.0])
    xc = np.array([5.0])  # exactly at the observation
    yc = np.array([5.0])

    ep = scaloa(xc, yc, x, y, corrlenx=2.0, corrleny=2.0, err=0.05)

    assert ep[0] < 0.1
