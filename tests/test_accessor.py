"""Tests for the xarray accessor."""

import numpy as np
import xarray as xr

import xobjmap


def test_scalar_interp_returns_dataset():
    """scalar_interp should return an xr.Dataset with var and error."""
    rng = np.random.default_rng(42)
    obs = xr.Dataset(
        {"temp": ("station", 2 * rng.uniform(0, 10, 30) + 3 * rng.uniform(0, 10, 30))},
        coords={
            "lon": ("station", rng.uniform(0, 10, 30)),
            "lat": ("station", rng.uniform(0, 10, 30)),
        },
    )
    target = xr.Dataset(
        coords={
            "lon": np.linspace(1, 9, 10),
            "lat": np.linspace(1, 9, 8),
        }
    )

    result = obs.xobjmap.scalar_interp("temp", target, corrlen={"lon": 3.0, "lat": 3.0}, err=0.1)

    assert isinstance(result, xr.Dataset)
    assert "temp" in result
    assert "error" in result
    assert result["temp"].dims == ("lat", "lon")
    assert result["temp"].shape == (8, 10)


def test_scalar_interp_isotropic_corrlen():
    """scalar_interp should accept a scalar corrlen for isotropic case."""
    rng = np.random.default_rng(42)
    obs = xr.Dataset(
        {"temp": ("station", rng.standard_normal(20))},
        coords={
            "lon": ("station", rng.uniform(0, 10, 20)),
            "lat": ("station", rng.uniform(0, 10, 20)),
        },
    )
    target = xr.Dataset(
        coords={"lon": np.linspace(1, 9, 5), "lat": np.linspace(1, 9, 5)}
    )

    result = obs.xobjmap.scalar_interp("temp", target, corrlen=3.0, err=0.1)

    assert result["temp"].shape == (5, 5)


def test_scalar_interp_error_bounded():
    """Error values should be between 0 and 1."""
    rng = np.random.default_rng(42)
    obs = xr.Dataset(
        {"temp": ("station", rng.standard_normal(30))},
        coords={
            "lon": ("station", rng.uniform(0, 10, 30)),
            "lat": ("station", rng.uniform(0, 10, 30)),
        },
    )
    target = xr.Dataset(
        coords={"lon": np.linspace(0, 10, 10), "lat": np.linspace(0, 10, 10)}
    )

    result = obs.xobjmap.scalar_interp("temp", target, corrlen=3.0, err=0.1)

    assert np.all(result["error"].values >= 0)
    assert np.all(result["error"].values <= 1)


def test_vector_interp_returns_psi():
    """vector_interp should return an xr.Dataset with psi."""
    rng = np.random.default_rng(42)
    obs = xr.Dataset(
        {
            "u": ("station", rng.standard_normal(20)),
            "v": ("station", rng.standard_normal(20)),
        },
        coords={
            "lon": ("station", rng.uniform(1, 9, 20)),
            "lat": ("station", rng.uniform(1, 9, 20)),
        },
    )
    target = xr.Dataset(
        coords={"lon": np.linspace(2, 8, 6), "lat": np.linspace(2, 8, 5)}
    )

    result = obs.xobjmap.vector_interp("u", "v", target, corrlen={"lon": 3.0, "lat": 3.0}, err=0.1)

    assert isinstance(result, xr.Dataset)
    assert "psi" in result
    assert result["psi"].dims == ("lat", "lon")
    assert result["psi"].shape == (5, 6)
