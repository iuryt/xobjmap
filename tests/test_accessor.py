"""Tests for the xarray accessor."""

import importlib.util

import numpy as np
import pytest
import xarray as xr

import xobjmap


def test_scalar_returns_dataset(backend):
    """scalar should return an xr.Dataset with var and error."""
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

    result = obs.xobjmap.scalar("temp", target, corrlen={"lon": 3.0, "lat": 3.0}, err=0.1, backend=backend)

    assert isinstance(result, xr.Dataset)
    assert "temp" in result
    assert "error" in result
    assert result["temp"].dims == ("lat", "lon")
    assert result["temp"].shape == (8, 10)


def test_scalar_isotropic_corrlen(backend):
    """scalar should accept a scalar corrlen for isotropic case."""
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

    result = obs.xobjmap.scalar("temp", target, corrlen=3.0, err=0.1, backend=backend)

    assert result["temp"].shape == (5, 5)


def test_scalar_error_bounded(backend):
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

    result = obs.xobjmap.scalar("temp", target, corrlen=3.0, err=0.1, backend=backend)

    assert np.all(result["error"].values >= 0)
    assert np.all(result["error"].values <= 1)


def test_streamfunction_returns_psi(backend):
    """streamfunction should return an xr.Dataset with psi and psi_error."""
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

    result = obs.xobjmap.streamfunction("u", "v", target, corrlen={"lon": 3.0, "lat": 3.0}, err=0.1, backend=backend)

    assert isinstance(result, xr.Dataset)
    assert "psi" in result
    assert "psi_error" in result
    assert result["psi"].dims == ("lat", "lon")
    assert result["psi"].shape == (5, 6)
    assert result["psi_error"].shape == (5, 6)
    assert np.all(result["psi_error"].values >= 0)
    assert np.all(result["psi_error"].values <= 1)


def test_velocity_potential_returns_chi(backend):
    """velocity_potential should return an xr.Dataset with chi and chi_error."""
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

    result = obs.xobjmap.velocity_potential("u", "v", target, corrlen={"lon": 3.0, "lat": 3.0}, err=0.1, backend=backend)

    assert isinstance(result, xr.Dataset)
    assert "chi" in result
    assert "chi_error" in result
    assert result["chi"].dims == ("lat", "lon")
    assert result["chi"].shape == (5, 6)
    assert result["chi_error"].shape == (5, 6)
    assert np.all(result["chi_error"].values >= 0)
    assert np.all(result["chi_error"].values <= 1)


def test_helmholtz_returns_psi_and_chi(backend):
    """helmholtz should return an xr.Dataset with fields and errors."""
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

    result = obs.xobjmap.helmholtz(
        "u", "v", target,
        corrlen_psi={"lon": 3.0, "lat": 3.0},
        corrlen_chi={"lon": 3.0, "lat": 3.0},
        err=0.1, backend=backend,
    )

    assert isinstance(result, xr.Dataset)
    assert "psi" in result
    assert "chi" in result
    assert "psi_error" in result
    assert "chi_error" in result
    assert result["psi"].dims == ("lat", "lon")
    assert result["psi"].shape == (5, 6)
    assert result["chi"].shape == (5, 6)
    assert result["psi_error"].shape == (5, 6)
    assert result["chi_error"].shape == (5, 6)
    assert np.all(result["psi_error"].values >= 0)
    assert np.all(result["psi_error"].values <= 1)
    assert np.all(result["chi_error"].values >= 0)
    assert np.all(result["chi_error"].values <= 1)


def test_accessors_can_skip_error_computation(backend):
    """return_error=False should omit error variables from accessor outputs."""
    rng = np.random.default_rng(42)
    obs = xr.Dataset(
        {
            "temp": ("station", rng.standard_normal(20)),
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

    scalar = obs.xobjmap.scalar(
        "temp", target, corrlen=3.0, err=0.1, backend=backend, return_error=False
    )
    psi = obs.xobjmap.streamfunction(
        "u", "v", target, corrlen=3.0, err=0.1, backend=backend, return_error=False
    )
    chi = obs.xobjmap.velocity_potential(
        "u", "v", target, corrlen=3.0, err=0.1, backend=backend, return_error=False
    )
    helm = obs.xobjmap.helmholtz(
        "u", "v", target,
        corrlen_psi=3.0, corrlen_chi=3.0,
        err=0.1, backend=backend, return_error=False,
    )

    assert "error" not in scalar
    assert "psi_error" not in psi
    assert "chi_error" not in chi
    assert "psi_error" not in helm
    assert "chi_error" not in helm


def test_error_only_accessors_return_expected_variables(backend):
    """Dedicated error accessors should return only the documented error fields."""
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
    scalar_obs = xr.Dataset(
        {"temp": ("station", rng.standard_normal(20))},
        coords={
            "lon": ("station", rng.uniform(1, 9, 20)),
            "lat": ("station", rng.uniform(1, 9, 20)),
        },
    )
    target = xr.Dataset(
        coords={"lon": np.linspace(2, 8, 6), "lat": np.linspace(2, 8, 5)}
    )

    scalar_error = scalar_obs.xobjmap.scalar_error(target, corrlen=3.0, err=0.1, backend=backend)
    psi_error = obs.xobjmap.streamfunction_error(
        "u", "v", target, corrlen=3.0, err=0.1, backend=backend
    )
    chi_error = obs.xobjmap.velocity_potential_error(
        "u", "v", target, corrlen=3.0, err=0.1, backend=backend
    )
    helm_error = obs.xobjmap.helmholtz_error(
        "u", "v", target, corrlen_psi=3.0, corrlen_chi=3.0, err=0.1, backend=backend
    )

    assert set(scalar_error.data_vars) == {"error"}
    assert set(psi_error.data_vars) == {"psi_error"}
    assert set(chi_error.data_vars) == {"chi_error"}
    assert set(helm_error.data_vars) == {"psi_error", "chi_error"}

    for field in [scalar_error.error, psi_error.psi_error, chi_error.chi_error,
                  helm_error.psi_error, helm_error.chi_error]:
        assert np.all(field.values >= 0)
        assert np.all(field.values <= 1)


def test_error_only_accessors_match_default_accessor_errors(backend):
    """Default accessor error fields should match the error-only accessors."""
    rng = np.random.default_rng(42)
    obs = xr.Dataset(
        {
            "temp": ("station", rng.standard_normal(30)),
            "u": ("station", rng.standard_normal(30)),
            "v": ("station", rng.standard_normal(30)),
        },
        coords={
            "lon": ("station", rng.uniform(0, 10, 30)),
            "lat": ("station", rng.uniform(0, 10, 30)),
        },
    )
    target = xr.Dataset(coords={"lon": np.linspace(1, 9, 8), "lat": np.linspace(1, 9, 7)})

    scalar = obs.xobjmap.scalar("temp", target, corrlen=3.0, err=0.1, backend=backend)
    scalar_error = obs[["temp"]].xobjmap.scalar_error(target, corrlen=3.0, err=0.1, backend=backend)
    psi = obs.xobjmap.streamfunction("u", "v", target, corrlen=3.0, err=0.1, backend=backend)
    psi_error = obs.xobjmap.streamfunction_error("u", "v", target, corrlen=3.0, err=0.1, backend=backend)
    chi = obs.xobjmap.velocity_potential("u", "v", target, corrlen=3.0, err=0.1, backend=backend)
    chi_error = obs.xobjmap.velocity_potential_error("u", "v", target, corrlen=3.0, err=0.1, backend=backend)
    helm = obs.xobjmap.helmholtz(
        "u", "v", target, corrlen_psi=3.0, corrlen_chi=3.0, err=0.1, backend=backend
    )
    helm_error = obs.xobjmap.helmholtz_error(
        "u", "v", target, corrlen_psi=3.0, corrlen_chi=3.0, err=0.1, backend=backend
    )

    np.testing.assert_allclose(scalar.error, scalar_error.error)
    np.testing.assert_allclose(psi.psi_error, psi_error.psi_error)
    np.testing.assert_allclose(chi.chi_error, chi_error.chi_error)
    np.testing.assert_allclose(helm.psi_error, helm_error.psi_error)
    np.testing.assert_allclose(helm.chi_error, helm_error.chi_error)


def test_accessors_support_2d_target_coordinates(backend):
    """Accessors should preserve 2D target coordinate structure."""
    rng = np.random.default_rng(42)
    obs = xr.Dataset(
        {
            "temp": ("station", rng.standard_normal(25)),
            "u": ("station", rng.standard_normal(25)),
            "v": ("station", rng.standard_normal(25)),
        },
        coords={
            "lon": ("station", rng.uniform(-5, 5, 25)),
            "lat": ("station", rng.uniform(-5, 5, 25)),
        },
    )

    lon_1d = np.linspace(-4, 4, 6)
    lat_1d = np.linspace(-3, 3, 5)
    lon2d, lat2d = np.meshgrid(lon_1d, lat_1d)
    target = xr.Dataset(
        coords={
            "lon": (("lat", "lon"), lon2d),
            "lat": (("lat", "lon"), lat2d),
        }
    )

    scalar = obs.xobjmap.scalar("temp", target, corrlen=3.0, err=0.1, backend=backend)
    helm = obs.xobjmap.helmholtz(
        "u", "v", target, corrlen_psi=3.0, corrlen_chi=3.0, err=0.1, backend=backend
    )

    assert scalar.temp.dims == ("lat", "lon")
    assert scalar.error.dims == ("lat", "lon")
    assert helm.psi.dims == ("lat", "lon")
    assert helm.chi.dims == ("lat", "lon")
    assert helm.psi_error.dims == ("lat", "lon")
    assert helm.chi_error.dims == ("lat", "lon")


@pytest.mark.skipif(importlib.util.find_spec("jax") is None, reason="jax not installed")
def test_accessor_numpy_and_jax_are_reasonably_close():
    """NumPy and JAX accessors should agree on small deterministic problems."""
    rng = np.random.default_rng(42)
    obs = xr.Dataset(
        {
            "temp": ("station", rng.standard_normal(35)),
            "u": ("station", rng.standard_normal(35)),
            "v": ("station", rng.standard_normal(35)),
        },
        coords={
            "lon": ("station", rng.uniform(-4, 4, 35)),
            "lat": ("station", rng.uniform(-4, 4, 35)),
        },
    )
    target = xr.Dataset(coords={"lon": np.linspace(-3, 3, 9), "lat": np.linspace(-3, 3, 8)})

    scalar_np = obs.xobjmap.scalar("temp", target, corrlen=2.5, err=0.1, backend="numpy")
    scalar_jax = obs.xobjmap.scalar("temp", target, corrlen=2.5, err=0.1, backend="jax")
    np.testing.assert_allclose(scalar_np.temp, scalar_jax.temp, atol=1e-3, rtol=5e-2)
    np.testing.assert_allclose(scalar_np.error, scalar_jax.error, atol=2e-2, rtol=1.5e-1)

    helm_np = obs.xobjmap.helmholtz(
        "u", "v", target, corrlen_psi=2.5, corrlen_chi=2.5, err=0.1, backend="numpy"
    )
    helm_jax = obs.xobjmap.helmholtz(
        "u", "v", target, corrlen_psi=2.5, corrlen_chi=2.5, err=0.1, backend="jax"
    )
    np.testing.assert_allclose(helm_np.psi, helm_jax.psi, atol=2e-2, rtol=2e-1)
    np.testing.assert_allclose(helm_np.chi, helm_jax.chi, atol=2e-2, rtol=2e-1)
    np.testing.assert_allclose(helm_np.psi_error, helm_jax.psi_error, atol=5e-2, rtol=2.5e-1)
    np.testing.assert_allclose(helm_np.chi_error, helm_jax.chi_error, atol=5e-2, rtol=2.5e-1)
