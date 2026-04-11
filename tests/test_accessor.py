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


def test_accessors_support_paired_1d_target_coordinates(backend):
    """Accessors should preserve paired 1D target coordinates on one dimension."""
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

    target = xr.Dataset(
        coords={
            "lon": ("point", np.linspace(-4, 4, 7)),
            "lat": ("point", np.linspace(-3, 3, 7)),
        }
    )

    scalar = obs.xobjmap.scalar("temp", target, corrlen=3.0, err=0.1, backend=backend)
    helm = obs.xobjmap.helmholtz(
        "u", "v", target, corrlen_psi=3.0, corrlen_chi=3.0, err=0.1, backend=backend
    )

    assert scalar.temp.dims == ("point",)
    assert scalar.temp.shape == (7,)
    assert scalar.error.dims == ("point",)
    assert helm.psi.dims == ("point",)
    assert helm.chi.dims == ("point",)
    assert helm.psi_error.dims == ("point",)
    assert helm.chi_error.dims == ("point",)


def test_accessors_support_broadcast_3d_target_coordinates(backend):
    """Accessors should preserve broadcast target coordinates with 3 or more dimensions."""
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

    time = xr.DataArray(np.array([0.0, 1.0]), dims="time")
    lat = xr.DataArray(np.linspace(-3, 3, 5), dims="lat")
    lon = xr.DataArray(np.linspace(-4, 4, 6), dims="lon")
    _, lat3d, lon3d = xr.broadcast(time, lat, lon)
    target = xr.Dataset(coords={"lon": lon3d, "lat": lat3d, "time": time})

    scalar = obs.xobjmap.scalar("temp", target, corrlen=3.0, err=0.1, backend=backend)
    helm = obs.xobjmap.helmholtz(
        "u", "v", target, corrlen_psi=3.0, corrlen_chi=3.0, err=0.1, backend=backend
    )

    assert scalar.temp.dims == ("time", "lat", "lon")
    assert scalar.temp.shape == (2, 5, 6)
    assert scalar.error.dims == ("time", "lat", "lon")
    assert helm.psi.dims == ("time", "lat", "lon")
    assert helm.chi.dims == ("time", "lat", "lon")
    assert helm.psi_error.dims == ("time", "lat", "lon")
    assert helm.chi_error.dims == ("time", "lat", "lon")


def test_scalar_supports_datetime64_time_only_interpolation(backend):
    """Scalar accessor should support time-only interpolation with explicit units."""
    rng = np.random.default_rng(42)
    time_obs = np.array(
        [
            "2020-01-01T00",
            "2020-01-01T06",
            "2020-01-01T12",
            "2020-01-01T18",
        ],
        dtype="datetime64[h]",
    )
    obs = xr.Dataset(
        {"temp": ("time", rng.standard_normal(time_obs.size))},
        coords={"time": ("time", time_obs)},
    )
    target = xr.Dataset(
        coords={
            "time": (
                "time",
                np.array(
                    [
                        "2020-01-01T03",
                        "2020-01-01T09",
                        "2020-01-01T15",
                    ],
                    dtype="datetime64[h]",
                ),
            )
        }
    )

    result = obs.xobjmap.scalar(
        "temp",
        target,
        corrlen={"time": 6.0},
        err=0.1,
        interp_dims=("time",),
        coord_units={"time": "h"},
        backend=backend,
    )

    assert result.temp.dims == ("time",)
    assert result.temp.shape == (3,)
    assert result.error.dims == ("time",)


def test_scalar_datetime64_requires_coord_units():
    """Datetime-like interpolation dims should require explicit units."""
    obs = xr.Dataset(
        {"temp": ("time", [1.0, 2.0])},
        coords={"time": ("time", np.array(["2020-01-01", "2020-01-02"], dtype="datetime64[D]"))},
    )
    target = xr.Dataset(
        coords={"time": ("time", np.array(["2020-01-03"], dtype="datetime64[D]"))}
    )

    with pytest.raises(ValueError, match="coord_units"):
        obs.xobjmap.scalar(
            "temp",
            target,
            corrlen={"time": 1.0},
            err=0.1,
            interp_dims=("time",),
        )


def test_scalar_datetime64_matches_numeric_time_equivalent():
    """Datetime64 time interpolation should match the equivalent numeric-hour problem."""
    obs_numeric = xr.Dataset(
        {"temp": ("time", np.array([0.0, 1.0, 0.0]))},
        coords={"time": ("time", np.array([0.0, 6.0, 12.0]))},
    )
    target_numeric = xr.Dataset(coords={"time": ("time", np.array([3.0, 9.0]))})
    numeric = obs_numeric.xobjmap.scalar(
        "temp",
        target_numeric,
        corrlen={"time": 6.0},
        err=0.1,
        interp_dims=("time",),
        backend="numpy",
    )

    t0 = np.array("2020-01-01T00", dtype="datetime64[h]")
    obs_dt = xr.Dataset(
        {"temp": ("time", np.array([0.0, 1.0, 0.0]))},
        coords={"time": ("time", t0 + np.array([0, 6, 12]).astype("timedelta64[h]"))},
    )
    target_dt = xr.Dataset(
        coords={"time": ("time", t0 + np.array([3, 9]).astype("timedelta64[h]"))}
    )
    dated = obs_dt.xobjmap.scalar(
        "temp",
        target_dt,
        corrlen={"time": 6.0},
        err=0.1,
        interp_dims=("time",),
        coord_units={"time": "h"},
        backend="numpy",
    )

    np.testing.assert_allclose(dated.temp, numeric.temp)
    np.testing.assert_allclose(dated.error, numeric.error)


def test_scalar_exact_hit_is_near_observed_value_in_1d(backend):
    """A target exactly at an observed 1D location should recover that value closely."""
    obs = xr.Dataset(
        {"temp": ("z", np.array([1.0, 3.0, -2.0]))},
        coords={"z": ("z", np.array([-10.0, 0.0, 10.0]))},
    )
    target = xr.Dataset(coords={"z": ("z", np.array([0.0]))})

    result = obs.xobjmap.scalar(
        "temp",
        target,
        corrlen={"z": 5.0},
        err=1e-3,
        interp_dims=("z",),
        backend=backend,
    )

    np.testing.assert_allclose(result.temp.values, np.array([3.0]), atol=1e-2)


def test_scalar_3d_field_depending_only_on_z_is_constant_in_xy(backend):
    """A scalar field depending only on z should remain nearly constant across x/y at fixed z."""
    rng = np.random.default_rng(42)
    n = 80
    z_obs = rng.uniform(-2, 2, n)
    obs = xr.Dataset(
        {"temp": ("obs", 4.0 * z_obs)},
        coords={
            "x": ("obs", rng.uniform(-1, 1, n)),
            "y": ("obs", rng.uniform(-1, 1, n)),
            "z": ("obs", z_obs),
        },
    )

    x = xr.DataArray(np.linspace(-1, 1, 5), dims="x")
    y = xr.DataArray(np.linspace(-1, 1, 4), dims="y")
    z = xr.DataArray(np.array([-1.0, 1.0]), dims="z")
    z3d, y3d, x3d = xr.broadcast(z, y, x)
    target = xr.Dataset(coords={"x": x3d, "y": y3d, "z": z3d})

    result = obs.xobjmap.scalar(
        "temp",
        target,
        corrlen={"x": 3.0, "y": 3.0, "z": 1.0},
        err=0.05,
        interp_dims=("x", "y", "z"),
        backend=backend,
    )

    for iz, z_val in enumerate(z.values):
        plane = result.temp.isel(z=iz).values
        assert np.std(plane) < 0.2
        assert abs(np.mean(plane) - 4.0 * z_val) < 0.35


def test_scalar_anisotropic_corrlen_changes_smoothing_by_dimension():
    """Changing x vs y correlation lengths should affect smoothing along those axes differently."""
    rng = np.random.default_rng(42)
    n = 120
    x_obs = rng.uniform(-2, 2, n)
    y_obs = rng.uniform(-2, 2, n)
    temp = np.sin(4.0 * x_obs) + 0.2 * np.sin(0.5 * y_obs)
    obs = xr.Dataset(
        {"temp": ("obs", temp)},
        coords={"x": ("obs", x_obs), "y": ("obs", y_obs)},
    )
    x = np.linspace(-2, 2, 40)
    y = np.linspace(-2, 2, 30)
    target = xr.Dataset(coords={"x": x, "y": y})

    smooth_x = obs.xobjmap.scalar(
        "temp",
        target,
        corrlen={"x": 1.2, "y": 0.2},
        err=0.05,
        interp_dims=("x", "y"),
        backend="numpy",
    )
    smooth_y = obs.xobjmap.scalar(
        "temp",
        target,
        corrlen={"x": 0.2, "y": 1.2},
        err=0.05,
        interp_dims=("x", "y"),
        backend="numpy",
    )

    var_x_grad_smooth_x = np.var(np.diff(smooth_x.temp.values, axis=1))
    var_x_grad_smooth_y = np.var(np.diff(smooth_y.temp.values, axis=1))
    var_y_grad_smooth_x = np.var(np.diff(smooth_x.temp.values, axis=0))
    var_y_grad_smooth_y = np.var(np.diff(smooth_y.temp.values, axis=0))

    assert var_x_grad_smooth_x < var_x_grad_smooth_y
    assert var_y_grad_smooth_y < var_y_grad_smooth_x


def test_accessors_support_covariance_only_time_dim_for_vector_methods(backend):
    """Vector accessors should support time as a covariance-only dimension."""
    rng = np.random.default_rng(42)
    n = 20
    obs = xr.Dataset(
        {
            "u": ("obs", rng.standard_normal(n)),
            "v": ("obs", rng.standard_normal(n)),
        },
        coords={
            "x": ("obs", rng.uniform(-3, 3, n)),
            "y": ("obs", rng.uniform(-2, 2, n)),
            "time": (
                "obs",
                np.array("2020-01-01T00", dtype="datetime64[h]")
                + rng.integers(0, 24, n).astype("timedelta64[h]"),
            ),
        },
    )

    x = xr.DataArray(np.linspace(-2.5, 2.5, 6), dims="x")
    y = xr.DataArray(np.linspace(-1.5, 1.5, 5), dims="y")
    time = xr.DataArray(
        np.array(["2020-01-01T06", "2020-01-01T18"], dtype="datetime64[h]"),
        dims="time",
    )
    time3d, y3d, x3d = xr.broadcast(time, y, x)
    target = xr.Dataset(coords={"x": x3d, "y": y3d, "time": time3d})

    result = obs.xobjmap.streamfunction(
        "u",
        "v",
        target,
        corrlen={"x": 2.0, "y": 2.0, "time": 12.0},
        err=0.1,
        derivative_dims=("x", "y"),
        interp_dims=("x", "y", "time"),
        coord_units={"time": "h"},
        backend=backend,
    )

    assert result.psi.dims == ("time", "y", "x")
    assert result.psi.shape == (2, 5, 6)
    assert result.psi_error.dims == ("time", "y", "x")


def test_streamfunction_xytime_tracks_time_varying_amplitude(backend):
    """Recovered streamfunction slices should track the imposed time-varying amplitude ordering."""
    rng = np.random.default_rng(42)
    n = 140
    x_obs = rng.uniform(-2, 2, n)
    y_obs = rng.uniform(-2, 2, n)
    t_obs = rng.choice(np.array([0.0, 1.0]), size=n)
    amp = 1.0 + 0.6 * t_obs
    u_obs = amp * np.sin(x_obs) * np.sin(y_obs)
    v_obs = amp * np.cos(x_obs) * np.cos(y_obs)

    obs = xr.Dataset(
        {"u": ("obs", u_obs), "v": ("obs", v_obs)},
        coords={"x": ("obs", x_obs), "y": ("obs", y_obs), "time": ("obs", t_obs)},
    )
    x = xr.DataArray(np.linspace(-1.5, 1.5, 24), dims="x")
    y = xr.DataArray(np.linspace(-1.5, 1.5, 20), dims="y")
    time = xr.DataArray(np.array([0.0, 1.0]), dims="time")
    time3d, y3d, x3d = xr.broadcast(time, y, x)
    target = xr.Dataset(coords={"x": x3d, "y": y3d, "time": time3d})

    result = obs.xobjmap.streamfunction(
        "u",
        "v",
        target,
        corrlen={"x": 1.0, "y": 1.0, "time": 0.35},
        err=0.02,
        derivative_dims=("x", "y"),
        interp_dims=("x", "y", "time"),
        backend=backend,
    )

    amp_rms_t0 = np.sqrt(np.mean(result.psi.isel(time=0).values ** 2))
    amp_rms_t1 = np.sqrt(np.mean(result.psi.isel(time=1).values ** 2))
    assert amp_rms_t1 > amp_rms_t0


def test_accessors_support_covariance_only_z_dim_for_vector_methods(backend):
    """Vector accessors should support z as a covariance-only dimension."""
    rng = np.random.default_rng(42)
    n = 20
    obs = xr.Dataset(
        {
            "u": ("obs", rng.standard_normal(n)),
            "v": ("obs", rng.standard_normal(n)),
        },
        coords={
            "x": ("obs", rng.uniform(-3, 3, n)),
            "y": ("obs", rng.uniform(-2, 2, n)),
            "z": ("obs", rng.uniform(-100, 0, n)),
        },
    )

    x = xr.DataArray(np.linspace(-2.5, 2.5, 6), dims="x")
    y = xr.DataArray(np.linspace(-1.5, 1.5, 5), dims="y")
    z = xr.DataArray(np.array([-80.0, -20.0]), dims="z")
    z3d, y3d, x3d = xr.broadcast(z, y, x)
    target = xr.Dataset(coords={"x": x3d, "y": y3d, "z": z3d})

    result = obs.xobjmap.velocity_potential(
        "u",
        "v",
        target,
        corrlen={"x": 2.0, "y": 2.0, "z": 50.0},
        err=0.1,
        derivative_dims=("x", "y"),
        interp_dims=("x", "y", "z"),
        backend=backend,
    )

    assert result.chi.dims == ("z", "y", "x")
    assert result.chi.shape == (2, 5, 6)
    assert result.chi_error.dims == ("z", "y", "x")


def test_velocity_potential_xyz_tracks_z_varying_amplitude(backend):
    """Recovered velocity-potential slices should track the imposed z-varying amplitude ordering."""
    rng = np.random.default_rng(42)
    n = 140
    x_obs = rng.uniform(-2, 2, n)
    y_obs = rng.uniform(-2, 2, n)
    z_obs = rng.choice(np.array([-1.0, 1.0]), size=n)
    amp = 1.0 + 0.5 * z_obs
    u_obs = -amp * np.sin(x_obs) * np.sin(y_obs)
    v_obs = amp * np.cos(x_obs) * np.cos(y_obs)

    obs = xr.Dataset(
        {"u": ("obs", u_obs), "v": ("obs", v_obs)},
        coords={"x": ("obs", x_obs), "y": ("obs", y_obs), "z": ("obs", z_obs)},
    )
    x = xr.DataArray(np.linspace(-1.5, 1.5, 24), dims="x")
    y = xr.DataArray(np.linspace(-1.5, 1.5, 20), dims="y")
    z = xr.DataArray(np.array([-1.0, 1.0]), dims="z")
    z3d, y3d, x3d = xr.broadcast(z, y, x)
    target = xr.Dataset(coords={"x": x3d, "y": y3d, "z": z3d})

    result = obs.xobjmap.velocity_potential(
        "u",
        "v",
        target,
        corrlen={"x": 1.0, "y": 1.0, "z": 0.35},
        err=0.02,
        derivative_dims=("x", "y"),
        interp_dims=("x", "y", "z"),
        backend=backend,
    )

    amp_rms_z0 = np.sqrt(np.mean(result.chi.isel(z=0).values ** 2))
    amp_rms_z1 = np.sqrt(np.mean(result.chi.isel(z=1).values ** 2))
    assert amp_rms_z1 > amp_rms_z0


def test_small_extra_dim_corrlen_preserves_slice_separation_in_streamfunction():
    """Small extra-dimension correlation length should reduce cross-slice contamination."""
    rng = np.random.default_rng(42)
    n = 160
    x_obs = rng.uniform(-2, 2, n)
    y_obs = rng.uniform(-2, 2, n)
    z_obs = rng.choice(np.array([-1.0, 1.0]), size=n)
    amp = z_obs
    u_obs = amp * np.sin(x_obs) * np.sin(y_obs)
    v_obs = amp * np.cos(x_obs) * np.cos(y_obs)
    obs = xr.Dataset(
        {"u": ("obs", u_obs), "v": ("obs", v_obs)},
        coords={"x": ("obs", x_obs), "y": ("obs", y_obs), "z": ("obs", z_obs)},
    )
    x = xr.DataArray(np.linspace(-1.0, 1.0, 16), dims="x")
    y = xr.DataArray(np.linspace(-1.0, 1.0, 14), dims="y")
    z = xr.DataArray(np.array([-1.0, 1.0]), dims="z")
    z3d, y3d, x3d = xr.broadcast(z, y, x)
    target = xr.Dataset(coords={"x": x3d, "y": y3d, "z": z3d})

    small = obs.xobjmap.streamfunction(
        "u", "v", target,
        corrlen={"x": 1.0, "y": 1.0, "z": 0.05},
        err=0.02,
        derivative_dims=("x", "y"),
        interp_dims=("x", "y", "z"),
        backend="numpy",
    )
    large = obs.xobjmap.streamfunction(
        "u", "v", target,
        corrlen={"x": 1.0, "y": 1.0, "z": 10.0},
        err=0.02,
        derivative_dims=("x", "y"),
        interp_dims=("x", "y", "z"),
        backend="numpy",
    )

    small_gap = abs(
        float(small.psi.isel(z=0).mean()) - float(small.psi.isel(z=1).mean())
    )
    large_gap = abs(
        float(large.psi.isel(z=0).mean()) - float(large.psi.isel(z=1).mean())
    )
    assert small_gap > large_gap


def test_small_time_corrlen_preserves_slice_separation_in_helmholtz():
    """Small time correlation length should reduce cross-time contamination in Helmholtz recovery."""
    rng = np.random.default_rng(42)
    n = 180
    x_obs = rng.uniform(-2, 2, n)
    y_obs = rng.uniform(-2, 2, n)
    t_obs = rng.choice(np.array([0.0, 1.0]), size=n)
    amp = 1.0 + t_obs
    psi_obs = amp * np.sin(x_obs) * np.cos(y_obs)
    chi_obs = amp * np.cos(x_obs) * np.sin(y_obs)
    u_obs = amp * np.sin(x_obs) * np.sin(y_obs) - amp * np.sin(x_obs) * np.sin(y_obs)
    v_obs = amp * np.cos(x_obs) * np.cos(y_obs) + amp * np.cos(x_obs) * np.cos(y_obs)
    obs = xr.Dataset(
        {"u": ("obs", u_obs), "v": ("obs", v_obs)},
        coords={"x": ("obs", x_obs), "y": ("obs", y_obs), "time": ("obs", t_obs)},
    )
    x = xr.DataArray(np.linspace(-1.0, 1.0, 16), dims="x")
    y = xr.DataArray(np.linspace(-1.0, 1.0, 14), dims="y")
    time = xr.DataArray(np.array([0.0, 1.0]), dims="time")
    time3d, y3d, x3d = xr.broadcast(time, y, x)
    target = xr.Dataset(coords={"x": x3d, "y": y3d, "time": time3d})

    small = obs.xobjmap.helmholtz(
        "u", "v", target,
        corrlen_psi={"x": 1.0, "y": 1.0, "time": 0.05},
        corrlen_chi={"x": 1.0, "y": 1.0, "time": 0.05},
        err=0.02,
        derivative_dims=("x", "y"),
        interp_dims=("x", "y", "time"),
        backend="numpy",
    )
    large = obs.xobjmap.helmholtz(
        "u", "v", target,
        corrlen_psi={"x": 1.0, "y": 1.0, "time": 10.0},
        corrlen_chi={"x": 1.0, "y": 1.0, "time": 10.0},
        err=0.02,
        derivative_dims=("x", "y"),
        interp_dims=("x", "y", "time"),
        backend="numpy",
    )

    small_gap = abs(
        float(small.chi.isel(time=0).mean()) - float(small.chi.isel(time=1).mean())
    )
    large_gap = abs(
        float(large.chi.isel(time=0).mean()) - float(large.chi.isel(time=1).mean())
    )
    assert small_gap > large_gap


def test_vector_methods_validate_derivative_dims():
    """Vector accessors should reject invalid derivative dimensions."""
    rng = np.random.default_rng(42)
    obs = xr.Dataset(
        {
            "u": ("obs", rng.standard_normal(10)),
            "v": ("obs", rng.standard_normal(10)),
        },
        coords={
            "x": ("obs", rng.uniform(-1, 1, 10)),
            "y": ("obs", rng.uniform(-1, 1, 10)),
            "time": ("obs", np.array("2020-01-01", dtype="datetime64[h]") + np.arange(10).astype("timedelta64[h]")),
        },
    )
    target = xr.Dataset(
        coords={
            "x": np.linspace(-1, 1, 4),
            "y": np.linspace(-1, 1, 3),
            "time": ("time", np.array(["2020-01-01T12"], dtype="datetime64[h]")),
        }
    )

    with pytest.raises(ValueError, match="exactly two derivative dimensions"):
        obs.xobjmap.streamfunction(
            "u", "v", target, corrlen=1.0, err=0.1, derivative_dims=("x",)
        )

    with pytest.raises(ValueError, match="must be included in interp_dims"):
        obs.xobjmap.streamfunction(
            "u",
            "v",
            target,
            corrlen={"x": 1.0, "y": 1.0},
            err=0.1,
            derivative_dims=("x", "z"),
            interp_dims=("x", "y"),
        )

    with pytest.raises(ValueError, match="cannot be datetime-like"):
        obs.xobjmap.streamfunction(
            "u",
            "v",
            target,
            corrlen={"x": 1.0, "time": 1.0},
            err=0.1,
            derivative_dims=("x", "time"),
            interp_dims=("x", "time"),
            coord_units={"time": "h"},
        )


def test_nd_accessor_matches_legacy_2d_public_functions():
    """The new N-D accessor path should reproduce the legacy 2D kernels."""
    rng = np.random.default_rng(42)
    n = 18
    obs = xr.Dataset(
        {
            "temp": ("obs", rng.standard_normal(n)),
            "u": ("obs", rng.standard_normal(n)),
            "v": ("obs", rng.standard_normal(n)),
        },
        coords={
            "x": ("obs", rng.uniform(-2, 2, n)),
            "y": ("obs", rng.uniform(-2, 2, n)),
        },
    )
    x = np.linspace(-1.5, 1.5, 5)
    y = np.linspace(-1.0, 1.0, 4)
    gx, gy = np.meshgrid(x, y)
    target = xr.Dataset(coords={"x": x, "y": y})

    scalar_ds = obs[["temp"]].xobjmap.scalar(
        "temp",
        target,
        corrlen={"x": 1.5, "y": 1.5},
        err=0.1,
        interp_dims=("x", "y"),
        backend="numpy",
    )
    scalar_legacy = xobjmap.scalar(
        gx.ravel(),
        gy.ravel(),
        obs.x.values,
        obs.y.values,
        obs.temp.values,
        corrlenx=1.5,
        corrleny=1.5,
        err=0.1,
        backend="numpy",
    ).reshape(gx.shape)
    scalar_err_legacy = xobjmap.scalar_error(
        gx.ravel(),
        gy.ravel(),
        obs.x.values,
        obs.y.values,
        corrlenx=1.5,
        corrleny=1.5,
        err=0.1,
        backend="numpy",
    ).reshape(gx.shape)
    np.testing.assert_allclose(scalar_ds.temp, scalar_legacy)
    np.testing.assert_allclose(scalar_ds.error, scalar_err_legacy)

    helm_ds = obs.xobjmap.helmholtz(
        "u",
        "v",
        target,
        corrlen_psi={"x": 1.5, "y": 1.5},
        corrlen_chi={"x": 1.5, "y": 1.5},
        err=0.1,
        derivative_dims=("x", "y"),
        interp_dims=("x", "y"),
        backend="numpy",
    )
    psi_legacy, chi_legacy = xobjmap.helmholtz(
        gx,
        gy,
        obs.x.values,
        obs.y.values,
        obs.u.values,
        obs.v.values,
        corrlenx_psi=1.5,
        corrleny_psi=1.5,
        corrlenx_chi=1.5,
        corrleny_chi=1.5,
        err=0.1,
        backend="numpy",
    )
    psi_err_legacy, chi_err_legacy = xobjmap.helmholtz_error(
        gx,
        gy,
        obs.x.values,
        obs.y.values,
        obs.u.values,
        obs.v.values,
        corrlenx_psi=1.5,
        corrleny_psi=1.5,
        corrlenx_chi=1.5,
        corrleny_chi=1.5,
        err=0.1,
        backend="numpy",
    )
    np.testing.assert_allclose(helm_ds.psi, psi_legacy)
    np.testing.assert_allclose(helm_ds.chi, chi_legacy)
    np.testing.assert_allclose(helm_ds.psi_error, psi_err_legacy)
    np.testing.assert_allclose(helm_ds.chi_error, chi_err_legacy)


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
