"""
Xarray accessor for xobjmap.

Provides the `ds.xobjmap` namespace for objective mapping operations
on xarray Datasets containing scattered observations.
"""

import numpy as np
import xarray as xr

from .interp import (
    _helmholtz_error_nd_jax,
    _helmholtz_error_nd_numpy,
    _helmholtz_nd_jax,
    _helmholtz_nd_numpy,
    _scalar_error_nd_jax,
    _scalar_error_nd_numpy,
    _scalar_nd_jax,
    _scalar_nd_numpy,
    _single_component_vector_error_nd_jax,
    _single_component_vector_error_nd_numpy,
    _streamfunction_nd_jax,
    _streamfunction_nd_numpy,
    _velocity_potential_nd_jax,
    _velocity_potential_nd_numpy,
)


_DATETIME_UNIT_TO_NS = {
    "ns": 1,
    "us": 1_000,
    "ms": 1_000_000,
    "s": 1_000_000_000,
    "m": 60 * 1_000_000_000,
    "h": 60 * 60 * 1_000_000_000,
    "D": 24 * 60 * 60 * 1_000_000_000,
}


def _find_obs_dim(ds):
    """Identify the shared observation dimension across data variables."""
    dims = set()
    for var in ds.data_vars:
        dims.update(ds[var].dims)
    if len(dims) == 1:
        return dims.pop()
    raise ValueError(
        f"Expected one observation dimension, found {dims}. "
        "Ensure all variables share a single dimension."
    )


def _infer_interp_dims(ds, target, interp_dims=None):
    """Infer interpolation dimensions from shared coordinate names."""
    if interp_dims is not None:
        return tuple(interp_dims)

    obs_coords = set(ds.coords)
    shared = [name for name in target.coords if name in obs_coords]
    if not shared:
        raise ValueError(
            "Need at least one shared coordinate between observations and target."
        )
    return tuple(shared)


def _parse_corrlen(corrlen, interp_dims):
    """Return per-dimension correlation lengths in interp-dim order."""
    if isinstance(corrlen, dict):
        missing = [dim for dim in interp_dims if dim not in corrlen]
        if missing:
            raise ValueError(
                f"Missing correlation lengths for dimensions {missing}."
            )
        return np.asarray([corrlen[dim] for dim in interp_dims], dtype=float)
    return np.asarray([corrlen for _ in interp_dims], dtype=float)


def _validate_derivative_dims(ds, interp_dims, derivative_dims):
    """Validate and return the two derivative dimensions for vector methods."""
    if derivative_dims is None:
        if len(interp_dims) < 2:
            raise ValueError(
                "Vector-potential methods require exactly two derivative dimensions."
            )
        derivative_dims = tuple(interp_dims[:2])
    else:
        derivative_dims = tuple(derivative_dims)

    if len(derivative_dims) != 2:
        raise ValueError(
            f"Expected exactly two derivative dimensions, got {derivative_dims}."
        )

    missing = [dim for dim in derivative_dims if dim not in interp_dims]
    if missing:
        raise ValueError(
            f"Derivative dimensions {missing} must be included in interp_dims."
        )

    for dim in derivative_dims:
        dtype = ds[dim].dtype
        if np.issubdtype(dtype, np.datetime64):
            raise ValueError(
                f"Derivative dimension {dim!r} cannot be datetime-like."
            )

    return derivative_dims


def _datetime_unit_factor(unit, dim):
    """Return conversion factor from nanoseconds to the requested unit."""
    try:
        return _DATETIME_UNIT_TO_NS[unit]
    except KeyError as exc:
        raise ValueError(
            f"Unsupported datetime unit {unit!r} for dimension {dim!r}. "
            f"Choose from {sorted(_DATETIME_UNIT_TO_NS)}."
        ) from exc


def _convert_coord_values(obs_values, target_values, dim, coord_units):
    """Convert coordinate arrays to numeric values in consistent units."""
    obs_arr = np.asarray(obs_values)
    target_arr = np.asarray(target_values)

    if np.issubdtype(obs_arr.dtype, np.datetime64) or np.issubdtype(target_arr.dtype, np.datetime64):
        if coord_units is None or dim not in coord_units:
            raise ValueError(
                f"Dimension {dim!r} is datetime-like; provide coord_units[{dim!r}]."
            )
        factor = _datetime_unit_factor(coord_units[dim], dim)
        obs_ns = obs_arr.astype("datetime64[ns]").astype(np.int64)
        target_ns = target_arr.astype("datetime64[ns]").astype(np.int64)
        origin = min(obs_ns.min(), target_ns.min())
        return (obs_ns - origin) / factor, (target_ns - origin) / factor

    if obs_arr.dtype == object or target_arr.dtype == object:
        raise ValueError(
            f"Unsupported coordinate dtype for dimension {dim!r}. "
            "Use numeric or numpy datetime64 coordinates."
        )

    return obs_arr.astype(float), target_arr.astype(float)


def _prepare_target(target, interp_dims):
    """Broadcast target coordinates and preserve output metadata."""
    arrays = [target[dim] for dim in interp_dims]

    if (
        len(arrays) == 2
        and arrays[0].ndim == 1
        and arrays[1].ndim == 1
        and arrays[0].dims != arrays[1].dims
    ):
        mesh = np.meshgrid(*(arr.values for arr in arrays), indexing="xy")
        target_values = [mesh[0], mesh[1]]
        output_dims = (interp_dims[1], interp_dims[0])
        output_coords = {
            interp_dims[0]: arrays[0].values,
            interp_dims[1]: arrays[1].values,
        }
        target_shape = mesh[0].shape
    else:
        broadcast = xr.broadcast(*arrays)
        target_values = [arr.values for arr in broadcast]
        output_dims = broadcast[0].dims
        output_coords = {k: v for k, v in target.coords.items()}
        target_shape = broadcast[0].shape

    return target_values, target_shape, output_dims, output_coords


def _prepare_geometry(ds, target, interp_dims=None, coord_units=None):
    """Normalize observation and target coordinates into numeric point arrays."""
    interp_dims = _infer_interp_dims(ds, target, interp_dims=interp_dims)
    obs_dim = _find_obs_dim(ds)
    obs_values = [ds[dim].values for dim in interp_dims]
    target_values, target_shape, output_dims, output_coords = _prepare_target(target, interp_dims)

    obs_numeric = []
    target_numeric = []
    for dim, obs, tgt in zip(interp_dims, obs_values, target_values):
        obs_num, tgt_num = _convert_coord_values(obs, tgt, dim, coord_units)
        obs_numeric.append(np.asarray(obs_num).reshape(-1))
        target_numeric.append(np.asarray(tgt_num).reshape(-1))

    obs_points = np.column_stack(obs_numeric)
    target_points = np.column_stack(target_numeric)

    return {
        "interp_dims": interp_dims,
        "obs_dim": obs_dim,
        "obs_points": obs_points,
        "target_points": target_points,
        "target_shape": target_shape,
        "output_dims": output_dims,
        "output_coords": output_coords,
    }


def _scalar_impl(backend):
    """Select scalar interpolation backend."""
    if backend == "numpy":
        return _scalar_nd_numpy, _scalar_error_nd_numpy
    if backend == "jax":
        return _scalar_nd_jax, _scalar_error_nd_jax
    raise ValueError(f"Unknown backend {backend!r}. Choose 'numpy' or 'jax'.")


def _vector_impl(backend):
    """Select vector interpolation backend."""
    if backend == "numpy":
        return {
            "streamfunction": _streamfunction_nd_numpy,
            "streamfunction_error": lambda *args, **kwargs: _single_component_vector_error_nd_numpy(
                *args, **kwargs, nondivergent=True
            ),
            "velocity_potential": _velocity_potential_nd_numpy,
            "velocity_potential_error": lambda *args, **kwargs: _single_component_vector_error_nd_numpy(
                *args, **kwargs, nondivergent=False
            ),
            "helmholtz": _helmholtz_nd_numpy,
            "helmholtz_error": _helmholtz_error_nd_numpy,
        }
    if backend == "jax":
        return {
            "streamfunction": _streamfunction_nd_jax,
            "streamfunction_error": lambda *args, **kwargs: _single_component_vector_error_nd_jax(
                *args, **kwargs, nondivergent=True
            ),
            "velocity_potential": _velocity_potential_nd_jax,
            "velocity_potential_error": lambda *args, **kwargs: _single_component_vector_error_nd_jax(
                *args, **kwargs, nondivergent=False
            ),
            "helmholtz": _helmholtz_nd_jax,
            "helmholtz_error": _helmholtz_error_nd_jax,
        }
    raise ValueError(f"Unknown backend {backend!r}. Choose 'numpy' or 'jax'.")


@xr.register_dataset_accessor("xobjmap")
class XobjmapAccessor:
    """Xarray Dataset accessor for objective mapping."""

    def __init__(self, ds):
        self._ds = ds

    def scalar(
        self,
        var,
        target,
        corrlen,
        err,
        interp_dims=None,
        coord_units=None,
        backend="numpy",
        return_error=True,
        k_local=None,
    ):
        """Scalar objective analysis of a variable onto target locations."""
        ds = self._ds
        geom = _prepare_geometry(ds, target, interp_dims=interp_dims, coord_units=coord_units)
        corrlen_values = _parse_corrlen(corrlen, geom["interp_dims"])
        scalar_fn, scalar_error_fn = _scalar_impl(backend)

        values = ds[var].values
        tp = scalar_fn(
            geom["target_points"], geom["obs_points"], values, corrlen_values, err
        )
        data_vars = {var: (geom["output_dims"], np.asarray(tp).reshape(geom["target_shape"]))}

        if return_error:
            ep = scalar_error_fn(
                geom["target_points"], geom["obs_points"], corrlen_values, err,
                k_local=k_local,
            )
            data_vars["error"] = (
                geom["output_dims"],
                np.asarray(ep).reshape(geom["target_shape"]),
            )

        return xr.Dataset(data_vars, coords=geom["output_coords"])

    def scalar_error(
        self, target, corrlen, err, interp_dims=None, coord_units=None,
        backend="numpy", k_local=None,
    ):
        """Return only scalar interpolation error for the target grid."""
        ds = self._ds
        geom = _prepare_geometry(ds, target, interp_dims=interp_dims, coord_units=coord_units)
        corrlen_values = _parse_corrlen(corrlen, geom["interp_dims"])
        _, scalar_error_fn = _scalar_impl(backend)
        ep = scalar_error_fn(
            geom["target_points"], geom["obs_points"], corrlen_values, err,
            k_local=k_local,
        )
        return xr.Dataset(
            {"error": (geom["output_dims"], np.asarray(ep).reshape(geom["target_shape"]))},
            coords=geom["output_coords"],
        )

    def streamfunction(
        self,
        u_var,
        v_var,
        target,
        corrlen,
        err,
        derivative_dims=None,
        interp_dims=None,
        coord_units=None,
        b=0,
        backend="numpy",
        return_error=True,
        k_local=None,
    ):
        """Recover the streamfunction from scattered velocity observations."""
        ds = self._ds
        interp_dims = _infer_interp_dims(ds, target, interp_dims=interp_dims)
        derivative_dims = _validate_derivative_dims(ds, interp_dims, derivative_dims)
        geom = _prepare_geometry(ds, target, interp_dims=interp_dims, coord_units=coord_units)
        derivative_indices = tuple(geom["interp_dims"].index(dim) for dim in derivative_dims)
        corrlen_values = _parse_corrlen(corrlen, geom["interp_dims"])
        impl = _vector_impl(backend)

        psi = impl["streamfunction"](
            geom["target_points"],
            geom["obs_points"],
            ds[u_var].values,
            ds[v_var].values,
            corrlen_values,
            derivative_indices,
            err,
            b=b,
        )
        data_vars = {"psi": (geom["output_dims"], np.asarray(psi).reshape(geom["target_shape"]))}

        if return_error:
            psi_error = impl["streamfunction_error"](
                geom["target_points"],
                geom["obs_points"],
                corrlen_values,
                derivative_indices,
                err,
                b=b,
                k_local=k_local,
            )
            data_vars["psi_error"] = (
                geom["output_dims"],
                np.asarray(psi_error).reshape(geom["target_shape"]),
            )

        return xr.Dataset(data_vars, coords=geom["output_coords"])

    def streamfunction_error(
        self,
        u_var,
        v_var,
        target,
        corrlen,
        err,
        derivative_dims=None,
        interp_dims=None,
        coord_units=None,
        b=0,
        backend="numpy",
        k_local=None,
    ):
        """Return only streamfunction posterior error for the target grid."""
        ds = self._ds
        interp_dims = _infer_interp_dims(ds, target, interp_dims=interp_dims)
        derivative_dims = _validate_derivative_dims(ds, interp_dims, derivative_dims)
        geom = _prepare_geometry(ds, target, interp_dims=interp_dims, coord_units=coord_units)
        derivative_indices = tuple(geom["interp_dims"].index(dim) for dim in derivative_dims)
        corrlen_values = _parse_corrlen(corrlen, geom["interp_dims"])
        impl = _vector_impl(backend)
        psi_error = impl["streamfunction_error"](
            geom["target_points"],
            geom["obs_points"],
            corrlen_values,
            derivative_indices,
            err,
            b=b,
            k_local=k_local,
        )
        return xr.Dataset(
            {"psi_error": (geom["output_dims"], np.asarray(psi_error).reshape(geom["target_shape"]))},
            coords=geom["output_coords"],
        )

    def velocity_potential(
        self,
        u_var,
        v_var,
        target,
        corrlen,
        err,
        derivative_dims=None,
        interp_dims=None,
        coord_units=None,
        b=0,
        backend="numpy",
        return_error=True,
        k_local=None,
    ):
        """Recover the velocity potential from scattered velocity observations."""
        ds = self._ds
        interp_dims = _infer_interp_dims(ds, target, interp_dims=interp_dims)
        derivative_dims = _validate_derivative_dims(ds, interp_dims, derivative_dims)
        geom = _prepare_geometry(ds, target, interp_dims=interp_dims, coord_units=coord_units)
        derivative_indices = tuple(geom["interp_dims"].index(dim) for dim in derivative_dims)
        corrlen_values = _parse_corrlen(corrlen, geom["interp_dims"])
        impl = _vector_impl(backend)

        chi = impl["velocity_potential"](
            geom["target_points"],
            geom["obs_points"],
            ds[u_var].values,
            ds[v_var].values,
            corrlen_values,
            derivative_indices,
            err,
            b=b,
        )
        data_vars = {"chi": (geom["output_dims"], np.asarray(chi).reshape(geom["target_shape"]))}

        if return_error:
            chi_error = impl["velocity_potential_error"](
                geom["target_points"],
                geom["obs_points"],
                corrlen_values,
                derivative_indices,
                err,
                b=b,
                k_local=k_local,
            )
            data_vars["chi_error"] = (
                geom["output_dims"],
                np.asarray(chi_error).reshape(geom["target_shape"]),
            )

        return xr.Dataset(data_vars, coords=geom["output_coords"])

    def velocity_potential_error(
        self,
        u_var,
        v_var,
        target,
        corrlen,
        err,
        derivative_dims=None,
        interp_dims=None,
        coord_units=None,
        b=0,
        backend="numpy",
        k_local=None,
    ):
        """Return only velocity-potential posterior error for the target grid."""
        ds = self._ds
        interp_dims = _infer_interp_dims(ds, target, interp_dims=interp_dims)
        derivative_dims = _validate_derivative_dims(ds, interp_dims, derivative_dims)
        geom = _prepare_geometry(ds, target, interp_dims=interp_dims, coord_units=coord_units)
        derivative_indices = tuple(geom["interp_dims"].index(dim) for dim in derivative_dims)
        corrlen_values = _parse_corrlen(corrlen, geom["interp_dims"])
        impl = _vector_impl(backend)
        chi_error = impl["velocity_potential_error"](
            geom["target_points"],
            geom["obs_points"],
            corrlen_values,
            derivative_indices,
            err,
            b=b,
            k_local=k_local,
        )
        return xr.Dataset(
            {"chi_error": (geom["output_dims"], np.asarray(chi_error).reshape(geom["target_shape"]))},
            coords=geom["output_coords"],
        )

    def helmholtz(
        self,
        u_var,
        v_var,
        target,
        corrlen_psi,
        corrlen_chi,
        err,
        derivative_dims=None,
        interp_dims=None,
        coord_units=None,
        b=0,
        backend="numpy",
        return_error=True,
        k_local=None,
    ):
        """Recover streamfunction and velocity potential from scattered observations."""
        ds = self._ds
        interp_dims = _infer_interp_dims(ds, target, interp_dims=interp_dims)
        derivative_dims = _validate_derivative_dims(ds, interp_dims, derivative_dims)
        geom = _prepare_geometry(ds, target, interp_dims=interp_dims, coord_units=coord_units)
        derivative_indices = tuple(geom["interp_dims"].index(dim) for dim in derivative_dims)
        corrlen_psi_values = _parse_corrlen(corrlen_psi, geom["interp_dims"])
        corrlen_chi_values = _parse_corrlen(corrlen_chi, geom["interp_dims"])
        impl = _vector_impl(backend)

        psi, chi = impl["helmholtz"](
            geom["target_points"],
            geom["obs_points"],
            ds[u_var].values,
            ds[v_var].values,
            corrlen_psi_values,
            corrlen_chi_values,
            derivative_indices,
            err,
            b=b,
        )
        data_vars = {
            "psi": (geom["output_dims"], np.asarray(psi).reshape(geom["target_shape"])),
            "chi": (geom["output_dims"], np.asarray(chi).reshape(geom["target_shape"])),
        }

        if return_error:
            psi_error, chi_error = impl["helmholtz_error"](
                geom["target_points"],
                geom["obs_points"],
                corrlen_psi_values,
                corrlen_chi_values,
                derivative_indices,
                err,
                b=b,
                k_local=k_local,
            )
            data_vars["psi_error"] = (
                geom["output_dims"],
                np.asarray(psi_error).reshape(geom["target_shape"]),
            )
            data_vars["chi_error"] = (
                geom["output_dims"],
                np.asarray(chi_error).reshape(geom["target_shape"]),
            )

        return xr.Dataset(data_vars, coords=geom["output_coords"])

    def helmholtz_error(
        self,
        u_var,
        v_var,
        target,
        corrlen_psi,
        corrlen_chi,
        err,
        derivative_dims=None,
        interp_dims=None,
        coord_units=None,
        b=0,
        backend="numpy",
        k_local=None,
    ):
        """Return only Helmholtz posterior errors for the target grid."""
        ds = self._ds
        interp_dims = _infer_interp_dims(ds, target, interp_dims=interp_dims)
        derivative_dims = _validate_derivative_dims(ds, interp_dims, derivative_dims)
        geom = _prepare_geometry(ds, target, interp_dims=interp_dims, coord_units=coord_units)
        derivative_indices = tuple(geom["interp_dims"].index(dim) for dim in derivative_dims)
        corrlen_psi_values = _parse_corrlen(corrlen_psi, geom["interp_dims"])
        corrlen_chi_values = _parse_corrlen(corrlen_chi, geom["interp_dims"])
        impl = _vector_impl(backend)
        psi_error, chi_error = impl["helmholtz_error"](
            geom["target_points"],
            geom["obs_points"],
            corrlen_psi_values,
            corrlen_chi_values,
            derivative_indices,
            err,
            b=b,
            k_local=k_local,
        )
        return xr.Dataset(
            {
                "psi_error": (
                    geom["output_dims"],
                    np.asarray(psi_error).reshape(geom["target_shape"]),
                ),
                "chi_error": (
                    geom["output_dims"],
                    np.asarray(chi_error).reshape(geom["target_shape"]),
                ),
            },
            coords=geom["output_coords"],
        )
