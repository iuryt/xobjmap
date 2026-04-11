"""
Xarray accessor for xobjmap.

Provides the `ds.xobjmap` namespace for objective mapping operations
on xarray Datasets containing scattered observations.
"""

import numpy as np
import xarray as xr

from .interp import error as _error
from .interp import scalar as _scalar
from .interp import streamfunction as _streamfunction
from .interp import velocity_potential as _velocity_potential
from .interp import helmholtz as _helmholtz


def _parse_corrlen(corrlen, coord_names):
    """
    Parse correlation length into per-coordinate values.

    Parameters
    ----------
    corrlen : float or dict
        If float, isotropic. If dict, keys must match coord_names.
    coord_names : tuple of str
        Coordinate names, e.g. ("lon", "lat").

    Returns
    -------
    tuple of float
        Correlation lengths in the order of coord_names.
    """
    if isinstance(corrlen, dict):
        return tuple(corrlen[name] for name in coord_names)
    return tuple(corrlen for _ in coord_names)


def _find_obs_dim(ds):
    """
    Identify the observation dimension (the shared dim across data vars).
    """
    dims = set()
    for var in ds.data_vars:
        dims.update(ds[var].dims)
    if len(dims) == 1:
        return dims.pop()
    raise ValueError(
        f"Expected one observation dimension, found {dims}. "
        "Ensure all variables share a single dimension."
    )


def _find_coord_names(ds, target):
    """
    Find coordinate names shared between observations and target.

    Returns (x_coord, y_coord) preserving the order from the target
    Dataset's dimensions.
    """
    obs_coords = set(ds.coords)
    target_dims = list(target.dims)
    shared = [d for d in target_dims if d in obs_coords]
    if len(shared) < 2:
        raise ValueError(
            f"Need at least 2 shared coordinates between observations "
            f"and target. Found: {shared}"
        )
    # Return as (x_coord, y_coord): first dim in target is x, second is y
    return (shared[0], shared[1])


def _extract_grid(target, cx, cy):
    """Extract target grid arrays and build output metadata."""
    x_target = target[cx].values
    y_target = target[cy].values

    if target[cx].ndim == 1 and target[cy].ndim == 1:
        Xg, Yg = np.meshgrid(x_target, y_target)
        dims = (cy, cx)
        coords = {cx: x_target, cy: y_target}
    else:
        Xg = x_target
        Yg = y_target
        dims = target[cx].dims
        coords = {k: v for k, v in target.coords.items()}

    return Xg, Yg, dims, coords


@xr.register_dataset_accessor("xobjmap")
class XobjmapAccessor:
    """
    Xarray Dataset accessor for objective mapping.

    The source Dataset should contain scattered observations with
    coordinates along a single observation dimension.

    Examples
    --------
    >>> import xobjmap
    >>> obs = xr.Dataset(
    ...     {"temp": ("station", [20.1, 19.5, 21.0])},
    ...     coords={
    ...         "lon": ("station", [-40.0, -39.5, -38.0]),
    ...         "lat": ("station", [-23.0, -22.5, -23.5]),
    ...     },
    ... )
    >>> target = xr.Dataset(
    ...     coords={
    ...         "lon": np.linspace(-40, -38, 20),
    ...         "lat": np.linspace(-24, -22, 15),
    ...     }
    ... )
    >>> result = obs.xobjmap.scalar(
    ...     "temp", target, corrlen={"lon": 1.0, "lat": 1.0}, err=0.1,
    ... )
    """

    def __init__(self, ds):
        self._ds = ds

    def scalar(self, var, target, corrlen, err, backend="numpy"):
        """
        Scalar objective analysis of a variable onto target locations.

        Parameters
        ----------
        var : str
            Name of the variable in the dataset to interpolate.
        target : xr.Dataset
            Target locations with coordinates as dimensions. If the
            target has multiple dimensions (e.g., lon and lat), the
            output is mapped onto the full grid (meshgrid).
        corrlen : dict or float
            Correlation length scales. If a dict, keys should be
            coordinate names (e.g., ``{"lon": 1.0, "lat": 0.5}``).
            If a float, isotropic correlation is assumed. Must be in
            the same units as the coordinates.
        err : float
            Normalized random error variance (0 < err < 1).
        backend : {"numpy", "jax"}, optional
            Array backend to use. Use ``"jax"`` for lower memory
            usage and optional GPU acceleration (requires JAX to be
            installed). Default is ``"numpy"``.

        Returns
        -------
        xr.Dataset
            Dataset on the target coordinates with variables:
            - ``{var}``: interpolated field
            - ``error``: normalized mean squared error

        Notes
        -----
        The correlation lengths must be in the same units as the
        coordinates. If working with geographic coordinates (lon/lat
        in degrees), convert to a projected coordinate system first,
        or express corrlen in degrees.
        """
        ds = self._ds
        coord_names = _find_coord_names(ds, target)
        corrlenx, corrleny = _parse_corrlen(corrlen, coord_names)
        cx, cy = coord_names

        x_obs = ds[cx].values
        y_obs = ds[cy].values
        t_obs = ds[var].values

        Xg, Yg, dims, coords = _extract_grid(target, cx, cy)

        tp = _scalar(
            Xg.ravel(), Yg.ravel(), x_obs, y_obs, t_obs,
            corrlenx=corrlenx, corrleny=corrleny, err=err,
            backend=backend,
        )
        ep = _error(
            Xg.ravel(), Yg.ravel(), x_obs, y_obs,
            corrlenx=corrlenx, corrleny=corrleny, err=err,
            backend=backend,
        )

        tp = np.asarray(tp).reshape(Xg.shape)
        ep = np.asarray(ep).reshape(Xg.shape)

        return xr.Dataset(
            {var: (dims, tp), "error": (dims, ep)},
            coords=coords,
        )

    def streamfunction(self, u_var, v_var, target, corrlen, err, b=0, backend="numpy"):
        """
        Recover the streamfunction from scattered velocity observations.

        Parameters
        ----------
        u_var : str
            Name of the eastward velocity variable.
        v_var : str
            Name of the northward velocity variable.
        target : xr.Dataset
            Target grid with coordinates as dimensions.
        corrlen : dict or float
            Correlation length scales. Must be in the same units as
            the coordinates.
        err : float
            Normalized random error variance (0 < err < 1).
        b : float, optional
            Mean correction parameter. Default is 0.

        Returns
        -------
        xr.Dataset
            Dataset on the target grid with variable ``psi``
            (streamfunction).

        Notes
        -----
        The correlation lengths must be in the same units as the
        coordinates. If working with geographic coordinates (lon/lat
        in degrees), convert to a projected coordinate system first,
        or express corrlen in degrees.
        """
        ds = self._ds
        coord_names = _find_coord_names(ds, target)
        corrlenx, corrleny = _parse_corrlen(corrlen, coord_names)
        cx, cy = coord_names

        x_obs = ds[cx].values
        y_obs = ds[cy].values
        u_obs = ds[u_var].values
        v_obs = ds[v_var].values

        Xg, Yg, dims, coords = _extract_grid(target, cx, cy)

        psi = _streamfunction(
            Xg, Yg, x_obs, y_obs, u_obs, v_obs,
            corrlenx=corrlenx, corrleny=corrleny, err=err, b=b,
            backend=backend,
        )

        return xr.Dataset(
            {"psi": (dims, psi)},
            coords=coords,
        )

    def velocity_potential(self, u_var, v_var, target, corrlen, err, b=0, backend="numpy"):
        """
        Recover the velocity potential from scattered velocity observations.

        Parameters
        ----------
        u_var : str
            Name of the eastward velocity variable.
        v_var : str
            Name of the northward velocity variable.
        target : xr.Dataset
            Target grid with coordinates as dimensions.
        corrlen : dict or float
            Correlation length scales. Must be in the same units as
            the coordinates.
        err : float
            Normalized random error variance (0 < err < 1).
        b : float, optional
            Mean correction parameter. Default is 0.

        Returns
        -------
        xr.Dataset
            Dataset on the target grid with variable ``chi``
            (velocity potential).

        Notes
        -----
        The correlation lengths must be in the same units as the
        coordinates. If working with geographic coordinates (lon/lat
        in degrees), convert to a projected coordinate system first,
        or express corrlen in degrees.
        """
        ds = self._ds
        coord_names = _find_coord_names(ds, target)
        corrlenx, corrleny = _parse_corrlen(corrlen, coord_names)
        cx, cy = coord_names

        x_obs = ds[cx].values
        y_obs = ds[cy].values
        u_obs = ds[u_var].values
        v_obs = ds[v_var].values

        Xg, Yg, dims, coords = _extract_grid(target, cx, cy)

        chi = _velocity_potential(
            Xg, Yg, x_obs, y_obs, u_obs, v_obs,
            corrlenx=corrlenx, corrleny=corrleny, err=err, b=b,
            backend=backend,
        )

        return xr.Dataset(
            {"chi": (dims, chi)},
            coords=coords,
        )

    def helmholtz(self, u_var, v_var, target, corrlen_psi, corrlen_chi,
                  err, b=0, backend="numpy"):
        """
        Helmholtz decomposition: recover streamfunction and velocity
        potential from scattered velocity observations.

        Parameters
        ----------
        u_var : str
            Name of the eastward velocity variable.
        v_var : str
            Name of the northward velocity variable.
        target : xr.Dataset
            Target grid with coordinates as dimensions.
        corrlen_psi : dict or float
            Correlation length scales for the streamfunction.
        corrlen_chi : dict or float
            Correlation length scales for the velocity potential.
        err : float
            Normalized random error variance (0 < err < 1).
        b : float, optional
            Mean correction parameter. Default is 0.

        Returns
        -------
        xr.Dataset
            Dataset on the target grid with variables ``psi``
            (streamfunction) and ``chi`` (velocity potential).

        Notes
        -----
        The correlation lengths must be in the same units as the
        coordinates. If working with geographic coordinates (lon/lat
        in degrees), convert to a projected coordinate system first,
        or express corrlen in degrees.
        """
        ds = self._ds
        coord_names = _find_coord_names(ds, target)
        corrlenx_psi, corrleny_psi = _parse_corrlen(corrlen_psi, coord_names)
        corrlenx_chi, corrleny_chi = _parse_corrlen(corrlen_chi, coord_names)
        cx, cy = coord_names

        x_obs = ds[cx].values
        y_obs = ds[cy].values
        u_obs = ds[u_var].values
        v_obs = ds[v_var].values

        Xg, Yg, dims, coords = _extract_grid(target, cx, cy)

        psi, chi = _helmholtz(
            Xg, Yg, x_obs, y_obs, u_obs, v_obs,
            corrlenx_psi=corrlenx_psi, corrleny_psi=corrleny_psi,
            corrlenx_chi=corrlenx_chi, corrleny_chi=corrleny_chi,
            err=err, b=b, backend=backend,
        )

        return xr.Dataset(
            {"psi": (dims, psi), "chi": (dims, chi)},
            coords=coords,
        )
