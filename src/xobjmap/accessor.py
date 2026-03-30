"""
Xarray accessor for xobjmap.

Provides the `ds.xobjmap` namespace for objective mapping operations
on xarray Datasets containing scattered observations.
"""

import numpy as np
import xarray as xr

from .interp import scaloa, vectoa


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
    >>> result = obs.xobjmap.scalar_interp(
    ...     "temp", target, corrlen={"lon": 1.0, "lat": 1.0}, err=0.1,
    ... )
    """

    def __init__(self, ds):
        self._ds = ds

    def scalar_interp(self, var, target, corrlen, err):
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

        # Observation coordinates and values
        x_obs = ds[cx].values
        y_obs = ds[cy].values
        t_obs = ds[var].values

        # Target coordinates
        x_target = target[cx].values
        y_target = target[cy].values

        # Build target meshgrid if target has separate 1D coordinates
        if target[cx].ndim == 1 and target[cy].ndim == 1:
            xg, yg = np.meshgrid(x_target, y_target)
            xc_flat = xg.ravel()
            yc_flat = yg.ravel()
        else:
            xc_flat = x_target.ravel()
            yc_flat = y_target.ravel()

        tp, ep = scaloa(
            xc_flat, yc_flat, x_obs, y_obs, t_obs,
            corrlenx=corrlenx, corrleny=corrleny, err=err,
        )

        # Reshape onto target grid
        if target[cx].ndim == 1 and target[cy].ndim == 1:
            shape = (len(y_target), len(x_target))
            tp = tp.reshape(shape)
            ep = ep.reshape(shape)
            dims = (cy, cx)
            coords = {cx: x_target, cy: y_target}
        else:
            tp = tp.reshape(x_target.shape)
            ep = ep.reshape(x_target.shape)
            dims = target[cx].dims
            coords = {k: v for k, v in target.coords.items()}

        return xr.Dataset(
            {var: (dims, tp), "error": (dims, ep)},
            coords=coords,
        )

    def vector_interp(self, u_var, v_var, target, corrlen, err, b=0):
        """
        Vectorial objective analysis: recover streamfunction from
        scattered velocity observations.

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

        # Observation coordinates and velocities
        x_obs = ds[cx].values
        y_obs = ds[cy].values
        u_obs = ds[u_var].values
        v_obs = ds[v_var].values

        # Target grid
        x_target = target[cx].values
        y_target = target[cy].values

        if target[cx].ndim == 1 and target[cy].ndim == 1:
            Xg, Yg = np.meshgrid(x_target, y_target)
        else:
            Xg = x_target
            Yg = y_target

        psi = vectoa(
            Xg, Yg, x_obs, y_obs, u_obs, v_obs,
            corrlenx=corrlenx, corrleny=corrleny, err=err, b=b,
        )

        if target[cx].ndim == 1 and target[cy].ndim == 1:
            dims = (cy, cx)
            coords = {cx: x_target, cy: y_target}
        else:
            dims = target[cx].dims
            coords = {k: v for k, v in target.coords.items()}

        return xr.Dataset(
            {"psi": (dims, psi)},
            coords=coords,
        )
