# xobjmap

Xarray-native objective mapping and interpolation of scattered observations.

![Rankine vortex reconstruction](example_vortex.png)

## Installation

With pixi (recommended):
```bash
pixi add xobjmap
```

With conda:
```bash
conda install -c conda-forge xobjmap
```

## Quick start

```python
import numpy as np
import xarray as xr
import xobjmap

# Scattered observations
obs = xr.Dataset(
    {"temp": ("station", temp_data)},
    coords={"lon": ("station", lons), "lat": ("station", lats)},
)

# Target grid
target = xr.Dataset(
    coords={"lon": np.linspace(-40, -38, 50), "lat": np.linspace(-24, -22, 40)}
)

# Scalar objective analysis
result = obs.xobjmap.scalar(
    "temp", target, corrlen={"lon": 1.0, "lat": 0.5}, err=0.1
)
result.temp   # interpolated field
result.error  # normalized error map

# Streamfunction recovery
obs_vel = xr.Dataset(
    {"u": ("station", u_data), "v": ("station", v_data)},
    coords={"lon": ("station", lons), "lat": ("station", lats)},
)
psi = obs_vel.xobjmap.streamfunction(
    "u", "v", target, corrlen={"lon": 1.0, "lat": 0.5}, err=0.1
)

# Helmholtz decomposition
result = obs_vel.xobjmap.helmholtz(
    "u", "v", target,
    corrlen_psi={"lon": 1.0, "lat": 0.5},
    corrlen_chi={"lon": 1.0, "lat": 0.5},
    err=0.1,
)
```

## API Reference

### Accessor methods

#### `ds.xobjmap.scalar(var, target, corrlen, err)`

Interpolates a scalar variable from scattered observations onto target locations.

| Parameter | Type | Description |
|-----------|------|-------------|
| `var` | `str` | Variable name in the dataset |
| `target` | `xr.Dataset` | Target coordinates |
| `corrlen` | `dict` or `float` | Correlation length scales (same units as coordinates) |
| `err` | `float` | Normalized error variance (0 to 1) |

Returns an `xr.Dataset` with the interpolated field and an `error` variable.

#### `ds.xobjmap.streamfunction(u_var, v_var, target, corrlen, err, b=0)`

Recovers the streamfunction from scattered velocity observations, assuming purely nondivergent flow.

| Parameter | Type | Description |
|-----------|------|-------------|
| `u_var` | `str` | Eastward velocity variable name |
| `v_var` | `str` | Northward velocity variable name |
| `target` | `xr.Dataset` | Target grid coordinates |
| `corrlen` | `dict` or `float` | Correlation length scales (same units as coordinates) |
| `err` | `float` | Normalized error variance (0 to 1) |
| `b` | `float` | Mean correction parameter (default: 0) |

Returns an `xr.Dataset` with a `psi` (streamfunction) variable.

#### `ds.xobjmap.velocity_potential(u_var, v_var, target, corrlen, err, b=0)`

Recovers the velocity potential from scattered velocity observations, assuming purely irrotational flow.

| Parameter | Type | Description |
|-----------|------|-------------|
| `u_var` | `str` | Eastward velocity variable name |
| `v_var` | `str` | Northward velocity variable name |
| `target` | `xr.Dataset` | Target grid coordinates |
| `corrlen` | `dict` or `float` | Correlation length scales (same units as coordinates) |
| `err` | `float` | Normalized error variance (0 to 1) |
| `b` | `float` | Mean correction parameter (default: 0) |

Returns an `xr.Dataset` with a `chi` (velocity potential) variable.

#### `ds.xobjmap.helmholtz(u_var, v_var, target, corrlen_psi, corrlen_chi, err, b=0)`

Helmholtz decomposition: jointly recovers the streamfunction and velocity potential from scattered velocity observations.

| Parameter | Type | Description |
|-----------|------|-------------|
| `u_var` | `str` | Eastward velocity variable name |
| `v_var` | `str` | Northward velocity variable name |
| `target` | `xr.Dataset` | Target grid coordinates |
| `corrlen_psi` | `dict` or `float` | Correlation length scales for the streamfunction |
| `corrlen_chi` | `dict` or `float` | Correlation length scales for the velocity potential |
| `err` | `float` | Normalized error variance (0 to 1) |
| `b` | `float` | Mean correction parameter (default: 0) |

Returns an `xr.Dataset` with `psi` (streamfunction) and `chi` (velocity potential) variables.

### Low-level functions

#### `xobjmap.scalar(xc, yc, x, y, t, corrlenx, corrleny, err)`

Scalar Gauss-Markov estimation. Returns `(tp, ep)` if `t` is provided, or just `ep` (error map) if `t` is `None`.

#### `xobjmap.streamfunction(xc, yc, x, y, u, v, corrlenx, corrleny, err, b=0)`

Recovers the streamfunction on the target grid `(xc, yc)`, assuming nondivergent flow.

#### `xobjmap.velocity_potential(xc, yc, x, y, u, v, corrlenx, corrleny, err, b=0)`

Recovers the velocity potential on the target grid `(xc, yc)`, assuming irrotational flow.

#### `xobjmap.helmholtz(xc, yc, x, y, u, v, corrlenx_psi, corrleny_psi, corrlenx_chi, corrleny_chi, err, b=0)`

Helmholtz decomposition. Returns `(psi, chi)` on the target grid.

## Notes

- Correlation lengths must be in the **same units** as the coordinates. If working with lon/lat in degrees, either express corrlen in degrees or convert to a projected coordinate system first.
- The streamfunction convention follows Bretherton et al. (1976): `u = -dpsi/dy`, `v = dpsi/dx`.
- The velocity potential convention: `u = dchi/dx`, `v = dchi/dy`.

## References

Bretherton, F. P., Davis, R. E., & Fandry, C. B. (1976). A technique for
objective analysis and design of oceanographic experiments applied to MODE-73.
*Deep-Sea Research*, 23(7), 559-582.
