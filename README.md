# xobjmap

Xarray-native objective mapping and interpolation of scattered observations.

![Rankine vortex reconstruction from ADCP transects](docs/example_vortex.png)

## Overview

**xobjmap** provides optimal interpolation of scattered observations onto
arbitrary target locations using Gauss-Markov estimation. It supports:

- **Scalar objective analysis** — interpolate scalar fields (temperature,
  salinity, SSH, etc.) from scattered observations with analytical error maps
- **Streamfunction recovery** — recover the streamfunction from scattered
  velocity observations using physically derived cross-covariance
- **Velocity potential recovery** — recover the velocity potential from
  scattered velocity observations assuming irrotational flow
- **Helmholtz decomposition** — simultaneously estimate streamfunction and
  velocity potential from velocity observations
- **JAX backend** — matrix-free conjugate gradient solver that uses
  significantly less memory, with optional GPU acceleration

## Installation

With pip:
```bash
pip install xobjmap              # numpy only
pip install 'xobjmap[jax]'      # + JAX (CPU)
pip install 'xobjmap[jax-cuda]' # + JAX with GPU (CUDA 12)
```

With pixi (recommended for development):
```bash
pixi add xobjmap
```

## Quick start

```python
import numpy as np
import xarray as xr
import xobjmap  # registers the xarray accessor

# Scattered observations
obs = xr.Dataset(
    {"temp": ("station", temp_data)},
    coords={"lon": ("station", lons), "lat": ("station", lats)},
)

# Target grid
target = xr.Dataset(
    coords={"lon": np.linspace(-40, -38, 50), "lat": np.linspace(-24, -22, 40)}
)

# Scalar objective analysis → interpolated field + error map
result = obs.xobjmap.scalar(
    "temp", target, corrlen={"lon": 1.0, "lat": 0.5}, err=0.1
)
result.temp   # interpolated field
result.error  # normalized error map

# Streamfunction recovery from velocity observations
obs_vel = xr.Dataset(
    {"u": ("station", u_data), "v": ("station", v_data)},
    coords={"lon": ("station", lons), "lat": ("station", lats)},
)
result_psi = obs_vel.xobjmap.streamfunction(
    "u", "v", target, corrlen={"lon": 1.0, "lat": 0.5}, err=0.1
)
result_psi.psi        # streamfunction
result_psi.psi_error  # normalized posterior error

# Velocity potential recovery
result_chi = obs_vel.xobjmap.velocity_potential(
    "u", "v", target, corrlen={"lon": 1.0, "lat": 0.5}, err=0.1
)
result_chi.chi        # velocity potential
result_chi.chi_error  # normalized posterior error

# Helmholtz decomposition → streamfunction + velocity potential
result = obs_vel.xobjmap.helmholtz(
    "u", "v", target,
    corrlen_psi={"lon": 1.0, "lat": 0.5},
    corrlen_chi={"lon": 1.0, "lat": 0.5},
    err=0.1,
)
result.psi  # streamfunction
result.chi  # velocity potential
result.psi_error  # normalized posterior error for psi
result.chi_error  # normalized posterior error for chi

# Skip error computation when only the field is needed
fast = obs.xobjmap.scalar(
    "temp", target, corrlen={"lon": 1.0, "lat": 0.5}, err=0.1,
    return_error=False,
)
```

The low-level functions are also available directly:
```python
tp = xobjmap.scalar(xc, yc, x, y, t, corrlenx=1.0, corrleny=0.5, err=0.1)
ep = xobjmap.scalar_error(xc, yc, x, y, corrlenx=1.0, corrleny=0.5, err=0.1)
psi = xobjmap.streamfunction(xc, yc, x, y, u, v, corrlenx=1.0, corrleny=0.5, err=0.1)
psi_err = xobjmap.streamfunction_error(
    xc, yc, x, y, u, v, corrlenx=1.0, corrleny=0.5, err=0.1
)
chi = xobjmap.velocity_potential(xc, yc, x, y, u, v, corrlenx=1.0, corrleny=0.5, err=0.1)
chi_err = xobjmap.velocity_potential_error(
    xc, yc, x, y, u, v, corrlenx=1.0, corrleny=0.5, err=0.1
)
psi, chi = xobjmap.helmholtz(xc, yc, x, y, u, v,
    corrlenx_psi=1.0, corrleny_psi=0.5,
    corrlenx_chi=1.0, corrleny_chi=0.5, err=0.1)
psi_err, chi_err = xobjmap.helmholtz_error(
    xc, yc, x, y, u, v,
    corrlenx_psi=1.0, corrleny_psi=0.5,
    corrlenx_chi=1.0, corrleny_chi=0.5, err=0.1
)
```

All functions accept `backend="jax"` for lower memory usage and optional GPU acceleration (requires `pip install 'xobjmap[jax]'`).

## Error model

`xobjmap` exposes normalized posterior error estimates alongside the mapped
fields.

- `scalar_error` and `ds.xobjmap.scalar(...).error` are normalized mean squared
  interpolation errors for scalar objective analysis.
- `streamfunction_error`, `velocity_potential_error`, and `helmholtz_error`
  return normalized posterior errors for the recovered latent potential fields.
- These error fields depend on observation geometry, correlation scales, and
  backend, but not on the observed scalar values themselves.

Backend note:

- NumPy computes the direct Bretherton solve.
- JAX computes the fields with a matrix-free conjugate-gradient solve.
- JAX scalar and potential error fields use a local-neighborhood approximation
  (`k_local`) instead of a full dense solve, so they are expected to be close
  but not exactly identical to the NumPy error fields.

## References

Bretherton, F. P., Davis, R. E., & Fandry, C. B. (1976). A technique for
objective analysis and design of oceanographic experiments applied to MODE-73.
*Deep-Sea Research*, 23(7), 559-582.

## License

MIT
