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
psi = obs_vel.xobjmap.streamfunction(
    "u", "v", target, corrlen={"lon": 1.0, "lat": 0.5}, err=0.1
)

# Helmholtz decomposition → streamfunction + velocity potential
result = obs_vel.xobjmap.helmholtz(
    "u", "v", target,
    corrlen_psi={"lon": 1.0, "lat": 0.5},
    corrlen_chi={"lon": 1.0, "lat": 0.5},
    err=0.1,
)
result.psi  # streamfunction
result.chi  # velocity potential
```

The low-level functions are also available directly:
```python
tp = xobjmap.scalar(xc, yc, x, y, t, corrlenx=1.0, corrleny=0.5, err=0.1)
ep = xobjmap.error(xc, yc, x, y, corrlenx=1.0, corrleny=0.5, err=0.1)
psi = xobjmap.streamfunction(xc, yc, x, y, u, v, corrlenx=1.0, corrleny=0.5, err=0.1)
chi = xobjmap.velocity_potential(xc, yc, x, y, u, v, corrlenx=1.0, corrleny=0.5, err=0.1)
psi, chi = xobjmap.helmholtz(xc, yc, x, y, u, v,
    corrlenx_psi=1.0, corrleny_psi=0.5,
    corrlenx_chi=1.0, corrleny_chi=0.5, err=0.1)
```

All functions accept `backend="jax"` for lower memory usage and optional GPU acceleration (requires `pip install 'xobjmap[jax]'`).

## References

Bretherton, F. P., Davis, R. E., & Fandry, C. B. (1976). A technique for
objective analysis and design of oceanographic experiments applied to MODE-73.
*Deep-Sea Research*, 23(7), 559-582.

## License

MIT
