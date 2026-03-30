# xobjmap

Xarray-native objective mapping and interpolation of scattered observations.

## Overview

**xobjmap** provides optimal interpolation of scattered observations onto
arbitrary target locations using Gauss-Markov estimation. It supports:

- **Scalar objective analysis** — interpolate scalar fields (temperature,
  salinity, SSH, etc.) from scattered observations with analytical error maps
- **Vectorial objective analysis** — recover the streamfunction from scattered
  velocity observations using physically derived cross-covariance
- **Helmholtz decomposition** *(planned)* — simultaneously estimate
  streamfunction and velocity potential from velocity observations
- **Scalable JAX backend** *(planned)* — variational solver for large
  datasets with GPU acceleration

## Installation

```bash
conda install -c conda-forge xobjmap
```

## Quick start

```python
import numpy as np
import xobjmap

# Scalar objective analysis
tp, ep = xobjmap.scaloa(xc, yc, x, y, t, corrlenx=200e3, corrleny=100e3, err=0.1)

# Vectorial objective analysis (velocity → streamfunction)
psi = xobjmap.vectoa(Xg, Yg, X, Y, U, V, corrlenx=200e3, corrleny=100e3, err=0.1)
```

## References

Bretherton, F. P., Davis, R. E., & Fandry, C. B. (1976). A technique for
objective analysis and design of oceanographic experiments applied to MODE-73.
*Deep-Sea Research*, 23(7), 559-582.

## License

MIT
