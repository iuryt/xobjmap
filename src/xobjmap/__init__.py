"""
xobjmap: Xarray-native objective mapping and interpolation.

Optimal interpolation of scattered observations onto arbitrary target
locations using Gauss-Markov estimation, with support for scalar fields,
streamfunction recovery, velocity potential recovery, and Helmholtz
decomposition.
"""

__version__ = "0.0.1"

from .interp import (
    error,
    scalar_error,
    streamfunction_error,
    velocity_potential_error,
    helmholtz_error,
    scalar,
    streamfunction,
    velocity_potential,
    helmholtz,
)
from .accessor import XobjmapAccessor

__all__ = [
    "error",
    "scalar_error",
    "streamfunction_error",
    "velocity_potential_error",
    "helmholtz_error",
    "scalar",
    "streamfunction",
    "velocity_potential",
    "helmholtz",
    "XobjmapAccessor",
]
