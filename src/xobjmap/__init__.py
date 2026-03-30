"""
xobjmap: Xarray-native objective mapping and interpolation.

Optimal interpolation of scattered observations onto arbitrary target
locations using Gauss-Markov estimation, with support for scalar fields,
vector fields (streamfunction recovery), and Helmholtz decomposition.
"""

__version__ = "0.0.1"

from .interp import scaloa, vectoa
from .accessor import XobjmapAccessor

__all__ = ["scaloa", "vectoa", "XobjmapAccessor"]
