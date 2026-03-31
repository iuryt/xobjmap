"""
JAX/GPU benchmark for scalar interpolation
===========================================

Compares the numpy (direct solve) and JAX (matrix-free conjugate
gradient) backends for ``scalar`` at increasing problem sizes.

Requires JAX with CUDA support::

    pip install 'xobjmap[jax]'

Run::

    pixi run -e test-jax-cuda python examples/benchmark_jax.py
"""

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["XLA_FLAGS"] = "--xla_gpu_autotune_level=0"

import time
import numpy as np

import jax
print(f"JAX devices: {jax.devices()}")

from xobjmap import error, scalar  # noqa: E402

rng = np.random.default_rng(42)

# ── Problem sizes ──────────────────────────────────────────────
obs_sizes = [500, 1000, 5000, 10000, 100000]
n_target = 5000
n_repeats = 3

# ── Warmup JAX compilation ────────────────────────────────────
_x = rng.uniform(0, 10, 10)
scalar(_x, _x, _x, _x, _x, corrlenx=5., corrleny=5., err=0.05, backend="jax")
error(_x, _x, _x, _x, corrlenx=5., corrleny=5., err=0.05, backend="jax")

# ── Benchmark ─────────────────────────────────────────────────
header = f"{'n_obs':>7}  {'backend':>8}  {'time (s)':>9}  {'CPU mem (MB)':>12}  {'GPU mem (MB)':>12}"
print()
print(header)
print("─" * len(header))

import tracemalloc

for n_obs in obs_sizes:
    x = rng.uniform(0, 100, n_obs)
    y = rng.uniform(0, 100, n_obs)
    t = 2 * x + 3 * y + rng.normal(0, 1, n_obs)
    xc = rng.uniform(0, 100, n_target)
    yc = rng.uniform(0, 100, n_target)

    for backend in ("numpy", "jax"):
        # Clear state
        if backend == "jax":
            jax.clear_caches()

        tracemalloc.start()

        gpu_before = None
        if backend == "jax":
            stats = jax.devices()[0].memory_stats()
            if stats:
                gpu_before = stats["bytes_in_use"]

        times = []
        for _ in range(n_repeats):
            s = time.perf_counter()
            tp = scalar(
                xc, yc, x, y, t,
                corrlenx=10., corrleny=10., err=0.05,
                backend=backend,
            )
            ep = error(
                xc, yc, x, y,
                corrlenx=10., corrleny=10., err=0.05,
                backend=backend,
            )
            if backend == "jax":
                jax.block_until_ready(tp)
                jax.block_until_ready(ep)
            times.append(time.perf_counter() - s)

        _, peak_cpu = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        gpu_str = "—"
        if backend == "jax" and gpu_before is not None:
            stats = jax.devices()[0].memory_stats()
            if stats:
                gpu_str = f"{(stats['peak_bytes_in_use'] - gpu_before) / 1e6:.1f}"

        median_t = np.median(times)
        print(
            f"{n_obs:>7}  {backend:>8}  {median_t:>9.3f}  "
            f"{peak_cpu / 1e6:>12.1f}  {gpu_str:>12}"
        )

    # Check values match between backends
    tp_np = scalar(xc, yc, x, y, t, corrlenx=10., corrleny=10., err=0.05, backend="numpy")
    ep_np = error(xc, yc, x, y, corrlenx=10., corrleny=10., err=0.05, backend="numpy")
    tp_jax = scalar(xc, yc, x, y, t, corrlenx=10., corrleny=10., err=0.05, backend="jax")
    ep_jax = error(xc, yc, x, y, corrlenx=10., corrleny=10., err=0.05, backend="jax")
    tp_diff = np.max(np.abs(np.asarray(tp_np).ravel() - np.asarray(tp_jax).ravel()))
    ep_diff = np.max(np.abs(np.asarray(ep_np).ravel() - np.asarray(ep_jax).ravel()))
    print(f"         max |Δtp|={tp_diff:.4f}  max |Δep|={ep_diff:.6f}")
    print()
