"""
GPU-only scaling benchmark for JAX.

Sweeps ``n_obs`` and reports:
- runtime backend and devices
- compile time
- steady-state median time
- peak GPU memory
- fitted log-log slope for GPU memory vs n_obs

Run:

    pixi run -e test-jax-cuda python examples/benchmark_gpu_scaling.py

To force CUDA explicitly:

    pixi run -e test-jax-cuda python examples/benchmark_gpu_scaling.py --force-gpu
"""

from __future__ import annotations

import argparse
import json
import os
import resource
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]


DEFAULT_SIZES = [1000, 2000, 4000, 8000, 12000]


def _make_scalar_xy(n_obs, nx, ny, seed):
    import xobjmap  # noqa: F401
    import xarray as xr

    rng = np.random.default_rng(seed)
    x_obs = rng.uniform(-4.0, 4.0, n_obs)
    y_obs = rng.uniform(-4.0, 4.0, n_obs)
    temp = np.sin(1.7 * x_obs) + 0.35 * np.cos(0.9 * y_obs) + 0.1 * x_obs * y_obs
    obs = xr.Dataset(
        {"temp": ("obs", temp)},
        coords={"x": ("obs", x_obs), "y": ("obs", y_obs)},
    )
    target = xr.Dataset(
        coords={
            "x": np.linspace(-3.5, 3.5, nx),
            "y": np.linspace(-3.0, 3.0, ny),
        }
    )
    kwargs = {
        "corrlen": {"x": 1.2, "y": 1.0},
        "err": 0.05,
        "interp_dims": ("x", "y"),
    }
    return obs, target, kwargs


def _make_helmholtz_xy(n_obs, nx, ny, seed):
    import xobjmap  # noqa: F401
    import xarray as xr

    rng = np.random.default_rng(seed)
    x_obs = rng.uniform(-3.5, 3.5, n_obs)
    y_obs = rng.uniform(-3.5, 3.5, n_obs)
    u = np.sin(x_obs) * np.sin(y_obs) - 0.35 * np.sin(0.7 * x_obs) * np.sin(1.1 * y_obs)
    v = np.cos(x_obs) * np.cos(y_obs) + 0.55 * np.cos(0.7 * x_obs) * np.cos(1.1 * y_obs)
    obs = xr.Dataset(
        {"u": ("obs", u), "v": ("obs", v)},
        coords={"x": ("obs", x_obs), "y": ("obs", y_obs)},
    )
    target = xr.Dataset(
        coords={
            "x": np.linspace(-3.0, 3.0, nx),
            "y": np.linspace(-3.0, 3.0, ny),
        }
    )
    kwargs = {
        "corrlen_psi": {"x": 1.2, "y": 1.2},
        "corrlen_chi": {"x": 1.2, "y": 1.2},
        "err": 0.02,
        "derivative_dims": ("x", "y"),
        "interp_dims": ("x", "y"),
    }
    return obs, target, kwargs


def _run_case(kind, n_obs, nx, ny, repeats):
    if kind == "scalar_xy":
        obs, target, kwargs = _make_scalar_xy(n_obs, nx, ny, seed=42)
        runner = lambda: obs.xobjmap.scalar("temp", target, backend="jax", **kwargs)
    elif kind == "helmholtz_xy":
        obs, target, kwargs = _make_helmholtz_xy(n_obs, nx, ny, seed=42)
        runner = lambda: obs.xobjmap.helmholtz("u", "v", target, backend="jax", **kwargs)
    else:
        raise ValueError(f"Unknown kind {kind!r}")

    import jax

    runtime_backend = jax.default_backend()
    runtime_devices = [str(device) for device in jax.devices()]
    jax.clear_caches()
    stats0 = jax.devices()[0].memory_stats()
    before = stats0.get("bytes_in_use", 0) if stats0 else 0

    start = time.perf_counter()
    warm = runner()
    jax.block_until_ready(warm)
    compile_time = time.perf_counter() - start

    times = []
    for _ in range(repeats):
        start = time.perf_counter()
        out = runner()
        jax.block_until_ready(out)
        times.append(time.perf_counter() - start)

    stats1 = jax.devices()[0].memory_stats()
    peak_gpu_mb = None
    if stats1:
        peak_gpu_mb = stats1.get("peak_bytes_in_use", before) / 1e6

    ru = resource.getrusage(resource.RUSAGE_SELF)
    maxrss_mb = ru.ru_maxrss / 1024.0

    return {
        "kind": kind,
        "n_obs": n_obs,
        "n_target": int(nx * ny),
        "runtime_backend": runtime_backend,
        "runtime_devices": runtime_devices,
        "compile_time_s": compile_time,
        "median_time_s": float(np.median(times)),
        "peak_gpu_mem_mb": peak_gpu_mb,
        "peak_rss_mb": maxrss_mb,
    }


def _worker_main(args):
    os.environ.pop("JAX_PLATFORM_NAME", None)
    os.environ.pop("JAX_BACKEND_TARGET", None)
    if args.force_gpu:
        os.environ["JAX_PLATFORMS"] = "cuda"
    else:
        os.environ.pop("JAX_PLATFORMS", None)

    result = _run_case(args.kind, args.n_obs, args.nx, args.ny, args.repeats)
    print(json.dumps(result))


def _run_worker(kind, n_obs, nx, ny, repeats, force_gpu):
    cmd = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--worker",
        "--kind",
        kind,
        "--n-obs",
        str(n_obs),
        "--nx",
        str(nx),
        "--ny",
        str(ny),
        "--repeats",
        str(repeats),
    ]
    if force_gpu:
        cmd.append("--force-gpu")

    env = os.environ.copy()
    env.pop("JAX_PLATFORM_NAME", None)
    env.pop("JAX_BACKEND_TARGET", None)
    if force_gpu:
        env["JAX_PLATFORMS"] = "cuda"

    completed = subprocess.run(
        cmd,
        cwd=ROOT,
        capture_output=True,
        text=True,
        env=env,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            "GPU scaling worker failed.\n"
            f"Command: {' '.join(cmd)}\n"
            f"stdout:\n{completed.stdout}\n"
            f"stderr:\n{completed.stderr}"
        )
    return json.loads(completed.stdout.strip().splitlines()[-1])


def _fit_power_law(x, y):
    logx = np.log(np.asarray(x, dtype=float))
    logy = np.log(np.asarray(y, dtype=float))
    slope, intercept = np.polyfit(logx, logy, 1)
    return slope, intercept


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--kind", choices=("scalar_xy", "helmholtz_xy"), default="scalar_xy")
    parser.add_argument("--n-obs", type=int)
    parser.add_argument("--sizes", nargs="+", type=int, default=DEFAULT_SIZES)
    parser.add_argument("--nx", type=int, default=64)
    parser.add_argument("--ny", type=int, default=64)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--force-gpu", action="store_true")
    args = parser.parse_args()

    if args.worker:
        _worker_main(args)
        return

    header = (
        f"{'kind':<14} {'n_obs':>8} {'runtime':<8} {'compile(s)':>10} "
        f"{'median(s)':>10} {'GPU MB':>10} {'RSS MB':>10}"
    )
    print(header)
    print("-" * len(header))

    results = []
    for n_obs in args.sizes:
        result = _run_worker(
            args.kind,
            n_obs=n_obs,
            nx=args.nx,
            ny=args.ny,
            repeats=args.repeats,
            force_gpu=args.force_gpu,
        )
        print(
            f"{result['kind']:<14} {result['n_obs']:>8} {result['runtime_backend']:<8} "
            f"{result['compile_time_s']:>10.3f} {result['median_time_s']:>10.3f} "
            f"{result['peak_gpu_mem_mb']:>10.1f} {result['peak_rss_mb']:>10.1f}",
            flush=True,
        )
        print(f"  devices: {', '.join(result['runtime_devices'])}", flush=True)
        results.append(result)

    mem_slope, _ = _fit_power_law(
        [row["n_obs"] for row in results],
        [row["peak_gpu_mem_mb"] for row in results],
    )
    time_slope, _ = _fit_power_law(
        [row["n_obs"] for row in results],
        [row["median_time_s"] for row in results],
    )
    print("")
    print(f"Estimated GPU memory scaling exponent: n_obs^{mem_slope:.3f}")
    print(f"Estimated runtime scaling exponent:    n_obs^{time_slope:.3f}")


if __name__ == "__main__":
    main()
