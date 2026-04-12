"""
Focused 3-D Helmholtz benchmark for NumPy vs JAX GPU.

This benchmark targets the xarray accessor N-D Helmholtz path with
``interp_dims=('x', 'y', 'z')`` and ``derivative_dims=('x', 'y')``.

It measures:
- warmup/compile time
- steady-state runtime
- peak process RSS
- peak Python allocations tracked by ``tracemalloc``
- peak JAX-reported GPU memory

Typical runs:

    pixi run -e test-jax-cuda python examples/benchmark_helmholtz_3d.py

    pixi run -e test-jax-cuda python examples/benchmark_helmholtz_3d.py \
        --sizes 900 1400 2000 2600 --nx 80 --ny 80 --nz 3

If you want to avoid the dense NumPy path at very large sizes:

    pixi run -e test-jax-cuda python examples/benchmark_helmholtz_3d.py \
        --backends jax-gpu --sizes 2000 3000 4000
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import resource
import subprocess
import sys
import time
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SIZES = [900, 1400, 2000]


def _dense_memory_estimate_mb(n_obs, n_target):
    """Estimate dominant dense-memory footprint for NumPy Helmholtz."""
    bytes_per_float = 8
    a_bytes = (2 * n_obs) * (2 * n_obs) * bytes_per_float
    p_bytes = n_target * (2 * n_obs) * bytes_per_float
    return {
        "A_MB": a_bytes / 1e6,
        "cross_MB": p_bytes / 1e6,
        "total_MB": (a_bytes + 2 * p_bytes) / 1e6,
    }


def _maybe_float(value):
    if value is None:
        return None
    return float(value)


def _emit_json(path, rows):
    payload = {
        "rows": rows,
    }
    with open(path, "w", encoding="ascii") as f:
        json.dump(payload, f, indent=2)


def _emit_csv(path, rows):
    fieldnames = [
        "backend",
        "status",
        "skip_reason",
        "n_obs",
        "n_target",
        "target_shape",
        "runtime_backend",
        "runtime_devices",
        "compile_time_s",
        "median_time_s",
        "min_time_s",
        "max_time_s",
        "peak_rss_mb",
        "peak_python_mem_mb",
        "peak_gpu_mem_mb",
        "dense_A_mb",
        "dense_cross_mb",
        "dense_total_mb",
        "psi_mean",
        "chi_mean",
        "psi_error_mean",
        "chi_error_mean",
    ]
    with open(path, "w", newline="", encoding="ascii") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            out = dict(row)
            out["target_shape"] = json.dumps(row["target_shape"])
            out["runtime_devices"] = json.dumps(row["runtime_devices"])
            writer.writerow(out)


def _make_workload(n_obs, nx, ny, nz, seed, corrlen_xy, corrlen_z, err):
    import xobjmap  # noqa: F401
    import xarray as xr

    rng = np.random.default_rng(seed)
    x_obs = rng.uniform(-3.5, 3.5, n_obs)
    y_obs = rng.uniform(-3.5, 3.5, n_obs)
    z_levels = np.linspace(0.0, 1.0, nz)
    z_obs = rng.choice(z_levels, size=n_obs)

    amp = 1.0 + 0.35 * z_obs
    u = amp * (
        np.sin(x_obs) * np.sin(y_obs)
        - 0.35 * np.sin(0.7 * x_obs) * np.sin(1.1 * y_obs)
    )
    v = amp * (
        np.cos(x_obs) * np.cos(y_obs)
        + 0.55 * np.cos(0.7 * x_obs) * np.cos(1.1 * y_obs)
    )

    obs = xr.Dataset(
        {"u": ("obs", u), "v": ("obs", v)},
        coords={
            "x": ("obs", x_obs),
            "y": ("obs", y_obs),
            "z": ("obs", z_obs),
        },
    )

    x = xr.DataArray(np.linspace(-3.0, 3.0, nx), dims="x")
    y = xr.DataArray(np.linspace(-3.0, 3.0, ny), dims="y")
    z = xr.DataArray(z_levels, dims="z")
    z3, y3, x3 = xr.broadcast(z, y, x)
    target = xr.Dataset(coords={"x": x3, "y": y3, "z": z3})

    kwargs = {
        "corrlen_psi": {"x": corrlen_xy, "y": corrlen_xy, "z": corrlen_z},
        "corrlen_chi": {"x": corrlen_xy, "y": corrlen_xy, "z": corrlen_z},
        "err": err,
        "derivative_dims": ("x", "y"),
        "interp_dims": ("x", "y", "z"),
    }
    return obs, target, kwargs


def _block_helmholtz(result, backend_name):
    if not backend_name.startswith("jax"):
        return

    import jax

    jax.block_until_ready(result.psi.data)
    jax.block_until_ready(result.chi.data)
    jax.block_until_ready(result.psi_error.data)
    jax.block_until_ready(result.chi_error.data)


def _configure_worker_backend(backend_name):
    os.environ.pop("JAX_PLATFORM_NAME", None)
    os.environ.pop("JAX_BACKEND_TARGET", None)
    os.environ.pop("JAX_PLATFORMS", None)

    if backend_name == "jax-cpu":
        os.environ["JAX_PLATFORMS"] = "cpu"
    elif backend_name == "jax-gpu":
        os.environ["JAX_PLATFORMS"] = "cuda"


def _run_case(args):
    import tracemalloc

    _configure_worker_backend(args.backend)

    obs, target, kwargs = _make_workload(
        n_obs=args.n_obs,
        nx=args.nx,
        ny=args.ny,
        nz=args.nz,
        seed=args.seed,
        corrlen_xy=args.corrlen_xy,
        corrlen_z=args.corrlen_z,
        err=args.err,
    )

    accessor_backend = "numpy" if args.backend == "numpy" else "jax"
    runner = lambda: obs.xobjmap.helmholtz("u", "v", target, backend=accessor_backend, **kwargs)

    runtime_backend = "cpu" if args.backend == "numpy" else None
    runtime_devices = None
    peak_gpu_mb = None

    if args.backend.startswith("jax"):
        import jax

        jax.clear_caches()
        runtime_backend = jax.default_backend()
        runtime_devices = [str(device) for device in jax.devices()]

    tracemalloc.start()

    start = time.perf_counter()
    warm = runner()
    _block_helmholtz(warm, args.backend)
    compile_time_s = time.perf_counter() - start

    times = []
    result = warm
    for _ in range(args.repeats):
        start = time.perf_counter()
        result = runner()
        _block_helmholtz(result, args.backend)
        times.append(time.perf_counter() - start)

    _, peak_python_bytes = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    if args.backend.startswith("jax"):
        import jax

        stats = jax.devices()[0].memory_stats()
        if stats:
            peak_gpu_mb = stats.get(
                "peak_bytes_in_use",
                stats.get("bytes_in_use", 0),
            ) / 1e6

    ru = resource.getrusage(resource.RUSAGE_SELF)

    return {
        "backend": args.backend,
        "runtime_backend": runtime_backend,
        "runtime_devices": runtime_devices,
        "n_obs": args.n_obs,
        "n_target": int(args.nx * args.ny * args.nz),
        "target_shape": [args.nz, args.ny, args.nx],
        "compile_time_s": compile_time_s,
        "median_time_s": float(np.median(times)),
        "min_time_s": float(np.min(times)),
        "max_time_s": float(np.max(times)),
        "peak_rss_mb": ru.ru_maxrss / 1024.0,
        "peak_python_mem_mb": peak_python_bytes / 1e6,
        "peak_gpu_mem_mb": peak_gpu_mb,
        "psi_mean": float(np.asarray(result.psi.mean())),
        "chi_mean": float(np.asarray(result.chi.mean())),
        "psi_error_mean": float(np.asarray(result.psi_error.mean())),
        "chi_error_mean": float(np.asarray(result.chi_error.mean())),
    }


def _run_worker(
    backend,
    n_obs,
    nx,
    ny,
    nz,
    repeats,
    seed,
    corrlen_xy,
    corrlen_z,
    err,
):
    cmd = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--worker",
        "--backend",
        backend,
        "--n-obs",
        str(n_obs),
        "--nx",
        str(nx),
        "--ny",
        str(ny),
        "--nz",
        str(nz),
        "--repeats",
        str(repeats),
        "--seed",
        str(seed),
        "--corrlen-xy",
        str(corrlen_xy),
        "--corrlen-z",
        str(corrlen_z),
        "--err",
        str(err),
    ]

    env = os.environ.copy()
    env.pop("JAX_PLATFORM_NAME", None)
    env.pop("JAX_BACKEND_TARGET", None)
    env.pop("JAX_PLATFORMS", None)
    if backend == "jax-cpu":
        env["JAX_PLATFORMS"] = "cpu"
    elif backend == "jax-gpu":
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
            "Benchmark worker failed.\n"
            f"Command: {' '.join(cmd)}\n"
            f"stdout:\n{completed.stdout}\n"
            f"stderr:\n{completed.stderr}"
        )
    return json.loads(completed.stdout.strip().splitlines()[-1])


def _print_summary(results_by_backend):
    numpy_result = results_by_backend.get("numpy")
    jax_gpu_result = results_by_backend.get("jax-gpu")

    if numpy_result and jax_gpu_result:
        speedup = numpy_result["median_time_s"] / jax_gpu_result["median_time_s"]
        rss_ratio = jax_gpu_result["peak_rss_mb"] / numpy_result["peak_rss_mb"]
        print(
            f"  speedup(jax-gpu vs numpy): {speedup:.2f}x"
        )
        print(
            f"  host RSS ratio(jax-gpu / numpy): {rss_ratio:.2f}x"
        )


def _print_skip_row(backend, n_obs, n_target):
    print(
        f"{backend:<8} {n_obs:>8} {n_target:>10} {'skip':<8} "
        f"{'—':>10} {'—':>10} {'—':>10} {'—':>10} {'—':>10}"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", action="store_true")
    parser.add_argument(
        "--backends",
        nargs="+",
        choices=("numpy", "jax-cpu", "jax-gpu"),
        default=("numpy", "jax-gpu"),
        help="Backends to benchmark. Defaults to numpy and jax-gpu.",
    )
    parser.add_argument("--backend", choices=("numpy", "jax-cpu", "jax-gpu"))
    parser.add_argument("--sizes", nargs="+", type=int, default=DEFAULT_SIZES)
    parser.add_argument("--n-obs", type=int)
    parser.add_argument("--nx", type=int, default=80)
    parser.add_argument("--ny", type=int, default=80)
    parser.add_argument("--nz", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--corrlen-xy", type=float, default=1.0)
    parser.add_argument("--corrlen-z", type=float, default=0.35)
    parser.add_argument("--err", type=float, default=0.02)
    parser.add_argument(
        "--max-dense-gb",
        type=float,
        default=8.0,
        help="Skip numpy when the dense memory estimate exceeds this threshold.",
    )
    parser.add_argument(
        "--allow-large-numpy",
        action="store_true",
        help="Run numpy even when the dense memory estimate exceeds --max-dense-gb.",
    )
    parser.add_argument(
        "--json",
        dest="json_path",
        help="Write machine-readable benchmark results to this JSON file.",
    )
    parser.add_argument(
        "--csv",
        dest="csv_path",
        help="Write machine-readable benchmark results to this CSV file.",
    )
    args = parser.parse_args()

    if args.worker:
        print(json.dumps(_run_case(args)))
        return

    header = (
        f"{'backend':<8} {'n_obs':>8} {'n_target':>10} {'runtime':<8} "
        f"{'compile(s)':>10} {'median(s)':>10} {'RSS MB':>10} {'Py MB':>10} {'GPU MB':>10}"
    )
    print(header)
    print("-" * len(header))
    rows = []

    for n_obs in args.sizes:
        n_target = args.nx * args.ny * args.nz
        mem = _dense_memory_estimate_mb(n_obs, n_target)
        print(
            f"# n_obs={n_obs}, target_shape=({args.nz}, {args.ny}, {args.nx}), "
            f"n_target={n_target}, dense_numpy≈{mem['total_MB'] / 1024:.2f} GiB "
            f"(A={mem['A_MB'] / 1024:.2f} GiB, cross={mem['cross_MB'] / 1024:.2f} GiB)"
        )

        results_by_backend = {}
        for backend in args.backends:
            if (
                backend == "numpy"
                and not args.allow_large_numpy
                and mem["total_MB"] / 1024 > args.max_dense_gb
            ):
                _print_skip_row(backend, n_obs, n_target)
                print(
                    f"  skipped: dense lower-bound estimate {mem['total_MB'] / 1024:.2f} GiB "
                    f"exceeds threshold {args.max_dense_gb:.1f} GiB"
                )
                print(
                    "  note: real NumPy RSS can be much larger than this estimate due to "
                    "temporaries, solve workspace, and allocator overhead"
                )
                rows.append(
                    {
                        "backend": backend,
                        "status": "skipped",
                        "skip_reason": (
                            f"dense lower-bound estimate {mem['total_MB'] / 1024:.2f} GiB "
                            f"exceeds threshold {args.max_dense_gb:.1f} GiB"
                        ),
                        "n_obs": n_obs,
                        "n_target": n_target,
                        "target_shape": [args.nz, args.ny, args.nx],
                        "runtime_backend": None,
                        "runtime_devices": [],
                        "compile_time_s": None,
                        "median_time_s": None,
                        "min_time_s": None,
                        "max_time_s": None,
                        "peak_rss_mb": None,
                        "peak_python_mem_mb": None,
                        "peak_gpu_mem_mb": None,
                        "dense_A_mb": mem["A_MB"],
                        "dense_cross_mb": mem["cross_MB"],
                        "dense_total_mb": mem["total_MB"],
                        "psi_mean": None,
                        "chi_mean": None,
                        "psi_error_mean": None,
                        "chi_error_mean": None,
                    }
                )
                continue

            result = _run_worker(
                backend=backend,
                n_obs=n_obs,
                nx=args.nx,
                ny=args.ny,
                nz=args.nz,
                repeats=args.repeats,
                seed=args.seed,
                corrlen_xy=args.corrlen_xy,
                corrlen_z=args.corrlen_z,
                err=args.err,
            )
            results_by_backend[backend] = result
            gpu_str = "—" if result["peak_gpu_mem_mb"] is None else f"{result['peak_gpu_mem_mb']:.1f}"
            print(
                f"{backend:<8} {n_obs:>8} {n_target:>10} {result['runtime_backend']:<8} "
                f"{result['compile_time_s']:>10.3f} {result['median_time_s']:>10.3f} "
                f"{result['peak_rss_mb']:>10.1f} {result['peak_python_mem_mb']:>10.1f} "
                f"{gpu_str:>10}"
            )
            if result["runtime_devices"]:
                print(f"  devices: {', '.join(result['runtime_devices'])}")
            rows.append(
                {
                    "backend": result["backend"],
                    "status": "ok",
                    "skip_reason": None,
                    "n_obs": result["n_obs"],
                    "n_target": result["n_target"],
                    "target_shape": result["target_shape"],
                    "runtime_backend": result["runtime_backend"],
                    "runtime_devices": result["runtime_devices"] or [],
                    "compile_time_s": _maybe_float(result["compile_time_s"]),
                    "median_time_s": _maybe_float(result["median_time_s"]),
                    "min_time_s": _maybe_float(result["min_time_s"]),
                    "max_time_s": _maybe_float(result["max_time_s"]),
                    "peak_rss_mb": _maybe_float(result["peak_rss_mb"]),
                    "peak_python_mem_mb": _maybe_float(result["peak_python_mem_mb"]),
                    "peak_gpu_mem_mb": _maybe_float(result["peak_gpu_mem_mb"]),
                    "dense_A_mb": mem["A_MB"],
                    "dense_cross_mb": mem["cross_MB"],
                    "dense_total_mb": mem["total_MB"],
                    "psi_mean": _maybe_float(result["psi_mean"]),
                    "chi_mean": _maybe_float(result["chi_mean"]),
                    "psi_error_mean": _maybe_float(result["psi_error_mean"]),
                    "chi_error_mean": _maybe_float(result["chi_error_mean"]),
                }
            )

        _print_summary(results_by_backend)
        print("")

    if args.json_path:
        _emit_json(args.json_path, rows)
    if args.csv_path:
        _emit_csv(args.csv_path, rows)


if __name__ == "__main__":
    main()
