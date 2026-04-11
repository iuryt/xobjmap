"""
Backend benchmark for NumPy vs JAX GPU.

Measures:
- wall-clock time
- peak CPU RSS
- peak GPU memory reported by JAX
- output agreement between NumPy and JAX

Run with:

    pixi run -e test-jax-cuda python examples/benchmark_backends.py

To force GPU execution for JAX workers:

    JAX_PLATFORMS=gpu pixi run -e test-jax-cuda python examples/benchmark_backends.py
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


WORKLOADS = {
    "scalar_xy": {
        "kind": "scalar",
        "dims": ("x", "y"),
        "n_obs": 3000,
        "target_shape": (72, 64),
        "corrlen": {"x": 1.2, "y": 1.0},
        "err": 0.05,
        "repeats": 5,
    },
    "scalar_xy_large": {
        "kind": "scalar",
        "dims": ("x", "y"),
        "n_obs": 12000,
        "target_shape": (160, 160),
        "corrlen": {"x": 1.2, "y": 1.0},
        "err": 0.05,
        "repeats": 3,
    },
    "scalar_xytime": {
        "kind": "scalar",
        "dims": ("x", "y", "time"),
        "n_obs": 4000,
        "target_shape": (3, 48, 52),
        "corrlen": {"x": 1.2, "y": 1.0, "time": 0.5},
        "err": 0.05,
        "repeats": 5,
    },
    "scalar_xytime_large": {
        "kind": "scalar",
        "dims": ("x", "y", "time"),
        "n_obs": 12000,
        "target_shape": (4, 128, 128),
        "corrlen": {"x": 1.2, "y": 1.0, "time": 0.5},
        "err": 0.05,
        "repeats": 3,
    },
    "helmholtz_xy": {
        "kind": "helmholtz",
        "dims": ("x", "y"),
        "n_obs": 220,
        "target_shape": (42, 44),
        "corrlen_psi": {"x": 1.2, "y": 1.2},
        "corrlen_chi": {"x": 1.2, "y": 1.2},
        "err": 0.02,
        "repeats": 3,
    },
    "helmholtz_xy_large": {
        "kind": "helmholtz",
        "dims": ("x", "y"),
        "n_obs": 900,
        "target_shape": (96, 96),
        "corrlen_psi": {"x": 1.2, "y": 1.2},
        "corrlen_chi": {"x": 1.2, "y": 1.2},
        "err": 0.02,
        "repeats": 2,
    },
    "helmholtz_xytime": {
        "kind": "helmholtz",
        "dims": ("x", "y", "time"),
        "n_obs": 260,
        "target_shape": (2, 30, 32),
        "corrlen_psi": {"x": 1.0, "y": 1.0, "time": 0.35},
        "corrlen_chi": {"x": 1.0, "y": 1.0, "time": 0.35},
        "err": 0.02,
        "repeats": 3,
    },
    "helmholtz_xytime_large": {
        "kind": "helmholtz",
        "dims": ("x", "y", "time"),
        "n_obs": 900,
        "target_shape": (3, 64, 64),
        "corrlen_psi": {"x": 1.0, "y": 1.0, "time": 0.35},
        "corrlen_chi": {"x": 1.0, "y": 1.0, "time": 0.35},
        "err": 0.02,
        "repeats": 2,
    },
}


def _n_target(cfg):
    return int(np.prod(cfg["target_shape"]))


def _dense_memory_estimate_mb(cfg):
    """Estimate dense NumPy matrix memory for the dominant arrays."""
    n_obs = cfg["n_obs"]
    n_target = _n_target(cfg)
    bytes_per_float = 8

    if cfg["kind"] == "scalar":
        # A: (n_obs, n_obs), C: (n_target, n_obs)
        a_bytes = n_obs * n_obs * bytes_per_float
        c_bytes = n_target * n_obs * bytes_per_float
        return {
            "A_MB": a_bytes / 1e6,
            "cross_MB": c_bytes / 1e6,
            "total_MB": (a_bytes + c_bytes) / 1e6,
        }

    if cfg["kind"] == "helmholtz":
        # A: (2n_obs, 2n_obs), P_psi/P_chi: (n_target, 2n_obs)
        a_bytes = (2 * n_obs) * (2 * n_obs) * bytes_per_float
        p_bytes = n_target * (2 * n_obs) * bytes_per_float
        return {
            "A_MB": a_bytes / 1e6,
            "cross_MB": p_bytes / 1e6,
            "total_MB": (a_bytes + 2 * p_bytes) / 1e6,
        }

    raise ValueError(f"Unknown workload kind {cfg['kind']!r}")


def _make_scalar_workload(name, seed):
    import xobjmap  # noqa: F401
    import xarray as xr

    cfg = WORKLOADS[name]
    rng = np.random.default_rng(seed)

    if cfg["dims"] == ("x", "y"):
        n_obs = cfg["n_obs"]
        x_obs = rng.uniform(-4.0, 4.0, n_obs)
        y_obs = rng.uniform(-4.0, 4.0, n_obs)
        temp = np.sin(1.7 * x_obs) + 0.35 * np.cos(0.9 * y_obs) + 0.1 * x_obs * y_obs
        obs = xr.Dataset(
            {"temp": ("obs", temp)},
            coords={"x": ("obs", x_obs), "y": ("obs", y_obs)},
        )
        ny, nx = cfg["target_shape"]
        target = xr.Dataset(
            coords={
                "x": np.linspace(-3.5, 3.5, nx),
                "y": np.linspace(-3.0, 3.0, ny),
            }
        )
        kwargs = {
            "corrlen": cfg["corrlen"],
            "err": cfg["err"],
            "interp_dims": cfg["dims"],
        }
        return obs, target, kwargs

    if cfg["dims"] == ("x", "y", "time"):
        n_obs = cfg["n_obs"]
        x_obs = rng.uniform(-4.0, 4.0, n_obs)
        y_obs = rng.uniform(-4.0, 4.0, n_obs)
        t_obs = rng.choice(np.array([0.0, 0.5, 1.0]), size=n_obs)
        temp = (
            np.sin(1.7 * x_obs)
            + 0.35 * np.cos(0.9 * y_obs)
            + (1.0 + 0.4 * t_obs) * np.sin(0.6 * x_obs) * np.cos(0.7 * y_obs)
        )
        obs = xr.Dataset(
            {"temp": ("obs", temp)},
            coords={
                "x": ("obs", x_obs),
                "y": ("obs", y_obs),
                "time": ("obs", t_obs),
            },
        )
        nt, ny, nx = cfg["target_shape"]
        x = xr.DataArray(np.linspace(-3.5, 3.5, nx), dims="x")
        y = xr.DataArray(np.linspace(-3.0, 3.0, ny), dims="y")
        t = xr.DataArray(np.linspace(0.0, 1.0, nt), dims="time")
        t3, y3, x3 = xr.broadcast(t, y, x)
        target = xr.Dataset(coords={"x": x3, "y": y3, "time": t3})
        kwargs = {
            "corrlen": cfg["corrlen"],
            "err": cfg["err"],
            "interp_dims": cfg["dims"],
        }
        return obs, target, kwargs

    raise ValueError(f"Unknown scalar workload {name!r}")


def _make_helmholtz_workload(name, seed):
    import xobjmap  # noqa: F401
    import xarray as xr

    cfg = WORKLOADS[name]
    rng = np.random.default_rng(seed)

    if cfg["dims"] == ("x", "y"):
        n_obs = cfg["n_obs"]
        x_obs = rng.uniform(-3.5, 3.5, n_obs)
        y_obs = rng.uniform(-3.5, 3.5, n_obs)
        psi = np.sin(x_obs) * np.cos(y_obs)
        chi = 0.5 * np.cos(0.7 * x_obs) * np.sin(1.1 * y_obs)
        u = np.sin(x_obs) * np.sin(y_obs) - 0.35 * np.sin(0.7 * x_obs) * np.sin(1.1 * y_obs)
        v = np.cos(x_obs) * np.cos(y_obs) + 0.55 * np.cos(0.7 * x_obs) * np.cos(1.1 * y_obs)
        obs = xr.Dataset(
            {"u": ("obs", u), "v": ("obs", v)},
            coords={"x": ("obs", x_obs), "y": ("obs", y_obs)},
        )
        ny, nx = cfg["target_shape"]
        target = xr.Dataset(
            coords={
                "x": np.linspace(-3.0, 3.0, nx),
                "y": np.linspace(-3.0, 3.0, ny),
            }
        )
        kwargs = {
            "corrlen_psi": cfg["corrlen_psi"],
            "corrlen_chi": cfg["corrlen_chi"],
            "err": cfg["err"],
            "derivative_dims": ("x", "y"),
            "interp_dims": cfg["dims"],
        }
        return obs, target, kwargs

    if cfg["dims"] == ("x", "y", "time"):
        n_obs = cfg["n_obs"]
        x_obs = rng.uniform(-3.5, 3.5, n_obs)
        y_obs = rng.uniform(-3.5, 3.5, n_obs)
        t_obs = rng.choice(np.array([0.0, 1.0]), size=n_obs)
        amp = 1.0 + 0.35 * t_obs
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
                "time": ("obs", t_obs),
            },
        )
        nt, ny, nx = cfg["target_shape"]
        x = xr.DataArray(np.linspace(-3.0, 3.0, nx), dims="x")
        y = xr.DataArray(np.linspace(-3.0, 3.0, ny), dims="y")
        t = xr.DataArray(np.linspace(0.0, 1.0, nt), dims="time")
        t3, y3, x3 = xr.broadcast(t, y, x)
        target = xr.Dataset(coords={"x": x3, "y": y3, "time": t3})
        kwargs = {
            "corrlen_psi": cfg["corrlen_psi"],
            "corrlen_chi": cfg["corrlen_chi"],
            "err": cfg["err"],
            "derivative_dims": ("x", "y"),
            "interp_dims": cfg["dims"],
        }
        return obs, target, kwargs

    raise ValueError(f"Unknown helmholtz workload {name!r}")


def _result_arrays(kind, result):
    if kind == "scalar":
        return {
            "field": np.asarray(result.temp.values),
            "error": np.asarray(result.error.values),
        }
    if kind == "helmholtz":
        return {
            "psi": np.asarray(result.psi.values),
            "chi": np.asarray(result.chi.values),
            "psi_error": np.asarray(result.psi_error.values),
            "chi_error": np.asarray(result.chi_error.values),
        }
    raise ValueError(f"Unknown kind {kind!r}")


def _run_workload(name, backend, force_gpu):
    import xarray as xr
    import xobjmap

    cfg = WORKLOADS[name]
    seed = 42
    if cfg["kind"] == "scalar":
        obs, target, kwargs = _make_scalar_workload(name, seed)
        runner = lambda: obs.xobjmap.scalar("temp", target, backend=backend, **kwargs)
    elif cfg["kind"] == "helmholtz":
        obs, target, kwargs = _make_helmholtz_workload(name, seed)
        runner = lambda: obs.xobjmap.helmholtz("u", "v", target, backend=backend, **kwargs)
    else:
        raise ValueError(f"Unknown workload kind {cfg['kind']!r}")

    import tracemalloc

    gpu_before = None
    gpu_peak_mb = None
    compile_time = 0.0
    runtime_backend = None
    runtime_devices = None

    if backend == "jax":
        import jax

        runtime_backend = jax.default_backend()
        runtime_devices = [str(device) for device in jax.devices()]
        if force_gpu and runtime_backend != "gpu":
            raise RuntimeError(f"Expected GPU backend, got {runtime_backend!r}.")
        jax.clear_caches()
        stats = jax.devices()[0].memory_stats()
        if stats:
            gpu_before = stats.get("bytes_in_use", 0)

    tracemalloc.start()

    start_compile = time.perf_counter()
    warm = runner()
    if backend == "jax":
        import jax

        jax.block_until_ready(_result_arrays(cfg["kind"], warm))
    compile_time = time.perf_counter() - start_compile

    times = []
    result = None
    for _ in range(cfg["repeats"]):
        start = time.perf_counter()
        result = runner()
        if backend == "jax":
            import jax

            jax.block_until_ready(_result_arrays(cfg["kind"], result))
        times.append(time.perf_counter() - start)

    _, peak_tracemalloc = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    ru = resource.getrusage(resource.RUSAGE_SELF)
    maxrss_mb = ru.ru_maxrss / 1024.0

    if backend == "jax" and gpu_before is not None:
        import jax

        stats = jax.devices()[0].memory_stats()
        if stats:
            peak = stats.get("peak_bytes_in_use", stats.get("bytes_in_use", 0))
            gpu_peak_mb = peak / 1e6

    return {
        "workload": name,
        "backend": backend,
        "runtime_backend": runtime_backend,
        "runtime_devices": runtime_devices,
        "compile_time_s": compile_time,
        "median_time_s": float(np.median(times)),
        "min_time_s": float(np.min(times)),
        "max_time_s": float(np.max(times)),
        "peak_python_mem_mb": peak_tracemalloc / 1e6,
        "peak_rss_mb": maxrss_mb,
        "peak_gpu_mem_mb": gpu_peak_mb,
        "arrays": _result_arrays(cfg["kind"], result),
    }


def _worker_main(args):
    os.environ.pop("JAX_PLATFORM_NAME", None)
    os.environ.pop("JAX_BACKEND_TARGET", None)
    if args.force_gpu:
        os.environ["JAX_PLATFORMS"] = "cuda"
    else:
        os.environ.pop("JAX_PLATFORMS", None)

    result = _run_workload(args.workload, args.backend, args.force_gpu)
    np.savez_compressed(args.output, **result["arrays"])
    payload = {k: v for k, v in result.items() if k != "arrays"}
    print(json.dumps(payload))


def _run_worker(workload, backend, force_gpu):
    with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as tmp:
        output_path = Path(tmp.name)

    cmd = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--worker",
        "--workload",
        workload,
        "--backend",
        backend,
        "--output",
        str(output_path),
    ]
    if force_gpu:
        cmd.append("--force-gpu")

    env = os.environ.copy()
    env.pop("JAX_PLATFORM_NAME", None)
    env.pop("JAX_BACKEND_TARGET", None)
    if force_gpu:
        env["JAX_PLATFORMS"] = "cuda"
    else:
        env.pop("JAX_PLATFORMS", None)

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
    payload = json.loads(completed.stdout.strip().splitlines()[-1])
    arrays = dict(np.load(output_path))
    output_path.unlink(missing_ok=True)
    return payload, arrays


def _diff_metrics(arrays_np, arrays_jax):
    metrics = {}
    for key in arrays_np:
        a = np.asarray(arrays_np[key], dtype=float)
        b = np.asarray(arrays_jax[key], dtype=float)
        diff = a - b
        denom = np.sqrt(np.mean(a ** 2))
        metrics[key] = {
            "max_abs": float(np.max(np.abs(diff))),
            "rmse": float(np.sqrt(np.mean(diff ** 2))),
            "rel_rmse": float(np.sqrt(np.mean(diff ** 2)) / denom) if denom > 0 else 0.0,
        }
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--workload", choices=sorted(WORKLOADS))
    parser.add_argument("--backend", choices=("numpy", "jax"))
    parser.add_argument("--output")
    parser.add_argument("--force-gpu", action="store_true")
    parser.add_argument(
        "--workloads",
        nargs="+",
        choices=sorted(WORKLOADS),
        help="Subset of workloads to run in top-level benchmark mode.",
    )
    args = parser.parse_args()

    if args.worker:
        _worker_main(args)
        return

    header = (
        f"{'workload':<18} {'backend':<6} {'runtime':<8} {'compile(s)':>10} {'median(s)':>10} "
        f"{'RSS MB':>10} {'Py MB':>10} {'GPU MB':>10}"
    )
    print(header)
    print("-" * len(header))

    workloads = args.workloads or list(WORKLOADS)

    for workload in workloads:
        cfg = WORKLOADS[workload]
        mem = _dense_memory_estimate_mb(cfg)
        print(
            f"# {workload}: kind={cfg['kind']}, n_obs={cfg['n_obs']}, "
            f"n_target={_n_target(cfg)}, dense_numpy≈{mem['total_MB'] / 1024:.2f} GiB "
            f"(A={mem['A_MB'] / 1024:.2f} GiB, cross={mem['cross_MB'] / 1024:.2f} GiB)",
            flush=True,
        )
        results = []
        for backend, force_gpu in (("jax", args.force_gpu), ("numpy", False)):
            stats, arrays = _run_worker(workload, backend, force_gpu=force_gpu)
            results.append((stats, arrays))
            gpu_str = "—" if stats["peak_gpu_mem_mb"] is None else f"{stats['peak_gpu_mem_mb']:.1f}"
            print(
                f"{stats['workload']:<18} {stats['backend']:<6} "
                f"{(stats['runtime_backend'] or '—'):<8} "
                f"{stats['compile_time_s']:>10.3f} {stats['median_time_s']:>10.3f} "
                f"{stats['peak_rss_mb']:>10.1f} {stats['peak_python_mem_mb']:>10.1f} "
                f"{gpu_str:>10}",
                flush=True,
            )
            if stats["runtime_devices"]:
                print(f"  devices: {', '.join(stats['runtime_devices'])}", flush=True)

        jax_stats, jax_arrays = results[0]
        numpy_stats, numpy_arrays = results[1]
        diff = _diff_metrics(numpy_arrays, jax_arrays)
        for field, metrics in diff.items():
            print(
                f"  {field:<12} max|Δ|={metrics['max_abs']:.4e}  "
                f"rmse={metrics['rmse']:.4e}  rel_rmse={metrics['rel_rmse']:.4e}",
                flush=True,
            )
        print("", flush=True)


if __name__ == "__main__":
    main()
