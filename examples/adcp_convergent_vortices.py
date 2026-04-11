"""
ADCP transects across a simple Helmholtz flow
=============================================

Builds a synthetic ADCP sampling problem from a theoretical velocity field
with both:

- a non-divergent Gaussian vortex represented by the streamfunction ``psi``
- a non-rotational Gaussian convergent feature represented by the velocity
  potential ``chi``

It compares the classic Bretherton direct solve (``backend="numpy"``) and,
when available, the matrix-free JAX backend for:

- scalar objective mapping of current speed
- Helmholtz recovery of the velocity potential ``chi``

Run with:

    pixi run -e docs python examples/adcp_convergent_vortices.py
"""

from pathlib import Path
import os

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

import xobjmap
import jax  # noqa: F401


RNG = np.random.default_rng(42)
OUTFILE = Path("docs/example_adcp_convergent_vortices.png")
SPEED_ERROR_THRESHOLD = 0.10
CHI_ERROR_THRESHOLD = 0.35
PSI_ERROR_THRESHOLD = 0.35


def gaussian_streamfunction(x, y, x0, y0, scale, amp):
    """Gaussian streamfunction and its non-divergent velocity."""
    dx = x - x0
    dy = y - y0
    psi = amp * np.exp(-(dx**2 + dy**2) / (2 * scale**2))
    u = (dy / scale**2) * psi
    v = -(dx / scale**2) * psi
    return psi, u, v


def gaussian_velocity_potential(x, y, x0, y0, scale, amp):
    """Gaussian velocity potential and its irrotational velocity."""
    dx = x - x0
    dy = y - y0
    chi = amp * np.exp(-(dx**2 + dy**2) / (2 * scale**2))
    u = -(dx / scale**2) * chi
    v = -(dy / scale**2) * chi
    return chi, u, v


def build_transects():
    """Synthetic ship track with survey-style zig-zag transects."""
    waypoints = np.array(
        [
            [-180.0, 180.0],
            [180.0, 180.0],
            [180.0, 150.0],
            [-180.0, 150.0],
            [-180.0, 120.0],
            [180.0, 120.0],
            [180.0, 90.0],
            [-180.0, 90.0],
            [-180.0, 60.0],
            [180.0, 60.0],
            [180.0, 30.0],
            [-180.0, 30.0],
            [-180.0, 0.0],
            [180.0, 0.0],
            [180.0, -30.0],
            [-180.0, -30.0],
            [-180.0, -60.0],
            [180.0, -60.0],
            [180.0, -90.0],
            [-180.0, -90.0],
            [-180.0, -120.0],
            [180.0, -120.0],
            [180.0, -150.0],
            [-180.0, -150.0],
            [-180.0, -180.0],
            [180.0, -180.0],
        ]
    )

    pts = []
    segment_points = 16
    for start, end in zip(waypoints[:-1], waypoints[1:]):
        x_seg = np.linspace(start[0], end[0], segment_points, endpoint=False)
        y_seg = np.linspace(start[1], end[1], segment_points, endpoint=False)
        pts.append(np.column_stack([x_seg, y_seg]))
    pts.append(waypoints[-1][None, :])

    xy = np.vstack(pts)
    return xy[:, 0], xy[:, 1]


def synthesize_flow(x, y):
    """Combined theoretical field with rotational and divergent parts."""
    psi, u_psi, v_psi = gaussian_streamfunction(
        x, y, x0=-80.0, y0=60.0, scale=80.0, amp=42.0
    )
    chi, u_chi, v_chi = gaussian_velocity_potential(
        x, y, x0=95.0, y0=-70.0, scale=95.0, amp=52.0
    )
    return psi, chi, u_psi + u_chi, v_psi + v_chi


def demean(field):
    """Remove the arbitrary constant offset from psi or chi."""
    return field - field.mean()


def rmse(a, b):
    return np.sqrt(np.mean((np.asarray(a) - np.asarray(b)) ** 2))


def masked_rmse(a, b):
    mask = np.isfinite(np.asarray(a)) & np.isfinite(np.asarray(b))
    if not np.any(mask):
        return np.nan
    return np.sqrt(np.mean((np.asarray(a)[mask] - np.asarray(b)[mask]) ** 2))


def main():
    x_obs, y_obs = build_transects()
    _, _, u_true_obs, v_true_obs = synthesize_flow(x_obs, y_obs)
    u_obs = u_true_obs + RNG.normal(0, 0.015, size=u_true_obs.shape)
    v_obs = v_true_obs + RNG.normal(0, 0.015, size=v_true_obs.shape)
    speed_obs = np.sqrt(u_obs**2 + v_obs**2)

    obs = xr.Dataset(
        {
            "u": ("obs", u_obs),
            "v": ("obs", v_obs),
            "speed": ("obs", speed_obs),
        },
        coords={"x": ("obs", x_obs), "y": ("obs", y_obs)},
    )
    target = xr.Dataset(
        coords={
            "x": np.linspace(-260, 260, 61),
            "y": np.linspace(-260, 260, 61),
        }
    )

    xg, yg = np.meshgrid(target.x.values, target.y.values)
    psi_true_grid, chi_true_grid, _, _ = synthesize_flow(xg, yg)
    _, _, u_true_grid, v_true_grid = synthesize_flow(xg, yg)
    speed_true_grid = np.sqrt(u_true_grid**2 + v_true_grid**2)

    speed_np = obs.xobjmap.scalar(
        "speed",
        target,
        corrlen={"x": 80.0, "y": 80.0},
        err=0.03,
        backend="numpy",
    )

    helm_np = obs.xobjmap.helmholtz(
        "u",
        "v",
        target,
        corrlen_psi={"x": 90.0, "y": 90.0},
        corrlen_chi={"x": 100.0, "y": 100.0},
        err=0.03,
        backend="numpy",
    )

    psi_true = demean(psi_true_grid)
    chi_true = demean(chi_true_grid)
    psi_np = demean(helm_np.psi.values)
    chi_np = demean(helm_np.chi.values)
    psi_np_masked = xr.DataArray(
        psi_np, dims=("y", "x"), coords=target.coords
    ).where(helm_np.psi_error <= PSI_ERROR_THRESHOLD)
    chi_np_masked = xr.DataArray(
        chi_np, dims=("y", "x"), coords=target.coords
    ).where(helm_np.chi_error <= CHI_ERROR_THRESHOLD)

    speed_jax = obs.xobjmap.scalar(
        "speed",
        target,
        corrlen={"x": 80.0, "y": 80.0},
        err=0.03,
        backend="jax",
    )
    helm_jax = obs.xobjmap.helmholtz(
        "u",
        "v",
        target,
        corrlen_psi={"x": 90.0, "y": 90.0},
        corrlen_chi={"x": 100.0, "y": 100.0},
        err=0.03,
        backend="jax",
    )
    psi_jax = demean(helm_jax.psi.values)
    chi_jax = demean(helm_jax.chi.values)
    psi_jax_masked = xr.DataArray(
        psi_jax, dims=("y", "x"), coords=target.coords
    ).where(helm_jax.psi_error <= PSI_ERROR_THRESHOLD)
    chi_jax_masked = xr.DataArray(
        chi_jax, dims=("y", "x"), coords=target.coords
    ).where(helm_jax.chi_error <= CHI_ERROR_THRESHOLD)

    speed_np_masked = speed_np.speed.where(speed_np.error <= SPEED_ERROR_THRESHOLD)
    psi_np_error = xr.DataArray(helm_np.psi_error, dims=("y", "x"), coords=target.coords)
    chi_np_error = xr.DataArray(helm_np.chi_error, dims=("y", "x"), coords=target.coords)
    speed_np_error = speed_np.error
    speed_jax_masked = speed_jax.speed.where(speed_jax.error <= SPEED_ERROR_THRESHOLD)
    speed_jax_error = speed_jax.error
    chi_jax_error = xr.DataArray(helm_jax.chi_error, dims=("y", "x"), coords=target.coords)
    psi_jax_error = xr.DataArray(helm_jax.psi_error, dims=("y", "x"), coords=target.coords)

    fig, axes = plt.subplots(
        4, 3, figsize=(12.5, 13.4), constrained_layout=True
    )
    lims = dict(xlim=(-260, 260), ylim=(-260, 260))

    tracer_vmin = min(
        float(speed_true_grid.min()),
        float(speed_np_masked.min(skipna=True)),
    )
    tracer_vmax = max(
        float(speed_true_grid.max()),
        float(speed_np_masked.max(skipna=True)),
    )
    tracer_vmin = min(tracer_vmin, float(speed_jax_masked.min(skipna=True)))
    tracer_vmax = max(tracer_vmax, float(speed_jax_masked.max(skipna=True)))
    tracer_kwargs = dict(
        cmap="viridis",
        vmin=tracer_vmin,
        vmax=tracer_vmax,
        add_colorbar=False,
    )

    chi_vlim = max(
        np.abs(chi_true).max(),
        np.abs(chi_np_masked).max(skipna=True),
    )
    chi_vlim = max(chi_vlim, np.abs(chi_jax_masked).max(skipna=True))
    chi_kwargs = dict(
        cmap="RdBu_r",
        vmin=-chi_vlim,
        vmax=chi_vlim,
        add_colorbar=False,
    )

    speed_rel_err = np.abs(speed_np_masked - speed_true_grid) / np.abs(speed_true_grid).max()
    chi_rel_err = np.abs(chi_np_masked - chi_true) / np.abs(chi_true).max()
    psi_rel_err = np.abs(psi_np_masked - psi_true) / np.abs(psi_true).max()
    speed_jax_rel_err = np.abs(speed_jax_masked - speed_true_grid) / np.abs(speed_true_grid).max()
    chi_jax_rel_err = np.abs(chi_jax_masked - chi_true) / np.abs(chi_true).max()
    psi_jax_rel_err = np.abs(psi_jax_masked - psi_true) / np.abs(psi_true).max()
    rel_err_vmax = max(
        float(speed_rel_err.max(skipna=True)),
        float(chi_rel_err.max(skipna=True)),
        float(psi_rel_err.max(skipna=True)),
        float(speed_jax_rel_err.max(skipna=True)),
        float(chi_jax_rel_err.max(skipna=True)),
        float(psi_jax_rel_err.max(skipna=True)),
    )
    rel_err_kwargs = dict(
        cmap="magma",
        vmin=0.0,
        vmax=rel_err_vmax,
        add_colorbar=False,
    )

    speed_true_plot = xr.DataArray(
        speed_true_grid, dims=("y", "x"), coords=target.coords
    ).plot(
        ax=axes[0, 0], **tracer_kwargs
    )
    axes[0, 0].scatter(
        x_obs,
        y_obs,
        c=speed_obs,
        s=10,
        cmap=tracer_kwargs["cmap"],
        vmin=tracer_vmin,
        vmax=tracer_vmax,
        edgecolors="k",
        linewidths=0.15,
    )
    axes[0, 0].set_title("True speed + ship survey")
    axes[0, 0].set_aspect("equal")
    axes[0, 0].set(ylabel="y (km)", **lims)

    chi_true_plot = xr.DataArray(
        chi_true, dims=("y", "x"), coords=target.coords
    ).plot(
        ax=axes[1, 0], **chi_kwargs
    )
    axes[1, 0].contour(
        xg, yg, psi_true, levels=12, colors="k", linewidths=0.35, alpha=0.45
    )
    axes[1, 0].set_title(r"True $\chi$ with $\psi$ contours")
    axes[1, 0].set_aspect("equal")
    axes[1, 0].set(xlabel="x (km)", ylabel="y (km)", **lims)

    speed_np_masked.plot(ax=axes[0, 1], **tracer_kwargs)
    axes[0, 1].scatter(x_obs, y_obs, c="k", s=4, alpha=0.35)
    axes[0, 1].set_title(
        f"Bretherton speed\nRMSE={masked_rmse(speed_np_masked, speed_true_grid):.3f}"
    )
    axes[0, 1].set_aspect("equal")
    axes[0, 1].set(**lims)

    chi_np_masked.plot(ax=axes[1, 1], **chi_kwargs)
    axes[1, 1].contour(
        target.x,
        target.y,
        np.ma.masked_invalid(psi_np_masked.values),
        levels=12,
        colors="k",
        linewidths=0.35,
        alpha=0.45,
    )
    axes[1, 1].set_title(
        f"Bretherton $\\chi$ with $\\psi$\nRMSE={masked_rmse(chi_np_masked, chi_true):.3f}"
    )
    axes[1, 1].set_aspect("equal")
    axes[1, 1].set(xlabel="x (km)", **lims)

    speed_jax_masked.plot(ax=axes[0, 2], **tracer_kwargs)
    axes[0, 2].scatter(x_obs, y_obs, c="k", s=4, alpha=0.35)
    axes[0, 2].set_title(
        f"JAX speed\nRMSE={masked_rmse(speed_jax_masked, speed_true_grid):.3f}"
    )
    axes[0, 2].set_aspect("equal")
    axes[0, 2].set(**lims)

    chi_jax_masked.plot(ax=axes[1, 2], **chi_kwargs)
    axes[1, 2].contour(
        target.x,
        target.y,
        np.ma.masked_invalid(psi_jax_masked.values),
        levels=12,
        colors="k",
        linewidths=0.35,
        alpha=0.45,
    )
    axes[1, 2].set_title(
        f"JAX $\\chi$ with $\\psi$\nRMSE={masked_rmse(chi_jax_masked, chi_true):.3f}"
    )
    axes[1, 2].set_aspect("equal")
    axes[1, 2].set(xlabel="x (km)", **lims)

    rel_err_plot = speed_rel_err.plot(ax=axes[2, 0], **rel_err_kwargs)
    axes[2, 0].set_title("Bretherton speed rel. error")
    axes[2, 0].set_aspect("equal")
    axes[2, 0].set(ylabel="y (km)", **lims)

    chi_rel_err.plot(ax=axes[2, 1], **rel_err_kwargs)
    axes[2, 1].set_title(r"Bretherton $\chi$ rel. error")
    axes[2, 1].set_aspect("equal")
    axes[2, 1].set(**lims)

    psi_rel_err.plot(ax=axes[2, 2], **rel_err_kwargs)
    axes[2, 2].set_title(r"Bretherton $\psi$ rel. error")
    axes[2, 2].set_aspect("equal")
    axes[2, 2].set(**lims)

    jax_rel_err_plot = speed_jax_rel_err.plot(ax=axes[3, 0], **rel_err_kwargs)
    axes[3, 0].set_title("JAX speed rel. error")
    axes[3, 0].set_aspect("equal")
    axes[3, 0].set(xlabel="x (km)", ylabel="y (km)", **lims)

    chi_jax_rel_err.plot(ax=axes[3, 1], **rel_err_kwargs)
    axes[3, 1].set_title(r"JAX $\chi$ rel. error")
    axes[3, 1].set_aspect("equal")
    axes[3, 1].set(xlabel="x (km)", **lims)

    psi_jax_rel_err.plot(ax=axes[3, 2], **rel_err_kwargs)
    axes[3, 2].set_title(r"JAX $\psi$ rel. error")
    axes[3, 2].set_aspect("equal")
    axes[3, 2].set(xlabel="x (km)", **lims)

    for ax in axes.flat:
        ax.axhline(0, color="0.6", lw=0.35, alpha=0.35)
        ax.axvline(0, color="0.6", lw=0.35, alpha=0.35)

    fig.colorbar(speed_true_plot, ax=axes[0, :], shrink=0.82, aspect=30, label="speed")
    fig.colorbar(chi_true_plot, ax=axes[1, :], shrink=0.82, aspect=30, label=r"$\chi$ (demeaned)")
    fig.colorbar(rel_err_plot, ax=axes[2, :], shrink=0.82, aspect=30, label="field-relative error")
    fig.colorbar(jax_rel_err_plot, ax=axes[3, :], shrink=0.82, aspect=30, label="field-relative error")

    OUTFILE.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTFILE, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    main()
