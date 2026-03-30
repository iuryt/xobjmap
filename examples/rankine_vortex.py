"""
Rankine vortex reconstruction from ADCP transects
==================================================

Demonstrates vectorial objective analysis by reconstructing
the streamfunction of a Rankine vortex from synthetic ship
ADCP velocity observations along crossing transects.
"""

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import xobjmap

rng = np.random.default_rng(42)

# %% Rankine vortex definition
R = 100  # core radius (km)
Gamma = 2 * np.pi * R * 0.5  # circulation for v_max ~ 0.5 m/s


def rankine_uv(x, y):
    r = np.maximum(np.sqrt(x**2 + y**2), 1e-10)
    vt = np.where(r < R, Gamma * r / (2 * np.pi * R**2), Gamma / (2 * np.pi * r))
    theta = np.arctan2(y, x)
    return -vt * np.sin(theta), vt * np.cos(theta)


def rankine_psi(x, y):
    """Bretherton convention: u = -dpsi/dy, v = dpsi/dx."""
    r = np.maximum(np.sqrt(x**2 + y**2), 1e-10)
    return np.where(
        r < R,
        Gamma * r**2 / (4 * np.pi * R**2),
        Gamma * np.log(r / R) / (2 * np.pi) + Gamma / (4 * np.pi),
    )


# %% Synthetic ADCP transects (5 ship crossings)
lines = []
for angle, offset in [(10, -100), (80, 30), (150, -30), (40, 70), (110, -50)]:
    t = np.linspace(-300, 300, 40)
    rad = np.radians(angle)
    pts = np.outer(t, [np.cos(rad), np.sin(rad)]) + offset * np.array(
        [-np.sin(rad), np.cos(rad)]
    )
    lines.append(pts)

xy = np.vstack(lines)
x_obs, y_obs = xy[:, 0], xy[:, 1]
u_true, v_true = rankine_uv(x_obs, y_obs)
u_obs = u_true + rng.normal(0, 0.02, len(x_obs))
v_obs = v_true + rng.normal(0, 0.02, len(x_obs))

# %% Reconstruction with xobjmap
obs = xr.Dataset(
    {"u": ("obs", u_obs), "v": ("obs", v_obs)},
    coords={"x": ("obs", x_obs), "y": ("obs", y_obs)},
)
target = xr.Dataset(
    coords={"x": np.linspace(-300, 300, 30), "y": np.linspace(-300, 300, 30)}
)

result = obs.xobjmap.streamfunction(
    "u", "v", target, corrlen={"x": 100, "y": 100}, err=0.02
)

result_spd = obs.assign(
    speed=np.sqrt(obs.u**2 + obs.v**2)
).xobjmap.scalar("speed", target, corrlen={"x": 100, "y": 100}, err=0.05)

# %% Build analytical solution as xarray for comparison
xg, yg = np.meshgrid(target.x.values, target.y.values)
psi_true = xr.DataArray(
    rankine_psi(xg, yg), dims=("y", "x"), coords=result.psi.coords
)

# Remove median (absolute value is arbitrary; median centers the field
# better than mean for the logarithmic tail of the Rankine vortex)
result["psi"] = result.psi - result.psi.median()
psi_true = psi_true - psi_true.median()

# %% Plot
lims = dict(xlim=(-300, 300), ylim=(-300, 300))
cbar_kwargs = dict(shrink=0.75, aspect=20)

fig, axes = plt.subplots(2, 2, figsize=(8, 7), constrained_layout=True)

# ADCP transects
ax = axes[0, 0]
ax.set_title("ADCP transects", fontweight="bold")
spd = np.sqrt(u_obs**2 + v_obs**2)
q = ax.quiver(
    x_obs[::2], y_obs[::2], u_obs[::2], v_obs[::2],
    spd[::2], cmap="inferno", scale=5, width=0.005, clim=[0, 0.55],
)
plt.colorbar(q, ax=ax, label="speed (m/s)", **cbar_kwargs)
ax.set_aspect("equal")
ax.set(xlabel="x (km)", ylabel="y (km)", **lims)

# Shared ψ colorbar limits (symmetric around zero)
vlim = max(abs(psi_true.min().item()), abs(psi_true.max().item()),
           abs(result.psi.min().item()), abs(result.psi.max().item()))
psi_kwargs = dict(levels=25, vmin=-vlim, vmax=vlim, cmap="bwr",
                  cbar_kwargs={**cbar_kwargs, "label": "ψ"})

# Reconstructed ψ
result.psi.plot.contourf(ax=axes[0, 1], **psi_kwargs)
result.psi.plot.contour(ax=axes[0, 1], levels=25, colors="k", linewidths=0.3, alpha=0.4)
axes[0, 1].set_title("Reconstructed ψ", fontweight="bold")
axes[0, 1].set_aspect("equal")
axes[0, 1].set(**lims)

# True ψ
psi_true.plot.contourf(ax=axes[1, 0], **psi_kwargs)
psi_true.plot.contour(ax=axes[1, 0], levels=25, colors="k", linewidths=0.3, alpha=0.4)
axes[1, 0].set_title("True ψ (analytical)", fontweight="bold")
axes[1, 0].set_aspect("equal")
axes[1, 0].set(**lims)

# Interpolation error
err_levels = np.arange(0, 1.1, 0.1)
result_spd.error.plot.contourf(
    ax=axes[1, 1], levels=err_levels, vmin=0, vmax=1, cmap="Reds",
    cbar_kwargs={**cbar_kwargs, "label": "normalized error", "ticks": err_levels},
)
axes[1, 1].scatter(x_obs, y_obs, c="k", s=3, alpha=0.4, zorder=5)
axes[1, 1].set_title("Interpolation error", fontweight="bold")
axes[1, 1].set_aspect("equal")
axes[1, 1].set(**lims)

plt.savefig("docs/example_vortex.png", dpi=150, bbox_inches="tight", facecolor="white")
plt.close("all")
