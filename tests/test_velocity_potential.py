"""Tests for velocity potential recovery."""

import numpy as np

from xobjmap import velocity_potential


def test_velocity_potential_returns_correct_shape():
    """velocity_potential output shape should match the target grid."""
    gx, gy = np.meshgrid(np.linspace(0, 10, 8), np.linspace(0, 10, 6))

    rng = np.random.default_rng(42)
    x = rng.uniform(1, 9, 20)
    y = rng.uniform(1, 9, 20)
    u = rng.standard_normal(20)
    v = rng.standard_normal(20)

    chi = velocity_potential(gx, gy, x, y, u, v, corrlenx=3.0, corrleny=3.0, err=0.1)

    assert chi.shape == gx.shape


def test_velocity_potential_does_not_modify_inputs():
    """velocity_potential should not modify the input arrays."""
    gx, gy = np.meshgrid(np.linspace(0, 10, 5), np.linspace(0, 10, 5))
    x = np.array([2.0, 5.0, 8.0])
    y = np.array([2.0, 5.0, 8.0])
    u = np.array([1.0, 0.0, -1.0])
    v = np.array([0.0, 1.0, 0.0])

    gx_orig, x_orig, u_orig = gx.copy(), x.copy(), u.copy()

    velocity_potential(gx, gy, x, y, u, v, corrlenx=3.0, corrleny=1.5, err=0.1)

    np.testing.assert_array_equal(gx, gx_orig)
    np.testing.assert_array_equal(x, x_orig)
    np.testing.assert_array_equal(u, u_orig)


def test_velocity_potential_recovers_source():
    """Velocities derived from reconstructed chi should match the true field."""
    # Gaussian source: chi = exp(-r^2/2L^2)
    # u = dchi/dx = -(x/L^2) * chi
    # v = dchi/dy = -(y/L^2) * chi
    L = 3.0
    rng = np.random.default_rng(42)
    x_obs = rng.uniform(-8, 8, 60)
    y_obs = rng.uniform(-8, 8, 60)
    r2 = x_obs**2 + y_obs**2
    chi_obs = np.exp(-r2 / (2 * L**2))
    u_obs = -(x_obs / L**2) * chi_obs
    v_obs = -(y_obs / L**2) * chi_obs

    gx, gy = np.meshgrid(np.linspace(-6, 6, 20), np.linspace(-6, 6, 20))

    chi_recon = velocity_potential(gx, gy, x_obs, y_obs, u_obs, v_obs,
                                   corrlenx=4.0, corrleny=4.0, err=0.01)

    # True velocities on grid
    r2_grid = gx**2 + gy**2
    chi_true = np.exp(-r2_grid / (2 * L**2))
    u_true = -(gx / L**2) * chi_true
    v_true = -(gy / L**2) * chi_true

    # Reconstructed velocities from chi via finite differences
    dy = gy[1, 0] - gy[0, 0]
    dx = gx[0, 1] - gx[0, 0]
    dchi_dy, dchi_dx = np.gradient(chi_recon, dy, dx)
    u_recon = dchi_dx
    v_recon = dchi_dy

    nrmse_u = np.sqrt(np.mean((u_recon - u_true)**2)) / np.sqrt(np.mean(u_true**2))
    nrmse_v = np.sqrt(np.mean((v_recon - v_true)**2)) / np.sqrt(np.mean(v_true**2))
    assert nrmse_u < 0.3, f"u nRMSE = {nrmse_u:.3f}"
    assert nrmse_v < 0.3, f"v nRMSE = {nrmse_v:.3f}"
