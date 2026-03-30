"""Tests for Helmholtz decomposition."""

import numpy as np

from xobjmap import helmholtz


def test_helmholtz_returns_correct_shapes():
    """helmholtz should return psi and chi matching the target grid shape."""
    gx, gy = np.meshgrid(np.linspace(0, 10, 8), np.linspace(0, 10, 6))

    rng = np.random.default_rng(42)
    x = rng.uniform(1, 9, 20)
    y = rng.uniform(1, 9, 20)
    u = rng.standard_normal(20)
    v = rng.standard_normal(20)

    psi, chi = helmholtz(
        gx, gy, x, y, u, v,
        corrlenx_psi=3.0, corrleny_psi=3.0,
        corrlenx_chi=3.0, corrleny_chi=3.0,
        err=0.1,
    )

    assert psi.shape == gx.shape
    assert chi.shape == gx.shape


def test_helmholtz_does_not_modify_inputs():
    """helmholtz should not modify the input arrays."""
    gx, gy = np.meshgrid(np.linspace(0, 10, 5), np.linspace(0, 10, 5))
    x = np.array([2.0, 5.0, 8.0])
    y = np.array([2.0, 5.0, 8.0])
    u = np.array([1.0, 0.0, -1.0])
    v = np.array([0.0, 1.0, 0.0])

    gx_orig, x_orig, u_orig = gx.copy(), x.copy(), u.copy()

    helmholtz(
        gx, gy, x, y, u, v,
        corrlenx_psi=3.0, corrleny_psi=1.5,
        corrlenx_chi=2.0, corrleny_chi=1.0,
        err=0.1,
    )

    np.testing.assert_array_equal(gx, gx_orig)
    np.testing.assert_array_equal(x, x_orig)
    np.testing.assert_array_equal(u, u_orig)


def test_helmholtz_recovers_psi_and_chi():
    """helmholtz should recover both streamfunction and velocity potential."""
    # Gaussian vortex (nondivergent): psi = exp(-r^2 / 2L^2)
    #   u_psi = -dpsi/dy = (y/L^2) * psi
    #   v_psi =  dpsi/dx = -(x/L^2) * psi
    #
    # Gaussian source (irrotational): chi = exp(-r^2 / 2L^2)
    #   u_chi = dchi/dx = -(x/L^2) * chi
    #   v_chi = dchi/dy = -(y/L^2) * chi
    L = 3.0
    rng = np.random.default_rng(42)
    x_obs = rng.uniform(-8, 8, 100)
    y_obs = rng.uniform(-8, 8, 100)

    r2 = x_obs**2 + y_obs**2
    psi_obs = np.exp(-r2 / (2 * L**2))
    chi_obs = np.exp(-r2 / (2 * L**2))

    u_obs = (y_obs / L**2) * psi_obs + (-x_obs / L**2) * chi_obs
    v_obs = (-x_obs / L**2) * psi_obs + (-y_obs / L**2) * chi_obs

    gx, gy = np.meshgrid(np.linspace(-6, 6, 20), np.linspace(-6, 6, 20))

    psi_recon, chi_recon = helmholtz(
        gx, gy, x_obs, y_obs, u_obs, v_obs,
        corrlenx_psi=4.0, corrleny_psi=4.0,
        corrlenx_chi=4.0, corrleny_chi=4.0,
        err=0.01,
    )

    # True fields on grid
    r2_grid = gx**2 + gy**2
    psi_true = np.exp(-r2_grid / (2 * L**2))
    chi_true = np.exp(-r2_grid / (2 * L**2))

    for recon, true in [(psi_recon, psi_true), (chi_recon, chi_true)]:
        # Remove mean (absolute value is arbitrary) and normalize
        recon_n = recon - recon.mean()
        true_n = true - true.mean()
        recon_n = recon_n / np.abs(recon_n).max()
        true_n = true_n / np.abs(true_n).max()

        corr = np.corrcoef(recon_n.ravel(), true_n.ravel())[0, 1]
        assert corr > 0.9

        rmse = np.sqrt(np.mean((recon_n - true_n) ** 2))
        assert rmse < 0.3
