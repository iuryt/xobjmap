"""
Core interpolation functions using Gauss-Markov optimal estimation.
"""

import numpy as np


def _pairwise(x1, y1, x2, y2):
    """Pairwise differences and squared distances between two point sets.

    Returns
    -------
    dx : ndarray, shape (n1, n2)
        x2[j] - x1[i] for each pair.
    dy : ndarray, shape (n1, n2)
        y2[j] - y1[i] for each pair.
    d2 : ndarray, shape (n1, n2)
        Squared Euclidean distance for each pair.
    """
    n1, n2 = len(x1), len(x2)
    dx = np.tile(x2, (n1, 1)) - np.tile(x1, (n2, 1)).T
    dy = np.tile(y2, (n1, 1)) - np.tile(y1, (n2, 1)).T
    return dx, dy, dx**2 + dy**2


def _scalar_numpy(xc, yc, x, y, t, corrlenx, corrleny, err):
    """Scalar interpolation via direct solve (numpy)."""
    corrlen = corrleny
    xc = np.asarray(xc) * (corrleny / corrlenx)
    x = np.asarray(x) * (corrleny / corrlenx)
    yc = np.asarray(yc)
    y = np.asarray(y)

    n = len(x)

    _, _, d2 = _pairwise(x, y, x, y)
    _, _, dc2 = _pairwise(xc, yc, x, y)

    A = (1 - err) * np.exp(-d2 / corrlen**2) + err * np.eye(n)
    C = (1 - err) * np.exp(-dc2 / corrlen**2)

    t = np.asarray(t).reshape(n, 1)
    tp = np.dot(C, np.linalg.solve(A, t))
    return tp


def _scalar_jax(xc, yc, x, y, t, corrlenx, corrleny, err):
    """Scalar interpolation via matrix-free CG on GPU (JAX)."""
    import jax
    import jax.numpy as jnp
    from jax.scipy.sparse.linalg import cg

    corrlen = float(corrleny)
    ratio = corrleny / corrlenx
    x = jnp.asarray(x, dtype=jnp.float32) * ratio
    xc = jnp.asarray(xc, dtype=jnp.float32) * ratio
    y = jnp.asarray(y, dtype=jnp.float32)
    yc = jnp.asarray(yc, dtype=jnp.float32)

    n = x.shape[0]
    inv_l2 = 1.0 / (corrlen ** 2)
    one_m_err = 1.0 - err
    maxiter = min(n, 200)

    x_obs, y_obs = x, y

    # ── Chunked matvec for CG ────────────────────────────────
    if n <= 512:
        chunk = n
    else:
        chunk = 512
        rem = n % chunk
        if rem:
            pad = chunk - rem
            x = jnp.concatenate([x, jnp.zeros(pad)])
            y = jnp.concatenate([y, jnp.zeros(pad)])
    n_padded = x.shape[0]
    n_chunks = n_padded // chunk

    def _matvec_body(i, carry):
        v, result = carry
        i0 = i * chunk
        xi = jax.lax.dynamic_slice(x, (i0,), (chunk,))
        yi = jax.lax.dynamic_slice(y, (i0,), (chunk,))
        d2 = (xi[:, None] - x_obs[None, :]) ** 2 + (yi[:, None] - y_obs[None, :]) ** 2
        Kv = one_m_err * (jnp.exp(-d2 * inv_l2) @ v)
        prev = jax.lax.dynamic_slice(result, (i0,), (chunk,))
        result = jax.lax.dynamic_update_slice(result, prev + Kv, (i0,))
        return v, result

    @jax.jit
    def matvec(v):
        result = jnp.zeros(n_padded)
        result = result.at[:n].set(err * v)
        _, result = jax.lax.fori_loop(0, n_chunks, _matvec_body, (v, result))
        return result[:n]

    # ── Cross-covariance kernel sum ───────────────────────────
    @jax.jit
    def _cross_cov_vec(s):
        def _one(args):
            xc_j, yc_j = args
            dc2 = (xc_j - x_obs) ** 2 + (yc_j - y_obs) ** 2
            return one_m_err * jnp.dot(jnp.exp(-dc2 * inv_l2), s)
        return jax.lax.map(_one, (xc, yc))

    t_vec = jnp.asarray(t, dtype=jnp.float32).ravel()
    w, _ = cg(matvec, t_vec, maxiter=maxiter)
    tp = _cross_cov_vec(w).reshape(-1, 1)
    return tp


def _error_numpy(xc, yc, x, y, corrlenx, corrleny, err):
    """Error field via direct solve (numpy)."""
    corrlen = corrleny
    xc = np.asarray(xc) * (corrleny / corrlenx)
    x = np.asarray(x) * (corrleny / corrlenx)
    yc = np.asarray(yc)
    y = np.asarray(y)

    n = len(x)

    _, _, d2 = _pairwise(x, y, x, y)
    _, _, dc2 = _pairwise(xc, yc, x, y)

    A = (1 - err) * np.exp(-d2 / corrlen**2) + err * np.eye(n)
    C = (1 - err) * np.exp(-dc2 / corrlen**2)

    return 1 - np.sum(C.T * np.linalg.solve(A, C.T), axis=0) / (1 - err)


def _error_jax(xc, yc, x, y, corrlenx, corrleny, err, k_local=None):
    """Error field via local neighborhood solve on GPU (JAX)."""
    import jax
    import jax.numpy as jnp

    corrlen = float(corrleny)
    ratio = corrleny / corrlenx
    x = jnp.asarray(x, dtype=jnp.float32) * ratio
    xc = jnp.asarray(xc, dtype=jnp.float32) * ratio
    y = jnp.asarray(y, dtype=jnp.float32)
    yc = jnp.asarray(yc, dtype=jnp.float32)

    n = x.shape[0]
    inv_l2 = 1.0 / (corrlen ** 2)
    one_m_err = 1.0 - err
    k = min(n, k_local if k_local is not None else min(n, 100))

    @jax.jit
    def _local_error(args):
        xc_j, yc_j = args
        d2_all = (xc_j - x) ** 2 + (yc_j - y) ** 2
        _, idx = jax.lax.top_k(-d2_all, k)
        x_k, y_k = x[idx], y[idx]
        d2_kk = (x_k[:, None] - x_k[None, :]) ** 2 + (y_k[:, None] - y_k[None, :]) ** 2
        A_k = one_m_err * jnp.exp(-d2_kk * inv_l2) + err * jnp.eye(k)
        d2_ck = (xc_j - x_k) ** 2 + (yc_j - y_k) ** 2
        c_k = one_m_err * jnp.exp(-d2_ck * inv_l2)
        return 1.0 - jnp.dot(c_k, jnp.linalg.solve(A_k, c_k)) / one_m_err

    return jax.lax.map(_local_error, (xc, yc))


def _velocity_matvec_jax(x, y, n, err, lambd, bmo, nondivergent, chunk=512):
    """Build a chunked 2n-length matvec for velocity-velocity covariance.

    Returns a JIT-compiled function ``matvec(v)`` that computes
    ``(A_vel + err*I) @ v`` without forming the 2n × 2n matrix,
    where A_vel is the nondivergent or irrotational velocity covariance.
    """
    import jax
    import jax.numpy as jnp

    x_obs, y_obs = x, y
    chunk = min(n, chunk)

    # Pad obs coordinates so n is a multiple of chunk.
    if n > chunk:
        rem = n % chunk
        if rem:
            pad = chunk - rem
            x = jnp.concatenate([x, jnp.zeros(pad)])
            y = jnp.concatenate([y, jnp.zeros(pad)])
    n_padded = x.shape[0]
    n_chunks = n_padded // chunk

    def _body(i, carry):
        v_u, v_v, res_u, res_v = carry
        i0 = i * chunk
        xi = jax.lax.dynamic_slice(x, (i0,), (chunk,))
        yi = jax.lax.dynamic_slice(y, (i0,), (chunk,))
        # Pairwise distances and angles: chunk rows vs all n obs
        dxx = xi[:, None] - x_obs[None, :]
        dyy = yi[:, None] - y_obs[None, :]
        d2 = dxx ** 2 + dyy ** 2
        theta = jnp.arctan2(dyy, dxx)
        # Covariance kernels
        E = jnp.exp(-lambd * d2)
        if nondivergent:
            R = E + bmo
            S = E * (1 - 2 * lambd * d2) + bmo
        else:
            R = E * (1 - 2 * lambd * d2) + bmo
            S = E + bmo
        RmS = R - S
        ct = jnp.cos(theta)
        st = jnp.sin(theta)
        # Block matvec: [A_uu A_uv; A_vu A_vv] @ [v_u; v_v]
        A_uu = ct ** 2 * RmS + S
        A_uv = ct * st * RmS
        A_vv = st ** 2 * RmS + S
        chunk_u = A_uu @ v_u + A_uv @ v_v
        chunk_v = A_uv @ v_u + A_vv @ v_v
        prev_u = jax.lax.dynamic_slice(res_u, (i0,), (chunk,))
        prev_v = jax.lax.dynamic_slice(res_v, (i0,), (chunk,))
        res_u = jax.lax.dynamic_update_slice(res_u, prev_u + chunk_u, (i0,))
        res_v = jax.lax.dynamic_update_slice(res_v, prev_v + chunk_v, (i0,))
        return v_u, v_v, res_u, res_v

    @jax.jit
    def matvec(v):
        v_u, v_v = v[:n], v[n:]
        res_u = jnp.zeros(n_padded).at[:n].set(err * v_u)
        res_v = jnp.zeros(n_padded).at[:n].set(err * v_v)
        _, _, res_u, res_v = jax.lax.fori_loop(
            0, n_chunks, _body, (v_u, v_v, res_u, res_v)
        )
        return jnp.concatenate([res_u[:n], res_v[:n]])

    return matvec


def _streamfunction_jax(xc, yc, x, y, u, v, corrlenx, corrleny, err, b=0):
    """Streamfunction recovery via matrix-free CG on GPU (JAX)."""
    import jax
    import jax.numpy as jnp
    from jax.scipy.sparse.linalg import cg

    corrlen = float(corrleny)
    ratio = corrleny / corrlenx
    x = jnp.asarray(x, dtype=jnp.float32) * ratio
    xc = jnp.asarray(xc, dtype=jnp.float32) * ratio
    y = jnp.asarray(y, dtype=jnp.float32)
    yc = jnp.asarray(yc, dtype=jnp.float32)

    n = x.shape[0]
    lambd = 1.0 / corrlen ** 2
    bmo = b * err / lambd
    maxiter = min(2 * n, 400)

    uv = jnp.concatenate([
        jnp.asarray(u, dtype=jnp.float32).ravel(),
        jnp.asarray(v, dtype=jnp.float32).ravel(),
    ])

    matvec = _velocity_matvec_jax(x, y, n, err, lambd, bmo, nondivergent=True)
    w, _ = cg(matvec, uv, maxiter=maxiter)

    # Cross-covariance kernel sum: P @ w for streamfunction
    nv1, nv2 = xc.shape
    xc_flat = xc.T.ravel()
    yc_flat = yc.T.ravel()
    w_u, w_v = w[:n], w[n:]

    @jax.jit
    def _psi_vec(w_u, w_v):
        def _one(args):
            xc_j, yc_j = args
            dxx = x - xc_j
            dyy = y - yc_j
            dc2 = dxx ** 2 + dyy ** 2
            tc = jnp.arctan2(dyy, dxx)
            Rc = jnp.exp(-lambd * dc2) + bmo
            sqrt_dc2 = jnp.sqrt(dc2)
            p_u = jnp.sin(tc) * sqrt_dc2 * Rc
            p_v = -jnp.cos(tc) * sqrt_dc2 * Rc
            return jnp.dot(p_u, w_u) + jnp.dot(p_v, w_v)
        return jax.lax.map(_one, (xc_flat, yc_flat))

    psi = _psi_vec(w_u, w_v)
    return psi.reshape(nv2, nv1).T


def _velocity_potential_jax(xc, yc, x, y, u, v, corrlenx, corrleny, err, b=0):
    """Velocity potential recovery via matrix-free CG on GPU (JAX)."""
    import jax
    import jax.numpy as jnp
    from jax.scipy.sparse.linalg import cg

    corrlen = float(corrleny)
    ratio = corrleny / corrlenx
    x = jnp.asarray(x, dtype=jnp.float32) * ratio
    xc = jnp.asarray(xc, dtype=jnp.float32) * ratio
    y = jnp.asarray(y, dtype=jnp.float32)
    yc = jnp.asarray(yc, dtype=jnp.float32)

    n = x.shape[0]
    lambd = 1.0 / corrlen ** 2
    bmo = b * err / lambd
    maxiter = min(2 * n, 400)

    uv = jnp.concatenate([
        jnp.asarray(u, dtype=jnp.float32).ravel(),
        jnp.asarray(v, dtype=jnp.float32).ravel(),
    ])

    matvec = _velocity_matvec_jax(x, y, n, err, lambd, bmo, nondivergent=False)
    w, _ = cg(matvec, uv, maxiter=maxiter)

    # Cross-covariance kernel sum: P @ w for velocity potential
    nv1, nv2 = xc.shape
    xc_flat = xc.T.ravel()
    yc_flat = yc.T.ravel()
    w_u, w_v = w[:n], w[n:]

    @jax.jit
    def _chi_vec(w_u, w_v):
        def _one(args):
            xc_j, yc_j = args
            dxx = x - xc_j
            dyy = y - yc_j
            dc2 = dxx ** 2 + dyy ** 2
            tc = jnp.arctan2(dyy, dxx)
            Rc = jnp.exp(-lambd * dc2) + bmo
            sqrt_dc2 = jnp.sqrt(dc2)
            p_u = -jnp.cos(tc) * sqrt_dc2 * Rc
            p_v = -jnp.sin(tc) * sqrt_dc2 * Rc
            return jnp.dot(p_u, w_u) + jnp.dot(p_v, w_v)
        return jax.lax.map(_one, (xc_flat, yc_flat))

    chi = _chi_vec(w_u, w_v)
    return chi.reshape(nv2, nv1).T


def _helmholtz_jax(xc, yc, x, y, u, v,
                   corrlenx_psi, corrleny_psi,
                   corrlenx_chi, corrleny_chi,
                   err, b=0):
    """Helmholtz decomposition via matrix-free CG on GPU (JAX)."""
    import jax
    import jax.numpy as jnp
    from jax.scipy.sparse.linalg import cg

    y_arr = jnp.asarray(y, dtype=jnp.float32)
    yc_arr = jnp.asarray(yc, dtype=jnp.float32)

    n = len(y_arr)
    maxiter = min(2 * n, 400)

    # Psi-space rescaling
    corrlen_psi = float(corrleny_psi)
    ratio_psi = corrleny_psi / corrlenx_psi
    x_psi = jnp.asarray(x, dtype=jnp.float32) * ratio_psi
    xc_psi = jnp.asarray(xc, dtype=jnp.float32) * ratio_psi
    lambd_psi = 1.0 / corrlen_psi ** 2
    bmo_psi = b * err / lambd_psi

    # Chi-space rescaling
    corrlen_chi = float(corrleny_chi)
    ratio_chi = corrleny_chi / corrlenx_chi
    x_chi = jnp.asarray(x, dtype=jnp.float32) * ratio_chi
    xc_chi = jnp.asarray(xc, dtype=jnp.float32) * ratio_chi
    lambd_chi = 1.0 / corrlen_chi ** 2
    bmo_chi = b * err / lambd_chi

    uv = jnp.concatenate([
        jnp.asarray(u, dtype=jnp.float32).ravel(),
        jnp.asarray(v, dtype=jnp.float32).ravel(),
    ])

    # Combined matvec: (A_psi + A_chi + err*I) @ v
    # Each sub-matvec adds its own err*I, so we subtract one copy.
    matvec_psi = _velocity_matvec_jax(
        x_psi, y_arr, n, err, lambd_psi, bmo_psi, nondivergent=True
    )
    matvec_chi = _velocity_matvec_jax(
        x_chi, y_arr, n, err, lambd_chi, bmo_chi, nondivergent=False
    )

    @jax.jit
    def matvec(v):
        return matvec_psi(v) + matvec_chi(v) - err * v

    w, _ = cg(matvec, uv, maxiter=maxiter)
    w_u, w_v = w[:n], w[n:]

    # Target grid
    nv1, nv2 = xc_psi.shape
    xc_psi_flat = xc_psi.T.ravel()
    xc_chi_flat = xc_chi.T.ravel()
    yc_flat = yc_arr.T.ravel()

    @jax.jit
    def _psi_chi_vec(w_u, w_v):
        def _one_psi(args):
            xc_j, yc_j = args
            dxx = x_psi - xc_j
            dyy = y_arr - yc_j
            dc2 = dxx ** 2 + dyy ** 2
            tc = jnp.arctan2(dyy, dxx)
            Rc = jnp.exp(-lambd_psi * dc2) + bmo_psi
            sqrt_dc2 = jnp.sqrt(dc2)
            return (jnp.dot(jnp.sin(tc) * sqrt_dc2 * Rc, w_u)
                    + jnp.dot(-jnp.cos(tc) * sqrt_dc2 * Rc, w_v))

        def _one_chi(args):
            xc_j, yc_j = args
            dxx = x_chi - xc_j
            dyy = y_arr - yc_j
            dc2 = dxx ** 2 + dyy ** 2
            tc = jnp.arctan2(dyy, dxx)
            Rc = jnp.exp(-lambd_chi * dc2) + bmo_chi
            sqrt_dc2 = jnp.sqrt(dc2)
            return (jnp.dot(-jnp.cos(tc) * sqrt_dc2 * Rc, w_u)
                    + jnp.dot(-jnp.sin(tc) * sqrt_dc2 * Rc, w_v))

        psi = jax.lax.map(_one_psi, (xc_psi_flat, yc_flat))
        chi = jax.lax.map(_one_chi, (xc_chi_flat, yc_flat))
        return psi, chi

    psi, chi = _psi_chi_vec(w_u, w_v)
    return psi.reshape(nv2, nv1).T, chi.reshape(nv2, nv1).T


def _velocity_cov_block(t, d2, lambd, bmo, nondivergent=True):
    """Build a 2n x 2n velocity-velocity covariance contribution.

    For a nondivergent field (streamfunction), the longitudinal covariance
    is R = exp(-lambda*d2) and the transverse is S = R*(1 - 2*lambda*d2).
    For an irrotational field (velocity potential), R and S swap.

    Parameters
    ----------
    t : ndarray, shape (n, n)
        Angles between point pairs.
    d2 : ndarray, shape (n, n)
        Squared distances between point pairs.
    lambd : float
        Inverse squared correlation length (1 / corrlen**2).
    bmo : float
        Mean correction offset (b * err / lambd).
    nondivergent : bool
        True for streamfunction, False for velocity potential.

    Returns
    -------
    A : ndarray, shape (2n, 2n)
        Velocity covariance block (without diagonal noise).
    """
    n = t.shape[0]
    E = np.exp(-lambd * d2)
    if nondivergent:
        R = E + bmo
        S = E * (1 - 2 * lambd * d2) + bmo
    else:
        R = E * (1 - 2 * lambd * d2) + bmo
        S = E + bmo

    A = np.zeros((2 * n, 2 * n))
    A[0:n, 0:n] = (np.cos(t) ** 2) * (R - S) + S
    A[0:n, n : 2 * n] = np.cos(t) * np.sin(t) * (R - S)
    A[n : 2 * n, 0:n] = A[0:n, n : 2 * n]
    A[n : 2 * n, n : 2 * n] = (np.sin(t) ** 2) * (R - S) + S
    return A


def error(xc, yc, x, y, corrlenx=None, corrleny=None, err=None,
         backend="numpy", k_local=None):
    """
    Normalized mean squared error for scalar objective analysis.

    Computes the interpolation error at target locations (xc, yc) given
    observation locations (x, y) and correlation parameters. The error
    depends only on the geometry and correlation structure, not on the
    observed values.

    Parameters
    ----------
    xc : array_like
        x-coordinates of target (interpolation) points.
    yc : array_like
        y-coordinates of target (interpolation) points.
    x : array_like
        x-coordinates of observation points.
    y : array_like
        y-coordinates of observation points.
    corrlenx : float
        Correlation length scale in the x-direction.
    corrleny : float
        Correlation length scale in the y-direction.
    err : float
        Normalized random error variance (0 < err < 1).
    backend : {"numpy", "jax"}, optional
        Array backend. Default ``"numpy"`` (direct solve). ``"jax"``
        uses local neighborhood approximation (k nearest observations
        per target point) for GPU-friendly O(n) scaling.
    k_local : int, optional
        Number of nearest observations for the local error solve
        (JAX backend only). Default is min(n, 100).

    Returns
    -------
    ep : ndarray
        Normalized mean squared error at target locations. Taking the
        square root gives the interpolation error as a fraction.

    Notes
    -----
    Anisotropy is handled by rescaling the x-coordinate by the ratio
    corrleny / corrlenx before computing distances.

    References
    ----------
    Bretherton, F. P., Davis, R. E., & Fandry, C. B. (1976).
    A technique for objective analysis and design of oceanographic
    experiments applied to MODE-73. Deep-Sea Research, 23(7), 559-582.
    """
    if backend == "numpy":
        return _error_numpy(xc, yc, x, y, corrlenx, corrleny, err)
    elif backend == "jax":
        return _error_jax(xc, yc, x, y, corrlenx, corrleny, err, k_local=k_local)
    else:
        raise ValueError(f"Unknown backend {backend!r}. Choose 'numpy' or 'jax'.")


def scalar(xc, yc, x, y, t, corrlenx=None, corrleny=None, err=None, backend="numpy"):
    """
    Scalar objective analysis via Gauss-Markov estimation.

    Interpolates a scalar field t(x, y) onto target locations (xc, yc)
    assuming a Gaussian spatial correlation function of the form:

        C = (1 - err) * exp(-d^2 / corrlen^2)

    Parameters
    ----------
    xc : array_like
        x-coordinates of target (interpolation) points.
    yc : array_like
        y-coordinates of target (interpolation) points.
    x : array_like
        x-coordinates of observation points.
    y : array_like
        y-coordinates of observation points.
    t : array_like
        Observed scalar values at (x, y).
    corrlenx : float
        Correlation length scale in the x-direction.
    corrleny : float
        Correlation length scale in the y-direction.
    err : float
        Normalized random error variance (0 < err < 1).
    backend : {"numpy", "jax"}, optional
        Array backend to use. Default is ``"numpy"`` (direct solve).
        Use ``"jax"`` for GPU acceleration via matrix-free conjugate
        gradient — no large covariance matrix is formed. Requires JAX;
        install with ``pip install 'xobjmap[jax]'``.

    Returns
    -------
    tp : ndarray
        Interpolated field at target locations. Array type matches
        the backend.

    Notes
    -----
    Anisotropy is handled by rescaling the x-coordinate by the ratio
    corrleny / corrlenx before computing distances, so that the effective
    correlation length is isotropic in the rescaled space.

    Use :func:`error` to compute the interpolation error field separately.

    References
    ----------
    Bretherton, F. P., Davis, R. E., & Fandry, C. B. (1976).
    A technique for objective analysis and design of oceanographic
    experiments applied to MODE-73. Deep-Sea Research, 23(7), 559-582.
    """
    if backend == "numpy":
        return _scalar_numpy(xc, yc, x, y, t, corrlenx, corrleny, err)
    elif backend == "jax":
        return _scalar_jax(xc, yc, x, y, t, corrlenx, corrleny, err)
    else:
        raise ValueError(f"Unknown backend {backend!r}. Choose 'numpy' or 'jax'.")


def streamfunction(xc, yc, x, y, u, v, corrlenx, corrleny, err, b=0, backend="numpy"):
    """
    Recover the streamfunction from scattered velocity observations.

    Interpolates scattered velocity observations (u, v) onto a grid
    (xc, yc), returning the streamfunction field. The method assumes a
    Gaussian streamfunction covariance and derives the velocity-velocity
    and streamfunction-velocity cross-covariance matrices analytically,
    under the assumption of purely nondivergent flow.

    Parameters
    ----------
    xc : numpy.ndarray
        2D array of x-coordinates of the target grid (shape M x N).
    yc : numpy.ndarray
        2D array of y-coordinates of the target grid (shape M x N).
    x : array_like
        x-coordinates of velocity observation points.
    y : array_like
        y-coordinates of velocity observation points.
    u : array_like
        Observed eastward (x) velocity component.
    v : array_like
        Observed northward (y) velocity component.
    corrlenx : float
        Correlation length scale in the x-direction.
    corrleny : float
        Correlation length scale in the y-direction.
    err : float
        Normalized random error variance (0 < err < 1).
    b : float, optional
        Mean correction parameter. Default is 0 (no correction).

    Returns
    -------
    psi : numpy.ndarray
        Streamfunction field on the target grid (shape M x N).

    Notes
    -----
    The method solves for the streamfunction (psi) given velocity
    observations by exploiting the relationship:

        u = -d(psi)/dy,  v = d(psi)/dx

    The velocity-velocity covariance is decomposed into longitudinal (R)
    and transverse (S) components using the angles between observation
    pairs. The streamfunction-velocity cross-covariance is derived from
    the first derivative of the streamfunction covariance.

    Anisotropy is handled by rescaling the x-coordinate by the ratio
    corrleny / corrlenx.

    References
    ----------
    Bretherton, F. P., Davis, R. E., & Fandry, C. B. (1976).
    A technique for objective analysis and design of oceanographic
    experiments applied to MODE-73. Deep-Sea Research, 23(7), 559-582.
    """
    if backend == "jax":
        return _streamfunction_jax(xc, yc, x, y, u, v, corrlenx, corrleny, err, b)
    elif backend != "numpy":
        raise ValueError(f"Unknown backend {backend!r}. Choose 'numpy' or 'jax'.")

    xc = np.asarray(xc)
    yc = np.asarray(yc)
    x = np.asarray(x)
    y = np.asarray(y)
    u = np.asarray(u)
    v = np.asarray(v)

    # Rescale x-coordinates for anisotropy
    corrlen = corrleny
    xc = xc * (corrleny / corrlenx)
    x = x * (corrleny / corrlenx)

    n = len(x)
    uv = np.hstack((u, v)).reshape(-1, 1)

    # Observation-observation covariance
    dx, dy, d2 = _pairwise(x, y, x, y)
    t = np.arctan2(dy, dx)
    lambd = 1 / corrlen**2
    bmo = b * err / lambd
    A = _velocity_cov_block(t, d2, lambd, bmo) + err * np.eye(2 * n)

    # Target grid
    nv1, nv2 = xc.shape
    nv = nv1 * nv2
    xc_flat = xc.T.ravel()
    yc_flat = yc.T.ravel()

    # Target-observation cross-covariance
    dx_c, dy_c, dc2 = _pairwise(xc_flat, yc_flat, x, y)
    tc = np.arctan2(dy_c, dx_c)
    Rc = np.exp(-lambd * dc2) + bmo

    # Streamfunction-velocity cross-covariance (nv x 2n)
    P = np.zeros((nv, 2 * n))
    P[:, 0:n] = np.sin(tc) * np.sqrt(dc2) * Rc
    P[:, n : 2 * n] = -np.cos(tc) * np.sqrt(dc2) * Rc

    PSI = np.dot(P, np.linalg.solve(A, uv))
    return PSI.reshape(nv2, nv1).T


def velocity_potential(xc, yc, x, y, u, v, corrlenx, corrleny, err, b=0, backend="numpy"):
    """
    Recover the velocity potential from scattered velocity observations.

    Interpolates scattered velocity observations (u, v) onto a grid
    (xc, yc), returning the velocity potential field. The method assumes
    a Gaussian velocity potential covariance and derives the covariance
    matrices analytically, under the assumption of purely irrotational flow.

    Parameters
    ----------
    xc : numpy.ndarray
        2D array of x-coordinates of the target grid (shape M x N).
    yc : numpy.ndarray
        2D array of y-coordinates of the target grid (shape M x N).
    x : array_like
        x-coordinates of velocity observation points.
    y : array_like
        y-coordinates of velocity observation points.
    u : array_like
        Observed eastward (x) velocity component.
    v : array_like
        Observed northward (y) velocity component.
    corrlenx : float
        Correlation length scale in the x-direction.
    corrleny : float
        Correlation length scale in the y-direction.
    err : float
        Normalized random error variance (0 < err < 1).
    b : float, optional
        Mean correction parameter. Default is 0 (no correction).

    Returns
    -------
    chi : numpy.ndarray
        Velocity potential field on the target grid (shape M x N).

    Notes
    -----
    The method solves for the velocity potential (chi) given velocity
    observations by exploiting the relationship:

        u = d(chi)/dx,  v = d(chi)/dy

    For the irrotational case, the longitudinal and transverse covariance
    roles swap compared to the nondivergent (streamfunction) case.

    Anisotropy is handled by rescaling the x-coordinate by the ratio
    corrleny / corrlenx.

    References
    ----------
    Bretherton, F. P., Davis, R. E., & Fandry, C. B. (1976).
    A technique for objective analysis and design of oceanographic
    experiments applied to MODE-73. Deep-Sea Research, 23(7), 559-582.
    """
    if backend == "jax":
        return _velocity_potential_jax(xc, yc, x, y, u, v, corrlenx, corrleny, err, b)
    elif backend != "numpy":
        raise ValueError(f"Unknown backend {backend!r}. Choose 'numpy' or 'jax'.")

    xc = np.asarray(xc)
    yc = np.asarray(yc)
    x = np.asarray(x)
    y = np.asarray(y)
    u = np.asarray(u)
    v = np.asarray(v)

    # Rescale x-coordinates for anisotropy
    corrlen = corrleny
    xc = xc * (corrleny / corrlenx)
    x = x * (corrleny / corrlenx)

    n = len(x)
    uv = np.hstack((u, v)).reshape(-1, 1)

    # Observation-observation covariance (irrotational: swapped R/S)
    dx, dy, d2 = _pairwise(x, y, x, y)
    t = np.arctan2(dy, dx)
    lambd = 1 / corrlen**2
    bmo = b * err / lambd
    A = (_velocity_cov_block(t, d2, lambd, bmo, nondivergent=False)
         + err * np.eye(2 * n))

    # Target grid
    nv1, nv2 = xc.shape
    nv = nv1 * nv2
    xc_flat = xc.T.ravel()
    yc_flat = yc.T.ravel()

    # Target-observation cross-covariance
    dx_c, dy_c, dc2 = _pairwise(xc_flat, yc_flat, x, y)
    tc = np.arctan2(dy_c, dx_c)
    Rc = np.exp(-lambd * dc2) + bmo

    # Velocity-potential-velocity cross-covariance (nv x 2n)
    P = np.zeros((nv, 2 * n))
    P[:, 0:n] = -np.cos(tc) * np.sqrt(dc2) * Rc
    P[:, n : 2 * n] = -np.sin(tc) * np.sqrt(dc2) * Rc

    CHI = np.dot(P, np.linalg.solve(A, uv))
    return CHI.reshape(nv2, nv1).T


def helmholtz(xc, yc, x, y, u, v,
              corrlenx_psi, corrleny_psi,
              corrlenx_chi, corrleny_chi,
              err, b=0, backend="numpy"):
    """
    Helmholtz decomposition via Bretherton optimal estimation.

    Jointly recovers the streamfunction (psi) and velocity potential (chi)
    from scattered velocity observations. Models the velocity field as:

        u = -d(psi)/dy + d(chi)/dx
        v =  d(psi)/dx + d(chi)/dy

    The streamfunction and velocity potential can have independent
    correlation length scales, allowing different spatial structures
    for the nondivergent and irrotational components.

    Parameters
    ----------
    xc : numpy.ndarray
        2D array of x-coordinates of the target grid (shape M x N).
    yc : numpy.ndarray
        2D array of y-coordinates of the target grid (shape M x N).
    x : array_like
        x-coordinates of velocity observation points.
    y : array_like
        y-coordinates of velocity observation points.
    u : array_like
        Observed eastward (x) velocity component.
    v : array_like
        Observed northward (y) velocity component.
    corrlenx_psi : float
        Streamfunction correlation length in x.
    corrleny_psi : float
        Streamfunction correlation length in y.
    corrlenx_chi : float
        Velocity potential correlation length in x.
    corrleny_chi : float
        Velocity potential correlation length in y.
    err : float
        Normalized random error variance (0 < err < 1).
    b : float, optional
        Mean correction parameter. Default is 0 (no correction).

    Returns
    -------
    psi : numpy.ndarray
        Streamfunction field on the target grid (shape M x N).
    chi : numpy.ndarray
        Velocity potential field on the target grid (shape M x N).

    Notes
    -----
    The velocity-velocity covariance matrix is the sum of contributions
    from the nondivergent (psi) and irrotational (chi) parts, plus
    measurement noise. The system is solved once, then both fields are
    recovered via their respective cross-covariance matrices.

    Anisotropy is handled independently for each field by rescaling the
    x-coordinate by the ratio corrleny / corrlenx.

    References
    ----------
    Bretherton, F. P., Davis, R. E., & Fandry, C. B. (1976).
    A technique for objective analysis and design of oceanographic
    experiments applied to MODE-73. Deep-Sea Research, 23(7), 559-582.
    """
    if backend == "jax":
        return _helmholtz_jax(xc, yc, x, y, u, v,
                              corrlenx_psi, corrleny_psi,
                              corrlenx_chi, corrleny_chi, err, b)
    elif backend != "numpy":
        raise ValueError(f"Unknown backend {backend!r}. Choose 'numpy' or 'jax'.")

    xc = np.asarray(xc)
    yc = np.asarray(yc)
    x = np.asarray(x)
    y = np.asarray(y)
    u = np.asarray(u)
    v = np.asarray(v)

    n = len(x)
    uv = np.hstack((u, v)).reshape(-1, 1)

    # Psi-space rescaling
    corrlen_psi = corrleny_psi
    x_psi = x * (corrleny_psi / corrlenx_psi)
    xc_psi = xc * (corrleny_psi / corrlenx_psi)

    # Chi-space rescaling
    corrlen_chi = corrleny_chi
    x_chi = x * (corrleny_chi / corrlenx_chi)
    xc_chi = xc * (corrleny_chi / corrlenx_chi)

    # Observation-observation covariance: A = A_psi + A_chi + err*I
    dx_psi, dy, d2_psi = _pairwise(x_psi, y, x_psi, y)
    t_psi = np.arctan2(dy, dx_psi)
    lambd_psi = 1 / corrlen_psi**2
    bmo_psi = b * err / lambd_psi

    dx_chi, _, d2_chi = _pairwise(x_chi, y, x_chi, y)
    t_chi = np.arctan2(dy, dx_chi)
    lambd_chi = 1 / corrlen_chi**2
    bmo_chi = b * err / lambd_chi

    A = (_velocity_cov_block(t_psi, d2_psi, lambd_psi, bmo_psi, nondivergent=True)
         + _velocity_cov_block(t_chi, d2_chi, lambd_chi, bmo_chi, nondivergent=False)
         + err * np.eye(2 * n))

    w = np.linalg.solve(A, uv)

    # Target grid
    nv1, nv2 = xc.shape
    nv = nv1 * nv2
    yc_flat = yc.T.ravel()

    # Psi cross-covariance
    xc_psi_flat = xc_psi.T.ravel()
    dx_c, dy_c, dc2 = _pairwise(xc_psi_flat, yc_flat, x_psi, y)
    tc = np.arctan2(dy_c, dx_c)
    Rc = np.exp(-lambd_psi * dc2) + bmo_psi

    P_psi = np.zeros((nv, 2 * n))
    P_psi[:, 0:n] = np.sin(tc) * np.sqrt(dc2) * Rc
    P_psi[:, n : 2 * n] = -np.cos(tc) * np.sqrt(dc2) * Rc

    # Chi cross-covariance
    xc_chi_flat = xc_chi.T.ravel()
    dx_c, dy_c, dc2 = _pairwise(xc_chi_flat, yc_flat, x_chi, y)
    tc = np.arctan2(dy_c, dx_c)
    Rc = np.exp(-lambd_chi * dc2) + bmo_chi

    P_chi = np.zeros((nv, 2 * n))
    P_chi[:, 0:n] = -np.cos(tc) * np.sqrt(dc2) * Rc
    P_chi[:, n : 2 * n] = -np.sin(tc) * np.sqrt(dc2) * Rc

    PSI = np.dot(P_psi, w).reshape(nv2, nv1).T
    CHI = np.dot(P_chi, w).reshape(nv2, nv1).T

    return PSI, CHI
