"""
Core interpolation functions using Gauss-Markov optimal estimation.
"""

import numpy as np


def _velocity_cov_block(t, d2, lambd, bmo, nondivergent=True, d2_total=None):
    """Build a 2n x 2n velocity-velocity covariance contribution.

    For a nondivergent field (streamfunction), the longitudinal covariance
    is R = exp(-lambda*d2) and the transverse is S = R*(1 - 2*lambda*d2).
    For an irrotational field (velocity potential), R and S swap.

    Parameters
    ----------
    t : ndarray, shape (n, n)
        Angles between point pairs.
    d2 : ndarray, shape (n, n)
        Squared distances in the derivative plane.
    lambd : float
        Inverse squared correlation length (1 / corrlen**2).
    bmo : float
        Mean correction offset (b * err / lambd).
    nondivergent : bool
        True for streamfunction, False for velocity potential.
    d2_total : ndarray, shape (n, n), optional
        Full N-D squared distances for the Gaussian kernel.  When
        ``None`` (the default, backward-compatible with 2-D callers),
        ``d2`` is used for both the kernel and the derivative terms.

    Returns
    -------
    A : ndarray, shape (2n, 2n)
        Velocity covariance block (without diagonal noise).
    """
    if d2_total is None:
        d2_total = d2
    n = t.shape[0]
    E = np.exp(-lambd * d2_total)
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


def _velocity_cov_block_jax(t, d2, lambd, bmo, nondivergent=True, d2_total=None):
    """JAX version of `_velocity_cov_block`."""
    import jax.numpy as jnp

    if d2_total is None:
        d2_total = d2
    n = t.shape[0]
    E = jnp.exp(-lambd * d2_total)
    if nondivergent:
        R = E + bmo
        S = E * (1 - 2 * lambd * d2) + bmo
    else:
        R = E * (1 - 2 * lambd * d2) + bmo
        S = E + bmo

    A = jnp.zeros((2 * n, 2 * n), dtype=t.dtype)
    A = A.at[0:n, 0:n].set((jnp.cos(t) ** 2) * (R - S) + S)
    A = A.at[0:n, n : 2 * n].set(jnp.cos(t) * jnp.sin(t) * (R - S))
    A = A.at[n : 2 * n, 0:n].set(A[0:n, n : 2 * n])
    A = A.at[n : 2 * n, n : 2 * n].set((jnp.sin(t) ** 2) * (R - S) + S)
    return A


def _scaled_sqdist_numpy(points1, points2, corrlen):
    """Pairwise normalized squared distance for N-D scalar interpolation."""
    p1 = np.asarray(points1, dtype=float)
    p2 = np.asarray(points2, dtype=float)
    scale = np.asarray(corrlen, dtype=float)
    diff = (p2[None, :, :] - p1[:, None, :]) / scale[None, None, :]
    return np.sum(diff ** 2, axis=2)


def _scaled_sqdist_jax(points1, points2, corrlen):
    """JAX pairwise normalized squared distance for N-D scalar interpolation."""
    import jax.numpy as jnp

    p1 = jnp.asarray(points1, dtype=jnp.float32)
    p2 = jnp.asarray(points2, dtype=jnp.float32)
    scale = jnp.asarray(corrlen, dtype=jnp.float32)
    diff = (p2[None, :, :] - p1[:, None, :]) / scale[None, None, :]
    return jnp.sum(diff ** 2, axis=2)


def _scalar_nd_numpy(target_points, obs_points, values, corrlen, err):
    """Dense N-D scalar interpolation (numpy)."""
    d2 = _scaled_sqdist_numpy(obs_points, obs_points, corrlen)
    dc2 = _scaled_sqdist_numpy(target_points, obs_points, corrlen)
    a = (1.0 - err) * np.exp(-d2) + err * np.eye(len(obs_points))
    c = (1.0 - err) * np.exp(-dc2)
    t = np.asarray(values, dtype=float).reshape(len(obs_points), 1)
    return np.dot(c, np.linalg.solve(a, t)).reshape(-1)


def _scalar_nd_jax(target_points, obs_points, values, corrlen, err):
    """Matrix-free N-D scalar interpolation (JAX)."""
    import jax
    import jax.numpy as jnp
    from jax.scipy.sparse.linalg import cg

    obs_points = jnp.asarray(obs_points, dtype=jnp.float32)
    target_points = jnp.asarray(target_points, dtype=jnp.float32)
    corrlen = jnp.asarray(corrlen, dtype=jnp.float32)

    n = obs_points.shape[0]
    one_m_err = 1.0 - err
    maxiter = min(n, 200)

    if n <= 512:
        chunk = n
        obs_rows = obs_points
    else:
        chunk = 512
        rem = n % chunk
        if rem:
            pad = chunk - rem
            obs_rows = jnp.pad(obs_points, ((0, pad), (0, 0)))
        else:
            obs_rows = obs_points
    n_padded = obs_rows.shape[0]
    n_chunks = n_padded // chunk

    def _matvec_body(i, carry):
        v, result = carry
        i0 = i * chunk
        rows = jax.lax.dynamic_slice(obs_rows, (i0, 0), (chunk, obs_rows.shape[1]))
        diff = (rows[:, None, :] - obs_points[None, :, :]) / corrlen[None, None, :]
        d2 = jnp.sum(diff ** 2, axis=2)
        kv = one_m_err * (jnp.exp(-d2) @ v)
        prev = jax.lax.dynamic_slice(result, (i0,), (chunk,))
        result = jax.lax.dynamic_update_slice(result, prev + kv, (i0,))
        return v, result

    @jax.jit
    def matvec(v):
        result = jnp.zeros(n_padded, dtype=jnp.float32)
        result = result.at[:n].set(err * v)
        _, result = jax.lax.fori_loop(0, n_chunks, _matvec_body, (v, result))
        return result[:n]

    @jax.jit
    def _cross_cov_vec(weights):
        def _one(point):
            diff = (point[None, :] - obs_points) / corrlen[None, :]
            d2 = jnp.sum(diff ** 2, axis=1)
            return one_m_err * jnp.dot(jnp.exp(-d2), weights)
        return jax.lax.map(_one, target_points)

    t_vec = jnp.asarray(values, dtype=jnp.float32).ravel()
    w, _ = cg(matvec, t_vec, maxiter=maxiter)
    return _cross_cov_vec(w).reshape(-1)


def _scalar_error_nd_numpy(target_points, obs_points, corrlen, err, k_local=None):
    """Dense N-D scalar posterior error (numpy)."""
    d2 = _scaled_sqdist_numpy(obs_points, obs_points, corrlen)
    dc2 = _scaled_sqdist_numpy(target_points, obs_points, corrlen)
    a = (1.0 - err) * np.exp(-d2) + err * np.eye(len(obs_points))
    c = (1.0 - err) * np.exp(-dc2)
    return 1.0 - np.sum(c.T * np.linalg.solve(a, c.T), axis=0) / (1.0 - err)


def _scalar_error_nd_jax(target_points, obs_points, corrlen, err, k_local=None):
    """Local-neighborhood N-D scalar posterior error (JAX)."""
    import jax
    import jax.numpy as jnp

    obs_points = jnp.asarray(obs_points, dtype=jnp.float32)
    target_points = jnp.asarray(target_points, dtype=jnp.float32)
    corrlen = jnp.asarray(corrlen, dtype=jnp.float32)

    n = obs_points.shape[0]
    one_m_err = 1.0 - err
    k = min(n, k_local if k_local is not None else min(n, 100))

    @jax.jit
    def _local_error(point):
        diff_all = (point[None, :] - obs_points) / corrlen[None, :]
        d2_all = jnp.sum(diff_all ** 2, axis=1)
        _, idx = jax.lax.top_k(-d2_all, k)
        obs_local = obs_points[idx]
        diff_kk = (obs_local[None, :, :] - obs_local[:, None, :]) / corrlen[None, None, :]
        d2_kk = jnp.sum(diff_kk ** 2, axis=2)
        a_local = one_m_err * jnp.exp(-d2_kk) + err * jnp.eye(k, dtype=jnp.float32)
        diff_ck = (point[None, :] - obs_local) / corrlen[None, :]
        d2_ck = jnp.sum(diff_ck ** 2, axis=1)
        c_local = one_m_err * jnp.exp(-d2_ck)
        return 1.0 - jnp.dot(c_local, jnp.linalg.solve(a_local, c_local)) / one_m_err

    return jax.lax.map(_local_error, target_points)


def _vector_scale_params_numpy(corrlen, derivative_indices):
    """Return anisotropic coordinate scaling and reference lambda."""
    corrlen = np.asarray(corrlen, dtype=float)
    ref = float(corrlen[derivative_indices[1]])
    return ref / corrlen, 1.0 / ref ** 2


def _vector_scale_params_jax(corrlen, derivative_indices):
    """JAX version of `_vector_scale_params_numpy`."""
    import jax.numpy as jnp

    corrlen = jnp.asarray(corrlen, dtype=jnp.float32)
    ref = corrlen[derivative_indices[1]]
    return ref / corrlen, 1.0 / ref ** 2


def _vector_obs_geometry_numpy(obs_points, corrlen, derivative_indices):
    """Scaled observation geometry for vector-potential methods."""
    scale, lambd = _vector_scale_params_numpy(corrlen, derivative_indices)
    obs_scaled = np.asarray(obs_points, dtype=float) * scale[None, :]
    ix, iy = derivative_indices
    dx = obs_scaled[None, :, ix] - obs_scaled[:, None, ix]
    dy = obs_scaled[None, :, iy] - obs_scaled[:, None, iy]
    total_d2 = np.sum((obs_scaled[None, :, :] - obs_scaled[:, None, :]) ** 2, axis=2)
    return np.arctan2(dy, dx), dx ** 2 + dy ** 2, total_d2, lambd


def _vector_target_geometry_numpy(target_points, obs_points, corrlen, derivative_indices):
    """Scaled target-observation geometry for vector-potential methods."""
    scale, lambd = _vector_scale_params_numpy(corrlen, derivative_indices)
    target_scaled = np.asarray(target_points, dtype=float) * scale[None, :]
    obs_scaled = np.asarray(obs_points, dtype=float) * scale[None, :]
    ix, iy = derivative_indices
    dx = obs_scaled[None, :, ix] - target_scaled[:, None, ix]
    dy = obs_scaled[None, :, iy] - target_scaled[:, None, iy]
    total_d2 = np.sum((obs_scaled[None, :, :] - target_scaled[:, None, :]) ** 2, axis=2)
    return np.arctan2(dy, dx), dx ** 2 + dy ** 2, total_d2, lambd




def _cross_cov_nd_numpy(target_points, obs_points, corrlen, derivative_indices, bmo,
                        nondivergent=True):
    """Cross-covariance matrix for streamfunction/velocity potential in N-D."""
    theta, spatial_d2, total_d2, lambd = _vector_target_geometry_numpy(
        target_points, obs_points, corrlen, derivative_indices
    )
    rc = np.exp(-lambd * total_d2) + bmo
    sqrt_d2 = np.sqrt(spatial_d2)
    n_target, n_obs = total_d2.shape
    p = np.zeros((n_target, 2 * n_obs))
    if nondivergent:
        p[:, 0:n_obs] = np.sin(theta) * sqrt_d2 * rc
        p[:, n_obs:2 * n_obs] = -np.cos(theta) * sqrt_d2 * rc
    else:
        p[:, 0:n_obs] = -np.cos(theta) * sqrt_d2 * rc
        p[:, n_obs:2 * n_obs] = -np.sin(theta) * sqrt_d2 * rc
    return p, lambd



def _velocity_matvec_nd_jax(obs_scaled, derivative_indices, n, err, lambd, bmo,
                            nondivergent, chunk=512):
    """Build a chunked 2n-length matvec for N-D velocity-velocity covariance.

    Returns a JIT-compiled function ``matvec(v)`` that computes
    ``(A_vel + err*I) @ v`` without forming the 2n x 2n matrix.

    Parameters
    ----------
    obs_scaled : jnp.ndarray, shape (n, ndim)
        Observation coordinates already scaled by corrlen ratios.
    derivative_indices : tuple of int
        Indices of the x and y derivative dimensions.
    n : int
        Number of observations.
    err : float
        Noise variance.
    lambd : float
        Inverse squared reference correlation length (1 / corrlen_ref**2).
    bmo : float
        Mean correction offset (b * err / lambd).
    nondivergent : bool
        True for streamfunction, False for velocity potential.
    chunk : int
        Chunk size for memory-efficient computation.
    """
    import jax
    import jax.numpy as jnp

    ix, iy = derivative_indices
    obs_ref = obs_scaled
    ndim = obs_scaled.shape[1]

    chunk = min(n, chunk)
    if n > chunk:
        rem = n % chunk
        if rem:
            pad = chunk - rem
            obs_padded = jnp.pad(obs_scaled, ((0, pad), (0, 0)))
        else:
            obs_padded = obs_scaled
    else:
        obs_padded = obs_scaled
    n_padded = obs_padded.shape[0]
    n_chunks = n_padded // chunk

    def _body(i, carry):
        v_u, v_v, res_u, res_v = carry
        i0 = i * chunk
        rows = jax.lax.dynamic_slice(obs_padded, (i0, 0), (chunk, ndim))
        diff = rows[:, None, :] - obs_ref[None, :, :]
        dxx = diff[:, :, ix]
        dyy = diff[:, :, iy]
        spatial_d2 = dxx ** 2 + dyy ** 2
        total_d2 = jnp.sum(diff ** 2, axis=2)
        theta = jnp.arctan2(dyy, dxx)
        E = jnp.exp(-lambd * total_d2)
        if nondivergent:
            R = E + bmo
            S = E * (1 - 2 * lambd * spatial_d2) + bmo
        else:
            R = E * (1 - 2 * lambd * spatial_d2) + bmo
            S = E + bmo
        RmS = R - S
        ct = jnp.cos(theta)
        st = jnp.sin(theta)
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
        res_u = jnp.zeros(n_padded, dtype=jnp.float32).at[:n].set(err * v_u)
        res_v = jnp.zeros(n_padded, dtype=jnp.float32).at[:n].set(err * v_v)
        _, _, res_u, res_v = jax.lax.fori_loop(
            0, n_chunks, _body, (v_u, v_v, res_u, res_v)
        )
        return jnp.concatenate([res_u[:n], res_v[:n]])

    return matvec


def _streamfunction_nd_numpy(target_points, obs_points, u, v, corrlen, derivative_indices,
                             err, b=0):
    """Dense N-D streamfunction recovery (numpy)."""
    theta, spatial_d2, total_d2, lambd = _vector_obs_geometry_numpy(
        obs_points, corrlen, derivative_indices
    )
    n = len(obs_points)
    bmo = b * err / lambd
    a = _velocity_cov_block(theta, spatial_d2, lambd, bmo, nondivergent=True, d2_total=total_d2) + err * np.eye(2 * n)
    uv = np.hstack((u, v)).reshape(-1, 1)
    p, _ = _cross_cov_nd_numpy(
        target_points, obs_points, corrlen, derivative_indices, bmo, nondivergent=True
    )
    return np.dot(p, np.linalg.solve(a, uv)).reshape(-1)


def _streamfunction_nd_jax(target_points, obs_points, u, v, corrlen, derivative_indices,
                           err, b=0):
    """Matrix-free N-D streamfunction recovery (JAX)."""
    import jax
    import jax.numpy as jnp
    from jax.scipy.sparse.linalg import cg

    scale, lambd = _vector_scale_params_jax(corrlen, derivative_indices)
    obs_scaled = jnp.asarray(obs_points, dtype=jnp.float32) * scale[None, :]
    target_scaled = jnp.asarray(target_points, dtype=jnp.float32) * scale[None, :]
    ix, iy = derivative_indices

    n = obs_scaled.shape[0]
    bmo = b * err / lambd
    maxiter = min(2 * n, 400)

    uv = jnp.concatenate([
        jnp.asarray(u, dtype=jnp.float32).ravel(),
        jnp.asarray(v, dtype=jnp.float32).ravel(),
    ])

    matvec = _velocity_matvec_nd_jax(obs_scaled, derivative_indices, n, err, lambd, bmo,
                                     nondivergent=True)
    w, _ = cg(matvec, uv, maxiter=maxiter)
    w_u, w_v = w[:n], w[n:]

    @jax.jit
    def _cross_cov_vec(w_u, w_v):
        def _one(target_pt):
            diff = obs_scaled - target_pt[None, :]
            dxx = diff[:, ix]
            dyy = diff[:, iy]
            spatial_dc2 = dxx ** 2 + dyy ** 2
            total_dc2 = jnp.sum(diff ** 2, axis=1)
            tc = jnp.arctan2(dyy, dxx)
            Rc = jnp.exp(-lambd * total_dc2) + bmo
            sqrt_dc2 = jnp.sqrt(spatial_dc2)
            p_u = jnp.sin(tc) * sqrt_dc2 * Rc
            p_v = -jnp.cos(tc) * sqrt_dc2 * Rc
            return jnp.dot(p_u, w_u) + jnp.dot(p_v, w_v)
        return jax.lax.map(_one, target_scaled)

    return _cross_cov_vec(w_u, w_v)


def _velocity_potential_nd_numpy(target_points, obs_points, u, v, corrlen, derivative_indices,
                                 err, b=0):
    """Dense N-D velocity-potential recovery (numpy)."""
    theta, spatial_d2, total_d2, lambd = _vector_obs_geometry_numpy(
        obs_points, corrlen, derivative_indices
    )
    n = len(obs_points)
    bmo = b * err / lambd
    a = _velocity_cov_block(theta, spatial_d2, lambd, bmo, nondivergent=False, d2_total=total_d2) + err * np.eye(2 * n)
    uv = np.hstack((u, v)).reshape(-1, 1)
    p, _ = _cross_cov_nd_numpy(
        target_points, obs_points, corrlen, derivative_indices, bmo, nondivergent=False
    )
    return np.dot(p, np.linalg.solve(a, uv)).reshape(-1)


def _velocity_potential_nd_jax(target_points, obs_points, u, v, corrlen, derivative_indices,
                               err, b=0):
    """Matrix-free N-D velocity-potential recovery (JAX)."""
    import jax
    import jax.numpy as jnp
    from jax.scipy.sparse.linalg import cg

    scale, lambd = _vector_scale_params_jax(corrlen, derivative_indices)
    obs_scaled = jnp.asarray(obs_points, dtype=jnp.float32) * scale[None, :]
    target_scaled = jnp.asarray(target_points, dtype=jnp.float32) * scale[None, :]
    ix, iy = derivative_indices

    n = obs_scaled.shape[0]
    bmo = b * err / lambd
    maxiter = min(2 * n, 400)

    uv = jnp.concatenate([
        jnp.asarray(u, dtype=jnp.float32).ravel(),
        jnp.asarray(v, dtype=jnp.float32).ravel(),
    ])

    matvec = _velocity_matvec_nd_jax(obs_scaled, derivative_indices, n, err, lambd, bmo,
                                     nondivergent=False)
    w, _ = cg(matvec, uv, maxiter=maxiter)
    w_u, w_v = w[:n], w[n:]

    @jax.jit
    def _cross_cov_vec(w_u, w_v):
        def _one(target_pt):
            diff = obs_scaled - target_pt[None, :]
            dxx = diff[:, ix]
            dyy = diff[:, iy]
            spatial_dc2 = dxx ** 2 + dyy ** 2
            total_dc2 = jnp.sum(diff ** 2, axis=1)
            tc = jnp.arctan2(dyy, dxx)
            Rc = jnp.exp(-lambd * total_dc2) + bmo
            sqrt_dc2 = jnp.sqrt(spatial_dc2)
            p_u = -jnp.cos(tc) * sqrt_dc2 * Rc
            p_v = -jnp.sin(tc) * sqrt_dc2 * Rc
            return jnp.dot(p_u, w_u) + jnp.dot(p_v, w_v)
        return jax.lax.map(_one, target_scaled)

    return _cross_cov_vec(w_u, w_v)


def _single_component_vector_error_nd_numpy(target_points, obs_points, corrlen,
                                            derivative_indices, err, b=0,
                                            nondivergent=True, k_local=None):
    """Dense N-D posterior error for streamfunction/velocity potential."""
    theta, spatial_d2, total_d2, lambd = _vector_obs_geometry_numpy(
        obs_points, corrlen, derivative_indices
    )
    n = len(obs_points)
    bmo = b * err / lambd
    a = _velocity_cov_block(
        theta, spatial_d2, lambd, bmo, nondivergent=nondivergent, d2_total=total_d2,
    ) + err * np.eye(2 * n)
    p, _ = _cross_cov_nd_numpy(
        target_points, obs_points, corrlen, derivative_indices, bmo,
        nondivergent=nondivergent,
    )
    proj = np.sum(p.T * np.linalg.solve(a, p.T), axis=0)
    return np.clip(1.0 - (2.0 * lambd) * proj / (1.0 + bmo), 0.0, 1.0)


def _single_component_vector_error_nd_jax(target_points, obs_points, corrlen,
                                          derivative_indices, err, b=0,
                                          nondivergent=True, k_local=None):
    """Local-neighborhood N-D posterior error for streamfunction/velocity potential."""
    import jax
    import jax.numpy as jnp

    scale, lambd = _vector_scale_params_jax(corrlen, derivative_indices)
    obs_scaled = jnp.asarray(obs_points, dtype=jnp.float32) * scale[None, :]
    target_scaled = jnp.asarray(target_points, dtype=jnp.float32) * scale[None, :]
    ix, iy = derivative_indices

    n = obs_scaled.shape[0]
    bmo = b * err / lambd
    var0 = 1.0 + bmo
    k = min(n, k_local if k_local is not None else min(n, 100))

    @jax.jit
    def _local_error(target_pt):
        d2_all = jnp.sum((target_pt - obs_scaled) ** 2, axis=1)
        _, idx = jax.lax.top_k(-d2_all, k)
        obs_local = obs_scaled[idx]

        diff_kk = obs_local[None, :, :] - obs_local[:, None, :]
        dxx_kk = diff_kk[:, :, ix]
        dyy_kk = diff_kk[:, :, iy]
        spatial_d2_kk = dxx_kk ** 2 + dyy_kk ** 2
        total_d2_kk = jnp.sum(diff_kk ** 2, axis=2)
        theta_kk = jnp.arctan2(dyy_kk, dxx_kk)

        A_k = _velocity_cov_block_jax(
            theta_kk, spatial_d2_kk, lambd, bmo,
            nondivergent=nondivergent, d2_total=total_d2_kk,
        ) + err * jnp.eye(2 * k, dtype=jnp.float32)

        diff_c = obs_local - target_pt[None, :]
        dxx_c = diff_c[:, ix]
        dyy_c = diff_c[:, iy]
        spatial_dc2 = dxx_c ** 2 + dyy_c ** 2
        total_dc2 = jnp.sum(diff_c ** 2, axis=1)
        tc = jnp.arctan2(dyy_c, dxx_c)
        Rc = jnp.exp(-lambd * total_dc2) + bmo
        sqrt_dc2 = jnp.sqrt(spatial_dc2)

        if nondivergent:
            p = jnp.concatenate([
                jnp.sin(tc) * sqrt_dc2 * Rc,
                -jnp.cos(tc) * sqrt_dc2 * Rc,
            ])
        else:
            p = jnp.concatenate([
                -jnp.cos(tc) * sqrt_dc2 * Rc,
                -jnp.sin(tc) * sqrt_dc2 * Rc,
            ])

        err_local = 1.0 - (2.0 * lambd) * jnp.dot(p, jnp.linalg.solve(A_k, p)) / var0
        return jnp.clip(err_local, 0.0, 1.0)

    return jax.lax.map(_local_error, target_scaled)


def _helmholtz_nd_numpy(target_points, obs_points, u, v, corrlen_psi, corrlen_chi,
                        derivative_indices, err, b=0):
    """Dense N-D Helmholtz recovery (numpy)."""
    theta_psi, spatial_d2_psi, total_d2_psi, lambd_psi = _vector_obs_geometry_numpy(
        obs_points, corrlen_psi, derivative_indices
    )
    theta_chi, spatial_d2_chi, total_d2_chi, lambd_chi = _vector_obs_geometry_numpy(
        obs_points, corrlen_chi, derivative_indices
    )
    n = len(obs_points)
    bmo_psi = b * err / lambd_psi
    bmo_chi = b * err / lambd_chi
    a = (
        _velocity_cov_block(theta_psi, spatial_d2_psi, lambd_psi, bmo_psi, nondivergent=True, d2_total=total_d2_psi)
        + _velocity_cov_block(theta_chi, spatial_d2_chi, lambd_chi, bmo_chi, nondivergent=False, d2_total=total_d2_chi)
        + err * np.eye(2 * n)
    )
    uv = np.hstack((u, v)).reshape(-1, 1)
    w = np.linalg.solve(a, uv)
    p_psi, _ = _cross_cov_nd_numpy(
        target_points, obs_points, corrlen_psi, derivative_indices, bmo_psi,
        nondivergent=True,
    )
    p_chi, _ = _cross_cov_nd_numpy(
        target_points, obs_points, corrlen_chi, derivative_indices, bmo_chi,
        nondivergent=False,
    )
    return np.dot(p_psi, w).reshape(-1), np.dot(p_chi, w).reshape(-1)


def _helmholtz_nd_jax(target_points, obs_points, u, v, corrlen_psi, corrlen_chi,
                      derivative_indices, err, b=0):
    """Matrix-free N-D Helmholtz recovery (JAX)."""
    import jax
    import jax.numpy as jnp
    from jax.scipy.sparse.linalg import cg

    scale_psi, lambd_psi = _vector_scale_params_jax(corrlen_psi, derivative_indices)
    scale_chi, lambd_chi = _vector_scale_params_jax(corrlen_chi, derivative_indices)
    obs_scaled_psi = jnp.asarray(obs_points, dtype=jnp.float32) * scale_psi[None, :]
    obs_scaled_chi = jnp.asarray(obs_points, dtype=jnp.float32) * scale_chi[None, :]
    target_scaled_psi = jnp.asarray(target_points, dtype=jnp.float32) * scale_psi[None, :]
    target_scaled_chi = jnp.asarray(target_points, dtype=jnp.float32) * scale_chi[None, :]
    ix, iy = derivative_indices

    n = obs_scaled_psi.shape[0]
    bmo_psi = b * err / lambd_psi
    bmo_chi = b * err / lambd_chi
    maxiter = min(2 * n, 400)

    uv = jnp.concatenate([
        jnp.asarray(u, dtype=jnp.float32).ravel(),
        jnp.asarray(v, dtype=jnp.float32).ravel(),
    ])

    matvec_psi = _velocity_matvec_nd_jax(obs_scaled_psi, derivative_indices, n, err,
                                         lambd_psi, bmo_psi, nondivergent=True)
    matvec_chi = _velocity_matvec_nd_jax(obs_scaled_chi, derivative_indices, n, err,
                                         lambd_chi, bmo_chi, nondivergent=False)

    @jax.jit
    def matvec(v):
        return matvec_psi(v) + matvec_chi(v) - err * v

    w, _ = cg(matvec, uv, maxiter=maxiter)
    w_u, w_v = w[:n], w[n:]

    @jax.jit
    def _cross_cov_vec(w_u, w_v):
        def _one_psi(target_pt):
            diff = obs_scaled_psi - target_pt[None, :]
            dxx = diff[:, ix]
            dyy = diff[:, iy]
            spatial_dc2 = dxx ** 2 + dyy ** 2
            total_dc2 = jnp.sum(diff ** 2, axis=1)
            tc = jnp.arctan2(dyy, dxx)
            Rc = jnp.exp(-lambd_psi * total_dc2) + bmo_psi
            sqrt_dc2 = jnp.sqrt(spatial_dc2)
            return (jnp.dot(jnp.sin(tc) * sqrt_dc2 * Rc, w_u)
                    + jnp.dot(-jnp.cos(tc) * sqrt_dc2 * Rc, w_v))

        def _one_chi(target_pt):
            diff = obs_scaled_chi - target_pt[None, :]
            dxx = diff[:, ix]
            dyy = diff[:, iy]
            spatial_dc2 = dxx ** 2 + dyy ** 2
            total_dc2 = jnp.sum(diff ** 2, axis=1)
            tc = jnp.arctan2(dyy, dxx)
            Rc = jnp.exp(-lambd_chi * total_dc2) + bmo_chi
            sqrt_dc2 = jnp.sqrt(spatial_dc2)
            return (jnp.dot(-jnp.cos(tc) * sqrt_dc2 * Rc, w_u)
                    + jnp.dot(-jnp.sin(tc) * sqrt_dc2 * Rc, w_v))

        psi = jax.lax.map(_one_psi, target_scaled_psi)
        chi = jax.lax.map(_one_chi, target_scaled_chi)
        return psi, chi

    return _cross_cov_vec(w_u, w_v)


def _helmholtz_error_nd_numpy(target_points, obs_points, corrlen_psi, corrlen_chi,
                              derivative_indices, err, b=0, k_local=None):
    """Dense N-D Helmholtz posterior errors (numpy)."""
    theta_psi, spatial_d2_psi, total_d2_psi, lambd_psi = _vector_obs_geometry_numpy(
        obs_points, corrlen_psi, derivative_indices
    )
    theta_chi, spatial_d2_chi, total_d2_chi, lambd_chi = _vector_obs_geometry_numpy(
        obs_points, corrlen_chi, derivative_indices
    )
    n = len(obs_points)
    bmo_psi = b * err / lambd_psi
    bmo_chi = b * err / lambd_chi
    a = (
        _velocity_cov_block(theta_psi, spatial_d2_psi, lambd_psi, bmo_psi, nondivergent=True, d2_total=total_d2_psi)
        + _velocity_cov_block(theta_chi, spatial_d2_chi, lambd_chi, bmo_chi, nondivergent=False, d2_total=total_d2_chi)
        + err * np.eye(2 * n)
    )
    p_psi, _ = _cross_cov_nd_numpy(
        target_points, obs_points, corrlen_psi, derivative_indices, bmo_psi,
        nondivergent=True,
    )
    p_chi, _ = _cross_cov_nd_numpy(
        target_points, obs_points, corrlen_chi, derivative_indices, bmo_chi,
        nondivergent=False,
    )
    psi_proj = np.sum(p_psi.T * np.linalg.solve(a, p_psi.T), axis=0)
    chi_proj = np.sum(p_chi.T * np.linalg.solve(a, p_chi.T), axis=0)
    return (
        np.clip(1.0 - (2.0 * lambd_psi) * psi_proj / (1.0 + bmo_psi), 0.0, 1.0),
        np.clip(1.0 - (2.0 * lambd_chi) * chi_proj / (1.0 + bmo_chi), 0.0, 1.0),
    )


def _helmholtz_error_nd_jax(target_points, obs_points, corrlen_psi, corrlen_chi,
                            derivative_indices, err, b=0, k_local=None):
    """Local-neighborhood N-D Helmholtz posterior errors."""
    import jax
    import jax.numpy as jnp

    scale_psi, lambd_psi = _vector_scale_params_jax(corrlen_psi, derivative_indices)
    scale_chi, lambd_chi = _vector_scale_params_jax(corrlen_chi, derivative_indices)
    obs_pts = jnp.asarray(obs_points, dtype=jnp.float32)
    target_pts = jnp.asarray(target_points, dtype=jnp.float32)
    obs_scaled_psi = obs_pts * scale_psi[None, :]
    obs_scaled_chi = obs_pts * scale_chi[None, :]
    ix, iy = derivative_indices

    n = obs_pts.shape[0]
    bmo_psi = b * err / lambd_psi
    bmo_chi = b * err / lambd_chi
    var0_psi = 1.0 + bmo_psi
    var0_chi = 1.0 + bmo_chi
    k = min(n, k_local if k_local is not None else min(n, 100))

    @jax.jit
    def _local_error(target_pt):
        # Use unscaled distance for nearest-neighbor selection (psi/chi
        # have different scalings so no single scaled space is preferred).
        d2_all = jnp.sum((target_pt - obs_pts) ** 2, axis=1)
        _, idx = jax.lax.top_k(-d2_all, k)

        obs_psi_k = obs_scaled_psi[idx]
        obs_chi_k = obs_scaled_chi[idx]
        target_psi = target_pt * scale_psi
        target_chi = target_pt * scale_chi

        # Psi obs-obs geometry
        diff_psi = obs_psi_k[None, :, :] - obs_psi_k[:, None, :]
        dxx_psi = diff_psi[:, :, ix]
        dyy_psi = diff_psi[:, :, iy]
        spatial_d2_psi = dxx_psi ** 2 + dyy_psi ** 2
        total_d2_psi = jnp.sum(diff_psi ** 2, axis=2)
        theta_psi = jnp.arctan2(dyy_psi, dxx_psi)

        # Chi obs-obs geometry
        diff_chi = obs_chi_k[None, :, :] - obs_chi_k[:, None, :]
        dxx_chi = diff_chi[:, :, ix]
        dyy_chi = diff_chi[:, :, iy]
        spatial_d2_chi = dxx_chi ** 2 + dyy_chi ** 2
        total_d2_chi = jnp.sum(diff_chi ** 2, axis=2)
        theta_chi = jnp.arctan2(dyy_chi, dxx_chi)

        A_k = (
            _velocity_cov_block_jax(theta_psi, spatial_d2_psi, lambd_psi, bmo_psi,
                                    nondivergent=True, d2_total=total_d2_psi)
            + _velocity_cov_block_jax(theta_chi, spatial_d2_chi, lambd_chi, bmo_chi,
                                      nondivergent=False, d2_total=total_d2_chi)
            + err * jnp.eye(2 * k, dtype=jnp.float32)
        )

        # Psi cross-covariance
        diff_c_psi = obs_psi_k - target_psi[None, :]
        dxx_c_psi = diff_c_psi[:, ix]
        dyy_c_psi = diff_c_psi[:, iy]
        spatial_dc2_psi = dxx_c_psi ** 2 + dyy_c_psi ** 2
        total_dc2_psi = jnp.sum(diff_c_psi ** 2, axis=1)
        tc_psi = jnp.arctan2(dyy_c_psi, dxx_c_psi)
        Rc_psi = jnp.exp(-lambd_psi * total_dc2_psi) + bmo_psi
        sqrt_dc2_psi = jnp.sqrt(spatial_dc2_psi)
        p_psi = jnp.concatenate([
            jnp.sin(tc_psi) * sqrt_dc2_psi * Rc_psi,
            -jnp.cos(tc_psi) * sqrt_dc2_psi * Rc_psi,
        ])

        # Chi cross-covariance
        diff_c_chi = obs_chi_k - target_chi[None, :]
        dxx_c_chi = diff_c_chi[:, ix]
        dyy_c_chi = diff_c_chi[:, iy]
        spatial_dc2_chi = dxx_c_chi ** 2 + dyy_c_chi ** 2
        total_dc2_chi = jnp.sum(diff_c_chi ** 2, axis=1)
        tc_chi = jnp.arctan2(dyy_c_chi, dxx_c_chi)
        Rc_chi = jnp.exp(-lambd_chi * total_dc2_chi) + bmo_chi
        sqrt_dc2_chi = jnp.sqrt(spatial_dc2_chi)
        p_chi = jnp.concatenate([
            -jnp.cos(tc_chi) * sqrt_dc2_chi * Rc_chi,
            -jnp.sin(tc_chi) * sqrt_dc2_chi * Rc_chi,
        ])

        psi_err = 1.0 - (2.0 * lambd_psi) * jnp.dot(p_psi, jnp.linalg.solve(A_k, p_psi)) / var0_psi
        chi_err = 1.0 - (2.0 * lambd_chi) * jnp.dot(p_chi, jnp.linalg.solve(A_k, p_chi)) / var0_chi
        return jnp.clip(psi_err, 0.0, 1.0), jnp.clip(chi_err, 0.0, 1.0)

    psi_err, chi_err = jax.lax.map(_local_error, target_pts)
    return psi_err, chi_err


def _pack_2d(xc, yc, x, y):
    """Convert 2D public-API arguments into N-D point arrays.

    Returns target_points, target_shape, obs_points.
    """
    xc = np.asarray(xc)
    yc = np.asarray(yc)
    target_shape = xc.shape
    target_points = np.column_stack([xc.ravel(), yc.ravel()])
    obs_points = np.column_stack([np.asarray(x).ravel(), np.asarray(y).ravel()])
    return target_points, target_shape, obs_points


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
        per target point) for O(n) scaling.
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
    target_points, target_shape, obs_points = _pack_2d(xc, yc, x, y)
    corrlen = np.array([corrlenx, corrleny])
    if backend == "numpy":
        ep = _scalar_error_nd_numpy(target_points, obs_points, corrlen, err,
                                    k_local=k_local)
    elif backend == "jax":
        ep = _scalar_error_nd_jax(target_points, obs_points, corrlen, err,
                                  k_local=k_local)
    else:
        raise ValueError(f"Unknown backend {backend!r}. Choose 'numpy' or 'jax'.")
    return np.asarray(ep).reshape(target_shape)


def scalar_error(xc, yc, x, y, corrlenx=None, corrleny=None, err=None,
                 backend="numpy", k_local=None):
    """Alias for :func:`error` for API consistency with vector methods."""
    return error(
        xc, yc, x, y, corrlenx=corrlenx, corrleny=corrleny, err=err,
        backend=backend, k_local=k_local,
    )


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
        Use ``"jax"`` for lower memory usage via matrix-free conjugate
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
    target_points, target_shape, obs_points = _pack_2d(xc, yc, x, y)
    corrlen = np.array([corrlenx, corrleny])
    if backend == "numpy":
        tp = _scalar_nd_numpy(target_points, obs_points, t, corrlen, err)
    elif backend == "jax":
        tp = _scalar_nd_jax(target_points, obs_points, t, corrlen, err)
    else:
        raise ValueError(f"Unknown backend {backend!r}. Choose 'numpy' or 'jax'.")
    return np.asarray(tp).reshape(target_shape)


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
    target_points, target_shape, obs_points = _pack_2d(xc, yc, x, y)
    corrlen_arr = np.array([corrlenx, corrleny])
    derivative_indices = (0, 1)
    if backend == "numpy":
        psi = _streamfunction_nd_numpy(target_points, obs_points, u, v,
                                       corrlen_arr, derivative_indices, err, b=b)
    elif backend == "jax":
        psi = _streamfunction_nd_jax(target_points, obs_points, u, v,
                                     corrlen_arr, derivative_indices, err, b=b)
    else:
        raise ValueError(f"Unknown backend {backend!r}. Choose 'numpy' or 'jax'.")
    return np.asarray(psi).reshape(target_shape)


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
    target_points, target_shape, obs_points = _pack_2d(xc, yc, x, y)
    corrlen_arr = np.array([corrlenx, corrleny])
    derivative_indices = (0, 1)
    if backend == "numpy":
        chi = _velocity_potential_nd_numpy(target_points, obs_points, u, v,
                                           corrlen_arr, derivative_indices, err, b=b)
    elif backend == "jax":
        chi = _velocity_potential_nd_jax(target_points, obs_points, u, v,
                                         corrlen_arr, derivative_indices, err, b=b)
    else:
        raise ValueError(f"Unknown backend {backend!r}. Choose 'numpy' or 'jax'.")
    return np.asarray(chi).reshape(target_shape)


def streamfunction_error(xc, yc, x, y, corrlenx, corrleny, err,
                         b=0, backend="numpy", k_local=None):
    """Normalized posterior error for streamfunction recovery."""
    target_points, target_shape, obs_points = _pack_2d(xc, yc, x, y)
    corrlen_arr = np.array([corrlenx, corrleny])
    derivative_indices = (0, 1)
    if backend == "numpy":
        ep = _single_component_vector_error_nd_numpy(
            target_points, obs_points, corrlen_arr, derivative_indices, err,
            b=b, nondivergent=True,
        )
    elif backend == "jax":
        ep = _single_component_vector_error_nd_jax(
            target_points, obs_points, corrlen_arr, derivative_indices, err,
            b=b, nondivergent=True, k_local=k_local,
        )
    else:
        raise ValueError(f"Unknown backend {backend!r}. Choose 'numpy' or 'jax'.")
    return np.asarray(ep).reshape(target_shape)


def velocity_potential_error(xc, yc, x, y, corrlenx, corrleny, err,
                             b=0, backend="numpy", k_local=None):
    """Normalized posterior error for velocity-potential recovery."""
    target_points, target_shape, obs_points = _pack_2d(xc, yc, x, y)
    corrlen_arr = np.array([corrlenx, corrleny])
    derivative_indices = (0, 1)
    if backend == "numpy":
        ep = _single_component_vector_error_nd_numpy(
            target_points, obs_points, corrlen_arr, derivative_indices, err,
            b=b, nondivergent=False,
        )
    elif backend == "jax":
        ep = _single_component_vector_error_nd_jax(
            target_points, obs_points, corrlen_arr, derivative_indices, err,
            b=b, nondivergent=False, k_local=k_local,
        )
    else:
        raise ValueError(f"Unknown backend {backend!r}. Choose 'numpy' or 'jax'.")
    return np.asarray(ep).reshape(target_shape)


def helmholtz_error(xc, yc, x, y,
                    corrlenx_psi, corrleny_psi,
                    corrlenx_chi, corrleny_chi,
                    err, b=0, backend="numpy", k_local=None):
    """Normalized posterior errors for Helmholtz streamfunction and chi."""
    target_points, target_shape, obs_points = _pack_2d(xc, yc, x, y)
    corrlen_psi = np.array([corrlenx_psi, corrleny_psi])
    corrlen_chi = np.array([corrlenx_chi, corrleny_chi])
    derivative_indices = (0, 1)
    if backend == "numpy":
        psi_err, chi_err = _helmholtz_error_nd_numpy(
            target_points, obs_points, corrlen_psi, corrlen_chi,
            derivative_indices, err, b=b,
        )
    elif backend == "jax":
        psi_err, chi_err = _helmholtz_error_nd_jax(
            target_points, obs_points, corrlen_psi, corrlen_chi,
            derivative_indices, err, b=b, k_local=k_local,
        )
    else:
        raise ValueError(f"Unknown backend {backend!r}. Choose 'numpy' or 'jax'.")
    return (
        np.asarray(psi_err).reshape(target_shape),
        np.asarray(chi_err).reshape(target_shape),
    )


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
    target_points, target_shape, obs_points = _pack_2d(xc, yc, x, y)
    corrlen_psi_arr = np.array([corrlenx_psi, corrleny_psi])
    corrlen_chi_arr = np.array([corrlenx_chi, corrleny_chi])
    derivative_indices = (0, 1)
    if backend == "numpy":
        psi, chi = _helmholtz_nd_numpy(
            target_points, obs_points, u, v,
            corrlen_psi_arr, corrlen_chi_arr, derivative_indices, err, b=b,
        )
    elif backend == "jax":
        psi, chi = _helmholtz_nd_jax(
            target_points, obs_points, u, v,
            corrlen_psi_arr, corrlen_chi_arr, derivative_indices, err, b=b,
        )
    else:
        raise ValueError(f"Unknown backend {backend!r}. Choose 'numpy' or 'jax'.")
    return (
        np.asarray(psi).reshape(target_shape),
        np.asarray(chi).reshape(target_shape),
    )
