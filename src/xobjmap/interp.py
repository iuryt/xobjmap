"""
Core interpolation functions using Gauss-Markov optimal estimation.
"""

import numpy as np


def scaloa(xc, yc, x, y, t=None, corrlenx=None, corrleny=None, err=None):
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
    t : array_like, optional
        Observed scalar values at (x, y). If None, only the error map
        is returned.
    corrlenx : float
        Correlation length scale in the x-direction.
    corrleny : float
        Correlation length scale in the y-direction.
    err : float
        Normalized random error variance (0 < err < 1).

    Returns
    -------
    tp : numpy.ndarray
        Interpolated field at target locations. Only returned if t is
        provided.
    ep : numpy.ndarray
        Normalized mean squared error at target locations. Taking the
        square root gives the interpolation error as a fraction.

    Notes
    -----
    Anisotropy is handled by rescaling the x-coordinate by the ratio
    corrleny / corrlenx before computing distances, so that the effective
    correlation length is isotropic in the rescaled space.

    References
    ----------
    Bretherton, F. P., Davis, R. E., & Fandry, C. B. (1976).
    A technique for objective analysis and design of oceanographic
    experiments applied to MODE-73. Deep-Sea Research, 23(7), 559-582.
    """
    corrlen = corrleny
    xc = np.asarray(xc) * (corrleny / corrlenx)
    x = np.asarray(x) * (corrleny / corrlenx)
    yc = np.asarray(yc)
    y = np.asarray(y)

    n = len(x)
    x = x.reshape(1, n)
    y = y.reshape(1, n)

    # Squared distance matrix between observations
    d2 = (
        (np.tile(x, (n, 1)).T - np.tile(x, (n, 1))) ** 2
        + (np.tile(y, (n, 1)).T - np.tile(y, (n, 1))) ** 2
    )

    nv = len(xc)
    xc = xc.reshape(1, nv)
    yc = yc.reshape(1, nv)

    # Squared distance between observations and target points
    dc2 = (
        (np.tile(xc, (n, 1)).T - np.tile(x, (nv, 1))) ** 2
        + (np.tile(yc, (n, 1)).T - np.tile(y, (nv, 1))) ** 2
    )

    # Correlation matrix (A) and cross-correlation (C)
    A = (1 - err) * np.exp(-d2 / corrlen**2)
    C = (1 - err) * np.exp(-dc2 / corrlen**2)

    # Add diagonal sampling error
    A = A + err * np.eye(n)

    # Normalized mean error
    ep = 1 - np.sum(C.T * np.linalg.solve(A, C.T), axis=0) / (1 - err)

    if t is not None:
        t = np.asarray(t).reshape(n, 1)
        tp = np.dot(C, np.linalg.solve(A, t))
        return tp, ep

    return ep


def vectoa(xc, yc, x, y, u, v, corrlenx, corrleny, err, b=0):
    """
    Vectorial objective analysis for velocity fields.

    Interpolates scattered velocity observations (u, v) onto a grid
    (xc, yc), returning the streamfunction field. The method assumes a
    Gaussian streamfunction covariance and derives the velocity-velocity
    and streamfunction-velocity cross-covariance matrices analytically.

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

    # Stack velocity observations into a single column vector [u; v]
    uv = np.hstack((u, v)).reshape(-1, 1)

    # Angles and distances between all observation pairs
    dy_pairs = -np.tile(y, (n, 1)).T + np.tile(y, (n, 1))
    dx_pairs = -np.tile(x, (n, 1)).T + np.tile(x, (n, 1))
    t = np.arctan2(dy_pairs, dx_pairs)

    d2 = (
        (np.tile(x, (n, 1)).T - np.tile(x, (n, 1))) ** 2
        + (np.tile(y, (n, 1)).T - np.tile(y, (n, 1))) ** 2
    )

    lambd = 1 / (corrlen**2)
    bmo = b * err / lambd

    # Longitudinal and transverse covariance
    R = np.exp(-lambd * d2) + bmo
    S = np.exp(-lambd * d2) * (1 - 2 * lambd * d2) + bmo

    # Velocity-velocity covariance matrix (2n x 2n)
    A = np.zeros((2 * n, 2 * n))
    A[0:n, 0:n] = (np.cos(t) ** 2) * (R - S) + S
    A[0:n, n : 2 * n] = np.cos(t) * np.sin(t) * (R - S)
    A[n : 2 * n, 0:n] = A[0:n, n : 2 * n]
    A[n : 2 * n, n : 2 * n] = (np.sin(t) ** 2) * (R - S) + S
    A = A + err * np.eye(2 * n)

    # Target grid dimensions
    nv1, nv2 = xc.shape
    nv = nv1 * nv2
    xc_flat = xc.T.ravel()
    yc_flat = yc.T.ravel()

    # Angles and distances from target points to observation points
    dy_cross = -np.tile(yc_flat, (n, 1)).T + np.tile(y, (nv, 1))
    dx_cross = -np.tile(xc_flat, (n, 1)).T + np.tile(x, (nv, 1))
    tc = np.arctan2(dy_cross, dx_cross)

    dc2 = (
        (np.tile(xc_flat, (n, 1)).T - np.tile(x, (nv, 1))) ** 2
        + (np.tile(yc_flat, (n, 1)).T - np.tile(y, (nv, 1))) ** 2
    )
    Rc = np.exp(-lambd * dc2) + bmo

    # Streamfunction-velocity cross-covariance (nv x 2n)
    P = np.zeros((nv, 2 * n))
    P[:, 0:n] = np.sin(tc) * np.sqrt(dc2) * Rc
    P[:, n : 2 * n] = -np.cos(tc) * np.sqrt(dc2) * Rc

    # Solve for streamfunction
    PSI = np.dot(P, np.linalg.solve(A, uv))
    PSI = PSI.reshape(nv2, nv1).T

    return PSI
