"""
This module provides some commonly used functions.
"""


__all__ = [
    "Rotation",
    "ValidRotationForXBond",
    "ValidRotationForYBond",
    "ValidRotationForZBond",
]


import numpy as np
from numba import jit, int64, float64


@jit(float64[:, :](float64, float64, float64), nopython=True, cache=True)
def Rotation(alpha, beta, theta):
    """
    Proper rotation in 3 dimensions.

    `alpha` and `beta` specify the rotation axis and `theta` is the rotation
    angle.

    Parameters
    ----------
    alpha : float
        The angle between the positive x-axis and the projection of the
        rotation axis in the xy-plane.
    beta : float
        The angle between the positive z-axis and the rotation axis.
    theta : float
        The rotation angle.

    Returns
    -------
    R : (3, 3) array
        The corresponding transformation matrix.
    """

    sin_alpha = np.sin(alpha)
    cos_alpha = np.cos(alpha)
    sin_beta = np.sin(beta)
    cos_beta = np.cos(beta)
    sin_theta = np.sin(theta)
    cos_theta = np.cos(theta)

    RzAlpha = np.zeros((3, 3), dtype=np.float64)
    RzAlpha[0, 0] = cos_alpha
    RzAlpha[1, 1] = cos_alpha
    RzAlpha[2, 2] = 1.0
    RzAlpha[0, 1] = -sin_alpha
    RzAlpha[1, 0] = sin_alpha

    RyBeta = np.zeros((3, 3), dtype=np.float64)
    RyBeta[0, 0] = cos_beta
    RyBeta[1, 1] = 1.0
    RyBeta[2, 2] = cos_beta
    RyBeta[0, 2] = sin_beta
    RyBeta[2, 0] = -sin_beta

    RzTheta = np.zeros((3, 3), dtype=np.float64)
    RzTheta[0, 0] = cos_theta
    RzTheta[1, 1] = cos_theta
    RzTheta[2, 2] = 1.0
    RzTheta[0, 1] = -sin_theta
    RzTheta[1, 0] = sin_theta

    RAxis = np.dot(RzAlpha, RyBeta)
    return np.dot(RAxis, np.dot(RzTheta, RAxis.T))


@jit(int64(float64, float64, float64, float64), nopython=True, cache=True)
def isclose(a, b, rtol, atol):
    """
    Return True if `a` and `b` are equal within a tolerance.

    If the following equation is True, then `isclose` returns True.
        `absolute(a - b) <= (atol + rtol * absolute(b))`
    The above equation is not symmetric in `a` and `b`, so that
    `isclose(a, b, rtol, atol)` might be different from
    `isclose(b, a, rtol, atol)` in some rare case.

    Parameters
    ----------
    a, b : float
        Input numbers to compare.
    rtol : float
        The relative tolerance parameter.
    atol : float
        The absolute tolerance parameter.

    Returns
    -------
    res : bool
        Returns True if the two numbers are equal within the given tolerance.
    """

    return np.absolute(a - b) <= (atol + rtol * np.absolute(b))


@jit(
    int64(float64[:, :], float64[:, :], float64[:, :], float64, float64),
    nopython=True, cache=True
)
def ValidRotationForXBond(R0, R1, hijx, rtol, atol):
    """
    Whether `R0` and `R1` specifies self-dual rotation for the given `hijx`.

    The `hijx` should be of the following form:
        hijx = [[J+K, G1, G1], [G1, J, G0], [G1, G0, J]]
    where J, K, G0 and G1 are arbitrary real numbers.
    If the rotated hijx: hijx_new = np.dot(R0, np.dot(hijx, R1.T)) is of the
    following form:
        hijx_new = [
            [JNew+KNew, G1New, G1New],
            [G1New, JNew, G0New],
            [G1New, G0New, JNew],
        ]
    where JNew, KNew, G0New and G1New are arbitrary real numbers, then `R0`
    and `R1` specify a self-dual transformation and this function returns True.

    Parameters
    ----------
    R0, R1 : (3, 3) array
        The rotation matrices for spin operators on site-i and site-j.
    hijx : (3, 3) array
        The corresponding coefficients matrix on bond-ij.
    rtol : float
        The relative tolerance parameter.
    atol : float
        The absolute tolerance parameter.

    Returns
    -------
    res : bool
        Return True if the rotated coefficient matrix be of the right form.
    """

    hijx_new = np.dot(R0, np.dot(hijx, R1.T))
    G1 = hijx_new[0, 1]
    judge0 = isclose(G1, hijx_new[0, 2], rtol, atol)
    judge1 = isclose(G1, hijx_new[1, 0], rtol, atol)
    judge2 = isclose(G1, hijx_new[2, 0], rtol, atol)
    judge3 = isclose(hijx_new[1, 1], hijx_new[2, 2], rtol, atol)
    judge4 = isclose(hijx_new[1, 2], hijx_new[2, 1], rtol, atol)
    return judge0 and judge1 and judge2 and judge3 and judge4


@jit(
    int64(float64[:, :], float64[:, :], float64[:, :], float64, float64),
    nopython=True, cache=True
)
def ValidRotationForYBond(R0, R1, hijy, rtol, atol):
    """
    Whether `R0` and `R1` specifies self-dual rotation for the given `hijy`.

    The `hijy` should be of the following form:
        hijy = [[J, G1, G0], [G1, J+K, G1], [G0, G1, J]]
    where J, K, G0 and G1 are arbitrary real numbers.
    If the rotated hijy: hijy_new = np.dot(R0, np.dot(hijy, R1.T)) is of the
    following form:
        hijy_new = [
            [JNew, G1New, G0new],
            [G1New, JNew+KNew, G1New],
            [G0New, G1New, JNew],
        ]
    where JNew, KNew, G0New and G1New are arbitrary real numbers, then `R0`
    and `R1` specify a self-dual transformation and this function returns True.

    Parameters
    ----------
    R0, R1 : (3, 3) array
        The rotation matrices for spin operators on site-i and site-j.
    hijy : (3, 3) array
        The corresponding coefficients matrix on bond-ij.
    rtol : float
        The relative tolerance parameter.
    atol : float
        The absolute tolerance parameter.

    Returns
    -------
    res : bool
        Return True if the rotated coefficient matrix be of the right form.
    """

    hijy_new = np.dot(R0, np.dot(hijy, R1.T))
    G1 = hijy_new[0, 1]
    judge0 = isclose(G1, hijy_new[1, 0], rtol, atol)
    judge1 = isclose(G1, hijy_new[1, 2], rtol, atol)
    judge2 = isclose(G1, hijy_new[2, 1], rtol, atol)
    judge3 = isclose(hijy_new[0, 0], hijy_new[2, 2], rtol, atol)
    judge4 = isclose(hijy_new[0, 2], hijy_new[2, 0], rtol, atol)
    return judge0 and judge1 and judge2 and judge3 and judge4


@jit(
    int64(float64[:, :], float64[:, :], float64[:, :], float64, float64),
    nopython=True, cache=True
)
def ValidRotationForZBond(R0, R1, hijz, rtol, atol):
    """
    Whether `R0` and `R1` specifies self-dual rotation for the given `hijz`.

    The `hijz` should be of the following form:
        hijz = [[J, G0, G1], [G0, J, G1], [G1, G1, J+K]]
    where J, K, G0, G1 are arbitrary real numbers.
    If the rotated hijz: hijz_new = np.dot(R0, np.dot(hijz, R1.T)) is of the
    following form:
        hijz_new = [
            [JNew, G0New, G1New],
            [G0New, JNew, G1New],
            [G1New, G1New, JNew+KNew],
        ]
    where JNew, KNew, G0New and G1New are arbitrary real numbers, then `R0`
    and `R1` specify a self-dual transformation and this function returns True.

    Parameters
    ----------
    R0, R1 : (3, 3) array
        The rotation matrix for spin operator on site-i and site-j.
    hijz : (3, 3) array
        The corresponding coefficients matrix on bond-ij.
    rtol : float
        The relative tolerance parameter.
    atol : float
        The absolute tolerance parameter.

    Returns
    -------
    res : bool
        Return True if the rotated coefficient matrix be of the right form.
    """

    hijz_new = np.dot(R0, np.dot(hijz, R1.T))
    G1 = hijz_new[0, 2]
    judge0 = isclose(G1, hijz_new[1, 2], rtol, atol)
    judge1 = isclose(G1, hijz_new[2, 0], rtol, atol)
    judge2 = isclose(G1, hijz_new[2, 1], rtol, atol)
    judge3 = isclose(hijz_new[0, 0], hijz_new[1, 1], rtol, atol)
    judge4 = isclose(hijz_new[0, 1], hijz_new[1, 0], rtol, atol)
    return judge0 and judge1 and judge2 and judge3 and judge4
