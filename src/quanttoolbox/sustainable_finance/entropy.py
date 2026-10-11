"""Shannon entropy measures and Markov generator repair.

Ported from HSF toolbox
``hsf/{shannon_entropy,shannon_entropy_markov_chain,estimate_markov_generator}.m``.

Entropy calculations use ``scipy.special.xlogy`` to handle ``0 * log(0)``
without warnings. Markov-chain entropy retains the source's long-horizon
stationary-distribution approximation, ``expm(Lambda * 1000)``.

Generator repair follows Israel, Rosenthal & Wei (2001). Diagonal
adjustment removes negative off-diagonal entries and adjusts the diagonal,
preserving the input row sums. Proportional redistribution allocates each
row's negative mass across its positive off-diagonal entries; rows with no
positive off-diagonal mass are unchanged.
"""

from __future__ import annotations

import numpy as np
from scipy.linalg import expm
from scipy.special import xlogy


def shannon_entropy(p_xy: np.ndarray) -> tuple[float, float, float, float]:
    """Shannon entropy/mutual-information decomposition of a discrete
    distribution: `p_xy` may be a 1D probability vector (single variable,
    in which case `I_Y = I_XY = 0` and `I_X_Y = I_X`), or a 2D joint
    probability matrix (rows = X, columns = Y).

    Returns ``(I_X, I_Y, I_XY, I_X_Y)``: the marginal entropy of X, the
    marginal entropy of Y, the mutual information between X and Y, and the
    joint entropy of (X, Y).

    Original: hsf/shannon_entropy.m
    """
    p_xy = np.asarray(p_xy, dtype=float)

    if p_xy.ndim == 1:
        p_x = p_xy
        i_x = -float(np.sum(xlogy(p_x, p_x)))
        return i_x, 0.0, 0.0, i_x

    p_x = np.sum(p_xy, axis=1)
    p_y = np.sum(p_xy, axis=0)

    i_x = -float(np.sum(xlogy(p_x, p_x)))
    i_y = -float(np.sum(xlogy(p_y, p_y)))
    i_xy_joint = -float(np.sum(xlogy(p_xy, p_xy)))

    i_xy = i_x + i_y - i_xy_joint
    return i_x, i_y, i_xy, i_xy_joint


def shannon_entropy_markov_chain(
    lambda_: np.ndarray, t: np.ndarray | float
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Shannon entropy/mutual-information decomposition of the joint
    distribution of a continuous-time Markov chain's state at times 0 and
    `t`, with generator `lambda_` and stationary (long-run, ``t -> inf``)
    distribution approximated via ``expm(lambda_ * 1000)``.

    `t` may be an array of horizons; returns one value per horizon for
    each of ``(I_X, I_Y, I_XY, I_X_Y)``.

    Original: hsf/shannon_entropy_markov_chain.m
    """
    lambda_ = np.asarray(lambda_, dtype=float)
    t_arr = np.atleast_1d(np.asarray(t, dtype=float))
    n = lambda_.shape[0]

    p_inf = expm(lambda_ * 1000.0)
    pi = p_inf[0, :]

    n_t = t_arr.shape[0]
    i_x = np.zeros(n_t)
    i_y = np.zeros(n_t)
    i_xy = np.zeros(n_t)
    i_xy_joint = np.zeros(n_t)

    for it in range(n_t):
        if t_arr[it] == 0.0:
            p_t = np.eye(n)
        else:
            p_t = expm(lambda_ * t_arr[it])
        p_xy = pi[:, None] * p_t
        i_x[it], i_y[it], i_xy[it], i_xy_joint[it] = shannon_entropy(p_xy)

    return i_x, i_y, i_xy, i_xy_joint


def estimate_markov_generator(lam: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Apply diagonal adjustment and proportional redistribution to ``lam``.

    The input is an estimated Markov generator with zero row sums that
    may contain negative off-diagonal entries. Diagonal adjustment
    preserves the input row sums; it does not normalize them. Proportional
    redistribution leaves rows without positive off-diagonal mass unchanged.
    The methods follow Israel, Rosenthal & Wei (2001).

    Returns ``(Lambda1, Lambda2)`` -- see the module docstring for the
    method descriptions.

    Original: hsf/estimate_markov_generator.m
    """
    lam = np.asarray(lam, dtype=float)
    k = lam.shape[0]
    off_diag_mask = ~np.eye(k, dtype=bool)

    neg_off_diag = np.where(off_diag_mask, np.minimum(lam, 0.0), 0.0)
    pos_off_diag = np.where(off_diag_mask, np.maximum(lam, 0.0), 0.0)
    row_neg_sum = neg_off_diag.sum(axis=1)
    row_pos_sum = pos_off_diag.sum(axis=1)

    new_diag = np.diag(lam) + row_neg_sum
    lambda1 = pos_off_diag.copy()
    np.fill_diagonal(lambda1, new_diag)

    g = np.abs(np.diag(lam)) + row_pos_sum
    b = -row_neg_sum
    g_col = g[:, None]
    safe_g = np.where(g_col > 0, g_col, 1.0)
    lambda2 = lam - b[:, None] * np.abs(lam) / safe_g
    lambda2 = np.where(g_col > 0, lambda2, lam)

    neg_off_diag_mask = off_diag_mask & (lam < 0)
    lambda2 = np.where(neg_off_diag_mask, 0.0, lambda2)

    return lambda1, lambda2
