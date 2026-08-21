"""Shannon-entropy diversity/dependence measures, and the Israel-Rosenthal-Wei
(2001) technique for repairing an estimated Markov generator matrix that
fails to satisfy the generator constraints (non-negative off-diagonal
entries, zero row sums) because it was estimated from discretely-observed
transition data.

Ported from HSF toolbox `hsf/{shannon_entropy,
shannon_entropy_markov_chain,estimate_markov_generator}.m`.

Translation notes:

- MATLAB's ``missrv(x, v)`` (replace non-finite/flagged entries of `x` with
  `v`) is used throughout the originals to implement the ``0 * log(0) = 0``
  convention; translated as ``scipy.special.xlogy(p, p)``, which computes
  ``p * log(p)`` and returns exactly `0.0` at `p == 0` without ever
  evaluating ``log(0)`` (so, unlike ``np.where(p > 0, p * np.log(p), 0.0)``,
  it raises no divide-by-zero warning).
- `shannon_entropy_markov_chain.m` approximates the Markov chain's
  stationary distribution via ``expm(Lambda * 1000)`` (a long-horizon
  transition-probability matrix, every row of which has converged to the
  stationary distribution) -- kept as-is via `scipy.linalg.expm`.
- `estimate_markov_generator.m` implements Israel, Rosenthal & Wei (2001)'s
  two generator-repair methods for a possibly-invalid estimated generator
  `Lambda` (e.g. from `Lambda = logm(P) / dt` on an empirical transition
  matrix `P`, which need not itself be a valid generator):

  - ``Lambda1`` ("diagonal adjustment"): zero out negative off-diagonal
    entries and absorb the removed mass into the diagonal. This exactly
    preserves each row's original sum (``min(x, 0) + max(x, 0) = x``), so
    ``Lambda1``'s row sums equal ``Lambda``'s row sums (zero, if `Lambda`
    already had zero row sums as intended).
  - ``Lambda2`` ("proportional redistribution"): redistribute each row's
    negative mass proportionally across that row's positive off-diagonal
    entries, leaving rows with no positive off-diagonal mass (`g[i] == 0`)
    unchanged.

  Both are standard generator-matrix regularizations in credit-rating
  transition-intensity estimation (Roncalli's `hsf` toolbox applies this to
  rating-migration generators); the row-sum-preservation property of
  ``Lambda1`` is verified in this module's tests.
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
    """Repair a possibly-invalid estimated Markov generator `lam` (e.g. one
    with negative off-diagonal entries) into two valid generators
    (non-negative off-diagonal, zero row sums), via Israel, Rosenthal & Wei
    (2001)'s two methods.

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
