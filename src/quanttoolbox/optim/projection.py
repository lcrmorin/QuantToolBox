"""Projection operators onto L1/L2/Linf norm balls and a box-intersect-L2-ball set.

Ported from QuantToolbox/optim/{projection_L1,projection_L2,
projection_Linfinity,projection_box_L2}.m

Translation notes:

- ``projection_L2`` is literally ``v - proximal_l2(v, lambda)`` in the
  original (projection = identity minus the proximal/shrinkage step);
  reproduced as-is here rather than re-derived, to stay faithful.
- ``projection_box_L2`` (project onto a box intersected with an L2 ball
  centered at c) uses Dykstra's alternating projection algorithm, same
  pattern as ``proximal_linear_constraints``.
"""

from __future__ import annotations

import numpy as np

from quanttoolbox.config import ProximalConfig
from quanttoolbox.optim.proximal import proximal_bounds, proximal_l2, proximal_max


def projection_l1(v: np.ndarray, radius: float, method: int = 1) -> np.ndarray:
    """Euclidean projection of v onto the L1 ball of the given radius.

    method=1 (default): exact sorted-cumsum water-filling algorithm.
    method=2: equivalent computation via proximal_max.

    Original: optim/projection_L1.m
    """
    v = np.asarray(v, dtype=float)
    if np.sum(np.abs(v)) <= radius:
        return v.copy()

    if method == 1:
        n = v.shape[0]
        abs_v = np.sort(np.abs(v))[::-1]
        mu = (np.cumsum(abs_v) - radius) / np.arange(1, n + 1)
        rk = np.where(abs_v > mu)[0][-1]
        lambda_star = mu[rk]
        from quanttoolbox.optim.proximal import soft_thresholding

        return soft_thresholding(v, lambda_star)
    else:
        return v - proximal_max(np.abs(v), radius) * np.sign(v)


def projection_l2(v: np.ndarray, lambda_: float) -> np.ndarray:
    """Euclidean projection removing at most lambda_ of v's L2 length
    (identity minus the L2 proximal/shrinkage step).

    Original: optim/projection_L2.m
    """
    v = np.asarray(v, dtype=float)
    return v - proximal_l2(v, lambda_)


def projection_linfinity(v: np.ndarray, radius: float) -> np.ndarray:
    """Euclidean projection of v onto the L-infinity ball of the given radius
    (simple elementwise clip to [-radius, radius]).

    Original: optim/projection_Linfinity.m
    """
    v = np.asarray(v, dtype=float)
    return np.clip(v, -radius, radius)


def projection_box_l2(
    v: np.ndarray,
    x_minus: np.ndarray | float,
    x_plus: np.ndarray | float,
    c: np.ndarray,
    lambda_: float,
    config: ProximalConfig | None = None,
) -> tuple[np.ndarray, int]:
    """Projection onto the intersection of a box [x_minus, x_plus] and an L2
    ball of radius lambda_ centered at c, via Dykstra's alternating
    projection algorithm.

    Original: optim/projection_box_L2.m
    """
    if config is None:
        config = ProximalConfig()

    v = np.asarray(v, dtype=float)
    c = np.asarray(c, dtype=float)
    n = v.shape[0]

    x2 = v.copy()
    delta1 = np.zeros(n)
    delta2 = np.zeros(n)
    retcode = 1

    for _it in range(config.max_iters):
        x1 = proximal_bounds(x2 + delta1, x_minus, x_plus)
        delta1 = x2 + delta1 - x1

        x2_new = c + projection_l2((x1 + delta2 - c), lambda_)
        delta2 = x1 + delta2 - x2_new

        if np.allclose(x1, x2_new):
            x2 = x2_new
            break
        x2 = x2_new
    else:
        retcode = -1

    return x1, retcode
