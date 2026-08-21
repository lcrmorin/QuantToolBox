"""Reduced-form (intensity/hazard-based) credit models: default-time
survival/density/hazard functions implied by a continuous-time Markov
generator matrix (e.g. a credit-rating transition-intensity matrix, with
default as the absorbing state), and the (piecewise-)exponential default
model used elsewhere in the toolbox for simulating default times.

Ported from HSF toolbox `credit/{Survival_Markov_Generator,
Density_Markov_Generator,Hazard_Markov_Generator,survivalExponential,
cdfExponential,pdfExponential,invExponential,rndExponential}.m`.

Translation notes:

- `Hazard_Markov_Generator.m`'s function body is declared as ``function
  lambda = Density_Markov_Generator(t, Lambda)`` -- an apparent
  copy-paste error in the original (the internal function name doesn't
  match its own filename or its actual computation, ``f / S``). MATLAB
  still dispatches by *filename* when calling `Hazard_Markov_Generator(...)`
  from another script, so the bug is silently harmless in the original.
  Named `hazard_markov_generator` here, matching the filename and the
  actual computation rather than the erroneous internal name.
- `survivalExponential.m`/`pdfExponential.m`/`invExponential.m` all branch
  on ``size(lambda, 2) == 1`` to distinguish a *homogeneous* per-scenario
  hazard-rate vector from a *piecewise-constant* hazard matrix (knots in
  column 1, per-scenario rates in the remaining columns) -- but the
  homogeneous docstring also describes a "1 x C" row-vector case that, if
  ``C > 1``, would actually have ``size(lambda, 2) == C != 1`` and fall
  through to the (wrong) piecewise branch. That row-vector case is
  unreachable under the code's own dispatch condition; only the
  column-vector ("C x 1") reading is actually exercised anywhere in the
  original. This ambiguity doesn't translate cleanly to numpy (which has
  no row/column distinction for 1-D arrays), so the Python API instead
  dispatches on array dimensionality: a 1-D `lambda_` (shape ``(C,)``) is
  always the homogeneous case, and a 2-D `lambda_` (shape ``(M, 1+C)``)
  is always the piecewise case -- unambiguous, and consistent with the
  cases the original code actually exercises.
- MATLAB's ``discretize(t, edges)`` (returns the 1-based bin index of
  each element of `t`, or `NaN` outside all bins) is reimplemented as a
  private `_discretize_bin` helper via `numpy.searchsorted`, returning a
  0-based index and clamping out-of-range values to the first bin --
  matching the originals' own "safeguard for t < 0" `NaN`-clamping
  (`idx(isnan(idx)) = 1`), since every bin sequence here already extends
  to `+inf` on the right.
"""

from __future__ import annotations

import numpy as np
from scipy.linalg import expm


def survival_markov_generator(t: np.ndarray | float, lambda_matrix: np.ndarray) -> np.ndarray:
    """Survival probabilities ``S(t) = Pr(tau > t)`` implied by a
    continuous-time Markov generator `lambda_matrix` (K x K), where state
    K is the absorbing "default" state: ``S(t) = 1 - expm(t * Lambda)[:,
    K-1]``.

    Returns an array of shape ``(len(t), K)``.

    Original: credit/Survival_Markov_Generator.m
    """
    t_arr = np.atleast_1d(np.asarray(t, dtype=float))
    lambda_matrix = np.asarray(lambda_matrix, dtype=float)
    k = lambda_matrix.shape[1]
    n = t_arr.shape[0]

    s = np.zeros((n, k))
    for i in range(n):
        if t_arr[i] == 0.0:
            s[i, :] = 1.0
        else:
            m = expm(t_arr[i] * lambda_matrix)
            s[i, :] = 1.0 - m[:, k - 1]
    return s


def density_markov_generator(t: np.ndarray | float, lambda_matrix: np.ndarray) -> np.ndarray:
    """Default-time density ``f(t)`` implied by a continuous-time Markov
    generator `lambda_matrix`: ``f(t) = (Lambda @ expm(t * Lambda))[:,
    K-1]`` (``f(0) = Lambda[:, K-1]``).

    Returns an array of shape ``(len(t), K)``.

    Original: credit/Density_Markov_Generator.m
    """
    t_arr = np.atleast_1d(np.asarray(t, dtype=float))
    lambda_matrix = np.asarray(lambda_matrix, dtype=float)
    k = lambda_matrix.shape[1]
    n = t_arr.shape[0]

    pdf = np.zeros((n, k))
    for i in range(n):
        if t_arr[i] == 0.0:
            pdf[i, :] = lambda_matrix[:, k - 1]
        else:
            m = lambda_matrix @ expm(t_arr[i] * lambda_matrix)
            pdf[i, :] = m[:, k - 1]
    return pdf


def hazard_markov_generator(t: np.ndarray | float, lambda_matrix: np.ndarray) -> np.ndarray:
    """Hazard rate ``lambda(t) = f(t) / S(t)`` implied by a continuous-time
    Markov generator `lambda_matrix`.

    Original: credit/Hazard_Markov_Generator.m (see module docstring for
    the source function-name mismatch this resolves)
    """
    s = survival_markov_generator(t, lambda_matrix)
    f = density_markov_generator(t, lambda_matrix)
    return f / s


def _discretize_bin(t: np.ndarray, edges: np.ndarray) -> np.ndarray:
    """0-based bin index of each `t` among `edges` (bin k covers
    ``[edges[k], edges[k+1])``); values below `edges[0]` clamp to bin 0."""
    idx = np.searchsorted(edges, t, side="right") - 1
    return np.where(idx < 0, 0, idx)


def _piecewise_pieces(
    lambda_: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Shared setup for the piecewise-constant-hazard branch: knot left
    edges `tm0`, bin `edges` (extended to `+inf`), per-interval hazard
    rates `lam`, and cumulative hazard `hcum` accrued *before* each knot."""
    tm = lambda_[:, 0].copy()
    lam = lambda_[:, 1:]
    tm[-1] = np.inf
    tm0 = np.concatenate(([0.0], tm[:-1]))
    d = tm - tm0
    edges = np.concatenate((tm0, [np.inf]))
    hcum = np.vstack([np.zeros((1, lam.shape[1])), np.cumsum(lam[:-1, :] * d[:-1, None], axis=0)])
    return tm0, edges, lam, hcum


def survival_exponential(t: np.ndarray | float, lambda_: np.ndarray) -> np.ndarray:
    """Survival function ``S(t) = Pr(tau > t)`` of the (piecewise)
    exponential default model. `lambda_` is either a 1-D array of shape
    ``(C,)`` (homogeneous hazard rate per scenario) or a 2-D array of
    shape ``(M, 1+C)`` (column 0 = knots ``t*_1 < ... < t*_M``, columns
    1: = piecewise hazard rates per scenario, extended to `+inf` beyond
    the last knot).

    Returns an array of shape ``(len(t), C)``.

    Original: credit/survivalExponential.m
    """
    t_arr = np.atleast_1d(np.asarray(t, dtype=float))
    lambda_ = np.asarray(lambda_, dtype=float)

    if lambda_.ndim == 1:
        return np.exp(-np.outer(t_arr, lambda_))

    tm0, edges, lam, hcum = _piecewise_pieces(lambda_)
    idx = _discretize_bin(t_arr, edges)
    h = hcum[idx, :] + lam[idx, :] * (t_arr - tm0[idx])[:, None]
    return np.exp(-h)


def cdf_exponential(t: np.ndarray | float, lambda_: np.ndarray) -> np.ndarray:
    """Cumulative distribution function ``F(t) = 1 - S(t)`` of the
    (piecewise) exponential default model. See `survival_exponential`.

    Original: credit/cdfExponential.m
    """
    return 1.0 - survival_exponential(t, lambda_)


def pdf_exponential(t: np.ndarray | float, lambda_: np.ndarray) -> np.ndarray:
    """Density function ``f(t) = lambda_m(t) * S(t)`` of the (piecewise)
    exponential default model. See `survival_exponential`.

    Original: credit/pdfExponential.m
    """
    t_arr = np.atleast_1d(np.asarray(t, dtype=float))
    lambda_ = np.asarray(lambda_, dtype=float)
    s = survival_exponential(t_arr, lambda_)

    if lambda_.ndim == 1:
        return lambda_[None, :] * s

    tm0, edges, lam, _ = _piecewise_pieces(lambda_)
    idx = _discretize_bin(t_arr, edges)
    return lam[idx, :] * s


def inv_exponential(p: np.ndarray, lambda_: np.ndarray) -> np.ndarray:
    """Quantile function (inverse CDF): `t` such that
    ``Pr(tau <= t) = p``, for the (piecewise) exponential default model.
    `p` is an array of shape ``(N,)`` or ``(N, C)`` of probabilities in
    ``(0, 1)``; values too close to 0 or 1 to invert safely return `nan`.

    Original: credit/invExponential.m
    """
    p_arr = np.asarray(p, dtype=float)
    if p_arr.ndim == 1:
        p_arr = p_arr[:, None]
    lambda_ = np.asarray(lambda_, dtype=float)

    tolp = np.finfo(float).eps
    sp = 1.0 - p_arr
    bad = (sp >= 1.0 - tolp) | (sp <= tolp)
    sp_safe = np.where(bad, 0.5, sp)

    if lambda_.ndim == 1:
        t = -np.log(sp_safe) / lambda_[None, :]
        return np.where(bad, np.nan, t)

    tm = lambda_[:, 0]
    lam = lambda_[:, 1:]
    c = lam.shape[1]

    sm = survival_exponential(tm, lambda_)  # M x C
    sm = sm.copy()
    sm[-1, :] = 0.0
    sm = np.vstack([np.ones((1, c)), sm])  # (M+1) x C
    tm0 = np.concatenate(([0.0], tm))  # (M+1,)

    if sp_safe.shape[1] == 1 and c > 1:
        sp_safe = np.repeat(sp_safe, c, axis=1)
        bad = np.repeat(bad, c, axis=1)

    n = sp_safe.shape[0]
    t = np.zeros((n, c))
    for col in range(c):
        # count of knot-survival values (incl. S(0)=1) strictly above the
        # target -- indexes the bracketing interval's left endpoint.
        idx = np.sum(sm[:, col][:, None] > sp_safe[:, col][None, :], axis=0)
        idx = np.maximum(idx, 1) - 1  # 1-based -> 0-based

        s0 = sm[idx, col]
        t0 = tm0[idx]
        l0 = lam[idx, col]
        t[:, col] = t0 + (np.log(s0) - np.log(sp_safe[:, col])) / l0

    return np.where(bad, np.nan, t)


def rnd_exponential(
    r: np.ndarray | int, c: int, lambda_: np.ndarray, random_state: object = None
) -> np.ndarray:
    """Simulated default times for the (piecewise) exponential default
    model. If ``c != 0``, generates an `r` x `c` matrix of uniforms and
    inverts them via `inv_exponential`; if ``c == 0``, `r` is instead
    treated as a pre-generated matrix of uniforms (mirrors the original
    GAUSS calling convention).

    Original: credit/rndExponential.m
    """
    if c != 0:
        rng = np.random.default_rng(random_state)
        u = rng.random((r, c))
    else:
        u = np.asarray(r, dtype=float)
    return inv_exponential(u, lambda_)
