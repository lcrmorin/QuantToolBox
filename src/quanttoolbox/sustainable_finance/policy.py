"""Weitzman (1974) "prices vs. quantities": comparing a price instrument
(a Pigouvian tax fixing the marginal abatement cost) against a quantity
instrument (a cap fixing the abatement level) for correcting a linear
externality under uncertainty about the realized marginal-cost shock.

Not ported from the MATLAB HSF toolbox -- no `.m` file in `hfs-archive`
implements a single `weitzman_example` function. Promoted from
HSF-Notebooks chapter 10e, which packaged the repeated, near-identical
inline computations of `HSF/10. Transition Risk/chap10_externality5a.m`,
`chap10_externality5b.m`, and `chap10_externality5d.m` (three worked
examples at different cost-shock distributions/slope parameters) into
one parametrized Python helper -- the formulas here match those scripts'
own algebra directly (`q_star_i`, `q_ast`/`p_ast`, the quantity- and
price-instrument outcomes, and the analytical welfare-loss identity
`Delta_W = sigma_epsilon**2 * (1 - beta_B/beta_C) / (2*beta_C)` all
appear verbatim in `chap10_externality5a.m`).

Translation notes:

- The source scripts derive the quantity-instrument outcome `q1_ast_i`
  two different-looking ways -- as ``(p_ast - alpha_C - mean_epsilon) /
  beta_C`` in the `.m` scripts, and as a flat array equal to `q_ast` in
  the HSF-Notebooks helper this module ports from -- verified
  algebraically identical (substituting `p_ast = alpha_C + beta_C*q_ast +
  mean_epsilon` collapses the first form to `q_ast`). This module uses
  the simpler, already-verified `q_ast` form.
- The welfare integral ``W(q) = integral_0^q [SMB(q') - SMC(q',
  epsilon)] dq'`` is exposed here only via its closed form
  (`W_analytical` in the source); the source's own `scipy.integrate.quad`
  numerical cross-check (matching MATLAB's `integral`) is instead
  exercised in this module's test suite, the same treatment already
  established for `carbon.py`'s closed-form/`integral()` pairs.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class WeitzmanResult:
    """Weitzman (1974) prices-vs-quantities comparison at a given linear
    social-marginal-benefit (SMB) curve ``alpha_b - beta_b*q``, linear
    social-marginal-cost curve ``alpha_c + beta_c*q + epsilon`` under a
    discrete cost shock `epsilon_i` (realized with probability `p_i`),
    and the resulting ex-post-optimal, quantity-instrument, and
    price-instrument outcomes.

    `q_star_i` is the ex-post (realized-shock) optimal quantity in each
    state; `q_ast`/`p_ast` are the ex-ante optimum and its implied price
    under the *expected* cost curve; `q1_ast_i` is the outcome under a
    quantity instrument (a cap fixed at `q_ast` in every state);
    `q2_ast_i` is the outcome under a price instrument (a tax fixed at
    `p_ast` in every state, to which firms respond by equating it to
    their realized marginal cost); `w_q1_ast_i`/`w_q2_ast_i` are the
    realized welfare under each instrument in each state;
    `e_w_q1_ast`/`e_w_q2_ast` are their probability-weighted expectations;
    `delta_w` is the expected welfare gain of the price instrument over
    the quantity instrument, and `delta_w_analytical` is its closed form
    (Weitzman's classic result: the price instrument dominates in
    expectation iff `beta_c > beta_b`)."""

    epsilon_i: np.ndarray
    p_i: np.ndarray
    mean_epsilon: float
    sigma_epsilon: float
    q_star_i: np.ndarray
    q_ast: float
    p_ast: float
    q1_ast_i: np.ndarray
    q2_ast_i: np.ndarray
    w_q1_ast_i: np.ndarray
    w_q2_ast_i: np.ndarray
    e_w_q1_ast: float
    e_w_q2_ast: float
    delta_w: float
    delta_w_analytical: float


def weitzman_example(
    alpha_b: float,
    beta_b: float,
    alpha_c: float,
    beta_c: float,
    epsilon_i: np.ndarray,
    p_i: np.ndarray,
) -> WeitzmanResult:
    """Weitzman (1974) prices-vs-quantities instrument comparison for a
    linear SMB curve ``alpha_b - beta_b*q``, a linear SMC curve
    ``alpha_c + beta_c*q + epsilon`` shocked by a discrete random variable
    `epsilon_i` with probabilities `p_i`, and welfare
    ``W(q, epsilon) = ((alpha_b - alpha_c - epsilon) - 0.5*q*(beta_b +
    beta_c)) * q``.

    Not ported from a single MATLAB HSF toolbox function -- see the
    module docstring.
    """
    epsilon_i = np.asarray(epsilon_i, dtype=float)
    p_i = np.asarray(p_i, dtype=float)

    mean_epsilon = float(np.sum(p_i * epsilon_i))
    sigma_epsilon = float(np.sqrt(np.sum(p_i * (epsilon_i - mean_epsilon) ** 2)))

    # True (ex-post) optimal quantity under each realized shock.
    q_star_i = (alpha_b - alpha_c - epsilon_i) / (beta_b + beta_c)

    # Ex-ante optimum under the *expected* cost curve.
    q_ast = (alpha_b - alpha_c - mean_epsilon) / (beta_b + beta_c)
    p_ast = alpha_b - beta_b * q_ast

    # Quantity instrument: cap fixed at q_ast for every state.
    q1_ast_i = np.full_like(epsilon_i, q_ast)

    # Price instrument: firms equate SMC(q, epsilon_i) to the fixed price p_ast.
    q2_ast_i = (p_ast - alpha_c - epsilon_i) / beta_c

    def w_analytical(q: np.ndarray, epsilon: np.ndarray) -> np.ndarray:
        return ((alpha_b - alpha_c - epsilon) - 0.5 * q * (beta_b + beta_c)) * q

    w_q1_ast_i = w_analytical(q1_ast_i, epsilon_i)
    w_q2_ast_i = w_analytical(q2_ast_i, epsilon_i)
    e_w_q1_ast = float(np.sum(p_i * w_q1_ast_i))
    e_w_q2_ast = float(np.sum(p_i * w_q2_ast_i))
    delta_w = e_w_q2_ast - e_w_q1_ast
    delta_w_analytical = sigma_epsilon**2 * (1.0 - beta_b / beta_c) / (2.0 * beta_c)

    return WeitzmanResult(
        epsilon_i=epsilon_i,
        p_i=p_i,
        mean_epsilon=mean_epsilon,
        sigma_epsilon=sigma_epsilon,
        q_star_i=q_star_i,
        q_ast=q_ast,
        p_ast=p_ast,
        q1_ast_i=q1_ast_i,
        q2_ast_i=q2_ast_i,
        w_q1_ast_i=w_q1_ast_i,
        w_q2_ast_i=w_q2_ast_i,
        e_w_q1_ast=e_w_q1_ast,
        e_w_q2_ast=e_w_q2_ast,
        delta_w=delta_w,
        delta_w_analytical=delta_w_analytical,
    )
