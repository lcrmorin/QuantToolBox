"""Translated from Examples/rpb/test_bl5.m -- a second, independent 4-asset
Black-Litterman + tracking-error worked example (no book page cited in the
original). Every asset gets a direct view (`P=I`, `Q=mu`, full confidence
`Omega=Sigma_epsilon=covMatrix`) -- an unusually strong-conviction setup
that pulls the posterior most of the way from the equilibrium prior toward
the raw sample means.

Four blocks, each comparing the resulting active portfolio (x) against the
x0 benchmark:

1. gamma-problem MVO (mu_BL, Sigma_BL) at the implied risk-aversion gamma0.
2. sigma-problem TE-target=1% (mu_BL, Sigma_BL), i.e. `te_target_portfolio`.
3. gamma-problem MVO (mu_BL, covMatrix) -- same posterior mean, but
   optimized against the *original* (pre-view) covariance rather than the
   BL posterior covariance Sigma_BL.
4. sigma-problem TE-target=1% (mu_BL, covMatrix).

The original's final block repeats block 2 verbatim (mu_BL, Sigma_BL again)
-- included here too for a faithful translation, and it does reproduce
identical numbers, as expected."""

import numpy as np

from quanttoolbox.linalg.special_matrices import xpnd
from quanttoolbox.portfolio.black_litterman import black_litterman_moments, implied_risk_premia
from quanttoolbox.portfolio.mean_variance import mvo_portfolio
from quanttoolbox.portfolio.tracking_error import te_target_portfolio
from quanttoolbox.stats.moments import corr_to_cov

mu = np.array([0.03, 0.03, 0.08, 0.07])
sigma = np.array([0.06, 0.07, 0.18, 0.17])
rho = xpnd(np.array([1.00, 0.50, 1.00, -0.40, -0.40, 1.00, -0.40, -0.40, 0.80, 1.00]), method=1)
cov_matrix = corr_to_cov(sigma, rho)
print("Sigma_hat =\n", np.round(100 * cov_matrix, 2))

x0 = np.array([0.40, 0.40, 0.10, 0.10])
r = 0.02
irp = implied_risk_premia(x0, cov_matrix, 0.25)
mu_tilde = r + irp.pi
print("tilde(pi) =", np.round(100 * irp.pi, 2))
print("tilde(mu) =", np.round(100 * mu_tilde, 2))
print("gamma0 =", round(irp.gamma, 4))

p_matrix = np.eye(4)
bl = black_litterman_moments(mu_tilde, cov_matrix, p_matrix, mu, cov_matrix)
print("mu(BL) =", np.round(100 * bl.mu_bar, 2))
print("Sigma(BL) =\n", np.round(100 * bl.sigma_bar, 2))


def report(label: str, weights: np.ndarray, cov_for_te: np.ndarray) -> None:
    active = weights - x0
    te = np.sqrt(active @ cov_for_te @ active)
    print(f"{label}: x0={np.round(x0, 4)}  x={np.round(weights, 4)}  te={100 * te:.2f}")


gamma0 = irp.gamma

res = mvo_portfolio(
    bl.mu_bar,
    bl.sigma_bar,
    gamma=gamma0,
    a_eq=np.ones((1, 4)),
    b_eq=np.array([1.0]),
    lb=0.0,
    ub=1.0,
)
report("1. MVO(mu_BL, Sigma_BL) @ gamma0", res.weights, bl.sigma_bar)

te_res = te_target_portfolio(
    x0,
    bl.mu_bar,
    bl.sigma_bar,
    0.01,
    problem="sigma",
    a_eq=np.ones((1, 4)),
    b_eq=np.array([1.0]),
    lb=0.0,
    ub=1.0,
)[0]
report("2. TE-target 1% (mu_BL, Sigma_BL)", te_res.weights, bl.sigma_bar)

res2 = mvo_portfolio(
    bl.mu_bar, cov_matrix, gamma=gamma0, a_eq=np.ones((1, 4)), b_eq=np.array([1.0]), lb=0.0, ub=1.0
)
report("3. MVO(mu_BL, covMatrix) @ gamma0", res2.weights, cov_matrix)

te_res2 = te_target_portfolio(
    x0,
    bl.mu_bar,
    cov_matrix,
    0.01,
    problem="sigma",
    a_eq=np.ones((1, 4)),
    b_eq=np.array([1.0]),
    lb=0.0,
    ub=1.0,
)[0]
report("4. TE-target 1% (mu_BL, covMatrix)", te_res2.weights, cov_matrix)

te_res3 = te_target_portfolio(
    x0,
    bl.mu_bar,
    bl.sigma_bar,
    0.01,
    problem="sigma",
    a_eq=np.ones((1, 4)),
    b_eq=np.array([1.0]),
    lb=0.0,
    ub=1.0,
)[0]
report("5. TE-target 1% (mu_BL, Sigma_BL), repeated", te_res3.weights, bl.sigma_bar)
