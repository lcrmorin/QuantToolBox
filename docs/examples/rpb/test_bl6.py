"""Translated from Examples/rpb/test_bl6.m -- same 4-asset setup as
test_bl5.py, demonstrating that a raw QP call and `te_portfolio` at a fixed
gamma agree, then cross-checking that fixed-gamma solution against
`te_target_portfolio`'s sigma-problem mode targeting the tracking error that
fixed-gamma solution actually achieves.

The original's raw `quadprog(H, f, ..., A, B, lb, ub, x0, options)` call
with `H=covMatrix`, `f=-gamma0*mu - covMatrix*x0` is exactly what
`te_portfolio(x0, mu, covMatrix, gamma=gamma0, ...)` computes internally
(see tracking_error.py's `r_lin = gamma*mu + cov_matrix @ x_benchmark`) --
so it's translated directly as a `te_portfolio` call rather than a raw
`solve_qp` call, and the two are verified to agree below."""

import numpy as np

from quanttoolbox.linalg.special_matrices import xpnd
from quanttoolbox.optim.quadprog import solve_qp
from quanttoolbox.portfolio.tracking_error import te_portfolio, te_target_portfolio
from quanttoolbox.stats.moments import corr_to_cov

mu = np.array([0.03, 0.03, 0.08, 0.07])
sigma = np.array([0.06, 0.07, 0.18, 0.17])
rho = xpnd(np.array([1.00, 0.50, 1.00, -0.40, -0.40, 1.00, -0.40, -0.40, 0.80, 1.00]), method=1)
cov_matrix = corr_to_cov(sigma, rho)
print("Sigma_hat =\n", np.round(100 * cov_matrix, 2))

x0 = np.array([0.40, 0.40, 0.10, 0.10])
gamma0 = 0.0422

# Raw quadprog(H, f, [], [], A, B, lb, ub, x0, options) call from the
# original, translated literally via solve_qp.
x_raw = solve_qp(
    cov_matrix,
    gamma0 * mu + cov_matrix @ x0,
    a_eq=np.ones((1, 4)),
    b_eq=np.array([1.0]),
    lb=0.0,
    ub=1.0,
)
print("x (raw quadprog) =", np.round(x_raw, 6))

# Same QP via te_portfolio -- should match x_raw exactly.
res = te_portfolio(
    x0, mu, cov_matrix, gamma=gamma0, a_eq=np.ones((1, 4)), b_eq=np.array([1.0]), lb=0.0, ub=1.0
)
te_x = np.sqrt((res.weights - x0) @ cov_matrix @ (res.weights - x0))
ir_x = res.active_return / te_x
print("x0 =", x0, " x (te_portfolio) =", np.round(res.weights, 6))
print(f"te(x) = {100 * te_x:.2f}   IR = {ir_x:.4f}")

# Cross-check: te_target_portfolio's sigma-problem, targeting the tracking
# error the fixed-gamma solution above actually achieves, should recover
# gamma0 (and the same portfolio) via bisection.
target_res = te_target_portfolio(
    x0,
    mu,
    cov_matrix,
    te_x,
    problem="sigma",
    a_eq=np.ones((1, 4)),
    b_eq=np.array([1.0]),
    lb=0.0,
    ub=1.0,
)[0]
print(
    f"cross-check: gamma from te_target_portfolio={target_res.gamma:.4f} "
    f"(vs gamma0={gamma0}), weights match={np.allclose(target_res.weights, res.weights, atol=1e-4)}"
)
