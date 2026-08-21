# Black-Litterman

Translated from `Examples/rpb/test_bl1.m`, which reproduces the example
on page 24 of Roncalli, T. (2013), *Introduction to Risk Parity and
Budgeting*.

Four assets, a market-cap-weighted reference portfolio `x`, and two
investor views: (1) asset 1's absolute return, (2) the *spread* between
asset 2 and asset 3's returns.

```python
import numpy as np
from quanttoolbox.linalg.special_matrices import xpnd
from quanttoolbox.stats.moments import corr_to_cov
from quanttoolbox.portfolio.black_litterman import (
    implied_risk_premia,
    black_litterman_moments,
)

sigma = np.array([0.15, 0.20, 0.25, 0.30])
rho_vech = np.array(
    [1.00, 0.10, 1.00, 0.40, 0.70, 1.00, 0.50, 0.40, 0.80, 1.00]
)
rho = xpnd(rho_vech, method=1)
cov_matrix = corr_to_cov(sigma, rho)

x = np.array([0.40, 0.30, 0.20, 0.10])   # reference (e.g. market-cap) weights
r = 0.03                                  # risk-free rate
sharpe_ratio = 0.25                       # assumed market Sharpe ratio

# Step 1: back out the implied equilibrium excess returns
irp = implied_risk_premia(x, cov_matrix, sharpe_ratio)
mu_tilde = r + irp.pi
print("implied pi:", np.round(100 * irp.pi, 2))
print("implied mu_tilde:", np.round(100 * mu_tilde, 2))

# Step 2: express the investor views
#   view 1: asset 1's return = 4%
#   view 2: asset 2's return - asset 3's return = -1%
P = np.array([[1, 0, 0, 0], [0, 1, -1, 0]], dtype=float)
Q = np.array([0.04, -0.01])
Omega = np.diag([0.10**2, 0.05**2])  # view uncertainty
tau = 1
gamma_matrix = tau * cov_matrix

# Step 3: blend the equilibrium prior with the views
bl = black_litterman_moments(mu_tilde, gamma_matrix, P, Q, Omega)
print("posterior mu_bar:", np.round(100 * bl.mu_bar, 2))
```

Output:

```text
implied pi: [2.47 3.68 5.7  6.06]
implied mu_tilde: [5.47 6.68 8.7  9.06]
posterior mu_bar: [4.39 6.64 7.68 7.61]
```

Notice how the posterior expected returns move toward the views: asset
1's posterior return (`4.39%`) shifts down from its equilibrium value
(`5.47%`) toward the view target (`4%`), and the asset‑2‑minus‑asset‑3
spread narrows from its equilibrium value toward the `-1%` view target,
each moving proportionally to how confident the corresponding view's
`Omega` entry says it is.
