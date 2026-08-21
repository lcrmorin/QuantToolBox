# Risk budgeting: ERC, VaR, and ES contributions

Translated from `Examples/rpb/test_erc1.m`, which reproduces Tables 2.2,
2.3, and 2.4 (pages 81–82) of Roncalli, T. (2013), *Introduction to Risk
Parity and Budgeting*.

Three assets with given volatilities and correlations; a fixed
(non-equal-risk) weight vector `x = [0.50, 0.20, 0.30]`. The example
computes the risk contribution decomposition under three different risk
measures: plain volatility, 99% VaR, and 99% Expected Shortfall.

```python
import numpy as np
from quanttoolbox.linalg.special_matrices import xpnd
from quanttoolbox.stats.moments import corr_to_cov
from quanttoolbox.portfolio.risk_budgeting import (
    risk_contribution,
    risk_contribution_var,
    risk_contribution_es,
)

sigma = np.array([0.30, 0.20, 0.15])
rho_vech = np.array([1.00, 0.80, 1.00, 0.50, 0.30, 1.00])
rho = xpnd(rho_vech, method=1)
cov_matrix = corr_to_cov(sigma, rho)

x = np.array([0.50, 0.20, 0.30])

# --- Volatility-based risk contribution ---
rc_vol = risk_contribution(x, cov_matrix)
print("portfolio volatility:", round(100 * rc_vol.risk, 2))
print("pct risk contribution:", np.round(100 * rc_vol.pct_risk_contribution, 2))

# --- VaR-based (99% confidence) ---
mu = np.zeros(3)
alpha = 0.99
rc_var = risk_contribution_var(x, cov_matrix, mu, alpha)
print("99% VaR:", round(100 * rc_var.risk, 2))

# --- ES-based (99% confidence) ---
rc_es = risk_contribution_es(x, cov_matrix, mu, alpha)
print("99% ES:", round(100 * rc_es.risk, 2))
```

Output:

```text
portfolio volatility: 20.87
pct risk contribution: [70.43 15.93 13.64]
99% VaR: 48.55
99% ES: 55.62
```

The `20.87%` portfolio volatility and the `[70.43, 15.93, 13.64]` percent
risk contribution split match the published Table 2.2 values exactly.

## Equal Risk Contribution portfolio

The same covariance matrix, but solving *for* the weights that make each
asset's risk contribution equal (rather than decomposing a given,
unequal weight vector):

```python
from quanttoolbox.portfolio.risk_budgeting import erc_portfolio

erc = erc_portfolio(cov_matrix)
print("ERC weights:", np.round(erc.weights, 4))
print("pct risk contribution:", np.round(100 * erc.pct_risk_contribution, 2))
```

Output:

```text
ERC weights: [0.1969 0.3244 0.4787]
pct risk contribution: [33.33 33.33 33.33]
```

By construction, all three assets now contribute exactly one-third of
total portfolio risk.
