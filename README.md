# QuantToolBox

A Python port of Thierry Roncalli's MATLAB QuantToolBox and the
Handbook of Sustainable Finance toolbox: econometrics, portfolio
optimization, and risk analytics.

```bash
pip install quanttoolbox
```

## Example

```python
import numpy as np
from quanttoolbox.portfolio.risk_budgeting import erc_portfolio

cov_matrix = np.array([
    [0.09, 0.024, 0.018],
    [0.024, 0.04, 0.012],
    [0.018, 0.012, 0.0225],
])

result = erc_portfolio(cov_matrix)
print(result.weights)               # equal-risk-contribution weights
print(result.pct_risk_contribution)  # ~equal, by construction
```

## Docs

- [Examples](https://lcrmorin.github.io/QuantToolBox/examples/) —
  worked examples with real numeric output, ordered from simplest to
  most involved.
- [API reference](https://lcrmorin.github.io/QuantToolBox/api/) —
  generated from the docstrings.
- [Notes for translators](https://lcrmorin.github.io/QuantToolBox/migration_map/)
  — file-by-file mapping back to the original MATLAB source, and what's
  left to port.

## License

MIT.
