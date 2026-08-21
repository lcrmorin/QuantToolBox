# QuantToolBox

A Python port of the MATLAB QuantToolBox: econometrics, portfolio
optimization, and risk analytics.

```bash
pip install quanttoolbox
```

## What's here

- **[Examples](examples/index.md)** — worked examples with real numeric
  output, translated from the original MATLAB toolbox's own example
  scripts (many referencing Roncalli's *Introduction to Risk Parity and
  Budgeting*).
- **[API Reference](api/index.md)** — auto-generated from the docstrings
  throughout the package.
- **[Notes for translators](migration_map.md)** — the file-by-file
  mapping back to the original MATLAB source (including a per-example
  translation tracker), genuine bugs found in that source during
  porting, and an assessment of which modules should be replaced with
  mature Python libraries versus which justify staying custom — and, for
  those, whether the gap is worth contributing back upstream.

## Quick example

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
