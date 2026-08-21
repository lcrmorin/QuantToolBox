# Bond pricing and sector-level risk

Present value, yield to maturity, and current yield for a simple
cash-flow schedule, followed by a sector-level modified-duration/DTS
quadratic-risk example for a small bond portfolio.

## Price, yield to maturity, and current yield

A 5-year bond paying a 5 coupon annually plus 100 principal at maturity,
priced off a flat 3.7% continuously-compounded discount rate.

```python
import numpy as np
from quanttoolbox.bond.pricing import bond_price, bond_ytm, coupon_yield

t = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
cash_flows = np.array([5.0, 5.0, 5.0, 5.0, 105.0])
rate = 0.037

price = bond_price(t, cash_flows, rate, method=1)
print("price:", price)

ytm = bond_ytm(t, cash_flows, price)
print("ytm:", ytm)  # -> recovers 0.037, within the bisection tolerance

cy = coupon_yield(t, cash_flows, rate)
print("coupon yield:", cy)
```

Output:

```text
price: 105.51453826978027
ytm: 0.036998748779296875
coupon yield: 0.047386834856974556
```

`bond_ytm` recovers the original discount rate from the price alone,
via `optim.bisection.bisection` (see [Building
blocks](building_blocks.md)) rather than a hand-rolled search loop.

## Sector-level modified-duration/DTS risk

A 3-bond portfolio split across two sectors, penalized for deviating
its per-sector modified duration (MD) and duration-times-spread (DTS)
from targets, plus a linear carry term.

```python
from quanttoolbox.bond.pricing import bond_portfolio_quadratic_form

sector = np.array([1, 1, 2])          # bonds 0,1 in sector 1; bond 2 in sector 2
md = np.array([2.0, 4.0, 6.0])         # modified duration per bond
md_star = np.array([3.0, 6.0])         # per-sector MD target
dts = np.array([1.0, 1.0, 2.0])        # duration-times-spread per bond
dts_star = np.array([1.0, 2.0])        # per-sector DTS target
carry = np.array([0.01, 0.02, 0.03])   # carry per bond
w = np.array([0.5, 0.25, 0.25])        # portfolio weights

result = bond_portfolio_quadratic_form(
    sector,
    varphi_md=0.7,
    md=md,
    md_star=md_star,
    varphi_dts=0.3,
    dts=dts,
    dts_star=dts_star,
    gamma_carry=1.0,
    carry=carry,
    w=w,
)
print("risk (qf):", result.qf)
```

Output:

```text
risk (qf): 7.766875000000001
```

`result.md`/`result.dts` are the individual
`sustainable_finance.risk.QuadraticFormResult` terms the combined
`(q, r, c)` was built from — each sector's contribution is broken out
in `q_j`/`r_j`/`c_j` on those objects.
`bond_portfolio_quadratic_form_vs_benchmark` extends this with an
active-share term against a benchmark weight vector; see its docstring
for the full parameterization.
