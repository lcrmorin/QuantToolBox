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
print("ytm:", round(ytm, 5))  # -> recovers 0.037 within the solver tolerance

cy = coupon_yield(t, cash_flows, rate)
print("coupon yield:", cy)
```

Output:

```text
price: 105.51453826978027
ytm: 0.037
coupon yield: 0.047386834856974556
```

`bond_ytm` uses a bracketed solver with an expanding search interval. It
supports negative yields and yields above 100%. Cash flows must be
nonnegative, with at least one positive payment in the future, so the
solution is unique. A time-zero payment is allowed, but the supplied price
must exceed the total paid at time zero. Invalid schedules raise
`ValueError`; failure to bracket or converge raises `RuntimeError`.

Times are year fractions, and rates are decimals. `method=1` means
continuous compounding; `method=2` means annual compounding and requires a
rate greater than -1. Use the same method when pricing and recovering a
yield. The functions value exactly the payments supplied: they do not
construct coupon schedules or adjust for accrued interest or settlement.

`coupon_yield` retains the original MATLAB first-payment / price ratio.
This is an annual current yield only if the first payment is the annual
coupon, without principal. It does not identify coupons or annualize them.

## Duration and convexity

`bond_duration` returns the sensitivity `-P'/P` by default.
`modified=False` instead returns the present-value-weighted payment time
(Macaulay duration for annual compounding). `bond_convexity` returns
`P''/P`, using the same rate convention as pricing. Both require
nonnegative cash flows with at least one positive payment.

```python
from quanttoolbox.bond.pricing import bond_duration, bond_convexity

# A five-year zero-coupon bond at a 4% annually compounded yield.
duration = bond_duration(5.0, [100.0], 0.04, method=2)
convexity = bond_convexity(5.0, [100.0], 0.04, method=2)
print(round(duration, 6))   # 4.807692 = 5 / 1.04
print(round(convexity, 6))  # 27.736686 = 5 * 6 / 1.04**2

# Approximate relative price change for a 10-basis-point yield increase.
dr = 0.001
relative_change = -duration * dr + 0.5 * convexity * dr**2
```

With continuous compounding, duration is the weighted payment time and
convexity is the weighted squared payment time. The second-order price
approximation is useful for small yield changes, not a replacement for
repricing large moves.

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
