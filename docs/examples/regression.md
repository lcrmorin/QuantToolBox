# Ridge regression path

Translated from `Examples/stats/ridge1.m`. A small standardized dataset
(15 observations, 5 predictors); the example shows how ridge coefficients
shrink toward zero as the penalty `lambda` increases.

```python
import numpy as np
from quanttoolbox.stats.regression.ols import standardize
from quanttoolbox.stats.regression.ridge import ridge

data = np.array([
    [3.1, 2.8, 4.3, 0.3, 2.2, 3.5],
    [24.9, 5.9, 3.6, 3.2, 0.7, 6.4],
    [27.3, 6.0, 9.6, 7.6, 9.5, 0.9],
    [25.4, 8.4, 5.4, 1.8, 1.0, 7.1],
    [46.1, 5.2, 7.6, 8.3, 0.6, 4.5],
    [45.7, 6.0, 7.0, 9.6, 0.6, 0.6],
    [47.4, 6.1, 1.0, 8.5, 9.6, 8.6],
    [-1.8, 1.2, 9.6, 2.7, 4.8, 5.8],
    [20.8, 3.2, 5.0, 4.2, 2.7, 3.6],
    [6.8, 0.5, 9.2, 6.9, 9.3, 0.7],
    [12.9, 7.9, 9.1, 1.0, 5.9, 5.4],
    [37.0, 1.8, 1.3, 9.2, 6.1, 8.3],
    [14.7, 7.4, 5.6, 0.9, 5.6, 3.9],
    [-3.2, 2.3, 6.6, 0.0, 3.6, 6.4],
    [44.3, 7.7, 2.2, 6.5, 1.3, 0.7],
])
y = standardize(data[:, 0])
x = standardize(data[:, 1:6])

for lam in [0.0, 1.0, 10.0]:
    beta, df, complexity = ridge(y, x, lambda_=lam)
    rss = np.mean((y - x @ beta) ** 2)
    print(f"lambda={lam}: beta={np.round(beta, 3)}, df={round(df, 2)}, rss={round(rss, 4)}")
```

Output:

```text
lambda=0.0: beta=[ 0.459 -0.185  0.834 -0.189  0.093], df=5.0, rss=0.0118
lambda=1.0: beta=[ 0.426 -0.205  0.762 -0.166  0.063], df=4.59, rss=0.0164
lambda=10.0: beta=[ 0.267 -0.198  0.454 -0.088 -0.004], df=2.77, rss=0.1648
```

`lambda=0` recovers plain OLS exactly (5 effective degrees of freedom,
one per coefficient); as `lambda` grows, coefficients shrink toward zero,
effective degrees of freedom drops, and in-sample fit (RSS) worsens —
exactly the bias-variance tradeoff ridge regression is designed to
navigate.

## OLS and robust regression

Translated from `Examples/ects/ols1.m` and `Examples/ects/robust1.m`,
which share the same small dataset.

```python
from quanttoolbox.econometrics.estimation import ols_estimation
from quanttoolbox.stats.regression.robust import huber_regression, lad_regression

data = np.array([
    [1.5, 1.0, 2.4, 3.6, 0.3],
    [20.4, 1.0, 1.1, 3.8, 5.9],
    [17.1, 1.0, 5.1, 6.3, 6.1],
    [30.9, 1.0, 2.7, 2.4, 9.5],
    [22.2, 1.0, 3.3, 3.0, 7.4],
    [9.1, 1.0, 1.0, 5.4, 4.9],
    [39.2, 1.0, 9.6, 2.8, 8.1],
    [3.1, 1.0, 2.9, 4.4, 1.0],
    [7.2, 1.0, 4.2, 5.6, 1.7],
    [27.6, 1.0, 8.1, 1.7, 5.4],
])
y, x = data[:, 0], data[:, 1:5]

result = ols_estimation(y, x)
print("OLS beta:", np.round(result.beta, 4))
print("R2:", round(result.r_squared, 4))

print("Huber beta:", np.round(huber_regression(y, x, c=1.345).beta, 4))
print("LAD beta:", np.round(lad_regression(y, x).beta, 4))
```

Output:

```text
OLS beta: [ 3.4461  1.5442 -1.6454  2.8951]
R2: 0.9913
Huber beta: [ 3.4144  1.5276 -1.7484  2.8624]
LAD beta: [ 2.3756  1.8841 -1.7415  2.9071]
```

With a clean, un-contaminated dataset like this, OLS/Huber/LAD all agree
reasonably closely — the interesting divergence between them only shows
up with outliers, as demonstrated in the package's own test suite
(`tests/stats/test_regression_robust.py`).
