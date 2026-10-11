# Examples

Start with a guided example, then use the standalone scripts to investigate
one assumption at a time. There are **eight guided pages and 87 runnable scripts**.
The scripts retain their historical filenames; the titles below describe what
you will learn. Original MATLAB coverage is tracked separately in the
[translation map](../migration_map.md#example-translation-tracker).

## Set up and run

From a local checkout, install the package and plotting dependencies:

```sh
python -m pip install -e ".[viz]"
python docs/examples/backtest/backtest3.py
```

Guided pages contain Python blocks meant to run **in order in one session**.
Each standalone script starts fresh and embeds its data or uses a fixed random
seed. Numerical results may differ slightly with solver and library versions.
A seed makes a demonstration reproducible; it does not validate the model.

To check every script and guided page in fresh processes, without opening plots:

```sh
python scripts/check_examples.py
```

The report in `reports/examples-execution.json` records output, warnings,
completion status, and duration. A successful exit is an execution check;
read convergence flags and model diagnostics before interpreting estimates.

## Choose a learning path

| Path | Read first | Continue with |
| --- | --- | --- |
| Pricing and allocation | [Bonds](bond.md) → [Mean-variance](mean_variance.md) | [Risk budgets](risk_budgeting.md) → [Black-Litterman](black_litterman.md) |
| Numerical foundations | [Building blocks](building_blocks.md) | Matrix identities, constraint conversion, and finite differences below |
| Statistical estimation | [Regression](regression.md) | Quantile regression → local-level Kalman models → [Whittle](whittle.md) |
| Classification | [SVM geometry](svm.md) | Dual hard margins → soft margins → SVM regression |
| Backtesting | `backtest3.py` below | Funding round trip → rebalance schedules → transaction costs |

Use decimal returns and weights unless an output explicitly says percent.
Keep return and covariance horizons consistent. For fitted models, distinguish
training fit, numerical convergence, and performance on unseen observations.

## Standalone catalog

Each script opens with a learning goal, output interpretation, and a small
experiment. The [API reference](../api/econometrics.md) and other module pages
embed the same source files alongside the relevant functions.

### Backtesting

| Script | Learning goal |
| --- | --- |
| [backtest2.py](backtest/backtest2.py) | Rebalancing dates and missing prices |
| [backtest3.py](backtest/backtest3.py) | Rebalancing by hand |
| [backtest4.py](backtest/backtest4.py) | Turnover and trading costs |
| [backtest5.py](backtest/backtest5.py) | Compounding the cost of rebalancing |
| [mdd1.py](backtest/mdd1.py) | Maximum drawdown and its dates |
| [unfunded1.py](backtest/unfunded1.py) | Funded and excess-return backtests |
| [unfunded2.py](backtest/unfunded2.py) | A funding round trip |

### Econometrics

| Script | Learning goal |
| --- | --- |
| [gmm1.py](ects/gmm1.py) | OLS, likelihood, and moment conditions |
| [kalman1b.py](ects/kalman1b.py) | Likelihood sensitivity to initialization |
| [kalman1c.py](ects/kalman1c.py) | Time-domain and frequency-domain estimation |
| [kalman2a.py](ects/kalman2a.py) | Filtering a level and a slope |
| [kalman2b.py](ects/kalman2b.py) | Whittle estimation before trend filtering |
| [kalman2c.py](ects/kalman2c.py) | Likelihood estimation before trend filtering |
| [kalman3a.py](ects/kalman3a.py) | A multivariate state-space model |
| [kalman3b.py](ects/kalman3b.py) | Time-varying matrix representation |
| [kalman3c.py](ects/kalman3c.py) | Uncertainty around filtered states |
| [kalman3d.py](ects/kalman3d.py) | Identification before state-space estimation |
| [kalman4a.py](ects/kalman4a.py) | Regression coefficients as latent states |
| [kalman4b.py](ects/kalman4b.py) | Estimating coefficient process noise |
| [ml2.py](ects/ml2.py) | Maximum likelihood and covariance estimators |
| [ml3.py](ects/ml3.py) | A beta likelihood for bounded observations |
| [ml4.py](ects/ml4.py) | Analytical derivatives in maximum likelihood |
| [panel1.py](ects/panel1.py) | A first local-level filter |
| [quantile1.py](ects/quantile1.py) | Quantiles as asymmetric-loss minimizers |
| [quantile2.py](ects/quantile2.py) | Monte Carlo comparison of OLS and LAD |
| [robust2.py](ects/robust2.py) | OLS, median, LAD, and Huber regression |
| [robust3.py](ects/robust3.py) | Iterative and exact quantile regression |
| [varx1a.py](ects/varx1a.py) | A first vector autoregression |
| [varx1b.py](ects/varx1b.py) | Testing predictive restrictions in a VAR |
| [varx1c.py](ects/varx1c.py) | Contemporaneous residual dependence |
| [varx1d.py](ects/varx1d.py) | A restricted VAR |
| [varx1e.py](ects/varx1e.py) | Selecting a VAR lag order |
| [varx2a.py](ects/varx2a.py) | Adding exogenous regressors to a VAR |
| [varx2b.py](ects/varx2b.py) | Restricted VARX least squares |
| [varx2c.py](ects/varx2c.py) | Restricted VARX likelihood |
| [varx3.py](ects/varx3.py) | OLS as a zero-lag VAR |
| [varx4a.py](ects/varx4a.py) | Seemingly unrelated regressions |
| [varx4b.py](ects/varx4b.py) | Restrictions across a three-equation system |
| [varx5a.py](ects/varx5a.py) | Structural and reduced-form systems |
| [whittle1.py](ects/whittle1.py) | A spectral fit of a local-level model |
| [whittle2.py](ects/whittle2.py) | Supplying a custom spectral density |

### Numerical mathematics

| Script | Learning goal |
| --- | --- |
| [grad2.py](maths/grad2.py) | Finite differences for a coupled function |
| [grad3.py](maths/grad3.py) | Differentiating a separable exponential |
| [grad4.py](maths/grad4.py) | Checking a product-rule derivative |
| [grad5.py](maths/grad5.py) | Equivalent formulas, equivalent gradients |
| [hess2.py](maths/hess2.py) | Curvature and mixed derivatives |
| [pdgm1.py](maths/pdgm1.py) | FFT coefficients and spectral scaling |
| [pdgm2.py](maths/pdgm2.py) | Periodogram frequency resolution |

### Matrix operations

| Script | Learning goal |
| --- | --- |
| [matrix2.py](matrix/matrix2.py) | Vectorization matrix identities |
| [reshape1.py](matrix/reshape1.py) | Reshaping and symmetric reconstruction |

### Optimization

| Script | Learning goal |
| --- | --- |
| [explicit2.py](optim/explicit2.py) | Equality constraints as free coordinates |
| [explicit3.py](optim/explicit3.py) | Fixing selected coefficients at zero |
| [prox_turnover1.py](optim/prox_turnover1.py) | Penalty versus turnover budget |
| [proximal1.py](optim/proximal1.py) | Projection onto intersecting constraints |

### Portfolio construction

| Script | Learning goal |
| --- | --- |
| [test_bl2.py](rpb/test_bl2.py) | How views change portfolio weights |
| [test_bl3.py](rpb/test_bl3.py) | Comparing views at matched volatility |
| [test_bl4.py](rpb/test_bl4.py) | A tracking-error frontier |
| [test_bl5.py](rpb/test_bl5.py) | Prior uncertainty and portfolio risk |
| [test_bl6.py](rpb/test_bl6.py) | Cross-checking a tracking-error solution |
| [test_box1.py](rpb/test_box1.py) | Risk budgets under allocation limits |
| [test_erc2.py](rpb/test_erc2.py) | Capital weights versus risk budgets |
| [test_erc3.py](rpb/test_erc3.py) | Unequal risk-contribution targets |
| [test_lasso1.py](rpb/test_lasso1.py) | Budget constraints and portfolio penalties |
| [test_lasso2.py](rpb/test_lasso2.py) | Static and reference-based regularization |
| [test_lasso3.py](rpb/test_lasso3.py) | When an L1 penalty cannot help |
| [test_lasso5.py](rpb/test_lasso5.py) | Choosing the regularization target |
| [test_minvar2.py](rpb/test_minvar2.py) | Reading portfolio constraints |
| [test_mvo2.py](rpb/test_mvo2.py) | Three ways to specify mean-variance portfolios |
| [test_mvo3.py](rpb/test_mvo3.py) | Long-only and capped frontiers |

### Statistics

| Script | Learning goal |
| --- | --- |
| [cov1.py](stats/cov1.py) | Uncertainty estimates for regression |
| [elasticnet1.py](stats/elasticnet1.py) | Mixing L1 and L2 penalties |
| [kalman1.py](stats/kalman1.py) | Estimating changing regression coefficients |
| [kalman2.py](stats/kalman2.py) | Sensitivity to the initial state |
| [kernel1.py](stats/kernel1.py) | Bandwidth and marginal density estimates |
| [lasso1.py](stats/lasso1.py) | A lasso coefficient path |
| [lasso2.py](stats/lasso2.py) | Lasso versus unpenalized least squares |
| [pca1.py](stats/pca1.py) | Explained variance with correlated variables |
| [qreg1.py](stats/qreg1.py) | Conditional quantiles versus conditional means |
| [qreg2.py](stats/qreg2.py) | Local polynomial quantile curves |
| [ridge2.py](stats/ridge2.py) | Selecting ridge shrinkage by coefficient norm |

### Support vector machines

| Script | Learning goal |
| --- | --- |
| [svm2.py](svm/svm2.py) | The dual hard-margin classifier |
| [svm3.py](svm/svm3.py) | Soft margins and hinge penalties |
| [svm4.py](svm/svm4.py) | The dual soft-margin classifier |
| [svm5.py](svm/svm5.py) | Hard and soft margin examples |
| [svm6.py](svm/svm6.py) | Loss functions in SVM regression |
| [svm7.py](svm/svm7.py) | Dual formulations of regression losses |
| [svm8.py](svm/svm8.py) | A larger simulated regression comparison |
