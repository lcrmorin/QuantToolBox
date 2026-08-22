"""Tests for quanttoolbox.stats.regression.lasso."""

import numpy as np

from quanttoolbox.stats.regression.lasso import (
    elastic_net_ccd,
    lasso_admm,
    lasso_ccd,
    lasso_regularization_path,
    lasso_tau_constrained,
    soft_threshold,
)


def test_soft_threshold_basic():
    v = np.array([-3.0, -0.5, 0.0, 0.5, 3.0])
    out = soft_threshold(v, 1.0)
    assert np.allclose(out, [-2.0, 0.0, 0.0, 0.0, 2.0])


def test_lasso_ccd_induces_sparsity(rng):
    n, p = 200, 10
    x = rng.standard_normal((n, p))
    true_beta = np.zeros(p)
    true_beta[:3] = [2.0, -1.5, 1.0]
    y = x @ true_beta + rng.standard_normal(n) * 0.1

    beta, _ = lasso_ccd(y, x, lambda_=5.0, n_iters=200)
    n_nonzero = np.sum(np.abs(beta) > 1e-6)
    assert n_nonzero <= 5  # should be sparse, close to the true 3 nonzero


def test_lasso_ccd_at_zero_lambda_matches_ols(rng):
    n, p = 300, 3
    x = rng.standard_normal((n, p))
    y = x @ np.array([1.0, -1.0, 0.5]) + rng.standard_normal(n) * 0.01
    beta, _ = lasso_ccd(y, x, lambda_=0.0, n_iters=100)
    beta_ols = np.linalg.inv(x.T @ x) @ (x.T @ y)
    assert np.allclose(beta, beta_ols, atol=0.05)


def test_lasso_admm_matches_lasso_ccd(rng):
    n, p = 200, 5
    x = rng.standard_normal((n, p))
    true_beta = np.array([1.0, 0.0, -0.5, 0.0, 2.0])
    y = x @ true_beta + rng.standard_normal(n) * 0.1

    beta_ccd, _ = lasso_ccd(y, x, lambda_=3.0, n_iters=300)
    # ADMM's default tol=1e-10 is very tight for the fixed-step-size
    # variant (no adaptive varphi), so allow more iterations / a looser
    # tolerance here rather than asserting the strict `converged` flag --
    # the point of this test is numerical agreement with CCD, not the
    # exact convergence criterion.
    from quanttoolbox.config import ADMMConfig

    beta_admm, _, converged, n_iters_run = lasso_admm(
        y, x, lambda_=3.0, config=ADMMConfig(max_iters=3000, tol=1e-8)
    )
    assert converged
    assert np.allclose(beta_ccd, beta_admm, atol=0.01)


def test_lasso_tau_constrained_respects_budget(rng):
    n, p = 200, 5
    x = rng.standard_normal((n, p))
    y = x @ np.array([1.0, -1.0, 0.5, 0.0, 0.0]) + rng.standard_normal(n) * 0.1

    tau = 1.5
    beta, lam, df = lasso_tau_constrained(y, x, tau=tau)
    assert np.sum(np.abs(beta)) <= tau + 0.05


def test_lasso_tau_constrained_large_tau_matches_ols(rng):
    n, p = 200, 3
    x = rng.standard_normal((n, p))
    y = x @ np.array([1.0, -1.0, 0.5]) + rng.standard_normal(n) * 0.01
    beta_ols = np.linalg.inv(x.T @ x) @ (x.T @ y)

    tau = np.sum(np.abs(beta_ols)) * 2  # generous budget, should not bind
    beta, lam, df = lasso_tau_constrained(y, x, tau=tau)
    assert np.allclose(beta, beta_ols, atol=0.05)


def test_elastic_net_reduces_to_lasso_at_alpha_one(rng):
    n, p = 200, 5
    x = rng.standard_normal((n, p))
    y = x @ np.array([1.0, 0.0, -0.5, 0.0, 2.0]) + rng.standard_normal(n) * 0.1

    beta_lasso, _ = lasso_ccd(y, x, lambda_=3.0, n_iters=200)
    beta_en, _ = elastic_net_ccd(y, x, lambda_=3.0, alpha=1.0, n_iters=200)
    assert np.allclose(beta_lasso, beta_en, atol=0.05)


def _hsf_ch15b_data():
    # Same 15-obs dataset as HSF-Notebooks chapter 15b (reg_lasso2) and
    # docs/examples/stats/lasso2.py / Examples/stats/lasso2.m -- column 0
    # is y, columns 1:6 are x.
    data = np.array(
        [
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
        ]
    )

    # Notebook's own standardize helper (population std, ddof=0) -- not
    # quanttoolbox's own `standardize` (ddof=1), so numbers match the
    # notebook exactly.
    def standardize(a):
        a = np.asarray(a, dtype=float)
        if a.ndim == 1:
            return (a - a.mean()) / a.std(ddof=0)
        return (a - a.mean(axis=0)) / a.std(axis=0, ddof=0)

    y = standardize(data[:, 0])
    x = standardize(data[:, 1:6])
    return y, x


def test_lasso_regularization_path_matches_direct_lasso_ccd_calls():
    y, x = _hsf_ch15b_data()
    lambda_grid = np.array([0.0, 0.9, 2.5, 5.5, 7.5])

    result = lasso_regularization_path(y, x, lambda_grid, n_iters=1000)

    assert result.beta.shape == (len(lambda_grid), x.shape[1])
    for i, lam in enumerate(lambda_grid):
        beta_direct, _ = lasso_ccd(y, x, lam, n_iters=1000)
        assert np.allclose(result.beta[i], beta_direct)


def test_lasso_regularization_path_l1_norm_df_and_complexity():
    y, x = _hsf_ch15b_data()
    lambda_grid = np.array([0.0, 0.9, 2.5, 5.5, 7.5])

    result = lasso_regularization_path(y, x, lambda_grid, n_iters=1000)

    assert np.allclose(result.l1_norm, np.sum(np.abs(result.beta), axis=1))
    expected_df = np.sum(np.abs(result.beta) >= 1e-6, axis=1)
    assert np.array_equal(result.df, expected_df)
    with np.errstate(divide="ignore"):
        expected_complexity = 1.0 / expected_df
    assert np.allclose(result.complexity, expected_complexity, equal_nan=True)
    # df should be non-increasing (shrinkage removes variables, never adds).
    assert np.all(np.diff(result.df) <= 0)


def test_lasso_regularization_path_at_zero_lambda_matches_ols():
    y, x = _hsf_ch15b_data()

    result = lasso_regularization_path(y, x, lambda_grid=np.array([0.0]), n_iters=1000)
    beta_ols = np.linalg.inv(x.T @ x) @ (x.T @ y)
    assert np.allclose(result.beta[0], beta_ols)
    assert result.df[0] == x.shape[1]


def test_lasso_regularization_path_heavy_penalty_gives_infinite_complexity():
    y, x = _hsf_ch15b_data()
    huge_lambda = np.max(np.abs(x.T @ y)) * 10  # large enough to zero every coefficient

    result = lasso_regularization_path(y, x, lambda_grid=np.array([huge_lambda]), n_iters=500)
    assert result.df[0] == 0
    assert np.isposinf(result.complexity[0])
