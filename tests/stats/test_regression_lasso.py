"""Tests for quanttoolbox.stats.regression.lasso."""

import numpy as np

from quanttoolbox.stats.regression.lasso import (
    elastic_net_ccd,
    lasso_admm,
    lasso_ccd,
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
