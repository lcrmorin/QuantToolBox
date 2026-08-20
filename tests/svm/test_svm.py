"""Tests for quanttoolbox.svm.svm."""

import numpy as np
import pytest

from quanttoolbox.svm.svm import (
    svm_classification_dual,
    svm_classification_primal,
    svm_regression_dual,
    svm_regression_primal,
)


@pytest.fixture
def separable_classification_data(rng):
    x = np.vstack([rng.standard_normal((30, 2)) + 2, rng.standard_normal((30, 2)) - 2])
    y = np.concatenate([np.ones(30), -np.ones(30)])
    return x, y


def test_svm_classification_dual_matches_sklearn(separable_classification_data):
    sklearn = pytest.importorskip("sklearn.svm")
    x, y = separable_classification_data
    c = 1.0

    sk = sklearn.SVC(kernel="linear", C=c)
    sk.fit(x, y)

    result = svm_classification_dual(y, x, c=c, loss="hinge")
    assert np.allclose(result.beta, sk.coef_[0], atol=1e-3)
    assert np.isclose(result.beta0, sk.intercept_[0], atol=1e-3)


def test_svm_classification_primal_matches_dual(separable_classification_data):
    x, y = separable_classification_data
    dual = svm_classification_dual(y, x, c=1.0, loss="hinge")
    primal = svm_classification_primal(y, x, c=1.0, loss="hinge")
    assert np.allclose(dual.beta, primal.beta, atol=1e-3)
    assert np.isclose(dual.beta0, primal.beta0, atol=1e-3)


def test_svm_classification_hard_margin_primal_matches_dual(separable_classification_data):
    x, y = separable_classification_data
    dual = svm_classification_dual(y, x, c=None)
    primal = svm_classification_primal(y, x, c=None)
    assert np.allclose(dual.beta, primal.beta, atol=1e-3)


def test_svm_classification_squared_hinge_runs(separable_classification_data):
    x, y = separable_classification_data
    result = svm_classification_dual(y, x, c=1.0, loss="squared_hinge")
    assert result.beta.shape == (2,)
    # correctly classifies well-separated data
    preds = np.sign(result.beta0 + x @ result.beta)
    assert np.mean(preds == y) > 0.95


def test_svm_regression_epsilon_dual_matches_sklearn(rng):
    sklearn = pytest.importorskip("sklearn.svm")
    n = 100
    x = rng.standard_normal((n, 2))
    true_beta = np.array([1.5, -0.8])
    y = x @ true_beta + rng.standard_normal(n) * 0.1
    c, eps = 10.0, 0.1

    sk = sklearn.SVR(kernel="linear", C=c, epsilon=eps)
    sk.fit(x, y)

    result = svm_regression_dual(y, x, c=c, epsilon=eps)
    assert np.allclose(result.beta, sk.coef_[0], atol=0.05)
    assert np.isclose(result.beta0, sk.intercept_[0], atol=0.05)


def test_svm_regression_ls_svm_dual_matches_primal(rng):
    n = 100
    x = rng.standard_normal((n, 2))
    true_beta = np.array([1.5, -0.8])
    y = x @ true_beta + rng.standard_normal(n) * 0.05

    dual = svm_regression_dual(y, x, c=10.0, epsilon=None)
    primal = svm_regression_primal(y, x, c=10.0, epsilon=None)
    assert np.allclose(dual.beta, primal.beta, atol=1e-3)
    assert np.isclose(dual.beta0, primal.beta0, atol=1e-3)


def test_svm_regression_ls_svm_recovers_true_coefficients(rng):
    n = 200
    x = rng.standard_normal((n, 2))
    true_beta = np.array([1.5, -0.8])
    y = x @ true_beta + rng.standard_normal(n) * 0.05

    result = svm_regression_dual(y, x, c=100.0, epsilon=None)
    assert np.allclose(result.beta, true_beta, atol=0.1)


def test_svm_regression_epsilon_zero_dual_runs(rng):
    n = 100
    x = rng.standard_normal((n, 2))
    y = x @ np.array([1.0, -1.0]) + rng.standard_normal(n) * 0.1
    result = svm_regression_dual(y, x, c=10.0, epsilon=0.0)
    assert result.beta.shape == (2,)
