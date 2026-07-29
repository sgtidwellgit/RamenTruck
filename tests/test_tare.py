"""Unit tests for ramentruck.tare."""

import pandas as pd
import pytest
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from ramentruck import TareResult, tare


def test_tare_grid_search_returns_result():
    """Verify grid search returns a populated TareResult."""

    X, y = make_classification(n_samples=100, n_features=6, random_state=42)

    result = tare(
        LogisticRegression(max_iter=500),
        param_grid={"C": [0.1, 1.0, 10.0]},
        X=X,
        y=y,
        method="grid",
        cv=3,
        n_jobs=1,
        verbose=False,
    )

    assert isinstance(result, TareResult)
    assert result.best_params["C"] in {0.1, 1.0, 10.0}
    assert 0.0 <= result.best_score <= 1.0
    assert result.best_model is not None
    assert isinstance(result.cv_results, pd.DataFrame)
    assert result.search_time_s >= 0.0


def test_tare_grid_results_sorted_best_first():
    """Verify cv_results is sorted with the best rank first."""

    X, y = make_classification(n_samples=100, n_features=6, random_state=42)

    result = tare(
        LogisticRegression(max_iter=500),
        param_grid={"C": [0.01, 0.1, 1.0, 10.0]},
        X=X,
        y=y,
        method="grid",
        cv=3,
        n_jobs=1,
        verbose=False,
    )

    ranks = result.cv_results["rank_test_score"].tolist()
    assert ranks == sorted(ranks)
    assert ranks[0] == 1


def test_tare_random_search_uses_n_iter():
    """Verify random search samples n_iter combinations."""

    X, y = make_classification(n_samples=100, n_features=6, random_state=42)

    result = tare(
        RandomForestClassifier(random_state=42),
        param_grid={
            "n_estimators": [10, 20, 30],
            "max_depth": [2, 3, 4],
        },
        X=X,
        y=y,
        method="random",
        cv=3,
        n_iter=4,
        n_jobs=1,
        verbose=False,
        random_state=42,
    )

    assert isinstance(result, TareResult)
    assert len(result.cv_results) == 4


def test_tare_rejects_unsupported_method():
    """Verify an unsupported method name raises a descriptive ValueError."""

    X, y = make_classification(n_samples=40, random_state=42)

    with pytest.raises(ValueError, match="Unsupported method"):
        tare(
            LogisticRegression(max_iter=500),
            param_grid={"C": [1.0]},
            X=X,
            y=y,
            method="bayesian",
        )
