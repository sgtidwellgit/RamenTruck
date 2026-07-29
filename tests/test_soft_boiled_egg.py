"""Unit tests for ramentruck.soft_boiled_egg."""

import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_classification, make_regression
from sklearn.linear_model import LinearRegression, LogisticRegression

from ramentruck import EggResult, soft_boiled_egg


def test_soft_boiled_egg_stratified_default():
    """Verify the stratified default strategy returns a populated EggResult."""

    X, y = make_classification(n_samples=100, n_features=6, random_state=42)

    result = soft_boiled_egg(
        LogisticRegression(max_iter=500),
        X,
        y,
        n_splits=5,
        verbose=False,
    )

    assert isinstance(result, EggResult)
    assert len(result.scores) == 5
    assert 0.0 <= result.mean_score <= 1.0
    assert result.std_score >= 0.0
    assert isinstance(result.fold_results, pd.DataFrame)
    assert len(result.fold_results) == 5
    assert result.learning_curve_df is None


def test_soft_boiled_egg_multiple_scoring_metrics():
    """Verify fold_results has a column per requested metric."""

    X, y = make_classification(n_samples=100, n_features=6, random_state=42)

    result = soft_boiled_egg(
        LogisticRegression(max_iter=500),
        X,
        y,
        n_splits=5,
        scoring=["accuracy", "f1"],
        verbose=False,
    )

    assert set(result.fold_results.columns) == {"accuracy", "f1"}


def test_soft_boiled_egg_kfold_strategy_supports_regression():
    """Verify kfold strategy works for regression targets."""

    X, y = make_regression(n_samples=80, n_features=4, noise=0.1, random_state=42)

    result = soft_boiled_egg(
        LinearRegression(),
        X,
        y,
        strategy="kfold",
        n_splits=4,
        scoring="r2",
        verbose=False,
    )

    assert len(result.scores) == 4


def test_soft_boiled_egg_learning_curve():
    """Verify learning_curve=True populates learning_curve_df."""

    X, y = make_classification(n_samples=150, n_features=6, random_state=42)

    result = soft_boiled_egg(
        LogisticRegression(max_iter=500),
        X,
        y,
        n_splits=3,
        learning_curve=True,
        verbose=False,
    )

    assert result.learning_curve_df is not None
    expected_columns = {
        "train_size",
        "train_score_mean",
        "train_score_std",
        "val_score_mean",
        "val_score_std",
    }
    assert set(result.learning_curve_df.columns) == expected_columns


def test_soft_boiled_egg_timeseries_warns_on_shuffle():
    """Verify timeseries strategy warns when shuffle=True."""

    X, y = make_regression(n_samples=60, n_features=3, noise=0.1, random_state=42)

    with pytest.warns(UserWarning, match="never shuffles"):
        soft_boiled_egg(
            LinearRegression(),
            X,
            y,
            strategy="timeseries",
            n_splits=3,
            scoring="r2",
            shuffle=True,
            verbose=False,
        )


def test_soft_boiled_egg_timeseries_no_warning_without_shuffle(recwarn):
    """Verify timeseries strategy does not warn when shuffle=False."""

    X, y = make_regression(n_samples=60, n_features=3, noise=0.1, random_state=42)

    soft_boiled_egg(
        LinearRegression(),
        X,
        y,
        strategy="timeseries",
        n_splits=3,
        scoring="r2",
        shuffle=False,
        verbose=False,
    )

    messages = [str(warning.message) for warning in recwarn.list]
    assert not any("never shuffles" in message for message in messages)


def test_soft_boiled_egg_warns_on_small_minority_class():
    """Verify a warning fires when n_splits exceeds the minority class size."""

    X = np.random.RandomState(0).normal(size=(30, 4))
    y = np.array([0] * 27 + [1] * 3)

    with pytest.warns(UserWarning, match="minority class"):
        soft_boiled_egg(
            LogisticRegression(max_iter=500),
            X,
            y,
            strategy="stratified",
            n_splits=5,
            verbose=False,
        )


def test_soft_boiled_egg_rejects_unsupported_strategy():
    """Verify an unsupported strategy raises a descriptive ValueError."""

    X, y = make_classification(n_samples=40, random_state=42)

    with pytest.raises(ValueError, match="Unsupported strategy"):
        soft_boiled_egg(LogisticRegression(max_iter=500), X, y, strategy="bootstrap")
