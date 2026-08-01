"""Unit tests for ramentruck.toppings."""

import pytest
from sklearn.datasets import make_classification, make_regression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from ramentruck import toppings
from ramentruck.results import ToppingsResult


def _classification_data():
    """Build a small classification dataset."""

    return make_classification(n_samples=120, n_features=6, random_state=42)


def _regression_data():
    """Build a small regression dataset."""

    return make_regression(n_samples=120, n_features=6, random_state=42)


# ----------------------------------------------------------------------
# voting
# ----------------------------------------------------------------------


def test_voting_classification_returns_individual_scores():
    """Verify voting() scores the ensemble and each named member."""

    X, y = _classification_data()
    models = {
        "logreg": LogisticRegression(max_iter=500),
        "tree": DecisionTreeClassifier(random_state=42),
    }

    result = toppings.voting(models, X, y, task="classification", verbose=False)

    assert isinstance(result, ToppingsResult)
    assert 0.0 <= result.train_score <= 1.0
    assert set(result.individual_scores) == {"logreg", "tree"}
    assert result.val_score is None
    assert result.fit_time_s >= 0.0


def test_voting_regression_uses_regressor_ensemble():
    """Verify voting() builds a VotingRegressor for regression tasks."""

    X, y = _regression_data()
    models = [LinearRegression(), DecisionTreeRegressor(random_state=42)]

    result = toppings.voting(models, X, y, task="regression", verbose=False)

    assert set(result.individual_scores) == {"model_0", "model_1"}


def test_voting_uses_validation_data_when_provided():
    """Verify voting() scores against X_val/y_val when supplied."""

    X, y = _classification_data()
    X_train, X_val = X[:90], X[90:]
    y_train, y_val = y[:90], y[90:]
    models = {"logreg": LogisticRegression(max_iter=500), "tree": DecisionTreeClassifier(random_state=42)}

    result = toppings.voting(models, X_train, y_train, X_val=X_val, y_val=y_val, verbose=False)

    assert result.val_score is not None


def test_voting_rejects_unsupported_task():
    """Verify an unsupported task name raises a descriptive ValueError."""

    X, y = _classification_data()

    with pytest.raises(ValueError, match="Unsupported task"):
        toppings.voting({"logreg": LogisticRegression()}, X, y, task="clustering")


# ----------------------------------------------------------------------
# stack
# ----------------------------------------------------------------------


def test_stack_classification_returns_individual_scores():
    """Verify stack() scores the ensemble and each base model."""

    X, y = _classification_data()
    models = {"logreg": LogisticRegression(max_iter=500), "tree": DecisionTreeClassifier(random_state=42)}

    result = toppings.stack(
        models, LogisticRegression(max_iter=500), X, y, task="classification", cv=3, verbose=False
    )

    assert isinstance(result, ToppingsResult)
    assert set(result.individual_scores) == {"logreg", "tree"}


def test_stack_regression_returns_individual_scores():
    """Verify stack() supports regression tasks."""

    X, y = _regression_data()
    models = {"linear": LinearRegression(), "tree": DecisionTreeRegressor(random_state=42)}

    result = toppings.stack(models, LinearRegression(), X, y, task="regression", cv=3, verbose=False)

    assert set(result.individual_scores) == {"linear", "tree"}


# ----------------------------------------------------------------------
# bag
# ----------------------------------------------------------------------


def test_bag_classification_returns_per_estimator_scores():
    """Verify bag() scores the ensemble and each bootstrap-sampled member."""

    X, y = _classification_data()

    result = toppings.bag(
        DecisionTreeClassifier(random_state=42), X, y, task="classification", n_estimators=5, verbose=False
    )

    assert isinstance(result, ToppingsResult)
    assert len(result.individual_scores) == 5
    assert all(name.startswith("estimator_") for name in result.individual_scores)


def test_bag_regression_returns_per_estimator_scores():
    """Verify bag() supports regression tasks."""

    X, y = _regression_data()

    result = toppings.bag(
        DecisionTreeRegressor(random_state=42), X, y, task="regression", n_estimators=4, verbose=False
    )

    assert len(result.individual_scores) == 4


def test_bag_rejects_unsupported_task():
    """Verify an unsupported task name raises a descriptive ValueError."""

    X, y = _classification_data()

    with pytest.raises(ValueError, match="Unsupported task"):
        toppings.bag(DecisionTreeClassifier(), X, y, task="ranking")
