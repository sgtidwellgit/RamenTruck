"""Unit tests for ramentruck.nori."""

import matplotlib
import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_classification, make_regression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

from ramentruck import nori
from ramentruck.results import NoriResult, PDResult

matplotlib.use("Agg")


def _binary_classification_frame():
    """Build a small binary classification dataset as a DataFrame."""

    X, y = make_classification(n_samples=60, n_features=5, random_state=42)
    df = pd.DataFrame(X, columns=[f"f{i}" for i in range(5)])
    return df, y


def _multiclass_classification_frame():
    """Build a small multiclass classification dataset as a DataFrame."""

    X, y = make_classification(
        n_samples=90, n_features=5, n_informative=4, n_redundant=0, n_classes=3, random_state=42
    )
    df = pd.DataFrame(X, columns=[f"f{i}" for i in range(5)])
    return df, y


def _regression_frame():
    """Build a small regression dataset as a DataFrame."""

    X, y = make_regression(n_samples=60, n_features=5, random_state=42)
    df = pd.DataFrame(X, columns=[f"f{i}" for i in range(5)])
    return df, y


# ----------------------------------------------------------------------
# explain
# ----------------------------------------------------------------------


def test_explain_binary_classifier_returns_result():
    """Verify explain() returns a NoriResult with a 2D shap_values matrix."""

    X, y = _binary_classification_frame()
    model = RandomForestClassifier(n_estimators=10, random_state=42).fit(X, y)

    result = nori.explain(model, X, verbose=False)

    assert isinstance(result, NoriResult)
    assert result.shap_values.shape == (len(X), 5)
    assert result.feature_names == list(X.columns)
    assert set(result.importance["feature"]) == set(X.columns)
    assert result.class_index == 1


def test_explain_regressor_returns_2d_values():
    """Verify explain() handles regressors whose SHAP output is already 2D."""

    X, y = _regression_frame()
    model = RandomForestRegressor(n_estimators=10, random_state=42).fit(X, y)

    result = nori.explain(model, X, verbose=False)

    assert result.shap_values.shape == (len(X), 5)
    assert result.class_index is None


def test_explain_multiclass_requires_class_index():
    """Verify multiclass models require an explicit class_index."""

    X, y = _multiclass_classification_frame()
    model = RandomForestClassifier(n_estimators=10, random_state=42).fit(X, y)

    with pytest.raises(ValueError, match="class_index is required"):
        nori.explain(model, X, verbose=False)


def test_explain_multiclass_with_class_index():
    """Verify multiclass models can be explained for a specific class."""

    X, y = _multiclass_classification_frame()
    model = RandomForestClassifier(n_estimators=10, random_state=42).fit(X, y)

    result = nori.explain(model, X, class_index=2, verbose=False)

    assert result.shap_values.shape == (len(X), 5)
    assert result.class_index == 2


def test_explain_importance_sorted_descending():
    """Verify the importance table is sorted by mean absolute SHAP value."""

    X, y = _binary_classification_frame()
    model = RandomForestClassifier(n_estimators=10, random_state=42).fit(X, y)

    result = nori.explain(model, X, verbose=False)
    values = result.importance["mean_abs_shap"].tolist()

    assert values == sorted(values, reverse=True)


# ----------------------------------------------------------------------
# explain_instance
# ----------------------------------------------------------------------


def test_explain_instance_sorted_by_magnitude():
    """Verify explain_instance sorts contributions by absolute magnitude."""

    X, y = _binary_classification_frame()
    model = RandomForestClassifier(n_estimators=10, random_state=42).fit(X, y)
    result = nori.explain(model, X, verbose=False)

    contributions = nori.explain_instance(result, 0)

    assert list(contributions.index) == list(
        contributions.abs().sort_values(ascending=False).index
    )
    assert set(contributions.index) == set(X.columns)


# ----------------------------------------------------------------------
# plotting
# ----------------------------------------------------------------------


def test_plot_importance_returns_figure():
    """Verify plot_importance returns a matplotlib Figure."""

    X, y = _binary_classification_frame()
    model = RandomForestClassifier(n_estimators=10, random_state=42).fit(X, y)
    result = nori.explain(model, X, verbose=False)

    fig = nori.plot_importance(result, max_display=3)

    assert fig is not None
    assert len(fig.axes[0].patches) == 3


def test_plot_summary_returns_figure():
    """Verify plot_summary returns a matplotlib Figure with a colorbar."""

    X, y = _binary_classification_frame()
    model = RandomForestClassifier(n_estimators=10, random_state=42).fit(X, y)
    result = nori.explain(model, X, verbose=False)

    fig = nori.plot_summary(result, max_display=3)

    assert fig is not None
    assert len(fig.axes) == 2  # main axis + colorbar axis


# ----------------------------------------------------------------------
# partial_dependence
# ----------------------------------------------------------------------


def test_partial_dependence_returns_curve_per_feature():
    """Verify partial_dependence produces one DataFrame per requested feature."""

    X, y = _binary_classification_frame()
    model = RandomForestClassifier(n_estimators=10, random_state=42).fit(X, y)

    result = nori.partial_dependence(model, X, features=["f0", "f1"], grid_resolution=5, verbose=False)

    assert isinstance(result, PDResult)
    assert set(result.curves) == {"f0", "f1"}
    for curve in result.curves.values():
        assert list(curve.columns) == ["grid_value", "average"]
        assert len(curve) == 5


def test_partial_dependence_accepts_column_indices():
    """Verify partial_dependence resolves integer feature indices."""

    X, y = _regression_frame()
    model = RandomForestRegressor(n_estimators=10, random_state=42).fit(X, y)

    result = nori.partial_dependence(model, X, features=[0], grid_resolution=4, verbose=False)

    assert set(result.curves) == {"f0"}


def test_plot_partial_dependence_returns_figure():
    """Verify plot_partial_dependence returns one subplot per feature."""

    X, y = _binary_classification_frame()
    model = RandomForestClassifier(n_estimators=10, random_state=42).fit(X, y)
    pd_result = nori.partial_dependence(model, X, features=["f0", "f1"], grid_resolution=5, verbose=False)

    fig = nori.plot_partial_dependence(pd_result)

    assert len(fig.axes) == 2
