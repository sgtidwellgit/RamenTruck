"""Unit tests for ramentruck.kaeshi."""

import matplotlib
import pandas as pd
import pytest
from sklearn.datasets import make_classification
from sklearn.svm import SVC

from ramentruck import kaeshi, plot_calibration_curve
from ramentruck.results import KaeshiResult

matplotlib.use("Agg")


def _uncalibrated_svc_data():
    """Build a dataset with an SVC, whose raw decision scores are poorly calibrated."""

    X, y = make_classification(n_samples=200, n_features=8, random_state=42)
    return X, y


def test_kaeshi_returns_result_with_curve():
    """Verify kaeshi() returns brier scores and a calibration curve DataFrame."""

    X, y = _uncalibrated_svc_data()
    result = kaeshi(SVC(probability=True, random_state=42), X, y, cv=3, verbose=False)

    assert isinstance(result, KaeshiResult)
    assert result.method == "isotonic"
    assert result.brier_score_before >= 0.0
    assert result.brier_score_after >= 0.0
    assert isinstance(result.calibration_curve_df, pd.DataFrame)
    assert list(result.calibration_curve_df.columns) == ["prob_pred", "prob_true"]


def test_kaeshi_sigmoid_method():
    """Verify method='sigmoid' is accepted and recorded on the result."""

    X, y = _uncalibrated_svc_data()
    result = kaeshi(
        SVC(probability=True, random_state=42), X, y, method="sigmoid", cv=3, verbose=False
    )

    assert result.method == "sigmoid"


def test_kaeshi_rejects_unsupported_method():
    """Verify an unsupported method name raises a descriptive ValueError."""

    X, y = _uncalibrated_svc_data()

    with pytest.raises(ValueError, match="Unsupported method"):
        kaeshi(SVC(probability=True), X, y, method="platt", cv=3, verbose=False)


def test_kaeshi_model_is_fitted_and_predicts_probabilities():
    """Verify the returned model is fitted and predict_proba returns valid probabilities."""

    X, y = _uncalibrated_svc_data()
    result = kaeshi(SVC(probability=True, random_state=42), X, y, cv=3, verbose=False)

    probabilities = result.model.predict_proba(X)
    assert probabilities.shape == (len(X), 2)
    assert (probabilities >= 0).all() and (probabilities <= 1).all()


def test_plot_calibration_curve_returns_figure():
    """Verify plot_calibration_curve returns a matplotlib Figure."""

    X, y = _uncalibrated_svc_data()
    result = kaeshi(SVC(probability=True, random_state=42), X, y, cv=3, verbose=False)

    fig = plot_calibration_curve(result)

    assert fig is not None
    assert len(fig.axes) == 1
