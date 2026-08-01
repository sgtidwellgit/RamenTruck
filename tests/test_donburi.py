"""Unit tests for ramentruck.donburi."""

import numpy as np
import pytest
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression

from ramentruck import Donburi, chashu


def _fitted_classifier():
    """Build a small fitted binary classifier with named features."""

    X, y = make_classification(n_samples=60, n_features=3, n_informative=3, n_redundant=0, random_state=42)
    model = RandomForestClassifier(n_estimators=10, random_state=42).fit(X, y)
    return model, X, y


def test_rejects_model_without_predict():
    """Verify Donburi requires a model with a callable predict method."""

    with pytest.raises(TypeError, match="predict"):
        Donburi(object())


def test_predict_returns_array():
    """Verify predict() returns predictions for a batch of feature data."""

    model, X, y = _fitted_classifier()
    donburi = Donburi(model, feature_names=["a", "b", "c"])

    predictions = donburi.predict(X)

    assert isinstance(predictions, np.ndarray)
    assert len(predictions) == len(X)


def test_predict_proba_raises_without_support():
    """Verify predict_proba raises AttributeError for models lacking it."""

    X, y = make_classification(n_samples=40, n_features=3, n_informative=3, n_redundant=0, random_state=42)
    model = LinearRegression().fit(X, y.astype(float))
    donburi = Donburi(model)

    with pytest.raises(AttributeError, match="predict_proba"):
        donburi.predict_proba(X)


def test_predict_one_returns_single_value():
    """Verify predict_one accepts a dict record and returns one prediction."""

    model, X, y = _fitted_classifier()
    donburi = Donburi(model, feature_names=["a", "b", "c"])
    record = {"a": X[0, 0], "b": X[0, 1], "c": X[0, 2]}

    prediction = donburi.predict_one(record)

    assert prediction in (0, 1)


def test_predict_one_validates_missing_features():
    """Verify predict_one raises when a required feature is missing."""

    model, X, y = _fitted_classifier()
    donburi = Donburi(model, feature_names=["a", "b", "c"])

    with pytest.raises(ValueError, match="missing required feature"):
        donburi.predict_one({"a": 1.0, "b": 2.0})


def test_predict_batch_returns_list_per_record():
    """Verify predict_batch handles a list of records and empty input."""

    model, X, y = _fitted_classifier()
    donburi = Donburi(model, feature_names=["a", "b", "c"])
    records = [{"a": X[i, 0], "b": X[i, 1], "c": X[i, 2]} for i in range(3)]

    predictions = donburi.predict_batch(records)

    assert len(predictions) == 3
    assert donburi.predict_batch([]) == []


def test_serve_includes_probabilities_for_classifier():
    """Verify serve() returns a JSON-friendly dict with class probabilities."""

    model, X, y = _fitted_classifier()
    donburi = Donburi(model, feature_names=["a", "b", "c"])
    record = {"a": X[0, 0], "b": X[0, 1], "c": X[0, 2]}

    response = donburi.serve(record)

    assert "prediction" in response
    assert isinstance(response["prediction"], (int, float, str))
    assert "probabilities" in response
    assert set(response["probabilities"]) == {"0", "1"}
    assert pytest.approx(sum(response["probabilities"].values()), abs=1e-6) == 1.0


def test_serve_omits_probabilities_without_support():
    """Verify serve() omits probabilities when the model lacks predict_proba."""

    X, y = make_classification(n_samples=40, n_features=3, n_informative=3, n_redundant=0, random_state=42)
    model = LinearRegression().fit(X, y.astype(float))
    donburi = Donburi(model, feature_names=["a", "b", "c"])
    record = {"a": X[0, 0], "b": X[0, 1], "c": X[0, 2]}

    response = donburi.serve(record)

    assert "probabilities" not in response


def test_from_chashu_round_trips_model_and_feature_names(tmp_path):
    """Verify from_chashu loads a saved bundle and restores feature_names."""

    model, X, y = _fitted_classifier()
    bundle_path = tmp_path / "rf.chashu"
    chashu.save(model, bundle_path, metadata={"feature_names": ["a", "b", "c"]})

    donburi = Donburi.from_chashu(bundle_path)

    assert donburi.feature_names == ["a", "b", "c"]
    prediction = donburi.predict_one({"a": X[0, 0], "b": X[0, 1], "c": X[0, 2]})
    assert prediction in (0, 1)


def test_no_feature_names_skips_validation():
    """Verify records are accepted as-is when feature_names is not provided."""

    model, X, y = _fitted_classifier()
    donburi = Donburi(model)

    prediction = donburi.predict_one({"a": X[0, 0], "b": X[0, 1], "c": X[0, 2]})
    assert prediction in (0, 1)
