"""Unit tests for ramentruck.broth."""

import numpy as np
import pytest
from sklearn.datasets import make_classification, make_regression
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import train_test_split

from ramentruck import Broth, BrothResult


def test_broth_trains_classifier():
    """Verify classifier training returns a BrothResult."""

    X, y = make_classification(
        n_samples=120,
        n_features=6,
        n_informative=4,
        random_state=42,
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y,
        test_size=0.25,
        random_state=42,
    )

    trainer = Broth(LogisticRegression(max_iter=500))

    result = trainer.fit(
        X_train,
        y_train,
        X_val,
        y_val,
        metrics=["accuracy", "precision", "recall", "f1", "roc_auc"],
    )

    assert isinstance(result, BrothResult)
    assert result.model is trainer.estimator
    assert 0.0 <= result.train_score <= 1.0
    assert result.val_score is not None
    assert set(result.metrics) == {
        "accuracy",
        "precision",
        "recall",
        "f1",
        "roc_auc",
    }
    assert result.fit_time_s >= 0.0


def test_broth_trains_regressor():
    """Verify regression training supports regression metrics."""

    X, y = make_regression(
        n_samples=80,
        n_features=4,
        noise=0.1,
        random_state=42,
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y,
        test_size=0.25,
        random_state=42,
    )

    trainer = Broth(LinearRegression())
    result = trainer.fit(
        X_train,
        y_train,
        X_val,
        y_val,
        metrics=["mse", "rmse", "r2"],
    )

    assert result.val_score is not None
    assert result.metrics["mse"] >= 0.0
    assert result.metrics["rmse"] >= 0.0
    assert result.metrics["r2"] > 0.9


def test_broth_rejects_invalid_metric():
    """Verify unknown metrics raise a descriptive ValueError."""

    X, y = make_classification(random_state=42)
    trainer = Broth(LogisticRegression(max_iter=500))

    with pytest.raises(ValueError, match="Unsupported metric"):
        trainer.fit(X, y, metrics=["not_a_metric"])


def test_broth_rejects_mismatched_training_lengths():
    """Verify mismatched training X/y lengths are rejected."""

    X, y = make_classification(n_samples=20, random_state=42)
    trainer = Broth(LogisticRegression(max_iter=500))

    with pytest.raises(ValueError, match="training X and y lengths"):
        trainer.fit(X, y[:-1])


def test_broth_rejects_mismatched_validation_lengths():
    """Verify mismatched validation X/y lengths are rejected."""

    X, y = make_classification(n_samples=40, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y,
        test_size=0.25,
        random_state=42,
    )
    trainer = Broth(LogisticRegression(max_iter=500))

    with pytest.raises(ValueError, match="validation X and y lengths"):
        trainer.fit(X_train, y_train, X_val, y_val[:-1])


def test_broth_supports_no_validation_data():
    """Verify validation data is optional."""

    X, y = make_classification(random_state=42)
    trainer = Broth(LogisticRegression(max_iter=500))

    result = trainer.fit(X, y, metrics=["accuracy"])

    assert result.val_score is None
    assert "accuracy" in result.metrics


def test_broth_preserves_explicit_empty_metrics():
    """Verify an explicit empty metric list computes no extra metrics."""

    X, y = make_classification(random_state=42)
    trainer = Broth(LogisticRegression(max_iter=500))

    result = trainer.fit(X, y, metrics=[])

    assert result.metrics == {}


def test_broth_predict_after_fit():
    """Verify predict delegates to the fitted estimator."""

    X, y = make_classification(random_state=42)
    trainer = Broth(LogisticRegression(max_iter=500))
    trainer.fit(X, y)

    predictions = trainer.predict(X[:5])

    assert len(predictions) == 5


def test_broth_score_after_fit():
    """Verify score works after fitting."""

    X, y = make_classification(random_state=42)
    trainer = Broth(LogisticRegression(max_iter=500))
    trainer.fit(X, y)

    score = trainer.score(X, y)
    metric_score = trainer.score(X, y, metric="accuracy")

    assert 0.0 <= score <= 1.0
    assert 0.0 <= metric_score <= 1.0


def test_broth_warns_when_overfitting():
    """Verify a warning is emitted when train score greatly exceeds val score."""

    X_train, y_train = make_classification(
        n_samples=60,
        n_features=10,
        n_informative=6,
        random_state=42,
    )
    X_val = np.random.RandomState(7).normal(size=(60, 10))
    y_val = np.array([0, 1] * 30)
    trainer = Broth(RandomForestClassifier(random_state=42))

    with pytest.warns(UserWarning, match="Possible overfitting"):
        trainer.fit(X_train, y_train, X_val, y_val)


def test_broth_rejects_unpaired_validation_data():
    """Verify validation X/y must be supplied together."""

    X, y = make_classification(random_state=42)
    trainer = Broth(LogisticRegression(max_iter=500))

    with pytest.raises(ValueError, match="X_val and y_val"):
        trainer.fit(X, y, X_val=X)


def test_broth_rejects_estimator_without_predict():
    """Verify estimators must provide predict."""

    class FitOnly:
        """Estimator test double with no predict method."""

        def fit(self, X, y):
            """Fit test double."""

    with pytest.raises(TypeError, match="predict"):
        Broth(FitOnly())


def test_broth_roc_auc_requires_scores():
    """Verify roc_auc requires probability or decision scores."""

    X, y = make_classification(random_state=42)

    class PredictOnlyClassifier:
        """Classifier test double without probability scores."""

        def fit(self, X, y):
            """Fit and remember the majority class."""

            values, counts = np.unique(y, return_counts=True)
            self.majority_class_ = values[np.argmax(counts)]
            return self

        def predict(self, X):
            """Predict the majority class."""

            return np.full(len(X), self.majority_class_)

        def score(self, X, y):
            """Return accuracy."""

            return float(np.mean(self.predict(X) == y))

    trainer = Broth(PredictOnlyClassifier())

    with pytest.raises(ValueError, match="roc_auc requires"):
        trainer.fit(X, y, metrics=["roc_auc"])


def test_broth_roc_auc_uses_fitted_classes_for_multiclass_batches():
    """Verify multiclass ROC AUC does not fall back to a binary path."""

    X, y = make_classification(
        n_samples=300,
        n_features=8,
        n_informative=6,
        n_redundant=0,
        n_classes=3,
        n_clusters_per_class=1,
        random_state=42,
    )

    X_train = X
    y_train = y
    X_val = X[y != 2]
    y_val = y[y != 2]

    trainer = Broth(LogisticRegression(max_iter=1000))

    with pytest.raises(
        ValueError,
        match="multiclass estimator requires the evaluation target to contain all fitted classes",
    ):
        trainer.fit(X_train, y_train, X_val, y_val, metrics=["roc_auc"])


def test_broth_roc_auc_requires_two_evaluation_classes():
    """Verify ROC AUC raises clearly when evaluation data has one class."""

    X, y = make_classification(random_state=42)
    trainer = Broth(LogisticRegression(max_iter=500))
    trainer.fit(X, y, metrics=[])

    one_class_mask = y == y[0]

    with pytest.raises(
        ValueError,
        match="roc_auc requires at least two classes in the evaluation target",
    ):
        trainer.score(X[one_class_mask], y[one_class_mask], metric="roc_auc")
