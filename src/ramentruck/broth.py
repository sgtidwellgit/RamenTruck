"""Model training wrapper for RamenTruck."""

from __future__ import annotations

import math
import time
import warnings
from typing import Any, Callable, Iterable

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)

from .results import BrothResult


MetricFunction = Callable[[Any, Any, Any], float]

OVERFIT_WARNING_THRESHOLD = 0.10


class Broth:
    """Train and evaluate a scikit-learn compatible estimator."""

    def __init__(self, estimator: Any) -> None:
        """
        Initialize a Broth trainer.

        Parameters
        ----------
        estimator
            Estimator implementing ``fit`` and ``predict``.
        """

        self._validate_estimator(estimator)
        self.estimator = estimator
        self._is_fitted = False

    def fit(
        self,
        X_train: Any,
        y_train: Any,
        X_val: Any | None = None,
        y_val: Any | None = None,
        *,
        metrics: Iterable[str] | None = None,
    ) -> BrothResult:
        """
        Fit the estimator and return training metadata.

        Parameters
        ----------
        X_train
            Training features.
        y_train
            Training target values.
        X_val
            Optional validation features.
        y_val
            Optional validation target values.
        metrics
            Metric names to calculate. Metrics are calculated on validation
            data when available, otherwise on training data.

        Returns
        -------
        BrothResult
            Fitted model, train/validation scores, metric values, and elapsed
            fit time.
        """

        self._validate_matching_lengths(X_train, y_train, "training")
        has_validation = self._has_validation_data(X_val, y_val)

        if has_validation:
            self._validate_matching_lengths(X_val, y_val, "validation")

        if metrics is None:
            metric_names = ["accuracy"]
        else:
            metric_names = list(metrics)
        self._validate_metrics(metric_names)

        start = time.perf_counter()
        self.estimator.fit(X_train, y_train)
        fit_time_s = time.perf_counter() - start
        self._is_fitted = True

        train_score = float(self.estimator.score(X_train, y_train))
        val_score = None

        if has_validation:
            val_score = float(self.estimator.score(X_val, y_val))
            self._warn_if_overfit(train_score, val_score)

        metric_X = X_val if has_validation else X_train
        metric_y = y_val if has_validation else y_train
        metric_values = self._calculate_metrics(metric_names, metric_X, metric_y)

        return BrothResult(
            model=self.estimator,
            train_score=train_score,
            val_score=val_score,
            metrics=metric_values,
            fit_time_s=fit_time_s,
        )

    def predict(self, X: Any) -> Any:
        """
        Predict target values with the fitted estimator.

        Parameters
        ----------
        X
            Feature data.

        Returns
        -------
        Any
            Estimator predictions.
        """

        self._require_fitted()
        return self.estimator.predict(X)

    def score(
        self,
        X: Any,
        y: Any,
        *,
        metric: str | None = None,
    ) -> float:
        """
        Score the fitted estimator.

        Parameters
        ----------
        X
            Feature data.
        y
            Target values.
        metric
            Optional RamenTruck metric name. When omitted, the estimator's
            native ``score`` method is used.

        Returns
        -------
        float
            Score value.
        """

        self._require_fitted()
        self._validate_matching_lengths(X, y, "score")

        if metric is None:
            if not hasattr(self.estimator, "score"):
                raise ValueError(
                    "The estimator does not provide a score method. "
                    "Pass a supported metric name instead."
                )
            return float(self.estimator.score(X, y))

        self._validate_metrics([metric])
        return self._calculate_metric(metric, X, y)

    @staticmethod
    def _validate_estimator(estimator: Any) -> None:
        """Validate that an estimator implements the required API."""

        if not hasattr(estimator, "fit") or not callable(estimator.fit):
            raise TypeError("estimator must provide a callable fit method.")

        if not hasattr(estimator, "predict") or not callable(estimator.predict):
            raise TypeError("estimator must provide a callable predict method.")

    @staticmethod
    def _validate_matching_lengths(X: Any, y: Any, label: str) -> None:
        """Validate that feature and target inputs have matching lengths."""

        try:
            x_length = len(X)
            y_length = len(y)
        except TypeError as exc:
            raise TypeError(
                f"{label} X and y must both have a defined length."
            ) from exc

        if x_length != y_length:
            raise ValueError(
                f"{label} X and y lengths must match; got "
                f"{x_length} and {y_length}."
            )

    @staticmethod
    def _has_validation_data(X_val: Any | None, y_val: Any | None) -> bool:
        """Return whether validation data was provided as a complete pair."""

        if (X_val is None) != (y_val is None):
            raise ValueError(
                "X_val and y_val must either both be provided or both be None."
            )

        return X_val is not None and y_val is not None

    @staticmethod
    def _validate_metrics(metrics: Iterable[str]) -> None:
        """Validate metric names before training or scoring."""

        supported_metrics = set(METRIC_FUNCTIONS)
        unknown_metrics = [
            metric for metric in metrics if metric not in supported_metrics
        ]

        if unknown_metrics:
            supported = ", ".join(sorted(supported_metrics))
            unknown = ", ".join(unknown_metrics)
            raise ValueError(
                f"Unsupported metric(s): {unknown}. "
                f"Supported metrics are: {supported}."
            )

    def _calculate_metrics(
        self,
        metrics: Iterable[str],
        X: Any,
        y: Any,
    ) -> dict[str, float]:
        """Calculate a mapping of metric names to values."""

        return {
            metric: self._calculate_metric(metric, X, y)
            for metric in metrics
        }

    def _calculate_metric(self, metric: str, X: Any, y: Any) -> float:
        """Calculate a single named metric."""

        return float(METRIC_FUNCTIONS[metric](self.estimator, X, y))

    def _require_fitted(self) -> None:
        """Raise when the wrapped estimator has not been fitted by Broth."""

        if not self._is_fitted:
            raise RuntimeError("Broth estimator must be fitted before use.")

    @staticmethod
    def _warn_if_overfit(train_score: float, val_score: float) -> None:
        """Warn when validation score trails training score materially."""

        if train_score - val_score > OVERFIT_WARNING_THRESHOLD:
            warnings.warn(
                "Possible overfitting detected: training score exceeds "
                "validation score by more than 0.10.",
                UserWarning,
                stacklevel=2,
            )


def _accuracy(estimator: Any, X: Any, y: Any) -> float:
    """Calculate accuracy."""

    return float(accuracy_score(y, estimator.predict(X)))


def _precision(estimator: Any, X: Any, y: Any) -> float:
    """Calculate precision."""

    return float(
        precision_score(
            y,
            estimator.predict(X),
            average=_classification_average(y),
            zero_division=0,
        )
    )


def _recall(estimator: Any, X: Any, y: Any) -> float:
    """Calculate recall."""

    return float(
        recall_score(
            y,
            estimator.predict(X),
            average=_classification_average(y),
            zero_division=0,
        )
    )


def _f1(estimator: Any, X: Any, y: Any) -> float:
    """Calculate F1 score."""

    return float(
        f1_score(
            y,
            estimator.predict(X),
            average=_classification_average(y),
            zero_division=0,
        )
    )


def _roc_auc(estimator: Any, X: Any, y: Any) -> float:
    """Calculate ROC AUC from probabilities or decision scores."""

    scores = _prediction_scores(estimator, X)
    fitted_classes = _fitted_classes(estimator, y)
    evaluation_classes = np.unique(y)

    if len(evaluation_classes) < 2:
        raise ValueError(
            "roc_auc requires at least two classes in the evaluation target."
        )

    if len(fitted_classes) > 2:
        if len(evaluation_classes) != len(fitted_classes):
            raise ValueError(
                "roc_auc for a multiclass estimator requires the evaluation "
                "target to contain all fitted classes."
            )

        try:
            return float(
                roc_auc_score(
                    y,
                    scores,
                    multi_class="ovr",
                    labels=fitted_classes,
                )
            )
        except ValueError as exc:
            raise ValueError(
                "roc_auc could not be computed for this multiclass evaluation "
                "batch."
            ) from exc

    if _is_two_column_matrix(scores):
        scores = scores[:, 1]

    try:
        return float(roc_auc_score(y, scores))
    except ValueError as exc:
        raise ValueError(
            "roc_auc could not be computed for this binary evaluation batch."
        ) from exc


def _mse(estimator: Any, X: Any, y: Any) -> float:
    """Calculate mean squared error."""

    return float(mean_squared_error(y, estimator.predict(X)))


def _rmse(estimator: Any, X: Any, y: Any) -> float:
    """Calculate root mean squared error."""

    return float(math.sqrt(_mse(estimator, X, y)))


def _r2(estimator: Any, X: Any, y: Any) -> float:
    """Calculate R-squared."""

    return float(r2_score(y, estimator.predict(X)))


def _prediction_scores(estimator: Any, X: Any) -> Any:
    """Return scores suitable for ROC AUC."""

    if hasattr(estimator, "predict_proba"):
        return estimator.predict_proba(X)

    if hasattr(estimator, "decision_function"):
        return estimator.decision_function(X)

    raise ValueError(
        "roc_auc requires an estimator with predict_proba or "
        "decision_function."
    )


def _fitted_classes(estimator: Any, y: Any) -> np.ndarray:
    """Return fitted estimator classes when available."""

    classes = getattr(estimator, "classes_", None)

    if classes is not None:
        return np.asarray(classes)

    return np.unique(y)


def _classification_average(y: Any) -> str:
    """Choose a stable sklearn averaging strategy for classification metrics."""

    if _is_binary_target(y):
        return "binary"

    return "weighted"


def _is_binary_target(y: Any) -> bool:
    """Return whether a target has exactly two classes."""

    return len(np.unique(y)) == 2


def _is_two_column_matrix(values: Any) -> bool:
    """Return whether values look like two-column prediction scores."""

    shape = getattr(values, "shape", None)
    return shape is not None and len(shape) == 2 and shape[1] == 2


METRIC_FUNCTIONS: dict[str, MetricFunction] = {
    "accuracy": _accuracy,
    "precision": _precision,
    "recall": _recall,
    "f1": _f1,
    "roc_auc": _roc_auc,
    "mse": _mse,
    "rmse": _rmse,
    "r2": _r2,
}
