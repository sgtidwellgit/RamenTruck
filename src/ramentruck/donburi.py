"""Prediction and serving wrapper for RamenTruck.

Named after donburi, the bowl a finished dish is served in - here, the
wrapper that serves a trained model's predictions to a caller.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


class Donburi:
    """Serve predictions from a fitted model through a clean, validated API."""

    def __init__(self, model: Any, *, feature_names: list[str] | None = None) -> None:
        """
        Wrap a fitted model for prediction and serving.

        Parameters
        ----------
        model
            A fitted estimator implementing ``predict`` (and optionally
            ``predict_proba``).
        feature_names
            Expected feature names, in the order the model expects them.
            When provided, :meth:`predict_one`, :meth:`predict_batch`, and
            :meth:`serve` validate that incoming records contain every
            named feature. When omitted, incoming record keys are used
            as-is without validation.
        """

        if not hasattr(model, "predict") or not callable(model.predict):
            raise TypeError("model must provide a callable predict method.")

        self.model = model
        self.feature_names = list(feature_names) if feature_names is not None else None

    @classmethod
    def from_chashu(cls, path: str | Path) -> "Donburi":
        """
        Load a chashu bundle and wrap its model for serving.

        Parameters
        ----------
        path
            Path to a chashu bundle directory saved by
            :func:`ramentruck.chashu.save`.

        Returns
        -------
        Donburi
            A wrapper around the bundle's model. If the bundle's metadata
            contains a ``feature_names`` entry, it is used to validate
            incoming records.
        """

        from . import chashu

        bundle = chashu.load(path)
        feature_names = bundle.metadata.get("feature_names")
        return cls(bundle.model, feature_names=feature_names)

    def predict(self, X: Any) -> np.ndarray:
        """
        Predict target values for a batch of feature data.

        Parameters
        ----------
        X
            Feature data in whatever shape the wrapped model expects.

        Returns
        -------
        np.ndarray
            Model predictions.
        """

        return np.asarray(self.model.predict(X))

    def predict_proba(self, X: Any) -> np.ndarray:
        """
        Predict class probabilities for a batch of feature data.

        Parameters
        ----------
        X
            Feature data in whatever shape the wrapped model expects.

        Returns
        -------
        np.ndarray
            Predicted class probabilities.

        Raises
        ------
        AttributeError
            If the wrapped model does not implement ``predict_proba``.
        """

        if not hasattr(self.model, "predict_proba"):
            raise AttributeError(
                f"{type(self.model).__name__} does not support predict_proba."
            )

        return np.asarray(self.model.predict_proba(X))

    def predict_one(self, record: dict[str, Any]) -> Any:
        """
        Predict a target value for a single record.

        Parameters
        ----------
        record
            A mapping of feature name to value.

        Returns
        -------
        Any
            The single predicted value.
        """

        row = self._record_to_frame(record)
        return self.predict(row)[0]

    def predict_batch(self, records: list[dict[str, Any]]) -> list[Any]:
        """
        Predict target values for a batch of records.

        Parameters
        ----------
        records
            A list of mappings of feature name to value.

        Returns
        -------
        list
            One predicted value per input record, in order. Empty when
            ``records`` is empty.
        """

        if not records:
            return []

        rows = self._records_to_frame(records)
        return list(self.predict(rows))

    def serve(self, record: dict[str, Any]) -> dict[str, Any]:
        """
        Produce a JSON-friendly prediction response for a single record.

        Parameters
        ----------
        record
            A mapping of feature name to value.

        Returns
        -------
        dict
            A dict with a ``"prediction"`` key, and a ``"probabilities"``
            key (mapping class label to probability) when the wrapped
            model supports ``predict_proba``.
        """

        row = self._record_to_frame(record)
        response: dict[str, Any] = {"prediction": _to_jsonable(self.predict(row)[0])}

        if hasattr(self.model, "predict_proba"):
            probabilities = self.predict_proba(row)[0]
            classes = getattr(self.model, "classes_", range(len(probabilities)))
            response["probabilities"] = {
                str(_to_jsonable(cls)): float(prob) for cls, prob in zip(classes, probabilities)
            }

        return response

    def _record_to_frame(self, record: dict[str, Any]) -> pd.DataFrame:
        """Validate and convert a single record into a one-row DataFrame."""

        self._validate_record(record)
        columns = self.feature_names if self.feature_names is not None else list(record.keys())
        return pd.DataFrame([record], columns=columns)

    def _records_to_frame(self, records: list[dict[str, Any]]) -> pd.DataFrame:
        """Validate and convert a batch of records into a DataFrame."""

        for record in records:
            self._validate_record(record)

        columns = self.feature_names if self.feature_names is not None else list(records[0].keys())
        return pd.DataFrame(records, columns=columns)

    def _validate_record(self, record: dict[str, Any]) -> None:
        """Validate that a record contains every expected feature, when known."""

        if self.feature_names is None:
            return

        missing = set(self.feature_names) - set(record)

        if missing:
            raise ValueError(f"record is missing required feature(s): {sorted(missing)}.")


def _to_jsonable(value: Any) -> Any:
    """Convert a numpy scalar to a plain Python type for JSON serialization."""

    if isinstance(value, np.generic):
        return value.item()

    return value
