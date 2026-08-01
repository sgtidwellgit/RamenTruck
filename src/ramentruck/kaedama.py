"""Feature engineering pipeline for RamenTruck.

Named after kaedama, the extra serving of noodles a ramen shop drops into
a bowl of broth that's already been depleted - here, extra engineered
features dropped into a DataFrame that's already been through
:func:`ramentruck.slurp`.
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures


class Kaedama:
    """Fluent builder for adding engineered feature columns to a DataFrame."""

    def __init__(self) -> None:
        """Initialize an empty pipeline."""

        self._steps: list[tuple[str, dict[str, Any]]] = []
        self.added_columns_: list[str] = []

    def datetime_features(
        self,
        columns: str | Sequence[str],
        *,
        features: Sequence[str] = ("year", "month", "day", "dayofweek", "hour"),
    ) -> "Kaedama":
        """
        Extract calendar components from one or more datetime columns.

        Parameters
        ----------
        columns
            Column name, or names, to extract from. Coerced with
            ``pd.to_datetime`` if not already datetime-typed.
        features
            Calendar components to extract, e.g. ``"year"``, ``"month"``,
            ``"day"``, ``"dayofweek"``, ``"hour"``. Any attribute of the
            pandas ``.dt`` accessor is supported.

        Returns
        -------
        Kaedama
            ``self``, to allow chaining.
        """

        self._steps.append(
            ("datetime_features", {"columns": _as_list(columns), "features": list(features)})
        )
        return self

    def polynomial_features(
        self,
        columns: Sequence[str],
        *,
        degree: int = 2,
        interaction_only: bool = False,
    ) -> "Kaedama":
        """
        Add polynomial and/or interaction terms for a set of columns.

        Parameters
        ----------
        columns
            Numeric columns to expand.
        degree
            Highest polynomial degree to generate.
        interaction_only
            Whether to generate only interaction terms (e.g. ``a*b``),
            excluding pure powers (e.g. ``a**2``).

        Returns
        -------
        Kaedama
            ``self``, to allow chaining.
        """

        self._steps.append(
            (
                "polynomial_features",
                {
                    "columns": list(columns),
                    "degree": degree,
                    "interaction_only": interaction_only,
                },
            )
        )
        return self

    def ratio(
        self,
        numerator: str,
        denominator: str,
        *,
        name: str | None = None,
    ) -> "Kaedama":
        """
        Add a column dividing one numeric column by another.

        Division by zero produces ``NaN`` rather than raising or returning
        infinity.

        Parameters
        ----------
        numerator
            Column to use as the numerator.
        denominator
            Column to use as the denominator.
        name
            Name for the new column. Defaults to
            ``f"{numerator}_per_{denominator}"``.

        Returns
        -------
        Kaedama
            ``self``, to allow chaining.
        """

        self._steps.append(
            ("ratio", {"numerator": numerator, "denominator": denominator, "name": name})
        )
        return self

    def log_transform(self, columns: str | Sequence[str], *, offset: float = 1.0) -> "Kaedama":
        """
        Add natural-log transformed columns.

        Parameters
        ----------
        columns
            Column name, or names, to transform.
        offset
            Constant added before taking the log, to keep zero and small
            values finite.

        Returns
        -------
        Kaedama
            ``self``, to allow chaining.
        """

        self._steps.append(("log_transform", {"columns": _as_list(columns), "offset": offset}))
        return self

    def bin(
        self,
        column: str,
        *,
        bins: int | Sequence[float],
        labels: Sequence[Any] | None = None,
        name: str | None = None,
    ) -> "Kaedama":
        """
        Add a categorical column bucketing a numeric column into ranges.

        Bin edges are computed from whichever DataFrame is passed to
        :meth:`fit_transform`. Pass explicit ``bins`` edges (rather than a
        bin count) for consistent boundaries across separate calls, such as
        train and test splits.

        Parameters
        ----------
        column
            Numeric column to bucket.
        bins
            Number of equal-width bins, or an explicit sequence of bin edges
            (passed to ``pandas.cut``).
        labels
            Optional labels for the resulting categories.
        name
            Name for the new column. Defaults to ``f"{column}_bin"``.

        Returns
        -------
        Kaedama
            ``self``, to allow chaining.
        """

        self._steps.append(
            ("bin", {"column": column, "bins": bins, "labels": labels, "name": name})
        )
        return self

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply every queued step to a DataFrame and return the result.

        Parameters
        ----------
        df
            Input DataFrame. Not mutated; a copy is transformed and
            returned.

        Returns
        -------
        pd.DataFrame
            A copy of ``df`` with every engineered column appended. The
            names of the added columns are also recorded on
            ``self.added_columns_``.
        """

        result = df.copy()
        added: list[str] = []

        for step_name, kwargs in self._steps:
            result, new_columns = _STEP_HANDLERS[step_name](result, **kwargs)
            added.extend(new_columns)

        self.added_columns_ = added
        return result


def _as_list(columns: str | Sequence[str]) -> list[str]:
    """Normalize a single column name or a sequence of names to a list."""

    if isinstance(columns, str):
        return [columns]

    return list(columns)


def _apply_datetime_features(
    df: pd.DataFrame, *, columns: list[str], features: list[str]
) -> tuple[pd.DataFrame, list[str]]:
    """Extract calendar components from datetime columns."""

    added = []

    for column in columns:
        series = pd.to_datetime(df[column])

        for feature in features:
            new_column = f"{column}_{feature}"
            df[new_column] = getattr(series.dt, feature)
            added.append(new_column)

    return df, added


def _apply_polynomial_features(
    df: pd.DataFrame, *, columns: list[str], degree: int, interaction_only: bool
) -> tuple[pd.DataFrame, list[str]]:
    """Add polynomial and/or interaction terms via scikit-learn."""

    transformer = PolynomialFeatures(
        degree=degree, interaction_only=interaction_only, include_bias=False
    )
    expanded = transformer.fit_transform(df[columns])
    feature_names = transformer.get_feature_names_out(columns)

    added = []

    for feature_name, values in zip(feature_names, expanded.T):
        if feature_name in columns:
            continue

        safe_name = feature_name.replace(" ", "_").replace("^", "_pow")
        df[safe_name] = values
        added.append(safe_name)

    return df, added


def _apply_ratio(
    df: pd.DataFrame, *, numerator: str, denominator: str, name: str | None
) -> tuple[pd.DataFrame, list[str]]:
    """Add a numerator/denominator ratio column."""

    new_column = name or f"{numerator}_per_{denominator}"
    safe_denominator = df[denominator].replace(0, np.nan)
    df[new_column] = df[numerator] / safe_denominator
    return df, [new_column]


def _apply_log_transform(
    df: pd.DataFrame, *, columns: list[str], offset: float
) -> tuple[pd.DataFrame, list[str]]:
    """Add natural-log transformed columns."""

    added = []

    for column in columns:
        new_column = f"{column}_log"
        df[new_column] = np.log(df[column] + offset)
        added.append(new_column)

    return df, added


def _apply_bin(
    df: pd.DataFrame,
    *,
    column: str,
    bins: int | Sequence[float],
    labels: Sequence[Any] | None,
    name: str | None,
) -> tuple[pd.DataFrame, list[str]]:
    """Add a binned/bucketed version of a numeric column."""

    new_column = name or f"{column}_bin"
    df[new_column] = pd.cut(df[column], bins=bins, labels=labels)
    return df, [new_column]


_STEP_HANDLERS = {
    "datetime_features": _apply_datetime_features,
    "polynomial_features": _apply_polynomial_features,
    "ratio": _apply_ratio,
    "log_transform": _apply_log_transform,
    "bin": _apply_bin,
}
