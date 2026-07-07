"""
RamenTruck - noodles.py

Dataset inspection and preprocessing utilities.

The primary entry point is ``slurp()``, which inspects a dataset and
returns a DatasetMenu containing statistics and recommendations for
machine learning workflows.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd


# ----------------------------------------------------------------------
# Dataclasses
# ----------------------------------------------------------------------


@dataclass(slots=True, frozen=True)
class ChefRecommendation:
    """Suggested preprocessing configuration."""

    problem_type: str
    scaling: str
    encoding: str
    loss: str
    output_activation: str
    optimizer: str
    class_imbalance: bool


@dataclass(slots=True, frozen=True)
class DatasetMenu:
    """Summary of a dataset returned from slurp()."""

    rows: int
    columns: int

    memory_mb: float

    numeric_columns: list[str]
    categorical_columns: list[str]
    boolean_columns: list[str]
    datetime_columns: list[str]

    missing_values: pd.DataFrame

    duplicate_rows: int

    chef_recommendation: ChefRecommendation

    def __str__(self) -> str:

        lines = []

        lines.append("")
        lines.append("🍜 RamenTruck Dataset Menu")
        lines.append("=" * 45)

        lines.append(f"Rows:                {self.rows:,}")
        lines.append(f"Columns:             {self.columns}")
        lines.append(f"Memory Usage:        {self.memory_mb:.2f} MB")
        lines.append("")

        lines.append(f"Numeric:             {len(self.numeric_columns)}")
        lines.append(f"Categorical:         {len(self.categorical_columns)}")
        lines.append(f"Boolean:             {len(self.boolean_columns)}")
        lines.append(f"Datetime:            {len(self.datetime_columns)}")
        lines.append("")
        lines.append(f"Duplicate Rows:      {self.duplicate_rows}")

        if not self.missing_values.empty:

            lines.append("")
            lines.append("Missing Values")
            lines.append("-" * 45)

            for _, row in self.missing_values.iterrows():

                lines.append(
                    f"{row['column']:<25}"
                    f"{row['missing']:>6} "
                    f"({row['percent']:.2f}%)"
                )

        r = self.chef_recommendation

        lines.append("")
        lines.append("Chef's Recommendation")
        lines.append("-" * 45)
        lines.append(f"Problem Type:        {r.problem_type}")
        lines.append(f"Scaling:             {r.scaling}")
        lines.append(f"Encoding:            {r.encoding}")
        lines.append(f"Loss:                {r.loss}")
        lines.append(f"Output Activation:   {r.output_activation}")
        lines.append(f"Optimizer:           {r.optimizer}")
        lines.append(f"Class Imbalance:     {r.class_imbalance}")

        return "\n".join(lines)


# ----------------------------------------------------------------------
# Public API
# ----------------------------------------------------------------------


def slurp(
    df: pd.DataFrame,
    target: Optional[str] = None,
) -> DatasetMenu:
    """
    Inspect a DataFrame and return a DatasetMenu.

    Parameters
    ----------
    df
        Input pandas DataFrame.

    target
        Optional target column name.

    Returns
    -------
    DatasetMenu
    """

    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame.")

    if target is not None and target not in df.columns:
        raise ValueError(f"Target column '{target}' was not found.")

    rows, columns = df.shape

    memory_mb = df.memory_usage(deep=True).sum() / (1024 ** 2)

    numeric_columns = list(
        df.select_dtypes(include=np.number).columns
    )

    categorical_columns = list(
        df.select_dtypes(include=["object", "category"]).columns
    )

    boolean_columns = list(
        df.select_dtypes(include="bool").columns
    )

    datetime_columns = list(
        df.select_dtypes(include=["datetime64[ns]", "datetimetz"]).columns
    )

    duplicate_rows = int(df.duplicated().sum())

    # ----------------------------------------------------------
    # Missing values
    # ----------------------------------------------------------

    missing = df.isna().sum()
    missing = missing[missing > 0]

    missing_df = pd.DataFrame(
        {
            "column": missing.index,
            "missing": missing.values,
            "percent": (missing.values / rows) * 100,
        }
    )

    # ----------------------------------------------------------
    # Recommendation Engine
    # ----------------------------------------------------------

    problem_type = "unknown"
    loss = "unknown"
    output = "unknown"
    imbalance = False

    if target is not None:

        unique = df[target].nunique(dropna=True)

        if (
            pd.api.types.is_numeric_dtype(df[target])
            and unique > 20
        ):

            problem_type = "regression"

            loss = "mean_squared_error"
            output = "linear"

        else:

            problem_type = "classification"

            counts = df[target].value_counts(normalize=True)

            imbalance = bool(counts.min() < 0.10)

            if unique == 2:

                loss = "binary_crossentropy"
                output = "sigmoid"

            else:

                loss = "categorical_crossentropy"
                output = "softmax"

    recommendation = ChefRecommendation(
        problem_type=problem_type,
        scaling="StandardScaler",
        encoding="OneHotEncoder",
        loss=loss,
        output_activation=output,
        optimizer="Adam",
        class_imbalance=imbalance,
    )

    return DatasetMenu(
        rows=rows,
        columns=columns,
        memory_mb=memory_mb,
        numeric_columns=numeric_columns,
        categorical_columns=categorical_columns,
        boolean_columns=boolean_columns,
        datetime_columns=datetime_columns,
        missing_values=missing_df,
        duplicate_rows=duplicate_rows,
        chef_recommendation=recommendation,
    )
