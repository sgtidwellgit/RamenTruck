"""
Unit tests for ramentruck.noodles
"""

import pandas as pd
import pytest

from ramentruck import slurp


# ----------------------------------------------------------------------
# slurp()
# ----------------------------------------------------------------------


def test_slurp_returns_dataset_menu():
    """Verify a DatasetMenu is returned with basic metadata."""

    df = pd.DataFrame(
        {
            "age": [25, 30, 35],
            "salary": [50000, 60000, 70000],
            "target": [0, 1, 0],
        }
    )

    menu = slurp(df, target="target")

    assert menu.rows == 3
    assert menu.columns == 3
    assert menu.duplicate_rows == 0
    assert menu.chef_recommendation.problem_type == "classification"


def test_slurp_detects_missing_values():
    """Verify missing values are detected."""

    df = pd.DataFrame(
        {
            "age": [20, None, 30],
            "salary": [100, 200, None],
        }
    )

    menu = slurp(df)

    assert len(menu.missing_values) == 2

    assert "age" in menu.missing_values["column"].values
    assert "salary" in menu.missing_values["column"].values


def test_slurp_detects_duplicate_rows():
    """Verify duplicate rows are counted."""

    df = pd.DataFrame(
        {
            "a": [1, 1],
            "b": [2, 2],
        }
    )

    menu = slurp(df)

    assert menu.duplicate_rows == 1


def test_slurp_detects_regression_problem():
    """Verify regression datasets are recognized."""

    df = pd.DataFrame(
        {
            "x": range(100),
            "target": range(100),
        }
    )

    menu = slurp(df, target="target")

    assert menu.chef_recommendation.problem_type == "regression"
    assert menu.chef_recommendation.loss == "mean_squared_error"
    assert menu.chef_recommendation.output_activation == "linear"


def test_slurp_detects_binary_classification():
    """Verify binary classification is detected."""

    df = pd.DataFrame(
        {
            "x": range(10),
            "target": [0, 1] * 5,
        }
    )

    menu = slurp(df, target="target")

    assert menu.chef_recommendation.problem_type == "classification"
    assert menu.chef_recommendation.loss == "binary_crossentropy"
    assert menu.chef_recommendation.output_activation == "sigmoid"


def test_slurp_detects_multiclass_classification():
    """Verify multiclass classification is detected."""

    df = pd.DataFrame(
        {
            "x": range(9),
            "target": [0, 1, 2] * 3,
        }
    )

    menu = slurp(df, target="target")

    assert menu.chef_recommendation.problem_type == "classification"
    assert menu.chef_recommendation.loss == "categorical_crossentropy"
    assert menu.chef_recommendation.output_activation == "softmax"


def test_slurp_detects_class_imbalance():
    """Verify highly imbalanced datasets are detected."""

    df = pd.DataFrame(
        {
            "x": range(100),
            "target": [0] * 95 + [1] * 5,
        }
    )

    menu = slurp(df, target="target")

    assert menu.chef_recommendation.class_imbalance is True


def test_slurp_detects_no_class_imbalance():
    """Verify balanced datasets are not flagged."""

    df = pd.DataFrame(
        {
            "x": range(100),
            "target": [0] * 50 + [1] * 50,
        }
    )

    menu = slurp(df, target="target")

    assert menu.chef_recommendation.class_imbalance is False


def test_slurp_detects_column_types():
    """Verify each column type is categorized correctly."""

    df = pd.DataFrame(
        {
            "number": [1, 2, 3],
            "text": ["A", "B", "C"],
            "flag": [True, False, True],
            "date": pd.to_datetime(
                [
                    "2026-01-01",
                    "2026-01-02",
                    "2026-01-03",
                ]
            ),
        }
    )

    menu = slurp(df)

    assert "number" in menu.numeric_columns
    assert "text" in menu.categorical_columns
    assert "flag" in menu.boolean_columns
    assert "date" in menu.datetime_columns


def test_slurp_invalid_dataframe():
    """Verify invalid dataframe types raise TypeError."""

    with pytest.raises(TypeError):
        slurp("Hello World")


def test_slurp_invalid_target():
    """Verify invalid target column raises ValueError."""

    df = pd.DataFrame({"a": [1, 2, 3]})

    with pytest.raises(ValueError):
        slurp(df, target="missing")


def test_slurp_empty_dataframe():
    """Verify empty dataframes are handled correctly."""

    df = pd.DataFrame()

    menu = slurp(df)

    assert menu.rows == 0
    assert menu.columns == 0


def test_slurp_memory_usage():
    """Verify memory usage is reported."""

    df = pd.DataFrame(
        {
            "a": range(100),
        }
    )

    menu = slurp(df)

    assert menu.memory_mb >= 0


def test_slurp_string_representation():
    """Verify DatasetMenu can be printed."""

    df = pd.DataFrame(
        {
            "a": [1, 2, 3],
        }
    )

    menu = slurp(df)

    report = str(menu)

    assert "RamenTruck Dataset Menu" in report
    assert "Rows" in report
    assert "Columns" in report     