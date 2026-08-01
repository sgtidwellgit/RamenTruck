"""Unit tests for ramentruck.kaedama."""

import numpy as np
import pandas as pd
import pytest

from ramentruck import Kaedama


def _sample_df():
    """Build a small DataFrame covering every feature type kaedama handles."""

    return pd.DataFrame(
        {
            "signup_date": pd.to_datetime(
                ["2026-01-01", "2026-03-15", "2026-07-04"]
            ),
            "spend": [10.0, 0.0, 40.0],
            "visits": [2, 0, 4],
            "age": [22, 45, 67],
        }
    )


def test_datetime_features_extracts_calendar_components():
    """Verify datetime_features adds one column per requested component."""

    df = _sample_df()
    result = Kaedama().datetime_features("signup_date", features=("year", "month")).fit_transform(df)

    assert list(result["signup_date_year"]) == [2026, 2026, 2026]
    assert list(result["signup_date_month"]) == [1, 3, 7]


def test_polynomial_features_excludes_original_columns():
    """Verify polynomial_features only adds new higher-order terms."""

    df = _sample_df()
    kaedama = Kaedama().polynomial_features(["spend", "visits"], degree=2)
    result = kaedama.fit_transform(df)

    assert "spend" in result.columns
    assert "visits" in result.columns
    assert "spend^2" not in kaedama.added_columns_
    assert any("spend" in col and "2" in col for col in kaedama.added_columns_)
    assert any("visits" in col for col in kaedama.added_columns_)


def test_ratio_handles_division_by_zero():
    """Verify ratio() divides safely and produces NaN for zero denominators."""

    df = _sample_df()
    result = Kaedama().ratio("spend", "visits", name="spend_per_visit").fit_transform(df)

    assert result["spend_per_visit"].iloc[0] == pytest.approx(5.0)
    assert np.isnan(result["spend_per_visit"].iloc[1])


def test_ratio_default_name():
    """Verify ratio() defaults to a descriptive column name."""

    df = _sample_df()
    result = Kaedama().ratio("spend", "visits").fit_transform(df)

    assert "spend_per_visits" in result.columns


def test_log_transform_applies_offset():
    """Verify log_transform adds offset before taking the natural log."""

    df = _sample_df()
    result = Kaedama().log_transform("spend", offset=1.0).fit_transform(df)

    expected = np.log(df["spend"] + 1.0)
    assert result["spend_log"].tolist() == pytest.approx(expected.tolist())


def test_bin_buckets_numeric_column():
    """Verify bin() produces a categorical column with the requested bin count."""

    df = _sample_df()
    result = Kaedama().bin("age", bins=3, name="age_group").fit_transform(df)

    assert "age_group" in result.columns
    assert result["age_group"].nunique() <= 3


def test_chained_steps_track_added_columns():
    """Verify chaining multiple steps records every added column in order."""

    df = _sample_df()
    kaedama = (
        Kaedama()
        .datetime_features("signup_date", features=("year",))
        .ratio("spend", "visits")
        .log_transform("spend")
    )
    result = kaedama.fit_transform(df)

    assert kaedama.added_columns_ == ["signup_date_year", "spend_per_visits", "spend_log"]
    assert set(kaedama.added_columns_).issubset(result.columns)


def test_fit_transform_does_not_mutate_input():
    """Verify fit_transform leaves the original DataFrame untouched."""

    df = _sample_df()
    original_columns = list(df.columns)

    Kaedama().log_transform("spend").fit_transform(df)

    assert list(df.columns) == original_columns
