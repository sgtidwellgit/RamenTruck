"""Unit tests for ramentruck.diagnostics."""

from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from ramentruck.diagnostics import (
    DiagnosticCategory,
    DiagnosticEngine,
    DiagnosticReport,
    DiagnosticSeverity,
    Recommendation,
)


def test_recommendation_validates_confidence_bounds() -> None:
    """Confidence must stay within the closed unit interval."""

    with pytest.raises(ValueError, match="confidence must be between 0.0 and 1.0"):
        Recommendation(
            message="Bad confidence.",
            category=DiagnosticCategory.UNKNOWN,
            severity=DiagnosticSeverity.ERROR,
            confidence=1.1,
        )


def test_diagnostic_report_freezes_dataclass_fields() -> None:
    """The report dataclass should reject direct attribute mutation."""

    report = DiagnosticReport(
        diagnosis="No major issues detected.",
        recommendations=(),
        warnings=(),
        metadata={},
    )

    with pytest.raises(FrozenInstanceError):
        report.diagnosis = "Changed"


def test_overfitting_rule_returns_expected_diagnosis() -> None:
    """A large train/validation gap should trigger overfitting guidance."""

    report = DiagnosticEngine().evaluate(
        train_score=0.96,
        validation_score=0.80,
    )

    assert report.diagnosis == "Mild overfitting detected."
    assert report.warnings == ()
    assert [item.message for item in report.recommendations] == [
        "Reduce model complexity.",
        "Increase regularization.",
        "Collect more training data.",
    ]
    assert all(
        item.category is DiagnosticCategory.OVERFITTING
        for item in report.recommendations
    )
    assert report.metadata["score_gap"] == pytest.approx(0.16)


def test_underfitting_rule_returns_expected_diagnosis() -> None:
    """Low train and validation scores should trigger underfitting guidance."""

    report = DiagnosticEngine().evaluate(
        train_score=0.61,
        validation_score=0.59,
    )

    assert report.diagnosis == "Model may be underfitting."
    assert [item.message for item in report.recommendations] == [
        "Increase model capacity.",
        "Engineer additional features.",
        "Train longer.",
    ]
    assert report.metadata["triggered_categories"] == ("underfitting",)


def test_small_dataset_adds_warning_and_recommendation() -> None:
    """Small datasets should produce a warning and cross-validation advice."""

    report = DiagnosticEngine().evaluate(
        train_score=0.81,
        validation_score=0.79,
        dataset_size=48,
    )

    assert report.diagnosis == "No major issues detected."
    assert report.warnings == ("Small dataset detected.",)
    assert report.recommendations == (
        Recommendation(
            message="Use cross-validation.",
            category=DiagnosticCategory.SMALL_DATASET,
            severity=DiagnosticSeverity.WARNING,
            confidence=0.95,
        ),
    )


def test_high_variance_adds_warning_and_recommendation() -> None:
    """High validation variance should produce stable guidance."""

    report = DiagnosticEngine().evaluate(
        train_score=0.83,
        validation_score=0.80,
        variance=0.08,
    )

    assert report.warnings == ("High validation variance.",)
    assert report.recommendations == (
        Recommendation(
            message="Increase folds or gather more data.",
            category=DiagnosticCategory.HIGH_VARIANCE,
            severity=DiagnosticSeverity.WARNING,
            confidence=0.87,
        ),
    )


def test_class_imbalance_adds_warning_and_two_recommendations() -> None:
    """Class imbalance should return both split and weighting guidance."""

    report = DiagnosticEngine().evaluate(
        train_score=0.88,
        validation_score=0.84,
        class_imbalance=True,
    )

    assert report.warnings == ("Class imbalance detected.",)
    assert [item.message for item in report.recommendations] == [
        "Use StratifiedKFold.",
        "Consider class_weight.",
    ]
    assert [item.confidence for item in report.recommendations] == [0.97, 0.91]


def test_multiple_conditions_accumulate_deterministically() -> None:
    """Simultaneous conditions should preserve a stable warning and rec order."""

    report = DiagnosticEngine().evaluate(
        train_score=0.97,
        validation_score=0.80,
        class_imbalance=True,
        dataset_size=60,
        variance=0.09,
    )

    assert report.diagnosis == "Mild overfitting detected."
    assert report.warnings == (
        "Small dataset detected.",
        "High validation variance.",
        "Class imbalance detected.",
    )
    assert [item.category for item in report.recommendations] == [
        DiagnosticCategory.OVERFITTING,
        DiagnosticCategory.OVERFITTING,
        DiagnosticCategory.OVERFITTING,
        DiagnosticCategory.SMALL_DATASET,
        DiagnosticCategory.HIGH_VARIANCE,
        DiagnosticCategory.CLASS_IMBALANCE,
        DiagnosticCategory.CLASS_IMBALANCE,
    ]
    assert report.metadata["triggered_categories"] == (
        "overfitting",
        "small_dataset",
        "high_variance",
        "class_imbalance",
    )


def test_no_major_issues_returns_unknown_category_metadata() -> None:
    """A clean run should still expose deterministic metadata."""

    report = DiagnosticEngine().evaluate(
        train_score=0.84,
        validation_score=0.80,
    )

    assert report.diagnosis == "No major issues detected."
    assert report.recommendations == ()
    assert report.warnings == ()
    assert report.metadata["triggered_categories"] == ("unknown",)


def test_metadata_is_read_only() -> None:
    """The metadata mapping should be defensively frozen."""

    report = DiagnosticEngine().evaluate(
        train_score=0.84,
        validation_score=0.80,
        dataset_size=50,
    )

    with pytest.raises(TypeError):
        report.metadata["dataset_size"] = 100


def test_chef_report_renders_clean_multiline_summary() -> None:
    """chef_report should include each populated section once."""

    report = DiagnosticEngine().evaluate(
        train_score=0.97,
        validation_score=0.80,
        class_imbalance=True,
        dataset_size=60,
        variance=0.09,
    )

    expected = "\n".join(
        [
            "Diagnosis:",
            "Mild overfitting detected.",
            "",
            "Warnings:",
            "- Small dataset detected.",
            "- High validation variance.",
            "- Class imbalance detected.",
            "",
            "Recommendations:",
            "- [WARNING] Reduce model complexity. (OVERFITTING, confidence=0.92)",
            "- [WARNING] Increase regularization. (OVERFITTING, confidence=0.90)",
            "- [INFO] Collect more training data. (OVERFITTING, confidence=0.88)",
            "- [WARNING] Use cross-validation. (SMALL_DATASET, confidence=0.95)",
            "- [WARNING] Increase folds or gather more data. (HIGH_VARIANCE, confidence=0.87)",
            "- [WARNING] Use StratifiedKFold. (CLASS_IMBALANCE, confidence=0.97)",
            "- [INFO] Consider class_weight. (CLASS_IMBALANCE, confidence=0.91)",
            "",
            "Metadata:",
            "- class_imbalance: True",
            "- dataset_size: 60",
            "- score_gap: 0.170",
            "- train_score: 0.970",
            "- triggered_categories: ('overfitting', 'small_dataset', 'high_variance', 'class_imbalance')",
            "- validation_score: 0.800",
            "- variance: 0.090",
        ]
    )

    assert report.chef_report() == expected
