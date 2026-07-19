"""Shared diagnostic primitives and deterministic evaluation rules."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from types import MappingProxyType
from typing import Any


OVERFITTING_GAP_THRESHOLD = 0.10
LOW_SCORE_THRESHOLD = 0.70
SMALL_DATASET_THRESHOLD = 100
HIGH_VARIANCE_THRESHOLD = 0.05


class DiagnosticSeverity(str, Enum):
    """Severity level for a recommendation."""

    INFO = "info"
    WARNING = "warning"
    ERROR = "error"


class DiagnosticCategory(str, Enum):
    """Known diagnostic categories shared across RamenTruck modules."""

    OVERFITTING = "overfitting"
    UNDERFITTING = "underfitting"
    CLASS_IMBALANCE = "class_imbalance"
    SMALL_DATASET = "small_dataset"
    HIGH_VARIANCE = "high_variance"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class Recommendation:
    """Structured recommendation produced by a diagnostic rule."""

    message: str
    category: DiagnosticCategory
    severity: DiagnosticSeverity
    confidence: float

    def __post_init__(self) -> None:
        """Validate recommendation fields."""

        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("confidence must be between 0.0 and 1.0.")


@dataclass(frozen=True)
class DiagnosticReport:
    """Immutable report returned by :class:`DiagnosticEngine`."""

    diagnosis: str
    recommendations: tuple[Recommendation, ...]
    warnings: tuple[str, ...]
    metadata: dict[str, Any]

    def __post_init__(self) -> None:
        """Freeze metadata so the full report is immutable in practice."""

        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))

    def chef_report(self) -> str:
        """Render a deterministic human-readable report."""

        lines = [
            "Diagnosis:",
            self.diagnosis,
        ]

        if self.warnings:
            lines.extend(["", "Warnings:"])
            lines.extend(f"- {warning}" for warning in self.warnings)

        if self.recommendations:
            lines.extend(["", "Recommendations:"])
            lines.extend(
                (
                    f"- [{recommendation.severity.name}] "
                    f"{recommendation.message} "
                    f"({recommendation.category.name}, "
                    f"confidence={recommendation.confidence:.2f})"
                )
                for recommendation in self.recommendations
            )

        if self.metadata:
            lines.extend(["", "Metadata:"])
            for key in sorted(self.metadata):
                lines.append(
                    f"- {key}: {self._format_metadata_value(self.metadata[key])}"
                )

        return "\n".join(lines)

    @staticmethod
    def _format_metadata_value(value: Any) -> str:
        """Return a stable, human-readable metadata representation."""

        if isinstance(value, float):
            return f"{value:.3f}"

        return repr(value)


class DiagnosticEngine:
    """Deterministic rule engine for shared RamenTruck diagnostics."""

    def evaluate(
        self,
        *,
        train_score: float,
        validation_score: float | None,
        class_imbalance: bool = False,
        dataset_size: int | None = None,
        variance: float | None = None,
    ) -> DiagnosticReport:
        """Evaluate deterministic diagnostic rules from training signals."""

        warnings: list[str] = []
        recommendations: list[Recommendation] = []
        seen_recommendations: set[tuple[str, DiagnosticCategory]] = set()
        triggered_categories: list[DiagnosticCategory] = []

        score_gap = None
        if validation_score is not None:
            score_gap = train_score - validation_score

        diagnosis = "No major issues detected."

        if score_gap is not None and score_gap > OVERFITTING_GAP_THRESHOLD:
            diagnosis = "Mild overfitting detected."
            triggered_categories.append(DiagnosticCategory.OVERFITTING)
            self._add_recommendation(
                recommendations,
                seen_recommendations,
                "Reduce model complexity.",
                DiagnosticCategory.OVERFITTING,
                DiagnosticSeverity.WARNING,
                0.92,
            )
            self._add_recommendation(
                recommendations,
                seen_recommendations,
                "Increase regularization.",
                DiagnosticCategory.OVERFITTING,
                DiagnosticSeverity.WARNING,
                0.90,
            )
            self._add_recommendation(
                recommendations,
                seen_recommendations,
                "Collect more training data.",
                DiagnosticCategory.OVERFITTING,
                DiagnosticSeverity.INFO,
                0.88,
            )
        elif (
            validation_score is not None
            and train_score < LOW_SCORE_THRESHOLD
            and validation_score < LOW_SCORE_THRESHOLD
        ):
            diagnosis = "Model may be underfitting."
            triggered_categories.append(DiagnosticCategory.UNDERFITTING)
            self._add_recommendation(
                recommendations,
                seen_recommendations,
                "Increase model capacity.",
                DiagnosticCategory.UNDERFITTING,
                DiagnosticSeverity.WARNING,
                0.89,
            )
            self._add_recommendation(
                recommendations,
                seen_recommendations,
                "Engineer additional features.",
                DiagnosticCategory.UNDERFITTING,
                DiagnosticSeverity.INFO,
                0.84,
            )
            self._add_recommendation(
                recommendations,
                seen_recommendations,
                "Train longer.",
                DiagnosticCategory.UNDERFITTING,
                DiagnosticSeverity.INFO,
                0.78,
            )

        if dataset_size is not None and dataset_size < SMALL_DATASET_THRESHOLD:
            warnings.append("Small dataset detected.")
            triggered_categories.append(DiagnosticCategory.SMALL_DATASET)
            self._add_recommendation(
                recommendations,
                seen_recommendations,
                "Use cross-validation.",
                DiagnosticCategory.SMALL_DATASET,
                DiagnosticSeverity.WARNING,
                0.95,
            )

        if variance is not None and variance > HIGH_VARIANCE_THRESHOLD:
            warnings.append("High validation variance.")
            triggered_categories.append(DiagnosticCategory.HIGH_VARIANCE)
            self._add_recommendation(
                recommendations,
                seen_recommendations,
                "Increase folds or gather more data.",
                DiagnosticCategory.HIGH_VARIANCE,
                DiagnosticSeverity.WARNING,
                0.87,
            )

        if class_imbalance:
            warnings.append("Class imbalance detected.")
            triggered_categories.append(DiagnosticCategory.CLASS_IMBALANCE)
            self._add_recommendation(
                recommendations,
                seen_recommendations,
                "Use StratifiedKFold.",
                DiagnosticCategory.CLASS_IMBALANCE,
                DiagnosticSeverity.WARNING,
                0.97,
            )
            self._add_recommendation(
                recommendations,
                seen_recommendations,
                "Consider class_weight.",
                DiagnosticCategory.CLASS_IMBALANCE,
                DiagnosticSeverity.INFO,
                0.91,
            )

        if not triggered_categories:
            triggered_categories.append(DiagnosticCategory.UNKNOWN)

        metadata = {
            "class_imbalance": class_imbalance,
            "dataset_size": dataset_size,
            "score_gap": score_gap,
            "train_score": train_score,
            "triggered_categories": tuple(category.value for category in triggered_categories),
            "validation_score": validation_score,
            "variance": variance,
        }

        return DiagnosticReport(
            diagnosis=diagnosis,
            recommendations=tuple(recommendations),
            warnings=tuple(warnings),
            metadata=metadata,
        )

    @staticmethod
    def _add_recommendation(
        recommendations: list[Recommendation],
        seen_recommendations: set[tuple[str, DiagnosticCategory]],
        message: str,
        category: DiagnosticCategory,
        severity: DiagnosticSeverity,
        confidence: float,
    ) -> None:
        """Append a recommendation once while preserving insertion order."""

        key = (message, category)
        if key in seen_recommendations:
            return

        recommendations.append(
            Recommendation(
                message=message,
                category=category,
                severity=severity,
                confidence=confidence,
            )
        )
        seen_recommendations.add(key)


__all__ = [
    "DiagnosticCategory",
    "DiagnosticEngine",
    "DiagnosticReport",
    "DiagnosticSeverity",
    "Recommendation",
]
