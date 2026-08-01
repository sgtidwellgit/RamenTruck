"""Shared result containers for RamenTruck."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import pandas as pd


@dataclass(frozen=True)
class BrothResult:
    """Result returned by :class:`ramentruck.Broth` after fitting."""

    model: Any
    train_score: float
    val_score: float | None
    metrics: dict[str, float]
    fit_time_s: float


@dataclass(frozen=True)
class TareResult:
    """Result returned by :func:`ramentruck.tare` after a hyperparameter search."""

    best_params: dict[str, Any]
    best_score: float
    best_model: Any
    cv_results: pd.DataFrame
    search_time_s: float


@dataclass(frozen=True)
class EggResult:
    """Result returned by :func:`ramentruck.soft_boiled_egg` after cross-validation."""

    scores: Any
    mean_score: float
    std_score: float
    fold_results: pd.DataFrame
    learning_curve_df: pd.DataFrame | None


@dataclass(frozen=True)
class ChashuBundle:
    """Bundle returned by :func:`ramentruck.chashu.load`."""

    model: Any
    metadata: dict[str, Any]
    saved_at: str
    ramentruck_version: str
    python_version: str
    sklearn_version: str


@dataclass(frozen=True)
class NoriResult:
    """Result returned by :func:`ramentruck.nori.explain`."""

    shap_values: Any
    base_values: Any
    feature_names: list[str]
    data: Any
    importance: pd.DataFrame
    class_index: int | None


@dataclass(frozen=True)
class PDResult:
    """Result returned by :func:`ramentruck.nori.partial_dependence`."""

    curves: dict[str, pd.DataFrame]


@dataclass(frozen=True)
class MisoRunSummary:
    """Result returned by :func:`ramentruck.miso.best_run`."""

    run_id: str
    run_name: str | None
    params: dict[str, Any]
    metrics: dict[str, float]
    artifact_uri: str


@dataclass(frozen=True)
class ToppingsResult:
    """Result returned by :func:`ramentruck.toppings.voting`, `.stack`, and `.bag`."""

    model: Any
    train_score: float
    val_score: float | None
    individual_scores: dict[str, float]
    fit_time_s: float


@dataclass(frozen=True)
class KaeshiResult:
    """Result returned by :func:`ramentruck.kaeshi.kaeshi`."""

    model: Any
    method: str
    brier_score_before: float
    brier_score_after: float
    calibration_curve_df: pd.DataFrame
