"""Shared result containers for RamenTruck."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class BrothResult:
    """Result returned by :class:`ramentruck.Broth` after fitting."""

    model: Any
    train_score: float
    val_score: float | None
    metrics: dict[str, float]
    fit_time_s: float
