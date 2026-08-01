"""Ensemble methods for RamenTruck.

Named after toppings: no single one makes the bowl, but combined they're
better than any single ingredient alone.
"""

from __future__ import annotations

import time
from typing import Any, Sequence

from sklearn.ensemble import (
    BaggingClassifier,
    BaggingRegressor,
    StackingClassifier,
    StackingRegressor,
    VotingClassifier,
    VotingRegressor,
)

from .results import ToppingsResult

SUPPORTED_TASKS = frozenset({"classification", "regression"})


def voting(
    models: dict[str, Any] | Sequence[Any],
    X: Any,
    y: Any,
    *,
    X_val: Any | None = None,
    y_val: Any | None = None,
    task: str = "classification",
    voting_type: str = "soft",
    weights: Sequence[float] | None = None,
    n_jobs: int | None = None,
    verbose: bool = True,
) -> ToppingsResult:
    """
    Combine several fitted-or-not models into a voting ensemble.

    Parameters
    ----------
    models
        A dict mapping names to estimators, or a plain sequence of
        estimators (auto-named ``model_0``, ``model_1``, ...).
    X, y
        Training data. Each model is refit as part of the ensemble.
    X_val, y_val
        Optional validation data. When provided, ``val_score`` and
        ``individual_scores`` are computed on it instead of on training
        data.
    task
        ``"classification"`` or ``"regression"``.
    voting_type
        ``"soft"`` (average predicted probabilities) or ``"hard"``
        (majority vote of predicted labels). Classification only.
    weights
        Optional per-model weights, in the same order as ``models``.
    n_jobs
        Number of parallel jobs used to fit the ensemble members.
    verbose
        Whether to print the ensemble score and each member's individual
        score.

    Returns
    -------
    ToppingsResult
        The fitted ensemble, its train/validation scores, each member's
        individual score, and total fit time.
    """

    _validate_task(task)
    named_models = _named_models(models)

    if task == "classification":
        ensemble = VotingClassifier(
            estimators=named_models, voting=voting_type, weights=weights, n_jobs=n_jobs
        )
    else:
        ensemble = VotingRegressor(estimators=named_models, weights=weights, n_jobs=n_jobs)

    return _fit_and_score(ensemble, X, y, X_val=X_val, y_val=y_val, verbose=verbose, label="voting")


def stack(
    models: dict[str, Any] | Sequence[Any],
    meta_model: Any,
    X: Any,
    y: Any,
    *,
    X_val: Any | None = None,
    y_val: Any | None = None,
    task: str = "classification",
    cv: int = 5,
    n_jobs: int | None = None,
    verbose: bool = True,
) -> ToppingsResult:
    """
    Stack several models under a meta-model trained on their out-of-fold predictions.

    Parameters
    ----------
    models
        A dict mapping names to base estimators, or a plain sequence of
        estimators (auto-named ``model_0``, ``model_1``, ...).
    meta_model
        Estimator trained on the base models' out-of-fold predictions to
        produce the final output.
    X, y
        Training data.
    X_val, y_val
        Optional validation data. When provided, ``val_score`` and
        ``individual_scores`` are computed on it instead of on training
        data.
    task
        ``"classification"`` or ``"regression"``.
    cv
        Number of cross-validation folds used to generate out-of-fold base
        model predictions for training the meta-model.
    n_jobs
        Number of parallel jobs used to fit the base models.
    verbose
        Whether to print the ensemble score and each member's individual
        score.

    Returns
    -------
    ToppingsResult
        The fitted ensemble, its train/validation scores, each base
        model's individual score, and total fit time.
    """

    _validate_task(task)
    named_models = _named_models(models)

    if task == "classification":
        ensemble = StackingClassifier(
            estimators=named_models, final_estimator=meta_model, cv=cv, n_jobs=n_jobs
        )
    else:
        ensemble = StackingRegressor(
            estimators=named_models, final_estimator=meta_model, cv=cv, n_jobs=n_jobs
        )

    return _fit_and_score(ensemble, X, y, X_val=X_val, y_val=y_val, verbose=verbose, label="stack")


def bag(
    base_model: Any,
    X: Any,
    y: Any,
    *,
    X_val: Any | None = None,
    y_val: Any | None = None,
    task: str = "classification",
    n_estimators: int = 10,
    max_samples: float = 1.0,
    random_state: int | None = 42,
    verbose: bool = True,
) -> ToppingsResult:
    """
    Bag many bootstrap-sampled copies of a single base model.

    Parameters
    ----------
    base_model
        Estimator cloned and refit on each bootstrap sample.
    X, y
        Training data.
    X_val, y_val
        Optional validation data. When provided, ``val_score`` and
        ``individual_scores`` are computed on it instead of on training
        data.
    task
        ``"classification"`` or ``"regression"``.
    n_estimators
        Number of bootstrap-sampled copies to train.
    max_samples
        Fraction of the training set drawn (with replacement) for each
        copy.
    random_state
        Random seed controlling the bootstrap sampling.
    verbose
        Whether to print the ensemble score and each member's individual
        score.

    Returns
    -------
    ToppingsResult
        The fitted ensemble, its train/validation scores, each bagged
        member's individual score (keyed ``estimator_0``, ``estimator_1``,
        ...), and total fit time.
    """

    _validate_task(task)

    if task == "classification":
        ensemble = BaggingClassifier(
            estimator=base_model,
            n_estimators=n_estimators,
            max_samples=max_samples,
            random_state=random_state,
        )
    else:
        ensemble = BaggingRegressor(
            estimator=base_model,
            n_estimators=n_estimators,
            max_samples=max_samples,
            random_state=random_state,
        )

    return _fit_and_score(ensemble, X, y, X_val=X_val, y_val=y_val, verbose=verbose, label="bag")


def _fit_and_score(
    ensemble: Any,
    X: Any,
    y: Any,
    *,
    X_val: Any | None,
    y_val: Any | None,
    verbose: bool,
    label: str,
) -> ToppingsResult:
    """Fit an ensemble, score it, and score each of its members."""

    start = time.perf_counter()
    ensemble.fit(X, y)
    fit_time_s = time.perf_counter() - start

    train_score = float(ensemble.score(X, y))
    has_validation = (X_val is not None) and (y_val is not None)
    val_score = float(ensemble.score(X_val, y_val)) if has_validation else None

    eval_X = X_val if has_validation else X
    eval_y = y_val if has_validation else y
    individual_scores = _score_members(ensemble, eval_X, eval_y)

    if verbose:
        _report(label, train_score, val_score, individual_scores)

    return ToppingsResult(
        model=ensemble,
        train_score=train_score,
        val_score=val_score,
        individual_scores=individual_scores,
        fit_time_s=fit_time_s,
    )


def _score_members(ensemble: Any, X: Any, y: Any) -> dict[str, float]:
    """Score each fitted ensemble member individually."""

    named_estimators = getattr(ensemble, "named_estimators_", None)

    if named_estimators is not None:
        return {name: float(model.score(X, y)) for name, model in named_estimators.items()}

    estimators = getattr(ensemble, "estimators_", [])
    return {f"estimator_{i}": float(model.score(X, y)) for i, model in enumerate(estimators)}


def _named_models(models: dict[str, Any] | Sequence[Any]) -> list[tuple[str, Any]]:
    """Normalize a dict or sequence of models into (name, model) pairs."""

    if isinstance(models, dict):
        return list(models.items())

    return [(f"model_{i}", model) for i, model in enumerate(models)]


def _validate_task(task: str) -> None:
    """Validate the requested task type."""

    if task not in SUPPORTED_TASKS:
        supported = ", ".join(sorted(SUPPORTED_TASKS))
        raise ValueError(f"Unsupported task: {task!r}. Supported tasks are: {supported}.")


def _report(
    label: str,
    train_score: float,
    val_score: float | None,
    individual_scores: dict[str, float],
) -> None:
    """Print the ensemble score and each member's individual score."""

    summary = f"toppings: {label} train_score={train_score:.4f}"

    if val_score is not None:
        summary += f" val_score={val_score:.4f}"

    print(summary)

    for name, score in individual_scores.items():
        print(f"toppings: {label}   {name}={score:.4f}")
