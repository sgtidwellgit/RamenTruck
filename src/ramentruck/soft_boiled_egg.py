"""Cross-validation module for RamenTruck."""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import (
    KFold,
    StratifiedKFold,
    TimeSeriesSplit,
    cross_validate,
)
from sklearn.model_selection import learning_curve as sk_learning_curve

from .results import EggResult

DEFAULT_TRAIN_SIZES = [0.1, 0.2, 0.4, 0.6, 0.8, 1.0]
HIGH_VARIANCE_THRESHOLD = 0.05

SPLITTER_STRATEGIES = frozenset({"kfold", "stratified", "timeseries"})


def soft_boiled_egg(
    model: Any,
    X: Any,
    y: Any,
    *,
    strategy: str = "stratified",
    n_splits: int = 5,
    scoring: str | list[str] = "accuracy",
    learning_curve: bool = False,
    train_sizes: list[float] | None = None,
    shuffle: bool = True,
    random_state: int | None = 42,
    verbose: bool = True,
) -> EggResult:
    """
    Cross-validate a model and surface learning curves as a first-class output.

    Parameters
    ----------
    model
        Estimator to evaluate. Not mutated.
    X
        Feature data.
    y
        Target values.
    strategy
        Splitting strategy: ``"kfold"``, ``"stratified"`` (default, preserves
        class balance per fold), or ``"timeseries"`` (no shuffling, respects
        temporal order).
    n_splits
        Number of cross-validation folds.
    scoring
        A single scikit-learn scoring string, or a list of them. The first
        entry is treated as the primary metric for ``scores``, ``mean_score``,
        and ``std_score``.
    learning_curve
        Whether to additionally compute a learning curve.
    train_sizes
        Fractions of the training set used for the learning curve. Defaults
        to ``[0.1, 0.2, 0.4, 0.6, 0.8, 1.0]``. Only used when
        ``learning_curve=True``.
    shuffle
        Whether to shuffle before splitting. Ignored (and warned on) for
        ``"timeseries"``, which never shuffles.
    random_state
        Random seed used when ``shuffle=True``.
    verbose
        Whether to print per-fold scores, mean +/- std, and a high-variance
        flag.

    Returns
    -------
    EggResult
        Per-fold scores for the primary metric, their mean and standard
        deviation, a DataFrame of per-fold scores for every requested metric,
        and (when requested) a learning curve DataFrame.

    Raises
    ------
    ValueError
        If ``strategy`` is not one of ``"kfold"``, ``"stratified"``, or
        ``"timeseries"``.
    """

    if strategy not in SPLITTER_STRATEGIES:
        supported = ", ".join(sorted(SPLITTER_STRATEGIES))
        raise ValueError(
            f"Unsupported strategy: {strategy!r}. Supported strategies are: {supported}."
        )

    if strategy == "timeseries" and shuffle:
        warnings.warn(
            "strategy='timeseries' never shuffles; the shuffle=True argument "
            "is ignored.",
            UserWarning,
            stacklevel=2,
        )

    splitter = _build_splitter(strategy, n_splits, shuffle, random_state)

    if strategy == "stratified":
        _warn_if_n_splits_exceeds_minority_class(y, n_splits)

    scoring_list = [scoring] if isinstance(scoring, str) else list(scoring)
    primary_metric = scoring_list[0]

    cv_results = cross_validate(model, X, y, cv=splitter, scoring=scoring_list)

    fold_results = pd.DataFrame(
        {metric: cv_results[f"test_{metric}"] for metric in scoring_list}
    )

    scores = np.asarray(cv_results[f"test_{primary_metric}"])
    mean_score = float(scores.mean())
    std_score = float(scores.std())

    learning_curve_df = None
    if learning_curve:
        learning_curve_df = _compute_learning_curve(
            model,
            X,
            y,
            splitter=splitter,
            scoring=primary_metric,
            train_sizes=train_sizes if train_sizes is not None else DEFAULT_TRAIN_SIZES,
        )

    if verbose:
        _report(scoring_list, fold_results, mean_score, std_score)

    return EggResult(
        scores=scores,
        mean_score=mean_score,
        std_score=std_score,
        fold_results=fold_results,
        learning_curve_df=learning_curve_df,
    )


def _build_splitter(
    strategy: str,
    n_splits: int,
    shuffle: bool,
    random_state: int | None,
) -> Any:
    """Build the scikit-learn splitter for the requested strategy."""

    if strategy == "kfold":
        return KFold(
            n_splits=n_splits,
            shuffle=shuffle,
            random_state=random_state if shuffle else None,
        )

    if strategy == "stratified":
        return StratifiedKFold(
            n_splits=n_splits,
            shuffle=shuffle,
            random_state=random_state if shuffle else None,
        )

    return TimeSeriesSplit(n_splits=n_splits)


def _warn_if_n_splits_exceeds_minority_class(y: Any, n_splits: int) -> None:
    """Warn when n_splits exceeds the smallest class count."""

    _, counts = np.unique(np.asarray(y), return_counts=True)
    minority_count = int(counts.min())

    if n_splits > minority_count:
        warnings.warn(
            f"n_splits={n_splits} exceeds the minority class count "
            f"({minority_count}); some folds may not contain every class.",
            UserWarning,
            stacklevel=3,
        )


def _compute_learning_curve(
    model: Any,
    X: Any,
    y: Any,
    *,
    splitter: Any,
    scoring: str,
    train_sizes: list[float],
) -> pd.DataFrame:
    """Compute a learning curve DataFrame for train vs. validation scores."""

    train_sizes_abs, train_scores, val_scores = sk_learning_curve(
        model,
        X,
        y,
        cv=splitter,
        train_sizes=train_sizes,
        scoring=scoring,
    )

    return pd.DataFrame(
        {
            "train_size": train_sizes_abs,
            "train_score_mean": train_scores.mean(axis=1),
            "train_score_std": train_scores.std(axis=1),
            "val_score_mean": val_scores.mean(axis=1),
            "val_score_std": val_scores.std(axis=1),
        }
    )


def _report(
    scoring_list: list[str],
    fold_results: pd.DataFrame,
    mean_score: float,
    std_score: float,
) -> None:
    """Print per-fold scores, mean +/- std, and a high-variance flag."""

    primary_metric = scoring_list[0]

    for fold_index, score in enumerate(fold_results[primary_metric]):
        print(f"soft_boiled_egg: fold {fold_index} {primary_metric}={score:.4f}")

    print(f"soft_boiled_egg: {primary_metric} {mean_score:.4f} +/- {std_score:.4f}")

    if std_score > HIGH_VARIANCE_THRESHOLD:
        print(
            f"soft_boiled_egg: high variance detected (std={std_score:.4f} "
            f"> {HIGH_VARIANCE_THRESHOLD})"
        )
