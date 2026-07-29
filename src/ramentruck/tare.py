"""Hyperparameter tuning wrapper for RamenTruck."""

from __future__ import annotations

import time
from typing import Any

import pandas as pd
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

from .results import TareResult

SEARCH_METHODS = {
    "grid": GridSearchCV,
    "random": RandomizedSearchCV,
}


def tare(
    model: Any,
    param_grid: dict[str, Any],
    X: Any,
    y: Any,
    *,
    method: str = "grid",
    cv: int = 5,
    scoring: str = "accuracy",
    n_iter: int = 50,
    n_jobs: int = -1,
    verbose: bool = True,
    random_state: int | None = 42,
) -> TareResult:
    """
    Search a hyperparameter grid for the best-scoring model configuration.

    Parameters
    ----------
    model
        Estimator to tune. Not mutated; a clone is refit internally by the
        underlying scikit-learn search.
    param_grid
        Dictionary mapping parameter names to candidate values.
    X
        Feature data.
    y
        Target values.
    method
        Search strategy: ``"grid"`` for exhaustive :class:`GridSearchCV`, or
        ``"random"`` for :class:`RandomizedSearchCV` sampling ``n_iter``
        combinations.
    cv
        Number of cross-validation folds.
    scoring
        scikit-learn scoring string.
    n_iter
        Number of parameter combinations sampled. Only used by ``"random"``.
    n_jobs
        Number of parallel jobs. ``-1`` uses all available cores.
    verbose
        Whether to print the best params and best score at completion.
    random_state
        Random seed for reproducibility. Only used by ``"random"``.

    Returns
    -------
    TareResult
        Best parameters, best score, the refit best model, full CV results
        (sorted best score first), and total search time.

    Raises
    ------
    ValueError
        If ``method`` is not one of ``"grid"`` or ``"random"``.
    """

    search_cls = SEARCH_METHODS.get(method)

    if search_cls is None:
        supported = ", ".join(sorted(SEARCH_METHODS))
        raise ValueError(
            f"Unsupported method: {method!r}. Supported methods are: {supported}."
        )

    search_kwargs: dict[str, Any] = {"cv": cv, "scoring": scoring, "n_jobs": n_jobs}

    if method == "random":
        search_kwargs["n_iter"] = n_iter
        search_kwargs["random_state"] = random_state

    search = search_cls(model, param_grid, **search_kwargs)

    start = time.perf_counter()
    search.fit(X, y)
    search_time_s = time.perf_counter() - start

    cv_results = (
        pd.DataFrame(search.cv_results_)
        .sort_values("rank_test_score")
        .reset_index(drop=True)
    )

    if verbose:
        print(
            f"tare: best_score={search.best_score_:.4f} "
            f"best_params={search.best_params_}"
        )

    return TareResult(
        best_params=search.best_params_,
        best_score=float(search.best_score_),
        best_model=search.best_estimator_,
        cv_results=cv_results,
        search_time_s=search_time_s,
    )
