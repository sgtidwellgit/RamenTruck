"""Experiment tracking for RamenTruck.

Requires MLflow, an optional dependency. Install with
``pip install ramentruck[tracking]``.
"""

from __future__ import annotations

import os
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

import pandas as pd

try:
    import mlflow
except ImportError as exc:
    raise ImportError(
        "miso requires mlflow. Install it with: pip install ramentruck[tracking]"
    ) from exc

from .results import MisoRunSummary

DEFAULT_TRACKING_DIR = Path(".miso")
PARAM_PREFIX = "params."
METRIC_PREFIX = "metrics."
RUN_NAME_TAG = "tags.mlflow.runName"


class MisoRun:
    """Handle for logging to a single miso run, yielded by :func:`brew`."""

    def __init__(self, active_run: Any) -> None:
        """Wrap an active MLflow run."""

        self._active_run = active_run

    @property
    def run_id(self) -> str:
        """The run's unique identifier."""

        return self._active_run.info.run_id

    def log_param(self, key: str, value: Any) -> None:
        """Log a single hyperparameter or configuration value."""

        mlflow.log_param(key, value)

    def log_params(self, params: dict[str, Any]) -> None:
        """Log a batch of hyperparameter or configuration values."""

        mlflow.log_params(params)

    def log_metric(self, key: str, value: float, *, step: int | None = None) -> None:
        """Log a single metric, optionally at a training step."""

        mlflow.log_metric(key, value, step=step)

    def log_metrics(self, metrics: dict[str, float], *, step: int | None = None) -> None:
        """Log a batch of metrics, optionally at a training step."""

        mlflow.log_metrics(metrics, step=step)

    def log_model(self, model: Any, name: str = "model") -> None:
        """Log a fitted scikit-learn compatible model as a run artifact."""

        mlflow.sklearn.log_model(model, name)

    def log_artifact(self, path: str | Path) -> None:
        """Log an arbitrary local file as a run artifact."""

        mlflow.log_artifact(str(path))


@contextmanager
def brew(
    experiment_name: str,
    *,
    run_name: str | None = None,
    tracking_uri: str | None = None,
    tags: dict[str, str] | None = None,
    verbose: bool = True,
) -> Iterator[MisoRun]:
    """
    Start a miso run for logging params, metrics, models, and artifacts.

    Runs accumulate in a local SQLite-backed store by default (``.miso/``
    under the current working directory), so tracking works without any
    external server or account. Point ``tracking_uri`` at a remote MLflow
    server to share runs across a team.

    Parameters
    ----------
    experiment_name
        Name of the experiment the run belongs to. Created automatically if
        it does not already exist.
    run_name
        Optional human-readable name for the run.
    tracking_uri
        MLflow tracking URI. Defaults to the ``MLFLOW_TRACKING_URI``
        environment variable when set, otherwise a local SQLite database
        under ``.miso/tracking.db``.
    tags
        Optional key/value tags attached to the run.
    verbose
        Whether to print the run ID at start and completion.

    Yields
    ------
    MisoRun
        A handle for logging params, metrics, models, and artifacts to this
        run.

    Examples
    --------
    >>> with brew("rf_experiment", run_name="rf_v1") as run:  # doctest: +SKIP
    ...     run.log_params({"n_estimators": 100})
    ...     run.log_metric("accuracy", 0.95)
    ...     run.log_model(model)
    """

    mlflow.set_tracking_uri(tracking_uri or os.environ.get("MLFLOW_TRACKING_URI") or _default_tracking_uri())
    mlflow.set_experiment(experiment_name)

    with mlflow.start_run(run_name=run_name, tags=tags) as active_run:
        run = MisoRun(active_run)

        if verbose:
            print(f"miso: brewing run {run.run_id!r} in experiment {experiment_name!r}")

        yield run

        if verbose:
            print(f"miso: run {run.run_id!r} complete")


def list_runs(experiment_name: str, *, tracking_uri: str | None = None) -> pd.DataFrame:
    """
    List every run recorded under an experiment, most recent first.

    Parameters
    ----------
    experiment_name
        Name of the experiment to list runs for.
    tracking_uri
        MLflow tracking URI. Defaults to the ``MLFLOW_TRACKING_URI``
        environment variable when set, otherwise the local ``.miso`` store.

    Returns
    -------
    pd.DataFrame
        One row per run with ``run_id``, ``params.*``, ``metrics.*``,
        ``tags.*``, and timing columns, sorted by ``start_time`` descending.
        Empty (with just a ``run_id`` column) if the experiment does not
        exist or has no runs.
    """

    mlflow.set_tracking_uri(tracking_uri or os.environ.get("MLFLOW_TRACKING_URI") or _default_tracking_uri())
    experiment = mlflow.get_experiment_by_name(experiment_name)

    if experiment is None:
        return pd.DataFrame(columns=["run_id"])

    runs = mlflow.search_runs(experiment_ids=[experiment.experiment_id])

    if runs.empty:
        return runs

    return runs.sort_values("start_time", ascending=False).reset_index(drop=True)


def best_run(
    experiment_name: str,
    metric: str,
    *,
    mode: str = "max",
    tracking_uri: str | None = None,
) -> MisoRunSummary:
    """
    Find the best run in an experiment by a logged metric.

    Parameters
    ----------
    experiment_name
        Name of the experiment to search.
    metric
        Metric name to rank runs by (without the ``metrics.`` prefix).
    mode
        ``"max"`` to select the highest value, ``"min"`` to select the
        lowest.
    tracking_uri
        MLflow tracking URI. Defaults to the ``MLFLOW_TRACKING_URI``
        environment variable when set, otherwise the local ``.miso`` store.

    Returns
    -------
    MisoRunSummary
        The best run's ID, name, params, metrics, and artifact URI.

    Raises
    ------
    ValueError
        If ``mode`` is not ``"max"`` or ``"min"``, or if no run in the
        experiment logged the requested metric.
    """

    if mode not in {"max", "min"}:
        raise ValueError(f"Unsupported mode: {mode!r}. Supported modes are: max, min.")

    runs = list_runs(experiment_name, tracking_uri=tracking_uri)
    metric_column = f"{METRIC_PREFIX}{metric}"

    if metric_column not in runs.columns or runs[metric_column].dropna().empty:
        raise ValueError(
            f"No runs in experiment {experiment_name!r} logged metric {metric!r}."
        )

    candidates = runs.dropna(subset=[metric_column])
    best_index = (
        candidates[metric_column].idxmax() if mode == "max" else candidates[metric_column].idxmin()
    )
    best = candidates.loc[best_index]

    return MisoRunSummary(
        run_id=best["run_id"],
        run_name=best.get(RUN_NAME_TAG),
        params=_extract_prefixed(best, PARAM_PREFIX),
        metrics=_extract_prefixed(best, METRIC_PREFIX),
        artifact_uri=best["artifact_uri"],
    )


def _extract_prefixed(row: pd.Series, prefix: str) -> dict[str, Any]:
    """Extract prefixed columns from a search_runs row into a plain dict."""

    return {
        column[len(prefix):]: row[column]
        for column in row.index
        if column.startswith(prefix) and pd.notna(row[column])
    }


def _default_tracking_uri() -> str:
    """Build a local SQLite-backed tracking URI under .miso/."""

    DEFAULT_TRACKING_DIR.mkdir(exist_ok=True)
    db_path = (DEFAULT_TRACKING_DIR / "tracking.db").resolve()
    return f"sqlite:///{db_path.as_posix()}"
