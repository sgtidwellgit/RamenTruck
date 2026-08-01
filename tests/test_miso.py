"""Unit tests for ramentruck.miso."""

from pathlib import Path

import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification

from ramentruck import miso
from ramentruck.results import MisoRunSummary


@pytest.fixture()
def tracking_uri(tmp_path: Path) -> str:
    """Return an isolated SQLite tracking URI for a single test."""

    db_path = tmp_path / "tracking.db"
    return f"sqlite:///{db_path.as_posix()}"


def test_brew_logs_params_metrics_and_model(tracking_uri):
    """Verify a brewed run records params, metrics, and a logged model."""

    X, y = make_classification(n_samples=40, n_features=4, random_state=42)
    model = RandomForestClassifier(n_estimators=5, random_state=42).fit(X, y)

    with miso.brew("exp_basic", run_name="run1", tracking_uri=tracking_uri, verbose=False) as run:
        run.log_params({"n_estimators": 5})
        run.log_metric("accuracy", 0.9)
        run.log_model(model)
        run_id = run.run_id

    runs = miso.list_runs("exp_basic", tracking_uri=tracking_uri)

    assert run_id in runs["run_id"].tolist()
    assert runs.loc[runs["run_id"] == run_id, "params.n_estimators"].iloc[0] == "5"
    assert runs.loc[runs["run_id"] == run_id, "metrics.accuracy"].iloc[0] == pytest.approx(0.9)


def test_brew_logs_artifact(tracking_uri, tmp_path):
    """Verify log_artifact attaches a local file to the run."""

    artifact_path = tmp_path / "notes.txt"
    artifact_path.write_text("hello miso")

    with miso.brew("exp_artifact", tracking_uri=tracking_uri, verbose=False) as run:
        run.log_artifact(artifact_path)

    runs = miso.list_runs("exp_artifact", tracking_uri=tracking_uri)
    assert len(runs) == 1


def test_list_runs_sorted_most_recent_first(tracking_uri):
    """Verify list_runs sorts by start_time descending."""

    with miso.brew("exp_order", run_name="first", tracking_uri=tracking_uri, verbose=False) as run:
        run.log_metric("accuracy", 0.5)

    with miso.brew("exp_order", run_name="second", tracking_uri=tracking_uri, verbose=False) as run:
        run.log_metric("accuracy", 0.6)

    runs = miso.list_runs("exp_order", tracking_uri=tracking_uri)

    assert list(runs["tags.mlflow.runName"]) == ["second", "first"]


def test_list_runs_missing_experiment_returns_empty(tracking_uri):
    """Verify list_runs returns an empty DataFrame for an unknown experiment."""

    runs = miso.list_runs("nonexistent", tracking_uri=tracking_uri)

    assert isinstance(runs, pd.DataFrame)
    assert runs.empty


def test_best_run_selects_max_metric(tracking_uri):
    """Verify best_run picks the run with the highest metric under mode=max."""

    with miso.brew("exp_best", run_name="low", tracking_uri=tracking_uri, verbose=False) as run:
        run.log_params({"C": 1.0})
        run.log_metric("accuracy", 0.7)

    with miso.brew("exp_best", run_name="high", tracking_uri=tracking_uri, verbose=False) as run:
        run.log_params({"C": 10.0})
        run.log_metric("accuracy", 0.9)

    result = miso.best_run("exp_best", "accuracy", mode="max", tracking_uri=tracking_uri)

    assert isinstance(result, MisoRunSummary)
    assert result.run_name == "high"
    assert result.metrics["accuracy"] == pytest.approx(0.9)
    assert result.params["C"] == "10.0"


def test_best_run_selects_min_metric(tracking_uri):
    """Verify best_run picks the run with the lowest metric under mode=min."""

    with miso.brew("exp_best_min", run_name="low_loss", tracking_uri=tracking_uri, verbose=False) as run:
        run.log_metric("loss", 0.1)

    with miso.brew("exp_best_min", run_name="high_loss", tracking_uri=tracking_uri, verbose=False) as run:
        run.log_metric("loss", 0.8)

    result = miso.best_run("exp_best_min", "loss", mode="min", tracking_uri=tracking_uri)

    assert result.run_name == "low_loss"


def test_best_run_rejects_unsupported_mode(tracking_uri):
    """Verify an unsupported mode raises a descriptive ValueError."""

    with miso.brew("exp_mode", tracking_uri=tracking_uri, verbose=False) as run:
        run.log_metric("accuracy", 0.9)

    with pytest.raises(ValueError, match="Unsupported mode"):
        miso.best_run("exp_mode", "accuracy", mode="median", tracking_uri=tracking_uri)


def test_best_run_missing_metric_raises(tracking_uri):
    """Verify requesting an unlogged metric raises a descriptive ValueError."""

    with miso.brew("exp_missing_metric", tracking_uri=tracking_uri, verbose=False) as run:
        run.log_metric("accuracy", 0.9)

    with pytest.raises(ValueError, match="No runs in experiment"):
        miso.best_run("exp_missing_metric", "f1", tracking_uri=tracking_uri)
