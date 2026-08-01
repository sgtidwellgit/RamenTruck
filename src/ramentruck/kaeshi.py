"""Probability calibration for RamenTruck.

Named after kaeshi, the concentrated sauce blended with dashi to balance
a bowl's final flavor - here, the correction blended into a model's raw
scores so its predicted probabilities can be trusted at face value.
"""

from __future__ import annotations

from typing import Any

import pandas as pd
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.metrics import brier_score_loss

from .results import KaeshiResult

SUPPORTED_METHODS = frozenset({"isotonic", "sigmoid"})


def kaeshi(
    model: Any,
    X: Any,
    y: Any,
    *,
    method: str = "isotonic",
    cv: int = 5,
    n_bins: int = 10,
    verbose: bool = True,
) -> KaeshiResult:
    """
    Calibrate a binary classifier's predicted probabilities.

    Fits two versions of ``model``: an uncalibrated clone, scored to
    establish a "before" baseline, and a calibrated version wrapped in
    scikit-learn's :class:`~sklearn.calibration.CalibratedClassifierCV`.
    Both are evaluated with the Brier score, and a reliability curve is
    computed from the calibrated model.

    Parameters
    ----------
    model
        An unfitted binary classifier implementing ``predict_proba``. Not
        mutated; clones are fit internally.
    X, y
        Training data. ``y`` must have exactly two classes.
    method
        Calibration method: ``"isotonic"`` (non-parametric, more flexible,
        needs more data) or ``"sigmoid"`` (Platt scaling, more stable on
        small datasets).
    cv
        Number of cross-validation folds used internally by
        ``CalibratedClassifierCV`` to fit the calibrator on held-out
        predictions.
    n_bins
        Number of bins used to compute the reliability (calibration)
        curve.
    verbose
        Whether to print the Brier score before and after calibration.

    Returns
    -------
    KaeshiResult
        The calibrated model, the calibration method used, Brier scores
        before and after calibration, and a calibration curve DataFrame
        with ``prob_pred`` and ``prob_true`` columns.

    Raises
    ------
    ValueError
        If ``method`` is not ``"isotonic"`` or ``"sigmoid"``.
    """

    _validate_method(method)

    uncalibrated = clone(model)
    uncalibrated.fit(X, y)
    raw_probability = uncalibrated.predict_proba(X)[:, 1]
    brier_before = float(brier_score_loss(y, raw_probability))

    calibrated = CalibratedClassifierCV(clone(model), method=method, cv=cv)
    calibrated.fit(X, y)
    calibrated_probability = calibrated.predict_proba(X)[:, 1]
    brier_after = float(brier_score_loss(y, calibrated_probability))

    prob_true, prob_pred = calibration_curve(y, calibrated_probability, n_bins=n_bins)
    calibration_curve_df = pd.DataFrame({"prob_pred": prob_pred, "prob_true": prob_true})

    if verbose:
        print(
            f"kaeshi: method={method} brier_before={brier_before:.4f} "
            f"brier_after={brier_after:.4f}"
        )

    return KaeshiResult(
        model=calibrated,
        method=method,
        brier_score_before=brier_before,
        brier_score_after=brier_after,
        calibration_curve_df=calibration_curve_df,
    )


def plot_calibration_curve(result: KaeshiResult, *, figsize: tuple[float, float] = (6, 6)) -> Any:
    """
    Plot a reliability diagram comparing predicted vs. observed probabilities.

    Parameters
    ----------
    result
        A :class:`KaeshiResult` returned by :func:`kaeshi`.
    figsize
        Figure size passed to ``matplotlib.pyplot.subplots``.

    Returns
    -------
    matplotlib.figure.Figure
    """

    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ImportError(
            "plot_calibration_curve requires matplotlib. Install it with: pip install matplotlib"
        ) from exc

    fig, ax = plt.subplots(figsize=figsize)
    ax.plot([0, 1], [0, 1], "k--", label="Perfectly calibrated")
    ax.plot(
        result.calibration_curve_df["prob_pred"],
        result.calibration_curve_df["prob_true"],
        "o-",
        label=f"{result.method} (brier={result.brier_score_after:.4f})",
    )
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Fraction of positives")
    ax.set_title("Calibration Curve")
    ax.legend()
    ax.grid()
    fig.tight_layout()
    return fig


def _validate_method(method: str) -> None:
    """Validate the requested calibration method."""

    if method not in SUPPORTED_METHODS:
        supported = ", ".join(sorted(SUPPORTED_METHODS))
        raise ValueError(f"Unsupported method: {method!r}. Supported methods are: {supported}.")
