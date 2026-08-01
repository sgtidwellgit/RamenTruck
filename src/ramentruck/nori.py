"""Model explainability for RamenTruck.

Requires SHAP, an optional dependency. Install with
``pip install ramentruck[explain]``.
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np
import pandas as pd

try:
    import shap
except ImportError as exc:
    raise ImportError(
        "nori requires shap. Install it with: pip install ramentruck[explain]"
    ) from exc

try:
    import matplotlib.pyplot as plt
except ImportError as exc:
    raise ImportError(
        "nori requires matplotlib. Install it with: pip install ramentruck[explain]"
    ) from exc

from sklearn.inspection import partial_dependence as sk_partial_dependence

from .results import NoriResult, PDResult


def explain(
    model: Any,
    X: Any,
    *,
    background: Any | None = None,
    feature_names: Sequence[str] | None = None,
    class_index: int | None = None,
    verbose: bool = True,
) -> NoriResult:
    """
    Compute SHAP values for a fitted model over a batch of samples.

    Parameters
    ----------
    model
        A fitted estimator. :func:`shap.Explainer` auto-selects an
        appropriate algorithm (exact ``TreeExplainer`` for tree ensembles,
        ``LinearExplainer`` for linear models, and a model-agnostic fallback
        otherwise).
    X
        Samples to explain.
    background
        Background dataset used to estimate feature "absence". Defaults to
        ``X`` itself. Keep this small for non-tree models, since some
        explainer algorithms scale with its size.
    feature_names
        Names for each column of ``X``. Defaults to ``X.columns`` when ``X``
        is a DataFrame, otherwise ``feature_0``, ``feature_1``, etc.
    class_index
        For classifiers whose SHAP values are per-class, the class to
        explain. Defaults to the positive class (index 1) for binary
        classifiers. Required for classifiers with more than two classes.
    verbose
        Whether to print the top features by mean absolute SHAP value.

    Returns
    -------
    NoriResult
        Per-sample, per-feature SHAP values, the explainer's base value,
        feature names, the explained data, and a feature importance
        DataFrame sorted by mean absolute SHAP value.

    Raises
    ------
    ValueError
        If ``class_index`` is required (more than two classes) but not
        provided.
    """

    resolved_feature_names = _resolve_feature_names(X, feature_names)
    explainer = shap.Explainer(model, background if background is not None else X)
    explanation = explainer(X)

    values = np.asarray(explanation.values)
    base_values = np.asarray(explanation.base_values)

    if values.ndim == 3:
        n_classes = values.shape[-1]

        if class_index is None:
            if n_classes != 2:
                raise ValueError(
                    "class_index is required to explain a model with more "
                    f"than two classes; got {n_classes} classes."
                )
            class_index = 1

        values = values[:, :, class_index]
        base_values = base_values[:, class_index] if base_values.ndim == 2 else base_values

    importance = _build_importance(values, resolved_feature_names)

    if verbose:
        _report(importance)

    return NoriResult(
        shap_values=values,
        base_values=base_values,
        feature_names=resolved_feature_names,
        data=X,
        importance=importance,
        class_index=class_index,
    )


def explain_instance(result: NoriResult, index: int) -> pd.Series:
    """
    Return the per-feature SHAP contribution for a single explained sample.

    Parameters
    ----------
    result
        A :class:`NoriResult` returned by :func:`explain`.
    index
        Row position within the explained batch.

    Returns
    -------
    pd.Series
        SHAP values indexed by feature name, sorted by absolute magnitude
        (largest contribution first).
    """

    row = pd.Series(result.shap_values[index], index=result.feature_names, name="shap_value")
    return row.reindex(row.abs().sort_values(ascending=False).index)


def plot_importance(
    result: NoriResult,
    *,
    max_display: int = 20,
    figsize: tuple[float, float] = (8, 6),
) -> plt.Figure:
    """
    Plot mean absolute SHAP value per feature as a horizontal bar chart.

    Parameters
    ----------
    result
        A :class:`NoriResult` returned by :func:`explain`.
    max_display
        Maximum number of top features to display.
    figsize
        Figure size passed to ``matplotlib.pyplot.subplots``.

    Returns
    -------
    matplotlib.figure.Figure
    """

    top = result.importance.head(max_display).iloc[::-1]

    fig, ax = plt.subplots(figsize=figsize)
    ax.barh(top["feature"], top["mean_abs_shap"], color="#c0392b")
    ax.set_xlabel("mean(|SHAP value|)")
    ax.set_title("Feature Importance")
    fig.tight_layout()
    return fig


def plot_summary(
    result: NoriResult,
    *,
    max_display: int = 20,
    figsize: tuple[float, float] = (10, 6),
) -> plt.Figure:
    """
    Plot a beeswarm-style summary of SHAP values across explained samples.

    Each row is a feature; each dot is one sample, positioned by its SHAP
    value and colored by that sample's (normalized) feature value.

    Parameters
    ----------
    result
        A :class:`NoriResult` returned by :func:`explain`.
    max_display
        Maximum number of top features to display, ranked by mean absolute
        SHAP value.
    figsize
        Figure size passed to ``matplotlib.pyplot.subplots``.

    Returns
    -------
    matplotlib.figure.Figure
    """

    top_features = result.importance.head(max_display)["feature"].tolist()[::-1]
    data = _as_dataframe(result.data, result.feature_names)
    rng = np.random.default_rng(0)

    fig, ax = plt.subplots(figsize=figsize)
    mappable = None

    for row_index, feature in enumerate(top_features):
        column_index = result.feature_names.index(feature)
        shap_column = result.shap_values[:, column_index]
        feature_values = _normalize(np.asarray(data[feature], dtype=float))
        jitter = rng.uniform(-0.3, 0.3, size=len(shap_column))

        mappable = ax.scatter(
            shap_column,
            np.full(len(shap_column), row_index) + jitter,
            c=feature_values,
            cmap="coolwarm",
            s=14,
            vmin=0.0,
            vmax=1.0,
        )

    ax.set_yticks(range(len(top_features)))
    ax.set_yticklabels(top_features)
    ax.axvline(0.0, color="grey", linewidth=0.8)
    ax.set_xlabel("SHAP value (impact on model output)")
    ax.set_title("SHAP Summary")

    if mappable is not None:
        fig.colorbar(mappable, ax=ax, label="Feature value (normalized)")

    fig.tight_layout()
    return fig


def partial_dependence(
    model: Any,
    X: Any,
    features: Sequence[str] | Sequence[int],
    *,
    grid_resolution: int = 20,
    class_index: int | None = None,
    verbose: bool = True,
) -> PDResult:
    """
    Compute one-way partial dependence curves for a fitted model.

    Parameters
    ----------
    model
        A fitted estimator supporting scikit-learn's partial dependence
        (``predict`` or ``predict_proba``/``decision_function``).
    X
        Reference data used to build the dependence grid.
    features
        Column names (when ``X`` is a DataFrame) or column indices to
        compute partial dependence for.
    grid_resolution
        Number of points sampled along each feature's value range.
    class_index
        For classifiers with more than one output row (multiclass), the
        class to use. Defaults to the first (and, for binary classifiers,
        only) output row.
    verbose
        Whether to print each feature's dependence range.

    Returns
    -------
    PDResult
        A mapping of feature name to a DataFrame with ``grid_value`` and
        ``average`` columns.
    """

    feature_names = _resolve_feature_names(X, None)
    curves: dict[str, pd.DataFrame] = {}

    for feature in features:
        column_index = _resolve_column_index(feature, feature_names)
        raw = sk_partial_dependence(
            model, X, features=[column_index], grid_resolution=grid_resolution
        )

        average = np.asarray(raw["average"])
        row_index = 0 if class_index is None else class_index
        curve = pd.DataFrame(
            {
                "grid_value": np.asarray(raw["grid_values"][0]),
                "average": average[row_index],
            }
        )
        curves[feature_names[column_index]] = curve

        if verbose:
            print(
                f"nori: {feature_names[column_index]} average ranges "
                f"[{curve['average'].min():.4f}, {curve['average'].max():.4f}]"
            )

    return PDResult(curves=curves)


def plot_partial_dependence(
    result: PDResult,
    *,
    figsize: tuple[float, float] = (4, 4),
) -> plt.Figure:
    """
    Plot one subplot per feature in a :class:`PDResult`.

    Parameters
    ----------
    result
        A :class:`PDResult` returned by :func:`partial_dependence`.
    figsize
        Per-subplot figure size; the overall figure width scales with the
        number of features.

    Returns
    -------
    matplotlib.figure.Figure
    """

    features = list(result.curves)
    width, height = figsize
    fig, axes = plt.subplots(1, len(features), figsize=(width * len(features), height), squeeze=False)

    for ax, feature in zip(axes[0], features):
        curve = result.curves[feature]
        ax.plot(curve["grid_value"], curve["average"], "b-")
        ax.set_xlabel(feature)
        ax.set_ylabel("Partial dependence")
        ax.grid()

    fig.tight_layout()
    return fig


def _resolve_feature_names(X: Any, feature_names: Sequence[str] | None) -> list[str]:
    """Resolve display names for each column of X."""

    if feature_names is not None:
        return list(feature_names)

    if isinstance(X, pd.DataFrame):
        return list(X.columns)

    n_features = np.asarray(X).shape[1]
    return [f"feature_{i}" for i in range(n_features)]


def _resolve_column_index(feature: str | int, feature_names: list[str]) -> int:
    """Resolve a feature name or index to a column index."""

    if isinstance(feature, str):
        if feature not in feature_names:
            raise ValueError(f"Unknown feature: {feature!r}.")
        return feature_names.index(feature)

    return int(feature)


def _as_dataframe(X: Any, feature_names: list[str]) -> pd.DataFrame:
    """Return X as a DataFrame labeled with feature_names."""

    if isinstance(X, pd.DataFrame):
        return X

    return pd.DataFrame(np.asarray(X), columns=feature_names)


def _build_importance(values: np.ndarray, feature_names: list[str]) -> pd.DataFrame:
    """Build a feature importance DataFrame sorted by mean absolute SHAP value."""

    mean_abs = np.abs(values).mean(axis=0)
    return (
        pd.DataFrame({"feature": feature_names, "mean_abs_shap": mean_abs})
        .sort_values("mean_abs_shap", ascending=False)
        .reset_index(drop=True)
    )


def _normalize(values: np.ndarray) -> np.ndarray:
    """Min-max normalize an array to [0, 1], collapsing to 0.5 when constant."""

    value_range = values.max() - values.min()

    if value_range == 0:
        return np.full_like(values, 0.5, dtype=float)

    return (values - values.min()) / value_range


def _report(importance: pd.DataFrame) -> None:
    """Print the top features by mean absolute SHAP value."""

    for _, row in importance.head(10).iterrows():
        print(f"nori: {row['feature']:<25}{row['mean_abs_shap']:.4f}")
