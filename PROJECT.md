# RamenTruck — Project Document

> **Current version:** 0.4.0 | **Python:** >= 3.9 | **Status:** Early development

---

## Table of Contents

1. [What RamenTruck Is](#what-ramentruck-is)
2. [The Philosophy](#the-philosophy)
3. [Current State](#current-state)
4. [Repository Layout (Target)](#repository-layout-target)
5. [Dependency & Extras Architecture](#dependency--extras-architecture)
6. [The Bowl - Implemented and Planned Modules](#the-bowl--implemented-and-planned-modules)
   - [Implemented: `noodles`](#implemented-noodles--dataset-inspection--preprocessing)
   - [Shared: `diagnostics`](#shared-diagnostics--deterministic-diagnostic-engine)
   - [Core: `broth`](#core-broth--the-foundation)
   - [Core: `tare`](#core-tare--hyperparameter-tuning)
   - [Core: `soft_boiled_egg`](#core-soft_boiled_egg--cross-validation)
   - [Core: `chashu`](#core-chashu--model-persistence)
   - [Explainability: `nori`](#explainability-nori--model-explainability)
   - [Experiment Tracking: `miso`](#experiment-tracking-miso--experiment-tracking)
   - [Deep Learning: `tonkotsu`](#deep-learning-tonkotsu--deep-learning-interface)
   - [To Be Named](#to-be-named)
7. [Design Principles](#design-principles)
8. [Inter-Op with ThaiTruck](#inter-op-with-thaitruck)
9. [The Food Truck Fleet](#the-food-truck-fleet)
10. [Build & Publish Plan](#build--publish-plan)

---

## What RamenTruck Is

RamenTruck is a deep, layered ML/AI toolkit. The current public entry points are `slurp()`, which inspects a pandas DataFrame and returns a `DatasetMenu`, and `Broth`, a training wrapper for scikit-learn compatible estimators that returns a `BrothResult`. The broader roadmap includes hyperparameter tuning, cross-validation, explainability, experiment tracking, model persistence, and deep learning architectures.

RamenTruck is **not** a thin scikit-learn wrapper or a tutorial notebook. It is designed with genuine ML depth — handling edge cases like small-N datasets, class imbalance, noisy labels, and overfit diagnosis as first-class concerns rather than afterthoughts.

```bash
pip install ramentruck
```

---

## The Philosophy

**Why ramen?** Ramen is deep, layered, and complex. Great ramen takes time, precision, and real expertise — you can't shortcut the broth. That maps directly to serious ML work. The dish names follow through: `broth` is the base, `tare` is the concentrated seasoning, `chashu` is the slow-preserved topping, `nori` is the thin layer of insight on the surface, and `tonkotsu` is the heavy, long-cooked deep variant.

The guiding design values:

- **Depth over convenience** — don't paper over important details; expose them cleanly
- **Composable, not monolithic** — each module is independent and useful on its own
- **Edge-case first** — small N, class imbalance, noisy labels, and overfit diagnosis are not footnotes
- **Optional extras, not mandatory bloat** — TensorFlow and SHAP are heavy dependencies; they are opt-in
- **Consistent with ThaiTruck** — similar module independence, same no-mutation conventions, same `src/` layout

---

## Current State

| Item | Status |
|---|---|
| PyPI name `ramentruck` | Secured |
| Version | 0.4.0 |
| `src/ramentruck/__init__.py` | Exists - exports `slurp`, `DatasetMenu`, `ChefRecommendation`, `Broth`, `BrothResult`, `DiagnosticEngine`, `DiagnosticReport`, `DiagnosticCategory`, `DiagnosticSeverity`, `Recommendation`, and `__version__ = "0.4.0"` (note: `tonkotsu` is not re-exported at the top level since it requires the optional `deep` extra — import it as `from ramentruck import tonkotsu`) |
| `src/ramentruck/noodles.py` | Implemented - dataset inspection and recommendation engine |
| `src/ramentruck/diagnostics.py` | Implemented - shared deterministic diagnostics engine and immutable report types; standalone, not yet consumed by `broth` or other modules |
| `src/ramentruck/broth.py` | Implemented - training wrapper with `fit`, `predict`, `score`, metrics, timing, and overfitting warning |
| `src/ramentruck/results.py` | Implemented - shared `BrothResult` container |
| `src/ramentruck/tonkotsu.py` | Implemented (foundation + CNN family) - `build_dense`, `simmer`, `plot_history`, `EveryNEpochs`, `residual_identity_block`, `residual_conv_block`, `build_resnet`; requires the `deep` extra (TensorFlow + matplotlib). RNN/sequence family still planned. |
| `pyproject.toml` | Exists - hatchling build, Python >= 3.9, MIT license, runtime dependencies, `dev` extra, and `deep` extra |
| `README.md` | Exists - current `slurp()` and `Broth` usage, module table, install instructions, fleet context |
| Core ML modules | `broth` implemented; `tare`, `soft_boiled_egg`, and `chashu` remain planned |
| Optional modules | `tonkotsu` implemented (foundation + CNN family; RNN/sequence family planned); `nori` and `miso` not yet implemented |
| Tests | `tests/test_noodles.py`, `tests/test_broth.py`, `tests/test_diagnostics.py`, and `tests/test_tonkotsu.py` exist |
| Optional extras in `pyproject.toml` | `dev` and `deep` exist; `explain`, `tracking`, and `all` are still planned |

The package is no longer a pure stub. The current implemented workflows are dataset inspection through `slurp()`, estimator training through `Broth`, and deep learning model building/training through `tonkotsu` (dense networks and ResNet-style CNNs). A shared `diagnostics` engine exists as reusable infrastructure for future modules, not yet wired into `broth` or `tonkotsu`. Tuning, cross-validation, persistence, explainability, tracking, and the RNN/sequence half of deep learning remain roadmap items.

> **Note on dataclasses:** result and report objects across the package (`DatasetMenu`, `ChefRecommendation`, `BrothResult`, `Recommendation`, `DiagnosticReport`) use `@dataclass(frozen=True)` without `slots=True`. The `slots` keyword on `dataclass()` requires Python 3.10+, and the package declares `requires-python = ">=3.9"`.

---

## Repository Layout (Target)

```
RamenTruck/
+-- pyproject.toml              # build config, metadata, optional extras
+-- README.md                   # user-facing install and usage guide
+-- PROJECT.md                  # this file - comprehensive project state + roadmap
+-- CHANGELOG.md                # per-version release notes (planned)
+-- src/
|   +-- ramentruck/
|       +-- __init__.py         # public re-exports + __version__
|       +-- noodles.py          # dataset inspection and preprocessing (implemented)
|       +-- diagnostics.py      # shared diagnostic rules and report types (implemented, standalone)
|       +-- broth.py            # model training / evaluation wrapper (implemented)
|       +-- results.py          # shared result containers (implemented)
|       +-- tare.py             # hyperparameter tuning (planned)
|       +-- soft_boiled_egg.py  # cross-validation (planned)
|       +-- chashu.py           # model serialization + versioning (planned)
|       +-- nori.py             # SHAP / explainability [explain extra] (planned)
|       +-- miso.py             # experiment tracking [tracking extra] (planned)
|       +-- tonkotsu.py         # deep learning [deep extra] (implemented: foundation + CNN family)
|       +-- sensei.py           # neural network advisor (planned)
+-- tests/
|   +-- __init__.py
|   +-- test_noodles.py         # implemented
|   +-- test_diagnostics.py     # implemented
|   +-- test_broth.py           # implemented
|   +-- test_tare.py            # planned
|   +-- test_soft_boiled_egg.py # planned
|   +-- test_chashu.py          # planned
|   +-- test_nori.py            # planned
|   +-- test_miso.py            # planned
|   +-- test_tonkotsu.py        # implemented (foundation + CNN family)
+-- benchmarks/                 # pytest-benchmark regressions (future)
```

---

## Dependency & Extras Architecture

The current package declares runtime dependencies for `numpy`, `pandas`, and `scikit-learn`, which match the implemented `noodles` and `broth` modules.

The intended core package should stay lightweight: pandas, numpy, and scikit-learn for core data inspection and classical ML. Heavy dependencies such as TensorFlow, SHAP, MLflow, and W&B should remain opt-in extras.

```toml
[project]
dependencies = [
    "pandas>=1.5",
    "numpy>=1.23",
    "scikit-learn>=1.3",
]

[project.optional-dependencies]
dev      = ["pytest>=8.0", "pytest-cov"]
explain  = ["shap>=0.44"]
tracking = ["mlflow>=2.0"]
deep     = ["tensorflow>=2.12"]
all      = ["ramentruck[explain,tracking,deep]"]
```

Install patterns:

```bash
pip install ramentruck                   # core only (noodles, broth, tare, soft_boiled_egg, chashu)
pip install ramentruck[explain]          # + nori  (SHAP)
pip install ramentruck[tracking]         # + miso  (MLflow / W&B)
pip install ramentruck[deep]             # + tonkotsu (TensorFlow / Keras)
pip install ramentruck[all]              # everything
```

**Import guard pattern** — modules in optional extras must not cause `ImportError` at package import time. Each optional module should guard its heavy dependency with a clear, actionable error:

```python
# nori.py
try:
    import shap
except ImportError as e:
    raise ImportError(
        "nori requires SHAP. Install it with: pip install ramentruck[explain]"
    ) from e
```

---

## The Bowl - Implemented and Planned Modules


### Implemented: `noodles` - Dataset Inspection / Preprocessing

**File:** `src/ramentruck/noodles.py`

`noodles` currently provides the first implemented RamenTruck workflow: dataset inspection through `slurp()`.

**Public API:**

```python
from ramentruck import slurp, DatasetMenu, ChefRecommendation

menu = slurp(df, target="Purchased")
```

**Implemented result objects:**

| Object | Description |
|---|---|
| `DatasetMenu` | Frozen dataclass containing row/column counts, memory usage, column type lists, missing-value summary, duplicate row count, and a `ChefRecommendation` |
| `ChefRecommendation` | Frozen dataclass containing inferred problem type, scaling recommendation, encoding recommendation, loss, output activation, optimizer, and class imbalance flag |

**Current behavior:**

- Requires a pandas DataFrame input
- Validates the optional target column
- Detects numeric, categorical, boolean, and datetime columns
- Reports missing values and duplicate rows
- Infers regression vs. classification when a target is provided
- Flags class imbalance when the minority class is below 10%
- Recommends `StandardScaler`, `OneHotEncoder`, and basic neural-network loss/output defaults

**Tests:** `tests/test_noodles.py` covers the implemented behavior.

---

### Shared: `diagnostics` — Deterministic Diagnostic Engine

**File:** `src/ramentruck/diagnostics.py`

`diagnostics` is a shared, typed rule engine for model-quality guidance
(overfitting, underfitting, small datasets, high variance, class
imbalance). It is designed to be reused by `broth`, `tare`,
`soft_boiled_egg`, `tonkotsu`, and future modules so heuristic
thresholds live in one place instead of being reimplemented per module.

**It is currently standalone** — `broth.py` does not call into it yet
and still contains its own independent overfitting/underfitting/small
dataset checks. Wiring `Broth` (and future modules) to `DiagnosticEngine`
is a follow-up task, not yet done.

**Public API:**

```python
from ramentruck import (
    DiagnosticCategory,
    DiagnosticEngine,
    DiagnosticReport,
    DiagnosticSeverity,
    Recommendation,
)
```

**Implemented objects:**

| Object | Description |
|---|---|
| `DiagnosticSeverity` | Enum with `INFO`, `WARNING`, and `ERROR` recommendation levels |
| `DiagnosticCategory` | Enum of reusable categories: `OVERFITTING`, `UNDERFITTING`, `CLASS_IMBALANCE`, `SMALL_DATASET`, `HIGH_VARIANCE`, `UNKNOWN` |
| `Recommendation` | Frozen dataclass containing message, category, severity, and bounded confidence (validated to `0.0`-`1.0`) |
| `DiagnosticReport` | Frozen dataclass containing diagnosis text, typed recommendations, deterministic warnings, metadata, and `chef_report()` |
| `DiagnosticEngine` | Deterministic evaluator for train/validation gaps, small datasets, variance, and class imbalance |

**Current behavior:**

- `evaluate()` takes `train_score`, `validation_score`, `class_imbalance`, `dataset_size`, and `variance`, and applies fixed thresholds: overfitting gap `> 0.10`, low score `< 0.70`, small dataset `< 100` rows, high variance `> 0.05`
- Returns tuple-based recommendations and warnings so ordering is stable and immutable
- Defensively freezes `metadata` (via `MappingProxyType`) so the report stays immutable in practice, even though it accepts a plain `dict` at construction
- Produces a clean multiline `chef_report()` for human-readable summaries
- Accumulates multiple simultaneous conditions in a deterministic order

**Tests:** `tests/test_diagnostics.py` covers rule evaluation, confidence validation, immutability expectations, and `chef_report()` rendering.

---

### Core: `broth` — The Foundation

**File:** `src/ramentruck/broth.py`

The base model training wrapper. `Broth` wraps the scikit-learn `fit` / `predict` / `score` cycle with explicit validation, metric calculation, fit timing, and an overfitting warning when validation score materially trails training score.

**Implemented API:**

```python
class Broth:
    def __init__(self, estimator) -> None: ...

    def fit(
        self,
        X_train,
        y_train,
        X_val=None,
        y_val=None,
        *,
        metrics: Iterable[str] | None = None,
    ) -> BrothResult: ...

    def predict(self, X): ...

    def score(self, X, y, *, metric: str | None = None) -> float: ...
```

**`BrothResult`** — implemented in `src/ramentruck/results.py` as a
frozen dataclass (no `slots=True`, to keep the package importable on
Python 3.9):

| Field | Description |
|---|---|
| `model` | The fitted estimator |
| `train_score` | Score on training data |
| `val_score` | Score on validation data (if provided) |
| `metrics` | Dict of computed metric name -> value |
| `fit_time_s` | Wall-clock seconds to fit |

**Supported metrics** (via scikit-learn metrics): `"accuracy"`, `"f1"`, `"roc_auc"`, `"precision"`, `"recall"`, `"mse"`, `"rmse"`, `"r2"`.

**Current behavior:**

- Works with any sklearn-compatible estimator (anything with `.fit()` / `.predict()`)
- Validates estimator API and matching `X` / `y` lengths
- Calculates metrics on validation data when provided, otherwise on training data
- Emits a `UserWarning` if `train_score - val_score > 0.10`
- Uses `time.perf_counter()` for fit timing
- Raises a descriptive `ValueError` for unsupported metrics
- Supports `roc_auc` through `predict_proba()` or `decision_function()` when available
- Distinguishes `metrics=None` from `metrics=[]`; an explicit empty list computes no extra metrics
- Uses fitted estimator `classes_` for ROC AUC branching when available instead of relying only on the evaluation batch
- Raises a descriptive `ValueError` when ROC AUC cannot be computed because the evaluation target lacks enough classes or a multiclass evaluation batch is missing fitted classes

**Example:**

```python
from sklearn.ensemble import RandomForestClassifier
from ramentruck import Broth

trainer = Broth(RandomForestClassifier(n_estimators=100))
result = trainer.fit(
    X_train,
    y_train,
    X_val,
    y_val,
    metrics=["accuracy", "f1", "roc_auc"],
)

print(result.val_score)   # e.g., 0.87
print(result.metrics)     # {"accuracy": 0.87, "f1": 0.85, "roc_auc": 0.91}
predictions = trainer.predict(X_val)
```

---

### Core: `tare` — Hyperparameter Tuning

**File:** `src/ramentruck/tare.py`

Named for the concentrated seasoning added to ramen — small adjustments with outsized impact. `tare` is the hyperparameter tuning wrapper. Ships with grid search and randomized search; Optuna support is deferred (see note below).

**Scope decision (locked in 2026-07-19):** `optuna` is dropped from the initial implementation. Optuna's API is trial-based (`suggest_*` calls inside an objective function), which doesn't fit the static `param_grid: dict` interface used by `"grid"`/`"random"`. Adding Optuna later means designing a separate search-space representation rather than forcing it through this signature — treated as a future follow-up, not part of this build.

**Planned signature:**

```python
def tare(
    model,
    param_grid: dict,
    X: pd.DataFrame | np.ndarray,
    y: pd.Series | np.ndarray,
    *,
    method: str = "grid",          # "grid", "random"
    cv: int = 5,
    scoring: str = "accuracy",
    n_iter: int = 50,              # for "random" only
    n_jobs: int = -1,
    verbose: bool = True,
    random_state: int | None = 42,
) -> TareResult
```

**`TareResult`** — result container holding:

| Field | Description |
|---|---|
| `best_params` | Dict of best hyperparameter values |
| `best_score` | Best CV score achieved |
| `best_model` | Refitted model using best params |
| `cv_results` | Full cross-validation results DataFrame |
| `search_time_s` | Wall-clock seconds for the full search |

**Method dispatch:**

| `method` | Backend | Notes |
|---|---|---|
| `"grid"` | `GridSearchCV` | Exhaustive search over all combinations |
| `"random"` | `RandomizedSearchCV` | Randomly samples `n_iter` combinations |

**Design notes:**

- `n_jobs=-1` by default — use all available cores
- `random_state=42` default for reproducibility
- Verbose mode logs the best params and best score at completion
- `cv_results` is always returned as a sorted DataFrame (best score first) for easy inspection
- Optuna support is a future follow-up (not part of this build) — see the scope decision above

**Example:**

```python
from ramentruck import tare
from sklearn.ensemble import GradientBoostingClassifier

result = tare(
    GradientBoostingClassifier(),
    param_grid={"n_estimators": [50, 100, 200], "learning_rate": [0.01, 0.1, 0.3]},
    X_train, y_train,
    method="random",
    n_iter=20,
    scoring="roc_auc",
)

print(result.best_params)  # {"n_estimators": 100, "learning_rate": 0.1}
print(result.best_score)   # 0.93
```

---

### Core: `soft_boiled_egg` — Cross-Validation

**File:** `src/ramentruck/soft_boiled_egg.py`

Named for a technique that is entirely about timing and calibration. `soft_boiled_egg` is the cross-validation module. Supports k-fold, stratified k-fold, and time-series splits. Critically, it surfaces learning curves and overfit diagnostics as first-class outputs — not just a score.

**Planned signature:**

```python
def soft_boiled_egg(
    model,
    X: pd.DataFrame | np.ndarray,
    y: pd.Series | np.ndarray,
    *,
    strategy: str = "stratified",    # "kfold", "stratified", "timeseries"
    n_splits: int = 5,
    scoring: str | list[str] = "accuracy",
    learning_curve: bool = False,
    train_sizes: list[float] | None = None,   # for learning curves
    shuffle: bool = True,
    random_state: int | None = 42,
    verbose: bool = True,
) -> EggResult
```

**`EggResult`** — result container holding:

| Field | Description |
|---|---|
| `scores` | Array of per-fold scores |
| `mean_score` | Mean CV score |
| `std_score` | Standard deviation of CV scores |
| `fold_results` | DataFrame — one row per fold, columns per metric |
| `learning_curve_df` | Learning curve data (only when `learning_curve=True`) |

**Strategy dispatch:**

| `strategy` | Backend | Notes |
|---|---|---|
| `"kfold"` | `KFold` | Standard k-fold |
| `"stratified"` | `StratifiedKFold` | Preserves class balance per fold — default |
| `"timeseries"` | `TimeSeriesSplit` | No shuffling; respects temporal order |

**Learning curve (`learning_curve=True`):**

Returns a `learning_curve_df` DataFrame with columns `train_size`, `train_score_mean`, `train_score_std`, `val_score_mean`, `val_score_std`. This is the primary diagnostic for spotting underfitting (both curves low) vs. overfitting (train high, val low) vs. good generalization (both high and converging).

`train_sizes` defaults to `[0.1, 0.2, 0.4, 0.6, 0.8, 1.0]` if not specified.

**Design notes:**

- `"stratified"` is the default — it's almost always the right choice for classification
- `"timeseries"` must not shuffle; warn if `shuffle=True` is passed alongside it
- Verbose mode prints mean ± std per fold and flags if std > 0.05 (high variance)
- Edge cases: warn if `n_splits` > number of samples in the minority class

**Example:**

```python
from ramentruck import soft_boiled_egg

result = soft_boiled_egg(
    my_model,
    X, y,
    strategy="stratified",
    n_splits=5,
    scoring=["accuracy", "f1"],
    learning_curve=True,
)

print(f"{result.mean_score:.3f} ± {result.std_score:.3f}")
print(result.learning_curve_df)
```

---

### Core: `chashu` — Model Persistence

**File:** `src/ramentruck/chashu.py`

Named for the slow-cooked, perfectly preserved pork topping. `chashu` handles model serialization and persistence — wrapping joblib (or pickle) with metadata versioning so you always know what you saved, when, and with what params.

**Planned signatures:**

```python
def save(
    model,
    path: str | Path,
    *,
    metadata: dict | None = None,
    overwrite: bool = False,
) -> Path

def load(
    path: str | Path,
    *,
    verify: bool = True,
) -> ChashuBundle

def list_models(directory: str | Path) -> pd.DataFrame
```

**`ChashuBundle`** — the loaded artifact:

| Field | Description |
|---|---|
| `model` | The deserialized model object |
| `metadata` | Dict of saved metadata |
| `saved_at` | ISO timestamp of when the model was saved |
| `ramentruck_version` | `ramentruck.__version__` at save time |
| `python_version` | Python version at save time |
| `sklearn_version` | scikit-learn version at save time |

**Storage format:**

Each saved model is a directory (or a single `.chashu` bundle file — TBD at implementation) containing:

```
my_model.chashu/
├── model.joblib     # the serialized model
└── meta.json        # metadata, versions, timestamps
```

**`list_models(directory)`** — scans a directory and returns a DataFrame summarizing all saved `.chashu` bundles: path, saved_at, model class name, and any user-supplied metadata fields.

**Design notes:**

- Always saves `ramentruck_version`, `python_version`, and `sklearn_version` automatically — no manual tracking required
- `overwrite=False` by default — raises if the path already exists to prevent silent clobbers
- `verify=True` on load — checks that the saved sklearn version matches the current environment and warns on mismatch
- Avoid `pickle` directly; prefer `joblib` for large numpy arrays (faster, more memory-efficient)
- User-supplied `metadata` dict is merged with automatic fields; user keys must not collide with reserved names

**Example:**

```python
from ramentruck import chashu

# Save
chashu.save(
    fitted_model,
    "models/rf_v1.chashu",
    metadata={"dataset": "prices_2025", "val_auc": 0.93},
)

# Load
bundle = chashu.load("models/rf_v1.chashu")
model = bundle.model
print(bundle.saved_at)        # "2026-06-17T14:32:01"
print(bundle.metadata)        # {"dataset": "prices_2025", "val_auc": 0.93, ...}

# Inventory
df = chashu.list_models("models/")
```

---

### Explainability: `nori` — Model Explainability

**File:** `src/ramentruck/nori.py`  
**Requires:** `pip install ramentruck[explain]` (pulls in `shap`)

Named for the thin sheet of nori on top of the bowl — a thin layer that adds a distinct layer of insight to whatever is beneath it. `nori` wraps SHAP to make model explainability approachable without hiding the mechanics.

**Planned signatures:**

```python
def explain(
    model,
    X: pd.DataFrame | np.ndarray,
    *,
    method: str = "auto",        # "auto", "tree", "linear", "kernel", "deep"
    sample_n: int | None = 200,  # subsample X for kernel/deep methods
    plot: bool = True,
    plot_type: str = "summary",  # "summary", "bar", "waterfall", "beeswarm"
    verbose: bool = True,
) -> NoriResult

def feature_importance(
    model,
    X: pd.DataFrame | np.ndarray,
    *,
    top_n: int | None = 20,
    normalize: bool = True,
) -> pd.DataFrame
```

**`NoriResult`** — result container:

| Field | Description |
|---|---|
| `shap_values` | Raw SHAP values array or list (for multi-class) |
| `expected_value` | SHAP base value(s) |
| `importance_df` | DataFrame of mean absolute SHAP values per feature, sorted descending |
| `explainer` | The underlying SHAP `Explainer` object |

**`method` auto-selection logic:**

| `method` | Condition |
|---|---|
| `"tree"` | Model has `feature_importances_` or is a tree-based estimator |
| `"linear"` | Model has `coef_` (linear models) |
| `"deep"` | Model is a Keras neural network |
| `"kernel"` | Fallback for any other model (slowest — subsamples by default) |

**Plot types (when `plot=True`):**

| `plot_type` | Output |
|---|---|
| `"summary"` | SHAP summary dot plot (all features, all observations) |
| `"bar"` | Mean absolute SHAP bar chart |
| `"waterfall"` | Single-observation waterfall chart (first sample by default) |
| `"beeswarm"` | Beeswarm plot — best for distribution of impact per feature |

**Design notes:**

- `sample_n=200` default for `"kernel"` and `"deep"` methods — SHAP kernel explainer is O(N²) and unusable on large datasets without sampling
- `feature_importance()` is a simpler entry point that returns a clean DataFrame without requiring knowledge of SHAP internals
- For tree models, `TreeExplainer` is exact (not sampled); `sample_n` is ignored
- Raise a clear `ImportError` with `pip install ramentruck[explain]` if `shap` is not installed

**Example:**

```python
from ramentruck import nori

result = nori.explain(
    fitted_rf,
    X_val,
    method="tree",
    plot=True,
    plot_type="summary",
)

print(result.importance_df.head(10))

imp = nori.feature_importance(fitted_rf, X_val, top_n=10)
```

---

### Experiment Tracking: `miso` — Experiment Tracking

**File:** `src/ramentruck/miso.py`  
**Requires:** `pip install ramentruck[tracking]` (pulls in `mlflow`)

Named for fermented miso paste — the result of accumulated wisdom over many runs. `miso` is the experiment tracking wrapper. Logs params, metrics, artifacts, and model objects to MLflow (default) or Weights & Biases (when available).

**Planned signatures:**

```python
# Context manager (recommended)
with miso.run(experiment="my_experiment", run_name="rf_v1") as run:
    run.log_params({"n_estimators": 100, "max_depth": 5})
    run.log_metrics({"accuracy": 0.87, "f1": 0.85})
    run.log_model(fitted_model, artifact_name="model")
    run.log_dataframe(X_val, artifact_name="validation_set")

# Convenience wrapper — integrates with broth/tare results
def log_broth(result: BrothResult, experiment: str, run_name: str) -> str
def log_tare(result: TareResult,  experiment: str, run_name: str) -> str
```

**`MisoRun`** — context manager object with:

| Method | Description |
|---|---|
| `.log_params(dict)` | Log hyperparameters |
| `.log_metrics(dict)` | Log scalar metrics |
| `.log_model(model, artifact_name)` | Serialize and log a model artifact |
| `.log_dataframe(df, artifact_name)` | Log a DataFrame as a CSV artifact |
| `.log_figure(fig, artifact_name)` | Log a matplotlib Figure |
| `.run_id` | The MLflow run ID |
| `.experiment_id` | The MLflow experiment ID |

**`log_broth()` and `log_tare()`** — one-line convenience wrappers that take a `BrothResult` or `TareResult` and log all fields automatically without boilerplate.

**Backend selection:**

```python
import ramentruck.miso as miso

miso.set_backend("mlflow")   # default
miso.set_backend("wandb")    # requires wandb installed
```

**Design notes:**

- Context manager pattern is preferred — it guarantees the run is closed (even on exceptions)
- MLflow is the default backend; W&B is optional (raise a clear `ImportError` if not installed)
- `log_broth` / `log_tare` are sugar for the most common pattern — integrate tightly with `broth` and `tare` result objects
- Raise a clear `ImportError` with `pip install ramentruck[tracking]` if `mlflow` is not installed

**Example:**

```python
from ramentruck import miso, broth

result = broth(my_model, X_train, y_train, X_val, y_val)

miso.log_broth(result, experiment="price_classifier", run_name="rf_baseline")

# Or manually with full control
with miso.run(experiment="price_classifier", run_name="rf_v2") as run:
    run.log_params({"n_estimators": 200})
    run.log_metrics({"val_auc": 0.93})
    run.log_model(result.model, "random_forest")
```

---

### Deep Learning: `tonkotsu` — Deep Learning Interface

**File:** `src/ramentruck/tonkotsu.py`  
**Requires:** `pip install ramentruck[deep]` (pulls in `tensorflow`)

Named for the heavy, rich, long-cooked pork-bone broth that takes all day and rewards patience. `tonkotsu` is the deep learning interface — wrapping TensorFlow/Keras for common architectures with consistent training loops, regularization hooks, and callback support.

**Primary design decision (pre-implementation):** Default to **Keras functional API**, not Sequential. The functional API is more flexible, handles multi-input/multi-output architectures, and is what serious practitioners use.

**Backend decision (locked in 2026-07-19):** TensorFlow/Keras, not PyTorch. The extras table previously left this TBD, but every planned signature already used `keras.Model` and Keras callback types, so the ambiguity was resolved in favor of what the spec already committed to.

---

#### Design Philosophy: Composable Blocks → One-Call Presets

`tonkotsu` is built in two layers so it serves two different users:

1. **Composable blocks** — low-level, reusable functional-API pieces (a residual block, an attention layer) that someone who *does* understand the architecture can wire together themselves, the same way `keras.layers` gives you `Conv2D` or `LSTM` as primitives.
2. **One-call presets** — higher-level builder functions (`build_resnet`, `build_lstm_seq2seq`, `build_rnn_generator`) that assemble the composable blocks into a working, compiled `keras.Model` with sensible defaults. A user who doesn't want to learn how ResNets or attention mechanisms work can call a preset bare and get a working model back; a user who wants to tweak filter counts, stage depth, or latent dimensions can override individual keyword arguments instead of rewriting the architecture.

Defaults for each preset are sourced from working reference implementations (e.g., the stage/filter progression of a standard ResNet50, typical LSTM latent-dim sizing for sequence tasks) rather than picked arbitrarily.

**Build order:** foundation (`build_dense`, `simmer`, `plot_history`, `EveryNEpochs`) first, then the **CNN family** (residual blocks → `build_resnet`), then the **RNN/sequence family** (RNN/LSTM/GRU cells → attention → `build_lstm_seq2seq` / `build_rnn_generator`). Each preset is layered on top of blocks implemented earlier, so later presets are cheaper once the block library exists.

---

#### Architecture Builders — Foundation

**Planned signatures:**

```python
# Dense feedforward network (most common starting point)
def build_dense(
    input_dim: int,
    hidden_layers: list[int],          # e.g. [128, 64, 32]
    output_dim: int,
    *,
    activation: str = "relu",
    output_activation: str = "sigmoid",  # "sigmoid" for binary, "softmax" for multi-class
    dropout_rate: float = 0.0,
    l2_lambda: float = 0.0,
    batch_norm: bool = False,
) -> keras.Model

# 1D convolutional model (time-series / sequence data)
def build_conv1d(
    input_shape: tuple[int, int],      # (timesteps, features)
    filters: list[int],
    kernel_sizes: list[int],
    dense_layers: list[int],
    output_dim: int,
    *,
    dropout_rate: float = 0.0,
    l2_lambda: float = 0.0,
) -> keras.Model
```

Both builders use the **functional API** internally. `build_dense` is the go-to entry point for tabular data (which is the ThaiTruck → RamenTruck pipeline target).

---

#### CNN Family — Composable Blocks

**Planned signatures:**

```python
def residual_identity_block(
    x,
    kernel_size: int,
    filters: tuple[int, int, int],
    stage: int,
    block: str,
) -> tf.Tensor
    # Standard ResNet identity block: three convolutions + a shortcut
    # connection back to the block input. Output shape matches input shape.

def residual_conv_block(
    x,
    kernel_size: int,
    filters: tuple[int, int, int],
    stage: int,
    block: str,
    *,
    strides: int = 2,
) -> tf.Tensor
    # ResNet convolutional block: three convolutions plus a projected
    # shortcut, used when the block needs to change the tensor shape
    # (downsampling between stages).
```

Both are functional-API building blocks — they take and return a tensor, so they compose into any CNN, not just `build_resnet`.

#### CNN Family — One-Call Preset

**Planned signature:**

```python
def build_resnet(
    input_shape: tuple[int, int, int],
    classes: int,
    *,
    stage_filters: list[tuple[int, int, int]] = [
        (64, 64, 256), (128, 128, 512), (256, 256, 1024), (512, 512, 2048),
    ],
    blocks_per_stage: list[int] = [3, 4, 6, 3],   # ResNet50 defaults
    output_activation: str = "softmax",
    optimizer: str = "rmsprop",
    loss: str = "categorical_crossentropy",
) -> keras.Model
```

Calling `build_resnet(input_shape=(64, 64, 3), classes=6)` with no other arguments returns a compiled ResNet50-equivalent model — the user never has to know what a residual block is. `stage_filters` and `blocks_per_stage` are overridable for anyone who wants a shallower/deeper variant.

---

#### RNN / Sequence Family — Composable Blocks (planned, after CNN family)

Low-level building blocks mirroring RNN/LSTM/GRU cell mechanics and a Bahdanau-style `AttentionLayer`, to be designed once the CNN family is implemented and the blocks → presets pattern is validated end to end.

#### RNN / Sequence Family — One-Call Presets (planned, after CNN family)

`build_rnn_generator(vocab_size, ...)` and `build_lstm_seq2seq(source_vocab_size, target_vocab_size, ...)` — one-call presets over the sequence blocks above, analogous to `build_resnet`. Signatures to be finalized when this family is scheduled.

---

#### Training Loop

**Planned signature:**

```python
def simmer(
    model: keras.Model,
    X_train, y_train,
    X_val=None, y_val=None,
    *,
    epochs: int = 100,
    batch_size: int = 32,
    optimizer: str | keras.Optimizer = "adam",
    loss: str = "binary_crossentropy",
    metrics: list[str] = ["accuracy"],
    early_stopping: bool = True,
    patience: int = 10,
    checkpoint_path: str | Path | None = None,
    callbacks: list | None = None,
    verbose: int = 1,
) -> SipResult
```

`simmer` is the training entry point — named because great broth simmers; it is not rushed.

**`SipResult`** — result container:

| Field | Description |
|---|---|
| `model` | The trained Keras model |
| `history` | Keras `History` object (loss and metric curves per epoch) |
| `history_df` | `history.history` as a DataFrame (train and val columns per metric) |
| `best_epoch` | Epoch at which best val metric occurred (when early stopping used) |
| `stopped_early` | Boolean — did early stopping fire? |
| `train_time_s` | Wall-clock training time |

**Default callbacks wired in automatically:**

| Callback | Default Condition |
|---|---|
| `EarlyStopping` | When `early_stopping=True` — monitors `val_loss`, `restore_best_weights=True` |
| `ModelCheckpoint` | When `checkpoint_path` is provided — saves best weights only |

**`EveryNEpochs` callback** — a built-in convenience callback:

```python
from ramentruck.tonkotsu import EveryNEpochs

cb = EveryNEpochs(n=10, fn=lambda epoch, logs: print(f"Epoch {epoch}: {logs}"))
```

Fires a user-supplied function every N epochs. Useful for custom logging, metric collection, or live plot updates without writing a full `Callback` subclass.

---

#### Regularization Utilities

```python
def l2_penalty(lambda_: float) -> keras.regularizers.L2
    # Thin wrapper — creates the regularizer and applies consistent naming

def dropout_schedule(
    initial_rate: float,
    final_rate: float,
    n_layers: int,
) -> list[float]
    # Returns a linearly interpolated dropout rate per layer
    # e.g. dropout_schedule(0.1, 0.4, 4) → [0.1, 0.2, 0.3, 0.4]
```

---

#### Learning Curve Visualization

```python
def plot_history(
    result: SipResult,
    metrics: list[str] | None = None,   # defaults to all metrics in history
    figsize: tuple = (12, 4),
) -> matplotlib.figure.Figure
```

Plots train vs. validation curves for each metric in the history. The primary visual tool for diagnosing:

- **Underfitting** — both train and val loss are high and flat
- **Overfitting** — train loss falls, val loss diverges or plateaus
- **Good generalization** — both curves fall and converge

Returns a `matplotlib.Figure` so it can be saved, displayed inline, or logged via `miso`.

---

**Full example (binary classification, tabular data):**

```python
from ramentruck.tonkotsu import build_dense, simmer, plot_history

model = build_dense(
    input_dim=X_train.shape[1],
    hidden_layers=[128, 64, 32],
    output_dim=1,
    activation="relu",
    output_activation="sigmoid",
    dropout_rate=0.3,
    l2_lambda=0.001,
)

model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

result = simmer(
    model,
    X_train, y_train,
    X_val, y_val,
    epochs=200,
    batch_size=32,
    early_stopping=True,
    patience=15,
    checkpoint_path="checkpoints/best_model.keras",
)

print(f"Stopped at epoch {result.best_epoch}")
fig = plot_history(result)
fig.savefig("learning_curves.png")
```

---

### To Be Named

The following modules are planned but have no final names yet:

| Concept | Description |
|---|---|
| **Feature engineering pipeline** | Automated or recipe-driven feature construction: polynomial features, interaction terms, target encoding, binning, and datetime feature extraction. Complementary to ThaiTruck's `orange_chicken` (which cleans) — this one creates new features. |
| **Ensemble methods module** | Wrappers for stacking, blending, and voting ensembles. Takes multiple fitted models and combines their predictions. Should support both classification and regression. |
| **Probability calibration module** | Wraps `CalibratedClassifierCV` and related methods. Binary classifiers often output poorly calibrated probabilities — this module fixes that. Includes calibration curve visualization. |
| **Prediction / inference API wrapper** | A thin serving layer that wraps a fitted model for prediction with input validation, schema checking, and optional `chashu`-based loading. The "output window" of the RamenTruck pipeline. |

Names should follow the ramen theme and fit the metaphor (e.g., noodles = structural foundation, toppings = additions on top of the base, etc.).

---

## Design Principles

### Edge Cases Are First-Class

The following should be handled gracefully and explicitly — not ignored, not crashed on:

- **Small N** — fewer than ~50 samples in train. `broth` should warn; `soft_boiled_egg` should handle gracefully without crashing on `n_splits > minority_class_size`
- **Class imbalance** — when minority class < 10%, `broth` logs a warning. `soft_boiled_egg` defaults to `StratifiedKFold` to preserve balance per fold
- **Noisy labels** — no automatic detection (it's hard), but `nori` SHAP plots help identify mislabeled or anomalous samples via unexpectedly high SHAP magnitudes
- **Overfit diagnosis** — `soft_boiled_egg` with `learning_curve=True` and `tonkotsu`'s `plot_history` are the primary tools

### No Mutation

Following ThaiTruck's convention: nothing mutates its inputs. `broth`, `tare`, and `soft_boiled_egg` all return result containers rather than modifying passed-in model objects.

### Consistent Result Containers

Every module returns a typed result object (dataclass or namedtuple) rather than bare tuples or dicts. This makes results inspectable, IDE-friendly, and easy to pass to `miso` for logging.

### sklearn Compatibility

Every model-accepting function works with any sklearn-compatible estimator (implements `.fit()` / `.predict()`). This includes sklearn's own estimators, XGBoost, LightGBM, CatBoost, and any custom estimator that follows the sklearn interface.

---

## Composing with the Fleet

RamenTruck, ThaiTruck, and SushiTruck are **fully independent packages**. RamenTruck has no dependency on ThaiTruck or SushiTruck and imports nothing from them. It accepts `pd.DataFrame` and numpy arrays — whatever the caller passes in. Where those DataFrames came from is not RamenTruck's concern.

The composition happens at the **application layer** in the user's code:

```python
# This is USER code — not RamenTruck internals.
# Each package is imported independently; none calls another.

import pandas as pd
from thaitruck import fried_rice, orange_chicken, larb
from ramentruck import broth, soft_boiled_egg, nori, chashu
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# --- ThaiTruck produces a clean DataFrame ---
raw = orange_chicken(raw_df, heat=3)
merged = fried_rice(raw, earnings_df, freq="D")
profile = larb(merged, heat=3)  # spot-check

# --- RamenTruck receives DataFrames / arrays ---
X = merged.drop(columns=["target"])
y = merged["target"]
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

result = broth(
    RandomForestClassifier(),
    X_train, y_train,
    X_val, y_val,
    metrics=["accuracy", "roc_auc"],
)

cv = soft_boiled_egg(result.model, X, y, strategy="stratified", learning_curve=True)

nori.explain(result.model, X_val, method="tree", plot=True)

chashu.save(result.model, "models/rf_v1.chashu", metadata={"val_auc": result.metrics["roc_auc"]})
```

---

## The Food Truck Fleet

| Package | Status | Focus |
|---|---|---|
| **thaitruck** | Live on PyPI (v0.2.2) | Batch DataFrame cleaning, merging, profiling, caching |
| **sushitruck** | PyPI name secured | Streaming ingestion, API connectors |
| **ramentruck** | PyPI name secured (v0.4.0 early development) | ML/AI toolkit - dataset inspection, classical training, and deep learning (dense + CNN) now; tuning, validation, persistence, explainability, and tracking planned |
| **bentotruck** | Planned | Statistical analysis, feature engineering, and predictive analytics |

Each package is fully independent — none imports from another. They compose at the application layer through `pd.DataFrame` and numpy arrays. SushiTruck produces them. ThaiTruck transforms them. RamenTruck models them. The user's code is the only thing that knows about all three.

---

## Build & Publish Plan

### Before Next Feature Release

1. Implement remaining core modules: `tare`, `soft_boiled_egg`, `chashu`
2. Add tests for each new module
3. Declare optional extras in `pyproject.toml` (`explain`, `tracking`, `deep`, `all`)
4. Add import guards in `nori.py`, `miso.py`, `tonkotsu.py`
5. Add `py.typed` marker (PEP 561) for type-checker support
6. Add `CHANGELOG.md`
7. Add `.github/workflows/tests.yml` - pytest on Python 3.9/3.10/3.11/3.12

### Build commands

```bash
python -m build          # dist/*.whl + dist/*.tar.gz
twine check dist/*       # verify metadata
twine upload dist/*      # publish to PyPI
```

### Version bump (two places)

- `pyproject.toml` → `version = "x.y.z"`
- `src/ramentruck/__init__.py` → `__version__ = "x.y.z"`

---

*Last updated: 2026-07-09*


