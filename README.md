# RamenTruck

**RamenTruck** is a machine learning and deep learning toolkit for
Python, designed to simplify the modeling workflow from dataset
inspection and preprocessing through feature engineering, training,
tuning, evaluation, calibration, ensembling, explainability, experiment
tracking, and serving.

Built as the machine learning companion to **ThaiTruck**, RamenTruck
emphasizes clean APIs, reproducible workflows, and production-ready
engineering practices rather than notebook-only examples.

> Great ramen isn't rushed. Neither is great machine learning.

---

# Table of Contents

- [Why RamenTruck?](#why-ramentruck)
- [Installation](#installation)
- [Quickstart](#quickstart)
- [Module Guide](#module-guide)
  - [noodles — dataset inspection](#noodles--dataset-inspection)
  - [kaedama — feature engineering](#kaedama--feature-engineering)
  - [broth — model training](#broth--model-training)
  - [tare — hyperparameter tuning](#tare--hyperparameter-tuning)
  - [soft_boiled_egg — cross-validation](#soft_boiled_egg--cross-validation)
  - [kaeshi — probability calibration](#kaeshi--probability-calibration)
  - [toppings — ensemble methods](#toppings--ensemble-methods)
  - [diagnostics — shared model-quality rules](#diagnostics--shared-model-quality-rules)
  - [nori — explainability](#nori--explainability)
  - [chashu — model persistence](#chashu--model-persistence)
  - [donburi — prediction and serving](#donburi--prediction-and-serving)
  - [miso — experiment tracking](#miso--experiment-tracking)
  - [tonkotsu — deep learning](#tonkotsu--deep-learning)
- [Module Reference Table](#module-reference-table)
- [Design Principles](#design-principles)
- [The Food Truck Fleet](#the-food-truck-fleet)
- [Current Status](#current-status)
- [License](#license)

---

# Why RamenTruck?

Machine learning projects often require dozens of disconnected libraries
and hundreds of lines of repetitive boilerplate before the first model
is ever trained.

RamenTruck provides a unified, opinionated toolkit that helps you:

- Inspect datasets and receive intelligent preprocessing recommendations
- Engineer new features from raw columns
- Train and evaluate classical machine learning models
- Tune hyperparameters
- Cross-validate and diagnose learning curves
- Calibrate predicted probabilities
- Combine multiple models into ensembles
- Explain model predictions
- Track experiments across runs
- Save, version, and reload trained models
- Serve predictions through a clean, validated API
- Build modern neural network architectures

The goal is to let data scientists spend less time wiring together
infrastructure and more time building better models.

Every module is named after something you'd find in an actual bowl of
ramen. The metaphors aren't just decoration - each one describes what
the module does:

| Ramen term | What it is, literally | What the module does |
|---|---|---|
| noodles | The base ingredient | Inspect and understand your raw dataset |
| kaedama | A refill of noodles dropped into remaining broth | Add engineered feature columns to a dataset |
| broth | The base liquid everything else builds on | Train and evaluate a model |
| tare | The concentrated seasoning that defines the bowl | Tune hyperparameters |
| soft_boiled_egg | A topping that's all about timing | Cross-validate and check learning curves |
| kaeshi | The sauce blended in to balance final flavor | Calibrate predicted probabilities |
| toppings | Many ingredients, combined, better together | Ensemble multiple models |
| nori | A thin layer that adds insight/flavor on top | Explain model predictions |
| chashu | Slow-cooked, preserved, sliced when needed | Persist and version trained models |
| donburi | The bowl the finished dish is served in | Serve predictions through an API |
| miso | Fermented; wisdom accumulated over time | Track experiments across runs |
| tonkotsu | Heavy, rich, long-cooked | Deep learning |

---

# Installation

Core installation:

```bash
pip install ramentruck
```

The core install pulls in `noodles`, `kaedama`, `broth`, `tare`,
`soft_boiled_egg`, `kaeshi`, `toppings`, `diagnostics`, `chashu`, and
`donburi`. These only require:

```text
numpy
pandas
scikit-learn
joblib
```

Three modules wrap heavier, optional libraries and are installed via
extras so you never pay for a dependency you don't use:

```bash
pip install ramentruck[deep]       # tonkotsu: tensorflow, matplotlib
pip install ramentruck[explain]    # nori: shap, matplotlib
pip install ramentruck[tracking]   # miso: mlflow
pip install ramentruck[all]        # everything above
```

Importing an extras-gated module without its dependency installed
raises a clear `ImportError` telling you exactly which extra to install
- there's no silent fallback or degraded behavior.

---

# Quickstart

An end-to-end walkthrough touching most of the toolkit, from a raw
DataFrame to a served prediction:

```python
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

from ramentruck import slurp, Kaedama, Broth, tare, soft_boiled_egg, chashu, Donburi

# 1. Inspect the dataset
menu = slurp(df, target="Purchased")
print(menu)

# 2. Engineer a couple of extra features
kaedama = Kaedama().ratio("income", "age").log_transform("income")
engineered = kaedama.fit_transform(df)

X = engineered.drop(columns=["Purchased"])
y = engineered["Purchased"]
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Tune hyperparameters
tuned = tare(
    RandomForestClassifier(random_state=42),
    param_grid={"n_estimators": [100, 200], "max_depth": [4, 8, None]},
    X=X_train, y=y_train, method="grid", cv=5,
)

# 4. Train and evaluate the best configuration
trainer = Broth(tuned.best_model)
result = trainer.fit(X_train, y_train, X_val, y_val, metrics=["accuracy", "f1", "roc_auc"])
print(result.metrics)

# 5. Cross-validate to sanity-check the split
cv_result = soft_boiled_egg(tuned.best_model, X_train, y_train, learning_curve=True)

# 6. Persist the model
chashu.save(trainer.estimator, "models/rf_v1.chashu", metadata={"feature_names": list(X.columns)})

# 7. Serve predictions
donburi = Donburi.from_chashu("models/rf_v1.chashu")
response = donburi.serve({col: X_val.iloc[0][col] for col in X.columns})
print(response)  # {"prediction": 1, "probabilities": {"0": 0.12, "1": 0.88}}
```

---

# Module Guide

## noodles — dataset inspection

The base ingredient: before anything else, you need to understand what
you're working with. `slurp()` inspects a DataFrame and returns a
`DatasetMenu` with statistics and preprocessing recommendations.

```python
from ramentruck import slurp

menu = slurp(df, target="Purchased")
print(menu)
```

`DatasetMenu` includes:

- Row/column counts and memory usage
- Column type breakdown (numeric, categorical, boolean, datetime)
- Missing value counts and percentages, per column
- Duplicate row count
- A `ChefRecommendation`: inferred problem type (classification vs.
  regression), suggested scaling/encoding, loss function, output
  activation, optimizer, and class imbalance detection

`print(menu)` renders a formatted text report; every field is also
available as a plain attribute (`menu.rows`, `menu.missing_values`,
`menu.chef_recommendation.problem_type`, etc.) for programmatic use.

## kaedama — feature engineering

An extra serving of noodles dropped into broth that's already been
depleted. `Kaedama` is a fluent builder for adding engineered columns
to a DataFrame without touching the original data.

```python
from ramentruck import Kaedama

kaedama = (
    Kaedama()
    .datetime_features("signup_date", features=("year", "month", "dayofweek"))
    .polynomial_features(["income", "age"], degree=2)
    .ratio("income", "household_size", name="income_per_person")
    .log_transform("income")
    .bin("age", bins=[0, 18, 35, 50, 65, 120], labels=["<18", "18-34", "35-49", "50-64", "65+"])
)

engineered = kaedama.fit_transform(df)
print(kaedama.added_columns_)  # every column name that was added, in order
```

Available steps:

| Method | Adds |
|---|---|
| `.datetime_features(cols, features=...)` | Calendar components (`year`, `month`, `day`, `dayofweek`, `hour`, ...) pulled from any `.dt` accessor attribute |
| `.polynomial_features(cols, degree=, interaction_only=)` | Powers and interaction terms via scikit-learn's `PolynomialFeatures` |
| `.ratio(numerator, denominator, name=)` | A safe ratio column (division by zero produces `NaN`, not an error) |
| `.log_transform(cols, offset=)` | Natural-log transformed columns |
| `.bin(col, bins=, labels=, name=)` | A categorical bucketed version of a numeric column |

`fit_transform()` never mutates the input DataFrame - it returns a new
one with the original columns plus everything added by each chained
step. Chain as many steps as you like before calling it once.

## broth — model training

The base liquid everything else is built on. `Broth` wraps any
scikit-learn compatible estimator (implementing `fit`/`predict`) with
consistent training, scoring, and overfitting diagnostics.

```python
from sklearn.ensemble import RandomForestClassifier
from ramentruck import Broth

trainer = Broth(RandomForestClassifier(random_state=42))
result = trainer.fit(
    X_train, y_train,
    X_val, y_val,
    metrics=["accuracy", "f1", "roc_auc"],
)

print(result.train_score, result.val_score, result.metrics, result.fit_time_s)

predictions = trainer.predict(X_val)
score = trainer.score(X_val, y_val, metric="f1")
```

Supported `metrics` names: `accuracy`, `precision`, `recall`, `f1`,
`roc_auc`, `mse`, `rmse`, `r2`. Precision/recall/F1 automatically pick
binary vs. weighted averaging based on the number of classes in `y`.
`roc_auc` supports both binary and multiclass (one-vs-rest) targets.

If validation data is supplied and the training score exceeds the
validation score by more than 0.10, `Broth.fit()` raises a `UserWarning`
so overfitting doesn't go unnoticed.

## tare — hyperparameter tuning

The concentrated seasoning that defines the bowl - small adjustments,
big impact. `tare()` wraps `GridSearchCV`/`RandomizedSearchCV` behind a
single function.

```python
from sklearn.ensemble import GradientBoostingClassifier
from ramentruck import tare

result = tare(
    GradientBoostingClassifier(),
    param_grid={"n_estimators": [50, 100, 200], "learning_rate": [0.01, 0.1, 0.3]},
    X=X_train, y=y_train,
    method="random",   # or "grid"
    n_iter=20,         # only used by method="random"
    cv=5,
    scoring="roc_auc",
)

print(result.best_params, result.best_score)
best_model = result.best_model      # already refit on the full training set
print(result.cv_results.head())     # every candidate, best rank first
print(result.search_time_s)
```

## soft_boiled_egg — cross-validation

A topping that's all about timing and calibration. `soft_boiled_egg()`
cross-validates a model and can compute learning curves in the same
call, surfacing overfitting/underfitting patterns directly.

```python
from ramentruck import soft_boiled_egg

result = soft_boiled_egg(
    my_model,
    X, y,
    strategy="stratified",   # "kfold", "stratified", or "timeseries"
    n_splits=5,
    scoring=["accuracy", "f1"],   # first entry is the primary metric
    learning_curve=True,
)

print(f"{result.mean_score:.3f} +/- {result.std_score:.3f}")
print(result.fold_results)        # per-fold scores for every requested metric
print(result.learning_curve_df)   # train_size, train/val score mean and std
```

`strategy="stratified"` preserves class balance per fold and is the
default. `strategy="timeseries"` never shuffles, respecting temporal
order. A `UserWarning` fires if fold-to-fold variance exceeds 0.05, or
if `n_splits` exceeds the minority class count under stratified
splitting.

## kaeshi — probability calibration

The concentrated sauce blended with dashi to balance a bowl's final
flavor. `kaeshi()` calibrates a binary classifier's predicted
probabilities so they can be trusted at face value (a model with
"90% confidence" should be right about 90% of the time).

```python
from sklearn.svm import SVC
from ramentruck import kaeshi, plot_calibration_curve

result = kaeshi(
    SVC(probability=True),
    X_train, y_train,
    method="isotonic",   # or "sigmoid" (Platt scaling) for smaller datasets
    cv=5,
)

print(result.brier_score_before, result.brier_score_after)
calibrated_model = result.model
fig = plot_calibration_curve(result)   # reliability diagram
```

`method="sigmoid"` is generally more stable on small or noisy datasets;
`method="isotonic"` is more flexible but needs more data to avoid
overfitting the calibration curve itself. Currently supports binary
classification only.

## toppings — ensemble methods

Many ingredients, combined, better together than any single one alone.
`toppings` wraps scikit-learn's voting, stacking, and bagging ensembles
behind a consistent API that also scores each individual member for
comparison.

```python
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from ramentruck import toppings

# Voting: combine independently-trained models
result = toppings.voting(
    {"logreg": LogisticRegression(max_iter=500), "tree": DecisionTreeClassifier()},
    X_train, y_train,
    task="classification",
    voting_type="soft",
)
print(result.train_score, result.individual_scores)

# Stacking: train a meta-model on the base models' out-of-fold predictions
result = toppings.stack(
    {"logreg": LogisticRegression(max_iter=500), "tree": DecisionTreeClassifier()},
    meta_model=LogisticRegression(max_iter=500),
    X=X_train, y=y_train,
    cv=5,
)

# Bagging: many bootstrap-sampled copies of one model
result = toppings.bag(
    DecisionTreeClassifier(),
    X_train, y_train,
    n_estimators=25,
)
```

All three accept `task="classification"` or `task="regression"`, and
optional `X_val`/`y_val` to score against held-out data instead of
training data. Every call returns a `ToppingsResult` with the fitted
ensemble, `train_score`, `val_score`, `individual_scores` (per member),
and `fit_time_s`.

## diagnostics — shared model-quality rules

Deterministic evaluation rules shared across the toolkit: overfitting,
underfitting, small-dataset, high-variance, and class-imbalance
detection, each with concrete recommendations and confidence scores.

```python
from ramentruck import DiagnosticEngine

report = DiagnosticEngine().evaluate(
    train_score=0.96,
    validation_score=0.80,
    dataset_size=80,
    variance=0.07,
    class_imbalance=True,
)

print(report.diagnosis)          # "Mild overfitting detected."
print(report.warnings)           # ("Small dataset detected.", "High validation variance.", ...)
print(report.recommendations)    # tuple of Recommendation(message, category, severity, confidence)
print(report.chef_report())      # formatted text report combining all of the above
```

`DiagnosticEngine` is standalone infrastructure - it isn't automatically
invoked by `Broth` or `soft_boiled_egg`, but is available any time you
want a consistent, rule-based second opinion on a set of scores.

## nori — explainability

*Requires `pip install ramentruck[explain]`.*

A thin layer that adds insight on top. `nori` wraps SHAP to explain
what any model is actually paying attention to, at both the dataset and
individual-prediction level.

```python
from ramentruck import nori

result = nori.explain(model, X_test)   # auto-picks Tree/Linear/model-agnostic SHAP explainer

print(result.importance)                          # feature, mean_abs_shap - sorted
fig = nori.plot_importance(result, max_display=15) # bar chart
fig = nori.plot_summary(result, max_display=15)    # beeswarm-style per-sample plot

# Explain one specific prediction
contributions = nori.explain_instance(result, index=0)
print(contributions)   # per-feature SHAP value, sorted by magnitude

# Partial dependence: how does the prediction change as one feature varies?
pd_result = nori.partial_dependence(model, X_test, features=["age", "income"], grid_resolution=20)
fig = nori.plot_partial_dependence(pd_result)
```

For binary classifiers, `explain()` defaults to explaining the positive
class. Multiclass models require an explicit `class_index` argument -
`nori` raises a clear `ValueError` rather than guessing which class you
meant.

## chashu — model persistence

Slow-cooked, preserved perfectly, sliced when needed. `chashu` wraps
`joblib` with versioning metadata so saved models are traceable months
later.

```python
from ramentruck import chashu

chashu.save(
    trainer.estimator,
    "models/rf_v1.chashu",
    metadata={"val_auc": 0.93, "feature_names": list(X.columns)},
)

bundle = chashu.load("models/rf_v1.chashu")
print(bundle.model, bundle.metadata, bundle.saved_at, bundle.sklearn_version)

catalog = chashu.list_models("models/")   # one row per bundle, with metadata as columns
```

Every bundle automatically records `saved_at`, the RamenTruck version,
Python version, and scikit-learn version at save time. `chashu.load()`
warns (rather than fails) if the current scikit-learn version doesn't
match what the bundle was saved with. `chashu.save()` refuses to
overwrite an existing bundle path unless you pass `overwrite=True`.

## donburi — prediction and serving

The bowl the finished dish is served in. `Donburi` wraps a fitted model
for prediction, adding record-level validation and a JSON-friendly
`serve()` method that plugs directly into a REST endpoint.

```python
from ramentruck import Donburi, chashu

# From an in-memory model:
donburi = Donburi(trainer.estimator, feature_names=list(X.columns))

# Or straight from a saved chashu bundle (feature_names is restored from its metadata):
donburi = Donburi.from_chashu("models/rf_v1.chashu")

donburi.predict(X_val)                              # batch predictions, as a numpy array
donburi.predict_proba(X_val)                         # raises AttributeError if unsupported
donburi.predict_one({"age": 34, "income": 52000})    # single record -> single prediction
donburi.predict_batch([{"age": 34, "income": 52000}, {"age": 61, "income": 88000}])

donburi.serve({"age": 34, "income": 52000})
# {"prediction": 1, "probabilities": {"0": 0.12, "1": 0.88}}
```

When `feature_names` is known (either passed explicitly or restored
from a chashu bundle's metadata), every record-based call validates
that all required features are present and raises a descriptive
`ValueError` naming exactly what's missing - so a malformed request
fails loudly instead of producing a silently wrong prediction.

## miso — experiment tracking

*Requires `pip install ramentruck[tracking]`.*

Fermented - wisdom accumulated over many runs. `miso` wraps MLflow for
experiment tracking that works out of the box with zero setup: runs are
recorded to a local SQLite-backed store (`.miso/tracking.db`) unless you
point it at a shared or remote tracking server.

```python
from ramentruck import miso

with miso.brew("rf_experiment", run_name="rf_v1") as run:
    run.log_params({"n_estimators": 200, "max_depth": 8})
    run.log_metric("accuracy", 0.93)
    run.log_metrics({"f1": 0.91, "roc_auc": 0.97})
    run.log_model(trainer.estimator)
    run.log_artifact("plots/learning_curve.png")

runs = miso.list_runs("rf_experiment")             # every run, most recent first
best = miso.best_run("rf_experiment", "accuracy", mode="max")
print(best.run_id, best.params, best.metrics)
```

Point `tracking_uri` (or the `MLFLOW_TRACKING_URI` environment variable)
at `http://your-mlflow-server:5000` to share runs across a team without
changing any other code.

## tonkotsu — deep learning

*Requires `pip install ramentruck[deep]`.*

Heavy, rich, long-cooked. `tonkotsu` wraps TensorFlow/Keras (functional
API) for building and training neural networks, from a plain dense
network to a full ResNet.

```python
from ramentruck import tonkotsu

# Dense feedforward network
model = tonkotsu.build_dense(
    input_dim=20, hidden_layers=[64, 32], output_dim=1,
    dropout_rate=0.3, l2_lambda=0.001, batch_norm=True,
)

result = tonkotsu.simmer(
    model, X_train, y_train, X_val, y_val,
    epochs=100, early_stopping=True, patience=10,
    checkpoint_path="checkpoints/best.keras",
)

print(result.best_epoch, result.stopped_early, result.train_time_s)
fig = tonkotsu.plot_history(result)

# A custom callback that fires every N epochs
callback = tonkotsu.EveryNEpochs(5, lambda epoch, logs: print(epoch, logs))

# CNN family: one-call ResNet50-equivalent, or compose your own
resnet = tonkotsu.build_resnet(input_shape=(64, 64, 3), classes=6)
result = tonkotsu.simmer(resnet, X_train_img, y_train_img, X_val_img, y_val_img, epochs=50)
```

`build_dense` and `build_resnet` return models built with the Keras
functional API, not `Sequential` - every layer is named, and the
returned model can be freely composed into larger graphs.
`residual_identity_block` and `residual_conv_block` are exposed
individually if you want to assemble a custom architecture rather than
use the `build_resnet` preset. An RNN/sequence family (LSTM, GRU,
attention) is planned next.

---

# Module Reference Table

| Module | Purpose | Extra required |
|---------|---------|---|
| **noodles** | Dataset inspection (`slurp`, `DatasetMenu`, `ChefRecommendation`) | none |
| **kaedama** | Feature engineering (`Kaedama`) | none |
| **broth** | Model training and evaluation (`Broth`, `BrothResult`) | none |
| **tare** | Hyperparameter tuning (`tare`, `TareResult`) | none |
| **soft_boiled_egg** | Cross-validation and learning curves (`soft_boiled_egg`, `EggResult`) | none |
| **kaeshi** | Probability calibration (`kaeshi`, `plot_calibration_curve`, `KaeshiResult`) | none |
| **toppings** | Ensemble methods (`toppings.voting`, `.stack`, `.bag`, `ToppingsResult`) | none |
| **diagnostics** | Shared deterministic diagnostics (`DiagnosticEngine`, `DiagnosticReport`, `Recommendation`) | none |
| **chashu** | Model persistence and versioning (`chashu.save`, `.load`, `.list_models`, `ChashuBundle`) | none |
| **donburi** | Prediction and serving (`Donburi`) | none |
| **nori** | Explainability (`nori.explain`, `.plot_summary`, `.plot_importance`, `.partial_dependence`, `NoriResult`, `PDResult`) | `[explain]` |
| **miso** | Experiment tracking (`miso.brew`, `.list_runs`, `.best_run`, `MisoRunSummary`) | `[tracking]` |
| **tonkotsu** | Deep learning (`build_dense`, `simmer`, `build_resnet`, `EveryNEpochs`, `SipResult`) | `[deep]` |

---

# Design Principles

RamenTruck is built around a few core ideas:

- **Composable modules** - every component can be used independently.
- **Immutable workflows** - functions return new objects rather than modifying inputs.
- **Strong typing** - type hints and dataclasses throughout.
- **Production-first** - built for real applications, not just notebooks.
- **Testing-first** - every public module includes automated unit tests.
- **Explainability matters** - model interpretation is a first-class feature.
- **Classical ML and Deep Learning** - one consistent API across both worlds.
- **Optional extras stay optional** - a missing `[extras]` dependency fails loudly with an install hint, never silently.

---

# The Food Truck Fleet

The Food Truck ecosystem consists of independent Python packages that
work well together while remaining completely decoupled.

| Package | Purpose | Status |
|----------|---------|--------|
| **ThaiTruck** | Data cleaning, transformation, and DataFrame utilities | Available |
| **RamenTruck** | Machine learning and deep learning toolkit | Available |
| **SushiTruck** | Streaming ingestion and API connectors | Planned |
| **BentoTruck** | Statistical analysis, feature engineering, and predictive analytics | Planned |

Each package can be installed independently and composes naturally with
the others through standard pandas DataFrames and NumPy arrays.

---

# Current Status

**Version:** 0.6.0

Every module from the original design is now implemented:

- Dataset inspection with `slurp()`
- Feature engineering with `Kaedama` (datetime features, polynomial/interaction terms, ratios, log transforms, binning)
- Classical model training with `Broth`
- Hyperparameter tuning with `tare` (grid and randomized search)
- Cross-validation and learning curves with `soft_boiled_egg`
- Probability calibration with `kaeshi` (isotonic and sigmoid, with reliability diagrams)
- Ensemble methods with `toppings` (voting, stacking, bagging)
- Shared deterministic diagnostics with `DiagnosticEngine` and `DiagnosticReport`
- Model persistence and versioning with `chashu`
- Prediction and serving with `Donburi` (including a `from_chashu` loader)
- Explainability with `nori` (SHAP-based feature importance, summary plots, instance explanations, partial dependence)
- Experiment tracking with `miso` (local-first, MLflow-backed, zero setup required)
- Deep learning with `tonkotsu`: `build_dense`, `simmer`, `plot_history`, `EveryNEpochs`, and a CNN family (`residual_identity_block`, `residual_conv_block`, `build_resnet`)
- Shared result objects (`DatasetMenu`, `ChefRecommendation`, `BrothResult`, `TareResult`, `EggResult`, `ChashuBundle`, `KaeshiResult`, `ToppingsResult`, `NoriResult`, `PDResult`, `MisoRunSummary`, `SipResult`) throughout
- Comprehensive unit testing across every module

Planned next: an RNN/sequence family for `tonkotsu` (LSTM, GRU,
attention).

---

## License

MIT License

---

**RamenTruck** is an open-source project built with the philosophy that
elegant APIs, reproducible workflows, and thoughtful engineering should
be available to every machine learning practitioner.
