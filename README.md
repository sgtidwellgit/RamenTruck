# RamenTruck

**RamenTruck** is a machine learning and deep learning toolkit for
Python, designed to simplify the modeling workflow from dataset
inspection and preprocessing through training, evaluation,
explainability, and experiment tracking.

Built as the machine learning companion to **ThaiTruck**, RamenTruck
emphasizes clean APIs, reproducible workflows, and production-ready
engineering practices rather than notebook-only examples.

> Great ramen isn't rushed. Neither is great machine learning.

---

# Why RamenTruck?

Machine learning projects often require dozens of disconnected libraries
and hundreds of lines of repetitive boilerplate before the first model
is ever trained.

RamenTruck provides a unified, opinionated toolkit that helps you:

- Inspect datasets and receive intelligent preprocessing recommendations
- Prepare data for machine learning and deep learning workflows
- Train and evaluate classical machine learning models
- Build modern neural network architectures
- Tune hyperparameters
- Track experiments
- Explain model predictions
- Save and version trained models

The goal is to let data scientists spend less time wiring together
infrastructure and more time building better models.

---

# Getting Started

The current public entry points are **`slurp()`** for dataset inspection
and **`Broth`** for classical model training, plus **`tare`**
(hyperparameter tuning), **`soft_boiled_egg`** (cross-validation and
learning curves), and **`chashu`** (model persistence). A shared
**`diagnostics`** engine (`DiagnosticEngine`, `DiagnosticReport`,
`Recommendation`) is also available as reusable infrastructure for
model-quality guidance; it is standalone for now and not yet wired into
`Broth` or other modules.

Deep learning model building and training is available through
**`tonkotsu`** (requires `pip install ramentruck[deep]`): a foundation of
`build_dense`, `simmer`, `plot_history`, and an `EveryNEpochs` callback,
plus a CNN family of composable residual blocks and a one-call
`build_resnet` preset. An RNN/sequence family (LSTM, GRU, attention) is
planned next.

Dataset inspection:

```python
from ramentruck import slurp

menu = slurp(df, target="Purchased")
print(menu)
```

`slurp()` analyzes a dataset and returns a `DatasetMenu` containing:

- Dataset dimensions
- Memory usage
- Missing value analysis
- Duplicate detection
- Column type identification
- Classification vs. regression inference
- Class imbalance detection
- Intelligent preprocessing recommendations

Model training:

```python
from sklearn.ensemble import RandomForestClassifier
from ramentruck import Broth

trainer = Broth(RandomForestClassifier(random_state=42))
result = trainer.fit(
    X_train,
    y_train,
    X_val,
    y_val,
    metrics=["accuracy", "f1", "roc_auc"],
)

predictions = trainer.predict(X_val)
score = trainer.score(X_val, y_val, metric="accuracy")
```

Hyperparameter tuning:

```python
from sklearn.ensemble import GradientBoostingClassifier
from ramentruck import tare

result = tare(
    GradientBoostingClassifier(),
    param_grid={"n_estimators": [50, 100, 200], "learning_rate": [0.01, 0.1, 0.3]},
    X=X_train,
    y=y_train,
    method="random",
    n_iter=20,
    scoring="roc_auc",
)

print(result.best_params, result.best_score)
```

Cross-validation and learning curves:

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

print(f"{result.mean_score:.3f} +/- {result.std_score:.3f}")
print(result.learning_curve_df)
```

Model persistence:

```python
from ramentruck import chashu

chashu.save(trainer.estimator, "models/rf_v1.chashu", metadata={"val_auc": 0.93})
bundle = chashu.load("models/rf_v1.chashu")
df = chashu.list_models("models/")
```

Deterministic diagnostics (standalone, not yet wired into `Broth`):

```python
from ramentruck import DiagnosticEngine

report = DiagnosticEngine().evaluate(
    train_score=0.96,
    validation_score=0.80,
)
print(report.chef_report())
```

Deep learning (requires `pip install ramentruck[deep]`):

```python
from ramentruck import tonkotsu

model = tonkotsu.build_resnet(input_shape=(64, 64, 3), classes=6)
result = tonkotsu.simmer(model, X_train, y_train, X_val, y_val, epochs=50)
fig = tonkotsu.plot_history(result)
```

---

# Modules

| Module | Purpose |
|---------|---------|
| **noodles** | Dataset inspection and preprocessing (`slurp`, scaling, encoding, missing values, dataset splitting) |
| **diagnostics** | Shared deterministic diagnostics (`DiagnosticEngine`, `DiagnosticReport`, `Recommendation`) — standalone, not yet consumed by other modules |
| **broth** | Model training and evaluation (`Broth`, `BrothResult`) |
| **tare** | Hyperparameter tuning (`tare`, `TareResult`; grid and random search) |
| **soft_boiled_egg** | Cross-validation and learning curves (`soft_boiled_egg`, `EggResult`) |
| **chashu** | Model persistence and version management (`chashu.save`, `chashu.load`, `chashu.list_models`, `ChashuBundle`) |
| **nori** | Explainability (SHAP, feature importance, partial dependence) |
| **miso** | Experiment tracking (MLflow / Weights & Biases) |
| **tonkotsu** | Deep learning (`build_dense`, `simmer`, `build_resnet`, and more; TensorFlow / Keras) |

---

# Installation

Core installation:

```bash
pip install ramentruck
```

Current core dependencies:

```text
numpy
pandas
scikit-learn
joblib
```

Optional extras:

```bash
pip install ramentruck[deep]       # tonkotsu: tensorflow, matplotlib
pip install ramentruck[explain]    # nori (planned)
pip install ramentruck[tracking]   # miso (planned)
pip install ramentruck[all]        # everything (planned)
```

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

---

# The Food Truck Fleet

The Food Truck ecosystem consists of independent Python packages that
work well together while remaining completely decoupled.

| Package | Purpose | Status |
|----------|---------|--------|
| **ThaiTruck** | Data cleaning, transformation, and DataFrame utilities | Available |
| **RamenTruck** | Machine learning and deep learning toolkit | In Development |
| **SushiTruck** | Streaming ingestion and API connectors | Planned |
| **BentoTruck** | Statistical analysis, feature engineering, and predictive analytics | Planned |

Each package can be installed independently and composes naturally with
the others through standard pandas DataFrames and NumPy arrays.

---

# Current Status

**Version:** 0.5.0

Current functionality includes:

- Dataset inspection with `slurp()`
- Classical model training with `Broth`
- Hyperparameter tuning with `tare` (grid and randomized search)
- Cross-validation and learning curves with `soft_boiled_egg`
- Model persistence and versioning with `chashu`
- Shared deterministic diagnostics with `DiagnosticEngine` and `DiagnosticReport` (standalone; not yet used by `Broth`)
- Deep learning with `tonkotsu`: `build_dense`, `simmer`, `plot_history`, `EveryNEpochs`, and a CNN family (`residual_identity_block`, `residual_conv_block`, `build_resnet`)
- Shared `DatasetMenu`, `ChefRecommendation`, `BrothResult`, `TareResult`, `EggResult`, `ChashuBundle`, and `SipResult` objects
- Intelligent preprocessing recommendations
- Comprehensive unit testing for implemented modules
- Explainability (`nori`) and experiment tracking (`miso`) remain planned, along with the RNN/sequence half of `tonkotsu`

---

## License

MIT License

---

**RamenTruck** is an open-source project built with the philosophy that
elegant APIs, reproducible workflows, and thoughtful engineering should
be available to every machine learning practitioner.
