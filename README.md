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
and **`Broth`** for classical model training.

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

---

# Modules

| Module | Purpose |
|---------|---------|
| **noodles** | Dataset inspection and preprocessing (`slurp`, scaling, encoding, missing values, dataset splitting) |
| **broth** | Model training and evaluation (`Broth`, `BrothResult`) |
| **tare** | Hyperparameter tuning |
| **soft_boiled_egg** | Cross-validation and learning curves |
| **chashu** | Model persistence and version management |
| **nori** | Explainability (SHAP, feature importance, partial dependence) |
| **miso** | Experiment tracking (MLflow / Weights & Biases) |
| **tonkotsu** | Deep learning (TensorFlow / Keras architectures) |

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
```

Optional extras:

```bash
pip install ramentruck[deep]
pip install ramentruck[explain]
pip install ramentruck[tracking]
pip install ramentruck[all]
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

**Version:** 0.3.0

Current functionality includes:

- Dataset inspection with `slurp()`
- Classical model training with `Broth`
- Shared `DatasetMenu`, `ChefRecommendation`, and `BrothResult` objects
- Intelligent preprocessing recommendations
- Comprehensive unit testing for implemented modules
- Hyperparameter tuning, cross-validation, persistence, and optional modules in active development

---

## License

MIT License

---

**RamenTruck** is an open-source project built with the philosophy that
elegant APIs, reproducible workflows, and thoughtful engineering should
be available to every machine learning practitioner.
