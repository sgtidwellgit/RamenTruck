"""Multi-label text classification for RamenTruck.

Named after the drive-through window: raw text comes in and is
classified into zero, one, or more requested categories on the way
out. The label vocabulary is entirely consumer-defined - this module
has no built-in notion of what any label means, and no notion of AI
agents, routing, or orchestration. A customer-support consumer might
configure ``["BILLING", "SALES", "TECH_SUPPORT", "CANCELLATION"]``; a
different consumer might configure something else entirely.

Multi-class vs. multi-label
----------------------------
This module deliberately does not use softmax. Softmax normalizes
probabilities across labels to sum to 1, which is only correct when
labels are mutually exclusive. Real text requests are often not: a
single query can require zero, one, or several labels at once. Every
backend here therefore produces independent per-label probabilities
(a sigmoid output layer for the neural backend, one binary classifier
per label for the linear backend), and each label is thresholded on
its own.

Abstention
----------
When no label clears its threshold, :meth:`MultiLabelTextClassifier`
returns an empty label list with ``abstained=True`` rather than
guessing a default label. This module has no opinion about what a
caller should do next - fallback behavior belongs to the consuming
application, not to the classifier.

Data leakage
------------
Fit the text vectorizer on training text only; never on validation or
test text. Tune thresholds with :meth:`MultiLabelTextClassifier.tune_thresholds`
against validation data only. Call :meth:`MultiLabelTextClassifier.evaluate`
against a held-out test set exactly once - repeated evaluation against
the same "held-out" data turns it into a second validation set.
:func:`check_for_leakage` helps verify that protected/held-out text
supplied by the caller has not leaked into a training or validation
corpus.

Linear vs. neural
------------------
The linear backend (independent per-label logistic regression over
TF-IDF features) is a legitimate baseline, not a placeholder. A dense
neural backend is not automatically superior - whether it measurably
outperforms the linear baseline is an empirical question the two
backends exist to let you answer, not an assumption this module makes
for you.
"""

from __future__ import annotations

import json
import platform
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Sequence

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, hamming_loss, precision_recall_fscore_support
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import MultiLabelBinarizer

DEFAULT_THRESHOLD = 0.5
DEFAULT_THRESHOLD_CANDIDATES = tuple(round(v, 2) for v in np.arange(0.05, 0.96, 0.05))

BUNDLE_FILES = (
    "labels.json",
    "thresholds.json",
    "vectorizer_config.json",
    "backend_config.json",
    "meta.json",
)


# ----------------------------------------------------------------------
# Text representation: a real, pluggable abstraction
# ----------------------------------------------------------------------


class TextVectorizer(ABC):
    """Base class for pluggable text-to-vector encoders.

    Implementations turn a batch of raw strings into a fixed-width
    numeric matrix. Configuration (constructor arguments) and fitted
    state are kept separate so a bundle can persist and reproduce
    both independently - see :meth:`get_config` and :meth:`save`.
    """

    @abstractmethod
    def fit(self, texts: Sequence[str]) -> "TextVectorizer":
        """Fit the vectorizer on a batch of training text, in place."""

    @abstractmethod
    def transform(self, texts: Sequence[str]) -> np.ndarray:
        """Transform a batch of text into a numeric feature matrix."""

    def fit_transform(self, texts: Sequence[str]) -> np.ndarray:
        """Fit on ``texts``, then transform them. A convenience alias."""

        return self.fit(texts).transform(texts)

    @property
    @abstractmethod
    def output_dim(self) -> int:
        """Dimensionality of each transformed vector."""

    @abstractmethod
    def get_config(self) -> dict[str, Any]:
        """Return this vectorizer's JSON-serializable constructor configuration.

        Must include a ``"type"`` key matching its :data:`VECTORIZER_REGISTRY`
        entry. Must not include fitted state - see :meth:`save`.
        """

    def save(self, directory: Path) -> None:
        """Persist fitted state (if any) into an already-created directory."""

        return None

    @classmethod
    def load(cls, directory: Path, config: dict[str, Any]) -> "TextVectorizer":
        """Reconstruct a fitted vectorizer from a directory and its config.

        The default implementation constructs a fresh instance from
        ``config`` alone, for stateless vectorizers with nothing to
        restore beyond configuration. Stateful vectorizers (e.g.
        :class:`TfidfTextVectorizer`) override this.
        """

        params = {key: value for key, value in config.items() if key != "type"}
        return cls(**params)


class TfidfTextVectorizer(TextVectorizer):
    """TF-IDF bag-of-words vectorizer - the lightweight, dependency-free default.

    Wraps :class:`sklearn.feature_extraction.text.TfidfVectorizer`, which
    is already part of RamenTruck's core dependencies, so this backend
    requires no optional extra.
    """

    STATE_FILENAME = "tfidf_state.joblib"

    def __init__(
        self,
        *,
        max_features: int | None = 20000,
        ngram_range: tuple[int, int] = (1, 1),
        min_df: int | float = 1,
        lowercase: bool = True,
    ) -> None:
        """
        Initialize an unfitted TF-IDF vectorizer.

        Parameters
        ----------
        max_features
            Maximum vocabulary size, keeping the highest document-frequency
            terms. ``None`` keeps every term.
        ngram_range
            Inclusive (min_n, max_n) range of n-gram sizes to extract.
        min_df
            Minimum document frequency (count or proportion) a term must
            meet to be kept in the vocabulary.
        lowercase
            Whether to lowercase text before tokenizing.
        """

        self.max_features = max_features
        self.ngram_range = tuple(ngram_range)
        self.min_df = min_df
        self.lowercase = lowercase
        self._vectorizer = TfidfVectorizer(
            max_features=self.max_features,
            ngram_range=self.ngram_range,
            min_df=self.min_df,
            lowercase=self.lowercase,
        )
        self._is_fitted = False

    def fit(self, texts: Sequence[str]) -> "TfidfTextVectorizer":
        """Fit the TF-IDF vocabulary and IDF weights on training text."""

        self._vectorizer.fit(texts)
        self._is_fitted = True
        return self

    def transform(self, texts: Sequence[str]) -> np.ndarray:
        """Transform text into a dense TF-IDF feature matrix."""

        self._require_fitted()
        return self._vectorizer.transform(texts).toarray().astype("float32")

    @property
    def output_dim(self) -> int:
        """Size of the fitted vocabulary."""

        self._require_fitted()
        return len(self._vectorizer.vocabulary_)

    def get_config(self) -> dict[str, Any]:
        """Return constructor configuration (not the fitted vocabulary)."""

        return {
            "type": "tfidf",
            "max_features": self.max_features,
            "ngram_range": list(self.ngram_range),
            "min_df": self.min_df,
            "lowercase": self.lowercase,
        }

    def save(self, directory: Path) -> None:
        """Persist the fitted vocabulary and IDF weights via joblib."""

        self._require_fitted()
        joblib.dump(self._vectorizer, Path(directory) / self.STATE_FILENAME)

    @classmethod
    def load(cls, directory: Path, config: dict[str, Any]) -> "TfidfTextVectorizer":
        """Reconstruct a fitted TF-IDF vectorizer from a saved bundle."""

        params = {key: value for key, value in config.items() if key != "type"}
        params["ngram_range"] = tuple(params["ngram_range"])
        instance = cls(**params)
        instance._vectorizer = joblib.load(Path(directory) / cls.STATE_FILENAME)
        instance._is_fitted = True
        return instance

    def _require_fitted(self) -> None:
        """Raise when used before fitting."""

        if not self._is_fitted:
            raise RuntimeError("TfidfTextVectorizer must be fitted before use.")


class SentenceEmbeddingTextVectorizer(TextVectorizer):
    """Frozen pretrained sentence-embedding encoder.

    Requires sentence-transformers, an optional dependency. Install
    with ``pip install ramentruck[nlp]``. The underlying encoder is
    pretrained and frozen: ``fit()`` trains nothing and exists only to
    satisfy the :class:`TextVectorizer` interface. The model is loaded
    lazily on first use so importing this class (or this module) never
    requires sentence-transformers to be installed.
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        """
        Initialize the encoder configuration.

        Parameters
        ----------
        model_name
            A sentence-transformers model identifier. Not hard-coded
            anywhere else in RamenTruck - persisted as metadata so a
            reloaded bundle re-instantiates the exact same model.
        """

        self.model_name = model_name
        self._model: Any = None

    def fit(self, texts: Sequence[str]) -> "SentenceEmbeddingTextVectorizer":
        """No-op: the encoder is pretrained and frozen. Loads the model lazily."""

        self._ensure_model_loaded()
        return self

    def transform(self, texts: Sequence[str]) -> np.ndarray:
        """Encode text into dense sentence embeddings."""

        model = self._ensure_model_loaded()
        embeddings = model.encode(list(texts), convert_to_numpy=True)
        return np.asarray(embeddings, dtype="float32")

    @property
    def output_dim(self) -> int:
        """Embedding dimensionality reported by the underlying model."""

        model = self._ensure_model_loaded()
        return int(model.get_sentence_embedding_dimension())

    def get_config(self) -> dict[str, Any]:
        """Return the model identifier needed to reproduce this encoder."""

        return {"type": "sentence_embedding", "model_name": self.model_name}

    def _ensure_model_loaded(self) -> Any:
        """Import sentence-transformers and load the model on first use."""

        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as exc:
                raise ImportError(
                    "SentenceEmbeddingTextVectorizer requires sentence-transformers. "
                    "Install it with: pip install ramentruck[nlp]"
                ) from exc

            self._model = SentenceTransformer(self.model_name)

        return self._model


VECTORIZER_REGISTRY: dict[str, Callable[..., TextVectorizer]] = {
    "tfidf": TfidfTextVectorizer,
    "sentence_embedding": SentenceEmbeddingTextVectorizer,
}


def _resolve_vectorizer(vectorizer: "TextVectorizer | str") -> TextVectorizer:
    """Resolve a vectorizer name or instance into a TextVectorizer instance."""

    if isinstance(vectorizer, str):
        factory = VECTORIZER_REGISTRY.get(vectorizer)
        if factory is None:
            supported = ", ".join(sorted(VECTORIZER_REGISTRY))
            raise ValueError(
                f"Unsupported vectorizer: {vectorizer!r}. Supported vectorizers are: {supported}."
            )
        return factory()

    return vectorizer


# ----------------------------------------------------------------------
# Classifier backends: pluggable, none of them privileged
# ----------------------------------------------------------------------


class ClassifierBackend(ABC):
    """Base class for pluggable multi-label classifier backends.

    Implementations consume a fixed-width feature matrix and an
    ``n_labels``-wide multi-hot indicator matrix, and must produce
    independent per-label probabilities - never a softmax-normalized
    distribution.
    """

    @abstractmethod
    def fit(
        self,
        X_train: np.ndarray,
        Y_train: np.ndarray,
        X_val: np.ndarray | None = None,
        Y_val: np.ndarray | None = None,
        *,
        random_state: int | None = None,
        verbose: bool = True,
    ) -> "ClassifierBackend":
        """Fit the backend on training data, in place."""

    @abstractmethod
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return independent per-label probabilities, shape (n_samples, n_labels)."""

    @abstractmethod
    def get_config(self) -> dict[str, Any]:
        """Return this backend's JSON-serializable constructor configuration."""

    def save(self, directory: Path) -> None:
        """Persist fitted state (if any) into an already-created directory."""

        return None

    @classmethod
    def load(cls, directory: Path, config: dict[str, Any]) -> "ClassifierBackend":
        """Reconstruct a fitted backend from a directory and its config."""

        params = {key: value for key, value in config.items() if key != "type"}
        return cls(**params)


class LinearMultiLabelBackend(ClassifierBackend):
    """Independent per-label logistic regression - a legitimate baseline.

    Wraps :class:`sklearn.multiclass.OneVsRestClassifier`, which fits one
    binary logistic regression per label and reports independent
    probabilities rather than a softmax-normalized distribution. This
    backend exists to let "text classification works" be distinguished
    from "a neural network measurably helps" - it is not a placeholder.
    """

    STATE_FILENAME = "linear_state.joblib"

    def __init__(
        self,
        *,
        C: float = 1.0,
        max_iter: int = 1000,
        class_weight: str | dict | None = None,
    ) -> None:
        """
        Initialize an unfitted linear backend.

        Parameters
        ----------
        C
            Inverse regularization strength, passed to each label's
            :class:`~sklearn.linear_model.LogisticRegression`.
        max_iter
            Maximum solver iterations.
        class_weight
            Optional class weighting strategy, passed through to each
            label's logistic regression (e.g. ``"balanced"``).
        """

        self.C = C
        self.max_iter = max_iter
        self.class_weight = class_weight
        self._model: OneVsRestClassifier | None = None

    def fit(
        self,
        X_train: np.ndarray,
        Y_train: np.ndarray,
        X_val: np.ndarray | None = None,
        Y_val: np.ndarray | None = None,
        *,
        random_state: int | None = None,
        verbose: bool = True,
    ) -> "LinearMultiLabelBackend":
        """Fit one logistic regression per label. X_val/Y_val are unused (no early stopping)."""

        base_estimator = LogisticRegression(
            C=self.C,
            max_iter=self.max_iter,
            class_weight=self.class_weight,
            random_state=random_state,
        )
        self._model = OneVsRestClassifier(base_estimator)
        self._model.fit(X_train, Y_train)

        if verbose:
            print(
                f"drivethrough: linear backend fit on {len(X_train)} examples, "
                f"{Y_train.shape[1]} labels"
            )

        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return each label's independent predicted probability."""

        self._require_fitted()
        return np.asarray(self._model.predict_proba(X))

    def get_config(self) -> dict[str, Any]:
        """Return constructor configuration."""

        return {
            "type": "linear",
            "C": self.C,
            "max_iter": self.max_iter,
            "class_weight": self.class_weight,
        }

    def save(self, directory: Path) -> None:
        """Persist the fitted OneVsRestClassifier via joblib."""

        self._require_fitted()
        joblib.dump(self._model, Path(directory) / self.STATE_FILENAME)

    @classmethod
    def load(cls, directory: Path, config: dict[str, Any]) -> "LinearMultiLabelBackend":
        """Reconstruct a fitted linear backend from a saved bundle."""

        params = {key: value for key, value in config.items() if key != "type"}
        instance = cls(**params)
        instance._model = joblib.load(Path(directory) / cls.STATE_FILENAME)
        return instance

    def _require_fitted(self) -> None:
        """Raise when used before fitting."""

        if self._model is None:
            raise RuntimeError("LinearMultiLabelBackend must be fitted before use.")


class DenseNeuralMultiLabelBackend(ClassifierBackend):
    """Dense feedforward multi-label classifier built on :mod:`ramentruck.tonkotsu`.

    Requires TensorFlow, an optional dependency. Install with
    ``pip install ramentruck[deep]``. TensorFlow and ``tonkotsu`` are
    imported lazily (inside :meth:`fit` / :meth:`load`) so that using
    the TF-IDF+linear path never requires TensorFlow to be installed.

    Reuses :func:`ramentruck.tonkotsu.build_dense` (sigmoid output by
    default) and :func:`ramentruck.tonkotsu.simmer` (binary
    cross-entropy by default) unmodified - independent per-label
    probabilities fall out of that combination without any
    multi-label-specific neural-network code in this module.
    """

    MODEL_FILENAME = "model.keras"

    def __init__(
        self,
        *,
        hidden_layers: Sequence[int] = (64,),
        dropout_rate: float = 0.0,
        l2_lambda: float = 0.0,
        batch_norm: bool = False,
        epochs: int = 100,
        batch_size: int = 32,
        patience: int = 10,
        verbose_training: int = 0,
    ) -> None:
        """
        Initialize an unfitted neural backend.

        Parameters
        ----------
        hidden_layers
            Hidden layer widths, passed to :func:`tonkotsu.build_dense`.
        dropout_rate, l2_lambda, batch_norm
            Regularization options, passed to :func:`tonkotsu.build_dense`.
        epochs, batch_size, patience
            Training options, passed to :func:`tonkotsu.simmer`. Early
            stopping is enabled automatically when validation data is
            supplied to :meth:`fit`, disabled otherwise.
        verbose_training
            Keras verbosity level passed to ``model.fit`` inside
            :func:`tonkotsu.simmer`. Separate from this backend's own
            ``verbose`` argument, which only controls a one-line summary.
        """

        self.hidden_layers = tuple(hidden_layers)
        self.dropout_rate = dropout_rate
        self.l2_lambda = l2_lambda
        self.batch_norm = batch_norm
        self.epochs = epochs
        self.batch_size = batch_size
        self.patience = patience
        self.verbose_training = verbose_training
        self._model: Any = None

    def fit(
        self,
        X_train: np.ndarray,
        Y_train: np.ndarray,
        X_val: np.ndarray | None = None,
        Y_val: np.ndarray | None = None,
        *,
        random_state: int | None = None,
        verbose: bool = True,
    ) -> "DenseNeuralMultiLabelBackend":
        """Build and train a sigmoid-output dense network via tonkotsu."""

        from . import tonkotsu

        input_dim = X_train.shape[1]
        output_dim = Y_train.shape[1]

        self._model = tonkotsu.build_dense(
            input_dim=input_dim,
            hidden_layers=self.hidden_layers,
            output_dim=output_dim,
            dropout_rate=self.dropout_rate,
            l2_lambda=self.l2_lambda,
            batch_norm=self.batch_norm,
            random_state=random_state,
        )

        has_validation = X_val is not None and Y_val is not None
        result = tonkotsu.simmer(
            self._model,
            X_train,
            Y_train,
            X_val if has_validation else None,
            Y_val if has_validation else None,
            epochs=self.epochs,
            batch_size=self.batch_size,
            early_stopping=has_validation,
            patience=self.patience,
            verbose=self.verbose_training,
        )

        if verbose:
            print(
                f"drivethrough: neural backend trained {len(result.history_df)} epoch(s) "
                f"(stopped_early={result.stopped_early}) in {result.train_time_s:.2f}s"
            )

        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return each label's independent predicted probability."""

        self._require_fitted()
        return np.asarray(self._model.predict(X, verbose=0))

    def get_config(self) -> dict[str, Any]:
        """Return constructor configuration (architecture + training options)."""

        return {
            "type": "neural",
            "hidden_layers": list(self.hidden_layers),
            "dropout_rate": self.dropout_rate,
            "l2_lambda": self.l2_lambda,
            "batch_norm": self.batch_norm,
            "epochs": self.epochs,
            "batch_size": self.batch_size,
            "patience": self.patience,
            "verbose_training": self.verbose_training,
        }

    def save(self, directory: Path) -> None:
        """Persist the Keras model using its native save format (not joblib)."""

        self._require_fitted()
        self._model.save(Path(directory) / self.MODEL_FILENAME)

    @classmethod
    def load(cls, directory: Path, config: dict[str, Any]) -> "DenseNeuralMultiLabelBackend":
        """Reconstruct a fitted neural backend from a saved bundle."""

        from tensorflow import keras

        params = {key: value for key, value in config.items() if key != "type"}
        instance = cls(**params)
        instance._model = keras.models.load_model(Path(directory) / cls.MODEL_FILENAME)
        return instance

    def _require_fitted(self) -> None:
        """Raise when used before fitting."""

        if self._model is None:
            raise RuntimeError("DenseNeuralMultiLabelBackend must be fitted before use.")


BACKEND_REGISTRY: dict[str, Callable[..., ClassifierBackend]] = {
    "linear": LinearMultiLabelBackend,
    "neural": DenseNeuralMultiLabelBackend,
}


def _resolve_backend(backend: "ClassifierBackend | str") -> ClassifierBackend:
    """Resolve a backend name or instance into a ClassifierBackend instance."""

    if isinstance(backend, str):
        factory = BACKEND_REGISTRY.get(backend)
        if factory is None:
            supported = ", ".join(sorted(BACKEND_REGISTRY))
            raise ValueError(
                f"Unsupported backend: {backend!r}. Supported backends are: {supported}."
            )
        return factory()

    return backend


# ----------------------------------------------------------------------
# Result dataclasses
# ----------------------------------------------------------------------


@dataclass(frozen=True)
class PredictionTiming:
    """Wall-clock timing for a single prediction.

    For a batch call, this is :class:`BatchTiming` divided evenly across
    the batch - it approximates per-example cost rather than measuring
    each example in isolation. Consult :class:`BatchTiming` on the
    surrounding :class:`BatchPredictionResult` for the authoritative
    aggregate and throughput figures.
    """

    vectorize_s: float
    inference_s: float
    total_s: float


@dataclass(frozen=True)
class PredictionResult:
    """Structured multi-label prediction for a single text input.

    ``scores`` always contains every configured label, in the
    classifier's label order, regardless of whether a label crossed
    its threshold - raw scores are never dropped. When ``labels`` is
    empty, ``abstained`` is ``True``: no label cleared its threshold,
    and this classifier does not guess a fallback on the caller's
    behalf.
    """

    labels: list[str]
    scores: dict[str, float]
    abstained: bool
    timing: PredictionTiming


@dataclass(frozen=True)
class BatchTiming:
    """Aggregate wall-clock timing and throughput for a batch prediction."""

    vectorize_s: float
    inference_s: float
    total_s: float
    throughput_per_s: float


@dataclass(frozen=True)
class BatchPredictionResult:
    """Structured multi-label predictions for a batch of text inputs."""

    results: list[PredictionResult]
    timing: BatchTiming


@dataclass(frozen=True)
class ThresholdTuningResult:
    """Result returned by :meth:`MultiLabelTextClassifier.tune_thresholds`."""

    thresholds: dict[str, float]
    metric: str
    per_label_scores: dict[str, float]


@dataclass(frozen=True)
class EvaluationReport:
    """Multi-label evaluation metrics and failure analysis.

    ``per_label`` has one row per label with ``precision``, ``recall``,
    ``f1``, and ``support`` columns. ``failures`` has one row per
    evaluated example with its true/predicted labels, false
    positives/negatives, over-/under-selection flags, and whether the
    classifier abstained on it.
    """

    n_examples: int
    n_abstained: int
    subset_accuracy: float
    hamming_loss: float
    micro_precision: float
    micro_recall: float
    micro_f1: float
    macro_precision: float
    macro_recall: float
    macro_f1: float
    per_label: pd.DataFrame
    failures: pd.DataFrame


@dataclass(frozen=True)
class LeakageMatch:
    """A single match found between a protected text and a corpus text."""

    protected_index: int
    corpus_index: int
    match_type: str
    protected_text: str
    corpus_text: str


@dataclass(frozen=True)
class LeakageReport:
    """Result returned by :func:`check_for_leakage`."""

    matches: list[LeakageMatch]
    n_exact: int
    n_normalized: int

    @property
    def has_leakage(self) -> bool:
        """Whether any exact or normalized match was found."""

        return bool(self.matches)


# ----------------------------------------------------------------------
# Leakage checking
# ----------------------------------------------------------------------


def _default_normalize(text: str) -> str:
    """Lowercase, strip, and collapse internal whitespace."""

    return " ".join(text.strip().lower().split())


def check_for_leakage(
    protected_texts: Sequence[str],
    corpus_texts: Sequence[str],
    *,
    normalize: Callable[[str], str] | None = _default_normalize,
) -> LeakageReport:
    """
    Check whether protected/held-out text appears in a training or validation corpus.

    This function has no built-in notion of what "protected" means -
    the caller supplies whatever text must not leak (e.g. a frozen
    evaluation benchmark) and whatever corpus to check it against.

    Parameters
    ----------
    protected_texts
        Text that must not leak into training/validation.
    corpus_texts
        Text to check for leakage, e.g. a training or validation corpus.
    normalize
        Callable applied to both sides for the normalized-duplicate
        check, catching superficial differences like casing and
        whitespace. Defaults to lowercase + collapsed whitespace. Pass
        a different callable to change matching behavior (e.g. also
        stripping punctuation, or - in the future - a semantic
        near-duplicate check), or ``None`` to skip the normalized
        check and only look for exact matches.

    Returns
    -------
    LeakageReport
        Every exact and normalized match found, plus counts. This is
        intentionally not exhaustive semantic near-duplicate detection
        (e.g. paraphrases) - only exact and normalized-text matches.
    """

    matches: list[LeakageMatch] = []
    corpus_index_by_text: dict[str, int] = {}
    for idx, text in enumerate(corpus_texts):
        corpus_index_by_text.setdefault(text, idx)

    for protected_idx, protected_text in enumerate(protected_texts):
        corpus_idx = corpus_index_by_text.get(protected_text)
        if corpus_idx is not None:
            matches.append(
                LeakageMatch(
                    protected_index=protected_idx,
                    corpus_index=corpus_idx,
                    match_type="exact",
                    protected_text=protected_text,
                    corpus_text=corpus_texts[corpus_idx],
                )
            )

    n_exact = len(matches)

    if normalize is not None:
        already_flagged = {(match.protected_index, match.corpus_index) for match in matches}
        normalized_corpus_index: dict[str, int] = {}
        for idx, text in enumerate(corpus_texts):
            normalized_corpus_index.setdefault(normalize(text), idx)

        for protected_idx, protected_text in enumerate(protected_texts):
            corpus_idx = normalized_corpus_index.get(normalize(protected_text))
            if corpus_idx is None or (protected_idx, corpus_idx) in already_flagged:
                continue
            matches.append(
                LeakageMatch(
                    protected_index=protected_idx,
                    corpus_index=corpus_idx,
                    match_type="normalized",
                    protected_text=protected_text,
                    corpus_text=corpus_texts[corpus_idx],
                )
            )

    n_normalized = len(matches) - n_exact

    return LeakageReport(matches=matches, n_exact=n_exact, n_normalized=n_normalized)


# ----------------------------------------------------------------------
# Threshold-tuning metric functions (selection, not calibration)
# ----------------------------------------------------------------------


def _binary_score(metric: str, y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute a single-label precision/recall/f1 score for threshold sweeps."""

    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", zero_division=0
    )
    return _select_metric(metric, precision, recall, f1)


def _multilabel_score(metric: str, Y_true: np.ndarray, Y_pred: np.ndarray) -> float:
    """Compute a micro-averaged multi-label precision/recall/f1 score for threshold sweeps."""

    precision, recall, f1, _ = precision_recall_fscore_support(
        Y_true, Y_pred, average="micro", zero_division=0
    )
    return _select_metric(metric, precision, recall, f1)


def _select_metric(metric: str, precision: float, recall: float, f1: float) -> float:
    """Pick one of precision/recall/f1 by name."""

    values = {"precision": precision, "recall": recall, "f1": f1}
    if metric not in values:
        supported = ", ".join(sorted(values))
        raise ValueError(f"Unsupported metric: {metric!r}. Supported metrics are: {supported}.")
    return float(values[metric])


# ----------------------------------------------------------------------
# Label validation
# ----------------------------------------------------------------------


def _validate_labels(labels: Sequence[str]) -> tuple[str, ...]:
    """Validate and freeze the explicit, ordered label vocabulary."""

    labels = tuple(labels)

    if len(labels) == 0:
        raise ValueError("labels must not be empty.")

    if len(set(labels)) != len(labels):
        raise ValueError("labels must not contain duplicates.")

    return labels


def _validate_label_sets(label_sets: Sequence[Sequence[str]], known_labels: tuple[str, ...]) -> None:
    """Validate that every example's labels are drawn from the known vocabulary."""

    known = set(known_labels)

    for index, label_set in enumerate(label_sets):
        unknown = set(label_set) - known
        if unknown:
            raise ValueError(
                f"example {index} contains unknown label(s) {sorted(unknown)}; "
                f"known labels are {list(known_labels)}."
            )


def _default_thresholds(labels: tuple[str, ...], threshold: float) -> dict[str, float]:
    """Build a global threshold dict, one entry per label."""

    return {label: float(threshold) for label in labels}


def _now_iso() -> str:
    """Current UTC time as an ISO-8601 string."""

    return datetime.now(timezone.utc).isoformat()


# ----------------------------------------------------------------------
# MultiLabelTextClassifier
# ----------------------------------------------------------------------


class MultiLabelTextClassifier:
    """
    Multi-label text classifier: independent per-label probabilities, not a softmax distribution.

    Consumers supply an explicit, ordered label vocabulary; this class
    has no built-in notion of what the labels mean. A single input may
    match zero, one, or several labels. When no label clears its
    threshold, the classifier abstains (see :class:`PredictionResult`)
    rather than guessing a default label - fallback behavior is the
    consuming application's responsibility.

    See the module docstring for the multi-class-vs-multi-label
    rationale, the abstention contract, and data-leakage guidance.
    """

    def __init__(
        self,
        labels: Sequence[str],
        *,
        vectorizer: TextVectorizer | str = "tfidf",
        backend: ClassifierBackend | str = "linear",
        threshold: float = DEFAULT_THRESHOLD,
        random_state: int | None = 42,
    ) -> None:
        """
        Configure an unfitted multi-label text classifier.

        Parameters
        ----------
        labels
            Explicit, ordered label vocabulary. This order fixes target
            encoding, output-neuron layout (neural backend), score
            dictionary key order, the threshold dictionary, and every
            persisted artifact. It is never inferred from a particular
            training split.
        vectorizer
            A :class:`TextVectorizer` instance, or one of ``"tfidf"``
            (default; no extra dependency) / ``"sentence_embedding"``
            (requires ``pip install ramentruck[nlp]``).
        backend
            A :class:`ClassifierBackend` instance, or one of ``"linear"``
            (default; independent logistic regression per label - a
            legitimate baseline) / ``"neural"`` (dense feedforward
            network via :mod:`tonkotsu`, requires
            ``pip install ramentruck[deep]``).
        threshold
            Global decision threshold applied to every label until
            :meth:`tune_thresholds` overrides some or all of them.
        random_state
            Seed threaded into the backend's training for reproducibility.
        """

        self.labels = _validate_labels(labels)
        self.vectorizer = _resolve_vectorizer(vectorizer)
        self.backend = _resolve_backend(backend)
        self.random_state = random_state
        self.thresholds: dict[str, float] = _default_thresholds(self.labels, threshold)

        self._label_binarizer = MultiLabelBinarizer(classes=list(self.labels))
        self._label_binarizer.fit([])
        self._is_fitted = False
        self._n_train: int | None = None
        self._n_val: int | None = None

    def fit(
        self,
        texts_train: Sequence[str],
        labels_train: Sequence[Sequence[str]],
        texts_val: Sequence[str] | None = None,
        labels_val: Sequence[Sequence[str]] | None = None,
        *,
        verbose: bool = True,
    ) -> "MultiLabelTextClassifier":
        """
        Fit the text vectorizer and classifier backend on training data.

        The vectorizer is fit exclusively on ``texts_train`` - there is
        no parameter through which validation or test text can reach
        vectorizer fitting, by design. Pass ``texts_val``/``labels_val``
        to enable early stopping (neural backend); they are transformed
        with the already-fitted vectorizer, never used to fit it.

        Parameters
        ----------
        texts_train
            Training text.
        labels_train
            One label sequence per training example, drawn from
            ``self.labels``. An empty sequence is a valid "no label
            applies" training example.
        texts_val
            Optional validation text, for early stopping. Must be
            provided together with ``labels_val`` or not at all.
        labels_val
            Optional validation labels.
        verbose
            Whether the backend prints a one-line training summary.

        Returns
        -------
        MultiLabelTextClassifier
            ``self``, fitted.

        Raises
        ------
        ValueError
            If text/label lengths mismatch, an unknown label appears,
            or only one of ``texts_val``/``labels_val`` is given.
        """

        if len(texts_train) != len(labels_train):
            raise ValueError("texts_train and labels_train must have the same length.")

        _validate_label_sets(labels_train, self.labels)

        X_train = self.vectorizer.fit_transform(texts_train)
        Y_train = self._label_binarizer.transform(labels_train)

        X_val = Y_val = None
        if self._has_validation_data(texts_val, labels_val):
            if len(texts_val) != len(labels_val):
                raise ValueError("texts_val and labels_val must have the same length.")
            _validate_label_sets(labels_val, self.labels)
            X_val = self.vectorizer.transform(texts_val)
            Y_val = self._label_binarizer.transform(labels_val)

        self.backend.fit(
            X_train,
            Y_train,
            X_val,
            Y_val,
            random_state=self.random_state,
            verbose=verbose,
        )

        self._is_fitted = True
        self._n_train = len(texts_train)
        self._n_val = len(texts_val) if X_val is not None else 0

        return self

    def predict_one(self, text: str) -> PredictionResult:
        """Predict labels for a single text input."""

        return self.predict_batch([text]).results[0]

    def predict_batch(self, texts: Sequence[str]) -> BatchPredictionResult:
        """
        Predict labels for a batch of text inputs.

        Every returned :class:`PredictionResult` carries scores for
        every configured label, regardless of threshold outcome, plus
        timing. No free-form text is ever produced - the return value
        is entirely structured data.
        """

        self._require_fitted()

        start = time.perf_counter()
        X = self.vectorizer.transform(texts)
        vectorize_s = time.perf_counter() - start

        inference_start = time.perf_counter()
        probabilities = self.backend.predict_proba(X)
        inference_s = time.perf_counter() - inference_start

        total_s = time.perf_counter() - start
        n = len(texts)

        batch_timing = BatchTiming(
            vectorize_s=vectorize_s,
            inference_s=inference_s,
            total_s=total_s,
            throughput_per_s=(n / total_s) if total_s > 0 and n > 0 else 0.0,
        )
        per_example_timing = PredictionTiming(
            vectorize_s=vectorize_s / n if n else 0.0,
            inference_s=inference_s / n if n else 0.0,
            total_s=total_s / n if n else 0.0,
        )

        results = [
            self._to_prediction_result(probabilities[i], per_example_timing) for i in range(n)
        ]

        return BatchPredictionResult(results=results, timing=batch_timing)

    def tune_thresholds(
        self,
        texts_val: Sequence[str],
        labels_val: Sequence[Sequence[str]],
        *,
        metric: str = "f1",
        per_label: bool = True,
        thresholds_to_try: Sequence[float] | None = None,
        verbose: bool = True,
    ) -> ThresholdTuningResult:
        """
        Tune decision thresholds against VALIDATION data only.

        This is threshold *selection* (choosing the cut point applied
        to already-computed probabilities), not probability
        calibration (see :mod:`ramentruck.kaeshi` for recalibrating the
        probabilities themselves) - the scores are unchanged, only the
        decision boundary moves.

        Never call this with test data: doing so contaminates a
        held-out test evaluation exactly as fitting the vectorizer on
        test data would. A small validation set can also make the
        chosen thresholds noisy - a handful of examples can shift the
        best-F1 threshold considerably - so prefer the default global
        threshold over aggressively per-label-tuned thresholds when
        validation data is scarce.

        Parameters
        ----------
        texts_val, labels_val
            Validation data. Never pass test data here.
        metric
            Objective to maximize: ``"f1"`` (default), ``"precision"``,
            or ``"recall"``.
        per_label
            Whether to tune an independent threshold per label
            (default) or a single global threshold shared by every
            label.
        thresholds_to_try
            Candidate thresholds to sweep. Defaults to 0.05 through
            0.95 in steps of 0.05.
        verbose
            Whether to print the tuned thresholds.

        Returns
        -------
        ThresholdTuningResult
            The thresholds selected (also applied to ``self.thresholds``),
            the metric used, and each label's achieved score.
        """

        self._require_fitted()
        _validate_label_sets(labels_val, self.labels)

        candidates = (
            list(thresholds_to_try) if thresholds_to_try is not None else list(DEFAULT_THRESHOLD_CANDIDATES)
        )

        batch = self.predict_batch(texts_val)
        scores_matrix = np.array(
            [[result.scores[label] for label in self.labels] for result in batch.results]
        )
        Y_true = self._label_binarizer.transform(labels_val)

        per_label_scores: dict[str, float] = {}

        if per_label:
            new_thresholds = dict(self.thresholds)
            for label_index, label in enumerate(self.labels):
                best_threshold, best_score = self._sweep_thresholds(
                    candidates,
                    lambda candidate, i=label_index: _binary_score(
                        metric,
                        Y_true[:, i],
                        (scores_matrix[:, i] >= candidate).astype(int),
                    ),
                )
                new_thresholds[label] = best_threshold
                per_label_scores[label] = best_score
        else:
            best_threshold, best_score = self._sweep_thresholds(
                candidates,
                lambda candidate: _multilabel_score(
                    metric, Y_true, (scores_matrix >= candidate).astype(int)
                ),
            )
            new_thresholds = {label: best_threshold for label in self.labels}
            per_label_scores = {label: best_score for label in self.labels}

        self.thresholds = new_thresholds

        if verbose:
            print(f"drivethrough: tuned thresholds ({metric}): {new_thresholds}")

        return ThresholdTuningResult(
            thresholds=dict(new_thresholds), metric=metric, per_label_scores=per_label_scores
        )

    @staticmethod
    def _sweep_thresholds(
        candidates: list[float], score_fn: Callable[[float], float]
    ) -> tuple[float, float]:
        """Return the candidate threshold maximizing score_fn, ties broken by first-seen."""

        best_threshold, best_score = candidates[0], -1.0
        for candidate in candidates:
            score = score_fn(candidate)
            if score > best_score:
                best_threshold, best_score = candidate, score
        return float(best_threshold), float(best_score)

    def evaluate(
        self, texts: Sequence[str], labels_true: Sequence[Sequence[str]]
    ) -> EvaluationReport:
        """
        Evaluate the classifier's current thresholds against labeled text.

        This method makes no assumption about which split it is called
        on - that discipline is the caller's responsibility. Call it
        against a held-out test set exactly once; using its output to
        further tune thresholds or preprocessing turns "test" into a
        second validation set.

        Returns
        -------
        EvaluationReport
            Subset accuracy, Hamming loss, micro/macro precision,
            recall, and F1, per-label precision/recall/F1/support, and
            a per-example failure-analysis DataFrame.
        """

        self._require_fitted()
        _validate_label_sets(labels_true, self.labels)

        batch = self.predict_batch(texts)
        Y_true = self._label_binarizer.transform(labels_true)
        Y_pred = np.array(
            [
                [1 if label in result.labels else 0 for label in self.labels]
                for result in batch.results
            ]
        )

        micro_precision, micro_recall, micro_f1, _ = precision_recall_fscore_support(
            Y_true, Y_pred, average="micro", zero_division=0
        )
        macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
            Y_true, Y_pred, average="macro", zero_division=0
        )
        label_precision, label_recall, label_f1, label_support = precision_recall_fscore_support(
            Y_true, Y_pred, average=None, zero_division=0
        )

        per_label = pd.DataFrame(
            {
                "label": self.labels,
                "precision": label_precision,
                "recall": label_recall,
                "f1": label_f1,
                "support": label_support,
            }
        )

        failures = self._build_failure_frame(texts, Y_true, batch.results)
        n_abstained = sum(1 for result in batch.results if result.abstained)

        return EvaluationReport(
            n_examples=len(texts),
            n_abstained=n_abstained,
            subset_accuracy=float(accuracy_score(Y_true, Y_pred)),
            hamming_loss=float(hamming_loss(Y_true, Y_pred)),
            micro_precision=float(micro_precision),
            micro_recall=float(micro_recall),
            micro_f1=float(micro_f1),
            macro_precision=float(macro_precision),
            macro_recall=float(macro_recall),
            macro_f1=float(macro_f1),
            per_label=per_label,
            failures=failures,
        )

    def _build_failure_frame(
        self, texts: Sequence[str], Y_true: np.ndarray, results: list[PredictionResult]
    ) -> pd.DataFrame:
        """Build a per-example failure-analysis DataFrame."""

        rows = []
        for index, text in enumerate(texts):
            true_labels = [label for j, label in enumerate(self.labels) if Y_true[index, j]]
            predicted_labels = results[index].labels
            false_positives = sorted(set(predicted_labels) - set(true_labels))
            false_negatives = sorted(set(true_labels) - set(predicted_labels))

            rows.append(
                {
                    "text": text,
                    "true_labels": true_labels,
                    "predicted_labels": predicted_labels,
                    "false_positives": false_positives,
                    "false_negatives": false_negatives,
                    "over_selected": len(predicted_labels) > len(true_labels),
                    "under_selected": len(predicted_labels) < len(true_labels),
                    "abstained": results[index].abstained,
                    "exact_match": set(predicted_labels) == set(true_labels),
                }
            )

        return pd.DataFrame(rows)

    def save(self, path: str | Path, *, overwrite: bool = False) -> Path:
        """
        Persist a complete, portable bundle to a directory.

        Includes the vectorizer's configuration and fitted state, the
        classifier backend's configuration and fitted state (using the
        Keras model's own native save format for the neural backend,
        never ``joblib``), the explicit ordered label vocabulary,
        thresholds, and reproducibility metadata (versions, seed,
        sample counts). A reloaded classifier (:meth:`load`) produces
        equivalent predictions to the classifier before saving.

        Parameters
        ----------
        path
            Directory path for the bundle. Created (including parents)
            if it does not exist.
        overwrite
            Whether to allow writing into a path that already exists.

        Returns
        -------
        Path
            The bundle directory path.

        Raises
        ------
        FileExistsError
            If ``path`` already exists and ``overwrite=False``.
        """

        self._require_fitted()
        path = Path(path)

        if path.exists() and not overwrite:
            raise FileExistsError(f"{path} already exists. Pass overwrite=True to replace it.")

        path.mkdir(parents=True, exist_ok=True)

        (path / "labels.json").write_text(json.dumps(list(self.labels), indent=2))
        (path / "thresholds.json").write_text(json.dumps(self.thresholds, indent=2))

        vectorizer_config = self.vectorizer.get_config()
        (path / "vectorizer_config.json").write_text(json.dumps(vectorizer_config, indent=2))
        self.vectorizer.save(path)

        backend_config = self.backend.get_config()
        (path / "backend_config.json").write_text(json.dumps(backend_config, indent=2))
        self.backend.save(path)

        (path / "meta.json").write_text(json.dumps(self._build_metadata(), indent=2))

        return path

    @classmethod
    def load(cls, path: str | Path) -> "MultiLabelTextClassifier":
        """
        Load a bundle saved by :meth:`save`.

        Raises
        ------
        FileNotFoundError
            If any required bundle file is missing.
        ValueError
            If the bundle references an unknown vectorizer or backend type.
        """

        path = Path(path)
        missing = [name for name in BUNDLE_FILES if not (path / name).exists()]
        if missing:
            raise FileNotFoundError(
                f"No complete drivethrough bundle found at {path}; missing {missing}."
            )

        labels = json.loads((path / "labels.json").read_text())
        thresholds = json.loads((path / "thresholds.json").read_text())
        vectorizer_config = json.loads((path / "vectorizer_config.json").read_text())
        backend_config = json.loads((path / "backend_config.json").read_text())
        meta = json.loads((path / "meta.json").read_text())

        vectorizer_cls = VECTORIZER_REGISTRY.get(vectorizer_config["type"])
        if vectorizer_cls is None:
            raise ValueError(f"Unknown persisted vectorizer type: {vectorizer_config['type']!r}.")
        vectorizer = vectorizer_cls.load(path, vectorizer_config)

        backend_cls = BACKEND_REGISTRY.get(backend_config["type"])
        if backend_cls is None:
            raise ValueError(f"Unknown persisted backend type: {backend_config['type']!r}.")
        backend = backend_cls.load(path, backend_config)

        instance = cls(
            labels,
            vectorizer=vectorizer,
            backend=backend,
            random_state=meta.get("random_state"),
        )
        instance.thresholds = thresholds
        instance._is_fitted = True
        instance._n_train = meta.get("n_train")
        instance._n_val = meta.get("n_val")

        return instance

    def _build_metadata(self) -> dict[str, Any]:
        """Build the reproducibility metadata block persisted alongside the bundle."""

        from . import __version__ as ramentruck_version

        meta: dict[str, Any] = {
            "saved_at": _now_iso(),
            "ramentruck_version": ramentruck_version,
            "python_version": platform.python_version(),
            "sklearn_version": sklearn.__version__,
            "backend_type": self.backend.get_config()["type"],
            "vectorizer_type": self.vectorizer.get_config()["type"],
            "labels": list(self.labels),
            "random_state": self.random_state,
            "n_train": self._n_train,
            "n_val": self._n_val,
        }

        if isinstance(self.backend, DenseNeuralMultiLabelBackend):
            try:
                import tensorflow

                meta["tensorflow_version"] = tensorflow.__version__
            except ImportError:
                pass

        if isinstance(self.vectorizer, SentenceEmbeddingTextVectorizer):
            try:
                import sentence_transformers

                meta["sentence_transformers_version"] = sentence_transformers.__version__
            except ImportError:
                pass

        return meta

    def _to_prediction_result(
        self, scores_row: np.ndarray, timing: PredictionTiming
    ) -> PredictionResult:
        """Build a PredictionResult from one row of predicted probabilities."""

        scores = {label: float(score) for label, score in zip(self.labels, scores_row)}
        predicted = [label for label in self.labels if scores[label] >= self.thresholds[label]]

        return PredictionResult(
            labels=predicted,
            scores=scores,
            abstained=(len(predicted) == 0),
            timing=timing,
        )

    @staticmethod
    def _has_validation_data(
        texts_val: Sequence[str] | None, labels_val: Sequence[Sequence[str]] | None
    ) -> bool:
        """Return whether validation data was provided as a complete pair."""

        if (texts_val is None) != (labels_val is None):
            raise ValueError(
                "texts_val and labels_val must either both be provided or both be None."
            )

        return texts_val is not None and labels_val is not None

    def _require_fitted(self) -> None:
        """Raise when used before fitting or loading."""

        if not self._is_fitted:
            raise RuntimeError("MultiLabelTextClassifier must be fitted before use.")


__all__ = [
    "BACKEND_REGISTRY",
    "VECTORIZER_REGISTRY",
    "BatchPredictionResult",
    "BatchTiming",
    "ClassifierBackend",
    "DenseNeuralMultiLabelBackend",
    "EvaluationReport",
    "LeakageMatch",
    "LeakageReport",
    "LinearMultiLabelBackend",
    "MultiLabelTextClassifier",
    "PredictionResult",
    "PredictionTiming",
    "SentenceEmbeddingTextVectorizer",
    "TextVectorizer",
    "TfidfTextVectorizer",
    "ThresholdTuningResult",
    "check_for_leakage",
]
