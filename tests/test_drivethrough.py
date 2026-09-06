"""Unit tests for ramentruck.drivethrough."""

from __future__ import annotations

import json
import sys

import numpy as np
import pytest

from ramentruck.drivethrough import (
    DenseNeuralMultiLabelBackend,
    LinearMultiLabelBackend,
    MultiLabelTextClassifier,
    PredictionResult,
    SentenceEmbeddingTextVectorizer,
    TfidfTextVectorizer,
    check_for_leakage,
)

LABELS = ["BILLING", "SALES", "TECH_SUPPORT", "CANCELLATION"]


def _toy_corpus():
    """Return a small, deterministic customer-support-style labeled corpus."""

    texts = [
        "I want to cancel my subscription",
        "please cancel my plan",
        "cancel my membership entirely",
        "how much does the premium plan cost",
        "what are your pricing tiers",
        "tell me about upgrading to the pro plan",
        "my app keeps crashing on startup",
        "the login page shows an error",
        "the app freezes when I open settings",
        "I was charged twice this month",
        "I see an unexpected charge on my invoice",
        "why was I billed extra this cycle",
        "cancel my subscription and refund the last charge",
    ]
    labels = [
        ["CANCELLATION"],
        ["CANCELLATION"],
        ["CANCELLATION"],
        ["SALES"],
        ["SALES"],
        ["SALES"],
        ["TECH_SUPPORT"],
        ["TECH_SUPPORT"],
        ["TECH_SUPPORT"],
        ["BILLING"],
        ["BILLING"],
        ["BILLING"],
        ["CANCELLATION", "BILLING"],
    ]
    return texts, labels


def _fitted_linear_classifier(**kwargs):
    """Fit a low-threshold linear classifier on the toy corpus."""

    texts, labels = _toy_corpus()
    clf = MultiLabelTextClassifier(
        LABELS, vectorizer="tfidf", backend="linear", threshold=0.3, random_state=42, **kwargs
    )
    clf.fit(texts, labels, verbose=False)
    return clf


# ----------------------------------------------------------------------
# Label validation
# ----------------------------------------------------------------------


def test_rejects_empty_labels():
    """Verify an empty label vocabulary raises ValueError."""

    with pytest.raises(ValueError, match="must not be empty"):
        MultiLabelTextClassifier([])


def test_rejects_duplicate_labels():
    """Verify duplicate labels in the vocabulary raise ValueError."""

    with pytest.raises(ValueError, match="duplicates"):
        MultiLabelTextClassifier(["A", "B", "A"])


def test_fit_rejects_unknown_label():
    """Verify a training example with an unlisted label raises ValueError."""

    texts, labels = _toy_corpus()
    labels[0] = ["NOT_A_REAL_LABEL"]
    clf = MultiLabelTextClassifier(LABELS)

    with pytest.raises(ValueError, match="unknown label"):
        clf.fit(texts, labels, verbose=False)


def test_fit_rejects_mismatched_lengths():
    """Verify mismatched texts/labels lengths raise ValueError."""

    texts, labels = _toy_corpus()
    clf = MultiLabelTextClassifier(LABELS)

    with pytest.raises(ValueError, match="same length"):
        clf.fit(texts, labels[:-1], verbose=False)


def test_fit_rejects_partial_validation_data():
    """Verify supplying only one of texts_val/labels_val raises ValueError."""

    texts, labels = _toy_corpus()
    clf = MultiLabelTextClassifier(LABELS)

    with pytest.raises(ValueError, match="both be provided or both be None"):
        clf.fit(texts, labels, texts_val=texts[:2], verbose=False)


def test_resolve_vectorizer_rejects_unknown_name():
    """Verify an unknown vectorizer name raises ValueError."""

    with pytest.raises(ValueError, match="Unsupported vectorizer"):
        MultiLabelTextClassifier(LABELS, vectorizer="not_a_real_vectorizer")


def test_resolve_backend_rejects_unknown_name():
    """Verify an unknown backend name raises ValueError."""

    with pytest.raises(ValueError, match="Unsupported backend"):
        MultiLabelTextClassifier(LABELS, backend="not_a_real_backend")


# ----------------------------------------------------------------------
# Label order preservation
# ----------------------------------------------------------------------


def test_label_order_is_preserved_in_scores_and_thresholds():
    """Verify the constructor's label order governs scores and thresholds regardless of training order."""

    reordered = ["TECH_SUPPORT", "CANCELLATION", "BILLING", "SALES"]
    texts, labels = _toy_corpus()
    clf = MultiLabelTextClassifier(reordered, backend="linear", threshold=0.3, random_state=42)
    clf.fit(texts, labels, verbose=False)

    result = clf.predict_one(texts[0])

    assert list(result.scores.keys()) == reordered
    assert list(clf.thresholds.keys()) == reordered


# ----------------------------------------------------------------------
# Prediction: single-label, multi-label, abstention, thresholds
# ----------------------------------------------------------------------


def test_predict_one_returns_single_label():
    """Verify a clearly single-topic query predicts exactly one label."""

    clf = _fitted_linear_classifier()

    result = clf.predict_one("how much does the pro plan cost")

    assert isinstance(result, PredictionResult)
    assert result.labels == ["SALES"]
    assert result.abstained is False
    assert set(result.scores) == set(LABELS)


def test_predict_one_returns_multiple_labels():
    """Verify a query spanning two topics predicts both labels."""

    clf = _fitted_linear_classifier()

    result = clf.predict_one("please cancel my subscription and refund my last charge")

    assert "CANCELLATION" in result.labels
    assert "BILLING" in result.labels
    assert len(result.labels) >= 2


def test_predict_one_abstains_when_nothing_clears_threshold():
    """Verify an unreachable threshold forces abstention while scores stay populated."""

    clf = _fitted_linear_classifier()
    clf.thresholds = {label: 1.01 for label in LABELS}

    result = clf.predict_one("please cancel my subscription")

    assert result.labels == []
    assert result.abstained is True
    assert set(result.scores) == set(LABELS)
    assert all(0.0 <= score <= 1.0 for score in result.scores.values())


def test_global_threshold_applies_to_every_label():
    """Verify a single global threshold value gates every label identically."""

    clf = _fitted_linear_classifier()

    assert set(clf.thresholds.values()) == {0.3}


def test_per_label_thresholds_gate_labels_independently():
    """Verify per-label thresholds independently gate each label's prediction."""

    clf = _fitted_linear_classifier()
    scores = clf.predict_one("please cancel my subscription and refund my last charge").scores

    clf.thresholds = {label: 1.01 for label in LABELS}
    clf.thresholds["CANCELLATION"] = min(scores["CANCELLATION"], 0.01)

    result = clf.predict_one("please cancel my subscription and refund my last charge")

    assert result.labels == ["CANCELLATION"]


def test_predict_before_fit_raises():
    """Verify predicting before fitting raises RuntimeError."""

    clf = MultiLabelTextClassifier(LABELS)

    with pytest.raises(RuntimeError, match="must be fitted"):
        clf.predict_one("anything")


# ----------------------------------------------------------------------
# Batch prediction
# ----------------------------------------------------------------------


def test_predict_batch_matches_predict_one():
    """Verify batch predictions match individually-predicted results.

    Scores are compared approximately: batch vs. single-row matrix
    operations can differ in the last few floating-point digits due to
    non-associative summation order, which is not a correctness issue.
    """

    clf = _fitted_linear_classifier()
    texts, _ = _toy_corpus()

    batch = clf.predict_batch(texts[:3])
    individual = [clf.predict_one(text) for text in texts[:3]]

    assert [r.labels for r in batch.results] == [r.labels for r in individual]
    for batch_result, individual_result in zip(batch.results, individual):
        assert batch_result.scores == pytest.approx(individual_result.scores)


def test_predict_batch_reports_timing_and_throughput():
    """Verify batch timing fields are populated and throughput is positive."""

    clf = _fitted_linear_classifier()
    texts, _ = _toy_corpus()

    batch = clf.predict_batch(texts)

    assert batch.timing.total_s >= 0.0
    assert batch.timing.throughput_per_s > 0.0
    assert len(batch.results) == len(texts)


def test_predict_batch_abstains_consistently():
    """Verify abstention behaves consistently for single vs. batch prediction."""

    clf = _fitted_linear_classifier()
    clf.thresholds = {label: 1.01 for label in LABELS}

    batch = clf.predict_batch(["please cancel my subscription", "how much does it cost"])

    assert all(r.abstained for r in batch.results)
    assert all(r.labels == [] for r in batch.results)


# ----------------------------------------------------------------------
# TF-IDF vectorizer and train-only fitting
# ----------------------------------------------------------------------


def test_tfidf_vectorizer_fits_and_transforms():
    """Verify TfidfTextVectorizer fits a vocabulary and transforms to a fixed-width matrix."""

    vectorizer = TfidfTextVectorizer()
    texts, _ = _toy_corpus()

    matrix = vectorizer.fit_transform(texts)

    assert matrix.shape[0] == len(texts)
    assert matrix.shape[1] == vectorizer.output_dim
    assert matrix.shape[1] > 0


def test_tfidf_vectorizer_requires_fit_before_transform():
    """Verify transforming before fitting raises RuntimeError."""

    vectorizer = TfidfTextVectorizer()

    with pytest.raises(RuntimeError, match="must be fitted"):
        vectorizer.transform(["hello"])


def test_vectorizer_is_fit_only_on_training_text(monkeypatch):
    """Verify fit() is called on the vectorizer exactly once, with training text only."""

    texts, labels = _toy_corpus()
    texts_val, labels_val = texts[:2], labels[:2]

    clf = MultiLabelTextClassifier(LABELS, backend="linear", random_state=42)
    fit_calls = []
    original_fit = TfidfTextVectorizer.fit

    def spy_fit(self, fit_texts):
        fit_calls.append(list(fit_texts))
        return original_fit(self, fit_texts)

    monkeypatch.setattr(TfidfTextVectorizer, "fit", spy_fit)

    clf.fit(texts, labels, texts_val, labels_val, verbose=False)

    assert len(fit_calls) == 1
    assert fit_calls[0] == list(texts)


# ----------------------------------------------------------------------
# Linear backend
# ----------------------------------------------------------------------


def test_linear_backend_probabilities_are_independent_not_normalized():
    """Verify linear backend probabilities need not sum to 1 across labels."""

    clf = _fitted_linear_classifier()
    result = clf.predict_one("please cancel my subscription and refund my last charge")

    assert sum(result.scores.values()) != pytest.approx(1.0)
    assert isinstance(clf.backend, LinearMultiLabelBackend)


def test_linear_backend_predict_proba_before_fit_raises():
    """Verify predicting with an unfitted linear backend raises RuntimeError."""

    backend = LinearMultiLabelBackend()

    with pytest.raises(RuntimeError, match="must be fitted"):
        backend.predict_proba(np.zeros((1, 4)))


# ----------------------------------------------------------------------
# Neural backend (skipped automatically if TensorFlow is unavailable)
# ----------------------------------------------------------------------

pytest.importorskip("tensorflow", reason="neural backend requires TensorFlow ([deep] extra)")


def test_neural_backend_trains_and_predicts():
    """Verify the neural backend fits via tonkotsu and predicts independent probabilities."""

    texts, labels = _toy_corpus()
    clf = MultiLabelTextClassifier(
        LABELS, vectorizer="tfidf", backend="neural", threshold=0.3, random_state=42
    )
    clf.backend.epochs = 5
    clf.fit(texts, labels, verbose=False)

    result = clf.predict_one(texts[0])

    assert isinstance(clf.backend, DenseNeuralMultiLabelBackend)
    assert set(result.scores) == set(LABELS)
    assert all(0.0 <= score <= 1.0 for score in result.scores.values())


def test_neural_backend_reproducible_with_fixed_seed():
    """Verify identical global and backend seeding reproduces identical predictions.

    Full reproducibility of neural training requires seeding NumPy and
    TensorFlow's global RNGs in addition to the classifier's
    random_state (see the tonkotsu.build_dense docstring) - this test
    exercises that documented combination rather than random_state alone.
    """

    import tensorflow as tf

    texts, labels = _toy_corpus()

    def _make():
        tf.random.set_seed(7)
        np.random.seed(7)
        clf = MultiLabelTextClassifier(LABELS, vectorizer="tfidf", backend="neural", random_state=7)
        clf.backend.epochs = 3
        clf.fit(texts, labels, verbose=False)
        return clf

    result_a = _make().predict_one(texts[0])
    result_b = _make().predict_one(texts[0])

    assert result_a.scores == result_b.scores


def test_save_and_load_neural_backend_predicts_equivalently(tmp_path):
    """Verify a reloaded neural-backed classifier predicts identically to the original."""

    texts, labels = _toy_corpus()
    clf = MultiLabelTextClassifier(LABELS, backend="neural", threshold=0.3, random_state=42)
    clf.backend.epochs = 5
    clf.fit(texts, labels, verbose=False)

    bundle_path = clf.save(tmp_path / "support_router.drivethrough")
    loaded = MultiLabelTextClassifier.load(bundle_path)

    for text in texts:
        assert clf.predict_one(text).scores == loaded.predict_one(text).scores


# ----------------------------------------------------------------------
# Sentence-embedding vectorizer / missing optional dependency
# ----------------------------------------------------------------------


def test_sentence_embedding_vectorizer_config_without_dependency():
    """Verify get_config() works without ever importing sentence-transformers."""

    vectorizer = SentenceEmbeddingTextVectorizer(model_name="all-MiniLM-L6-v2")

    assert vectorizer.get_config() == {
        "type": "sentence_embedding",
        "model_name": "all-MiniLM-L6-v2",
    }


def test_sentence_embedding_vectorizer_missing_dependency_raises(monkeypatch):
    """Verify a clear ImportError with an install hint when sentence-transformers is unavailable."""

    monkeypatch.setitem(sys.modules, "sentence_transformers", None)
    vectorizer = SentenceEmbeddingTextVectorizer()

    with pytest.raises(ImportError) as exc_info:
        vectorizer.fit(["hello"])

    assert "ramentruck[nlp]" in str(exc_info.value)


# ----------------------------------------------------------------------
# Threshold tuning
# ----------------------------------------------------------------------


def test_tune_thresholds_improves_or_maintains_f1_on_validation():
    """Verify tune_thresholds selects thresholds at least as good as the default on validation data."""

    texts, labels = _toy_corpus()
    clf = MultiLabelTextClassifier(LABELS, backend="linear", threshold=0.5, random_state=42)
    clf.fit(texts, labels, verbose=False)

    before = clf.evaluate(texts, labels).micro_f1
    clf.tune_thresholds(texts, labels, metric="f1", verbose=False)
    after = clf.evaluate(texts, labels).micro_f1

    assert after >= before


def test_tune_thresholds_rejects_unsupported_metric():
    """Verify an unsupported tuning metric raises ValueError."""

    clf = _fitted_linear_classifier()
    texts, labels = _toy_corpus()

    with pytest.raises(ValueError, match="Unsupported metric"):
        clf.tune_thresholds(texts, labels, metric="not_a_real_metric", verbose=False)


def test_tune_thresholds_global_mode_shares_one_threshold():
    """Verify per_label=False produces a single threshold shared by every label."""

    clf = _fitted_linear_classifier()
    texts, labels = _toy_corpus()

    result = clf.tune_thresholds(texts, labels, per_label=False, verbose=False)

    assert len(set(result.thresholds.values())) == 1
    assert set(clf.thresholds.values()) == set(result.thresholds.values())


# ----------------------------------------------------------------------
# Evaluation metrics and failure analysis
# ----------------------------------------------------------------------


def test_evaluate_returns_expected_metric_fields():
    """Verify evaluate() returns every required multi-label metric field."""

    clf = _fitted_linear_classifier()
    texts, labels = _toy_corpus()

    report = clf.evaluate(texts, labels)

    assert 0.0 <= report.subset_accuracy <= 1.0
    assert 0.0 <= report.hamming_loss <= 1.0
    for value in (
        report.micro_precision,
        report.micro_recall,
        report.micro_f1,
        report.macro_precision,
        report.macro_recall,
        report.macro_f1,
    ):
        assert 0.0 <= value <= 1.0
    assert list(report.per_label["label"]) == LABELS
    assert set(report.per_label.columns) == {"label", "precision", "recall", "f1", "support"}
    assert len(report.failures) == len(texts)
    assert report.n_examples == len(texts)


def test_evaluate_failure_frame_flags_false_negatives_when_abstaining():
    """Verify the failure frame records false negatives and under-selection on forced abstention."""

    clf = _fitted_linear_classifier()
    clf.thresholds = {label: 1.01 for label in LABELS}

    texts, labels = _toy_corpus()
    report = clf.evaluate(texts, labels)

    assert report.n_abstained == len(texts)
    assert all(report.failures["under_selected"])
    assert all(len(fn) > 0 for fn in report.failures["false_negatives"])
    assert not any(report.failures["over_selected"])


# ----------------------------------------------------------------------
# Leakage checking
# ----------------------------------------------------------------------


def test_check_for_leakage_detects_exact_duplicate():
    """Verify an exact-duplicate protected text is flagged."""

    protected = ["please cancel my plan"]
    corpus = ["please cancel my plan", "something unrelated"]

    report = check_for_leakage(protected, corpus)

    assert report.has_leakage
    assert report.n_exact == 1
    assert report.n_normalized == 0


def test_check_for_leakage_detects_normalized_duplicate():
    """Verify casing/whitespace differences are caught by normalized matching."""

    protected = ["Please   Cancel My Plan"]
    corpus = ["please cancel my plan"]

    report = check_for_leakage(protected, corpus)

    assert report.n_exact == 0
    assert report.n_normalized == 1
    assert report.matches[0].match_type == "normalized"


def test_check_for_leakage_reports_no_match_for_unrelated_text():
    """Verify unrelated text reports no leakage."""

    report = check_for_leakage(["something entirely different"], ["please cancel my plan"])

    assert not report.has_leakage
    assert report.n_exact == 0
    assert report.n_normalized == 0


def test_check_for_leakage_normalize_none_skips_normalized_check():
    """Verify passing normalize=None disables the normalized-duplicate check."""

    protected = ["Please   Cancel My Plan"]
    corpus = ["please cancel my plan"]

    report = check_for_leakage(protected, corpus, normalize=None)

    assert not report.has_leakage


# ----------------------------------------------------------------------
# Persistence
# ----------------------------------------------------------------------


def test_save_and_load_linear_backend_predicts_equivalently(tmp_path):
    """Verify a reloaded linear-backed classifier predicts identically to the original."""

    clf = _fitted_linear_classifier()
    texts, _ = _toy_corpus()

    bundle_path = clf.save(tmp_path / "support_router.drivethrough")
    loaded = MultiLabelTextClassifier.load(bundle_path)

    for text in texts:
        original = clf.predict_one(text)
        reloaded = loaded.predict_one(text)
        assert original.labels == reloaded.labels
        assert original.scores == reloaded.scores
        assert original.abstained == reloaded.abstained

    assert loaded.labels == clf.labels
    assert loaded.thresholds == clf.thresholds


def test_save_rejects_existing_path_without_overwrite(tmp_path):
    """Verify save() raises FileExistsError when the bundle path exists and overwrite=False."""

    clf = _fitted_linear_classifier()
    bundle_path = tmp_path / "support_router.drivethrough"
    clf.save(bundle_path)

    with pytest.raises(FileExistsError, match="already exists"):
        clf.save(bundle_path)


def test_save_allows_overwrite(tmp_path):
    """Verify overwrite=True allows replacing an existing bundle."""

    clf = _fitted_linear_classifier()
    bundle_path = tmp_path / "support_router.drivethrough"
    clf.save(bundle_path)
    clf.save(bundle_path, overwrite=True)

    loaded = MultiLabelTextClassifier.load(bundle_path)
    assert loaded.labels == clf.labels


def test_save_before_fit_raises(tmp_path):
    """Verify saving an unfitted classifier raises RuntimeError."""

    clf = MultiLabelTextClassifier(LABELS)

    with pytest.raises(RuntimeError, match="must be fitted"):
        clf.save(tmp_path / "unfitted.drivethrough")


def test_load_missing_bundle_raises(tmp_path):
    """Verify loading a nonexistent or incomplete bundle raises FileNotFoundError."""

    with pytest.raises(FileNotFoundError, match="No complete drivethrough bundle"):
        MultiLabelTextClassifier.load(tmp_path / "does_not_exist")


def test_saved_bundle_includes_reproducibility_metadata(tmp_path):
    """Verify meta.json records versions, seed, labels, and sample counts."""

    clf = _fitted_linear_classifier()
    bundle_path = clf.save(tmp_path / "support_router.drivethrough")
    meta = json.loads((bundle_path / "meta.json").read_text())

    assert meta["random_state"] == 42
    assert meta["labels"] == LABELS
    assert meta["backend_type"] == "linear"
    assert meta["vectorizer_type"] == "tfidf"
    assert meta["n_train"] == len(_toy_corpus()[0])
    assert "ramentruck_version" in meta
    assert "sklearn_version" in meta
    assert "saved_at" in meta
