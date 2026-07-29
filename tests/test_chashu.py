"""Unit tests for ramentruck.chashu."""

import pytest
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression

from ramentruck import ChashuBundle, chashu


def _fitted_model():
    """Return a small fitted classifier for persistence tests."""

    X, y = make_classification(n_samples=40, n_features=4, random_state=42)
    model = LogisticRegression(max_iter=500)
    model.fit(X, y)
    return model, X, y


def test_chashu_save_and_load_roundtrip(tmp_path):
    """Verify a saved model can be loaded back with matching predictions."""

    model, X, _ = _fitted_model()
    bundle_path = tmp_path / "model.chashu"

    saved_path = chashu.save(model, bundle_path, metadata={"dataset": "test"})
    bundle = chashu.load(saved_path)

    assert saved_path == bundle_path
    assert isinstance(bundle, ChashuBundle)
    assert bundle.metadata == {"dataset": "test"}
    assert bundle.sklearn_version
    assert bundle.ramentruck_version
    assert bundle.python_version
    assert list(bundle.model.predict(X)) == list(model.predict(X))


def test_chashu_save_rejects_existing_path_without_overwrite(tmp_path):
    """Verify save raises FileExistsError when the path exists and overwrite=False."""

    model, _, _ = _fitted_model()
    bundle_path = tmp_path / "model.chashu"
    chashu.save(model, bundle_path)

    with pytest.raises(FileExistsError, match="already exists"):
        chashu.save(model, bundle_path)


def test_chashu_save_allows_overwrite(tmp_path):
    """Verify overwrite=True allows replacing an existing bundle."""

    model, _, _ = _fitted_model()
    bundle_path = tmp_path / "model.chashu"
    chashu.save(model, bundle_path, metadata={"version": 1})
    chashu.save(model, bundle_path, metadata={"version": 2}, overwrite=True)

    bundle = chashu.load(bundle_path)
    assert bundle.metadata == {"version": 2}


def test_chashu_save_rejects_reserved_metadata_keys(tmp_path):
    """Verify metadata keys colliding with reserved names raise ValueError."""

    model, _, _ = _fitted_model()

    with pytest.raises(ValueError, match="reserved fields"):
        chashu.save(
            model,
            tmp_path / "model.chashu",
            metadata={"saved_at": "not allowed"},
        )


def test_chashu_load_missing_bundle_raises(tmp_path):
    """Verify loading a nonexistent bundle raises FileNotFoundError."""

    with pytest.raises(FileNotFoundError, match="No chashu bundle found"):
        chashu.load(tmp_path / "does_not_exist.chashu")


def test_chashu_list_models(tmp_path):
    """Verify list_models summarizes every bundle in a directory."""

    model, _, _ = _fitted_model()
    chashu.save(model, tmp_path / "a.chashu", metadata={"val_auc": 0.9})
    chashu.save(model, tmp_path / "b.chashu", metadata={"val_auc": 0.95})

    df = chashu.list_models(tmp_path)

    assert len(df) == 2
    assert set(df["path"]) == {
        str(tmp_path / "a.chashu"),
        str(tmp_path / "b.chashu"),
    }
    assert set(df["val_auc"]) == {0.9, 0.95}
    assert all(df["model_class"] == "LogisticRegression")


def test_chashu_list_models_empty_directory(tmp_path):
    """Verify list_models returns an empty DataFrame for a directory with no bundles."""

    df = chashu.list_models(tmp_path)

    assert len(df) == 0
    assert list(df.columns) == ["path", "saved_at", "model_class"]


def test_chashu_load_warns_on_sklearn_version_mismatch(tmp_path, monkeypatch):
    """Verify a mismatched sklearn version warns when verify=True."""

    model, _, _ = _fitted_model()
    bundle_path = tmp_path / "model.chashu"
    chashu.save(model, bundle_path)

    import json

    meta_path = bundle_path / "meta.json"
    data = json.loads(meta_path.read_text())
    data["sklearn_version"] = "0.0.0"
    meta_path.write_text(json.dumps(data))

    with pytest.warns(UserWarning, match="scikit-learn"):
        chashu.load(bundle_path)
