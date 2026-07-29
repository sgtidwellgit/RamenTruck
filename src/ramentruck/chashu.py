"""Model persistence for RamenTruck."""

from __future__ import annotations

import json
import platform
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
import sklearn

from .results import ChashuBundle

RESERVED_METADATA_KEYS = frozenset(
    {"saved_at", "ramentruck_version", "python_version", "sklearn_version", "model_class"}
)

MODEL_FILENAME = "model.joblib"
META_FILENAME = "meta.json"


def save(
    model: Any,
    path: str | Path,
    *,
    metadata: dict[str, Any] | None = None,
    overwrite: bool = False,
) -> Path:
    """
    Serialize a fitted model to a chashu bundle directory.

    Parameters
    ----------
    model
        The model object to serialize. Not mutated.
    path
        Directory path for the bundle (conventionally suffixed ``.chashu``).
        Created (including parents) if it does not exist.
    metadata
        Optional user-supplied metadata to store alongside the model. Keys
        must not collide with the automatically recorded reserved fields.
    overwrite
        Whether to allow writing into a path that already exists. Defaults
        to ``False`` to prevent silent clobbers.

    Returns
    -------
    Path
        The bundle directory path.

    Raises
    ------
    FileExistsError
        If ``path`` already exists and ``overwrite=False``.
    ValueError
        If ``metadata`` uses a reserved key name.
    """

    path = Path(path)

    if path.exists() and not overwrite:
        raise FileExistsError(
            f"{path} already exists. Pass overwrite=True to replace it."
        )

    user_metadata = dict(metadata) if metadata else {}
    reserved_collisions = RESERVED_METADATA_KEYS & user_metadata.keys()

    if reserved_collisions:
        raise ValueError(
            f"metadata keys collide with reserved fields: "
            f"{sorted(reserved_collisions)}."
        )

    from . import __version__ as ramentruck_version

    bundle_meta = {
        "saved_at": datetime.now(timezone.utc).isoformat(),
        "ramentruck_version": ramentruck_version,
        "python_version": platform.python_version(),
        "sklearn_version": sklearn.__version__,
        "model_class": type(model).__name__,
        "metadata": user_metadata,
    }

    path.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path / MODEL_FILENAME)
    (path / META_FILENAME).write_text(json.dumps(bundle_meta, indent=2))

    return path


def load(path: str | Path, *, verify: bool = True) -> ChashuBundle:
    """
    Load a chashu bundle saved by :func:`save`.

    Parameters
    ----------
    path
        Path to the bundle directory.
    verify
        Whether to warn when the saved scikit-learn version does not match
        the current environment.

    Returns
    -------
    ChashuBundle
        The deserialized model plus its saved metadata and versions.

    Raises
    ------
    FileNotFoundError
        If no bundle is found at ``path``.
    """

    path = Path(path)
    model_path = path / MODEL_FILENAME
    meta_path = path / META_FILENAME

    if not model_path.exists() or not meta_path.exists():
        raise FileNotFoundError(f"No chashu bundle found at {path}.")

    bundle_meta = json.loads(meta_path.read_text())
    model = joblib.load(model_path)

    if verify and bundle_meta["sklearn_version"] != sklearn.__version__:
        warnings.warn(
            f"chashu bundle was saved with scikit-learn "
            f"{bundle_meta['sklearn_version']}, but the current environment "
            f"has {sklearn.__version__}.",
            UserWarning,
            stacklevel=2,
        )

    return ChashuBundle(
        model=model,
        metadata=bundle_meta["metadata"],
        saved_at=bundle_meta["saved_at"],
        ramentruck_version=bundle_meta["ramentruck_version"],
        python_version=bundle_meta["python_version"],
        sklearn_version=bundle_meta["sklearn_version"],
    )


def list_models(directory: str | Path) -> pd.DataFrame:
    """
    Summarize every chashu bundle found directly under a directory.

    Parameters
    ----------
    directory
        Directory to scan for immediate subdirectories containing a chashu
        bundle.

    Returns
    -------
    pd.DataFrame
        One row per bundle with ``path``, ``saved_at``, ``model_class``, and
        any user-supplied metadata fields as additional columns.
    """

    directory = Path(directory)
    rows: list[dict[str, Any]] = []

    for meta_path in sorted(directory.glob(f"*/{META_FILENAME}")):
        bundle_meta = json.loads(meta_path.read_text())
        row = {
            "path": str(meta_path.parent),
            "saved_at": bundle_meta["saved_at"],
            "model_class": bundle_meta["model_class"],
            **bundle_meta["metadata"],
        }
        rows.append(row)

    if not rows:
        return pd.DataFrame(columns=["path", "saved_at", "model_class"])

    return pd.DataFrame(rows)
