"""Filesystem and serialization helpers."""

from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Any

import pandas as pd
try:
    import joblib
except ModuleNotFoundError:
    joblib = None

try:
    import yaml
except ModuleNotFoundError:
    yaml = None


def ensure_dir(path: Path | str) -> Path:
    """Create a directory if it does not already exist."""
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def save_json(data: Any, path: Path | str) -> None:
    """Persist JSON with UTF-8 encoding."""
    output_path = Path(path)
    output_path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def save_yaml(data: Any, path: Path | str) -> None:
    """Persist YAML with stable ordering."""
    output_path = Path(path)
    if yaml is not None:
        output_path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    else:
        output_path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def save_dataframe(frame: pd.DataFrame, path: Path | str) -> None:
    """Persist a dataframe as CSV."""
    frame.to_csv(path, index=False)


def save_joblib(data: Any, path: Path | str) -> None:
    """Persist a Python object with joblib."""
    if joblib is not None:
        joblib.dump(data, path)
        return
    with open(path, "wb") as handle:
        pickle.dump(data, handle)
