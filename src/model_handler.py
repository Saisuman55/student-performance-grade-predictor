from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler


@dataclass
class ModelBundle:
    model: LinearRegression
    scaler: StandardScaler
    feature_columns: list[str]


def save_model(bundle: ModelBundle, model_path: Path) -> None:
    """Save trained model artifacts as a pickle file."""
    model_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model": bundle.model,
        "scaler": bundle.scaler,
        "feature_columns": bundle.feature_columns,
    }
    with model_path.open("wb") as model_file:
        pickle.dump(payload, model_file)


def load_model(model_path: Path) -> ModelBundle:
    """Load model artifacts from a pickle file."""
    with model_path.open("rb") as model_file:
        payload = pickle.load(model_file)

    return ModelBundle(
        model=payload["model"],
        scaler=payload["scaler"],
        feature_columns=payload["feature_columns"],
    )
