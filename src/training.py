from __future__ import annotations

import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from src.config import DATA_PATH, FEATURE_COLUMNS, MODEL_PATH, TARGET_COLUMN
from src.model_handler import ModelBundle, save_model


def _load_and_clean_dataset(csv_path) -> pd.DataFrame:
    dataset = pd.read_csv(csv_path)
    cleaned = dataset.copy()
    for column in FEATURE_COLUMNS + [TARGET_COLUMN]:
        cleaned[column] = pd.to_numeric(cleaned[column], errors="coerce")
        cleaned[column] = cleaned[column].fillna(cleaned[column].median())
    return cleaned


def train_and_save_model() -> dict[str, float]:
    """Train a linear regression model and persist artifacts to disk."""
    dataset = _load_and_clean_dataset(DATA_PATH)
    features = dataset[FEATURE_COLUMNS]
    target = dataset[TARGET_COLUMN]

    x_train, x_test, y_train, y_test = train_test_split(
        features,
        target,
        test_size=0.2,
        random_state=42,
    )

    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_test_scaled = scaler.transform(x_test)

    model = LinearRegression()
    model.fit(x_train_scaled, y_train)

    predictions = model.predict(x_test_scaled)
    metrics = {
        "r2": float(r2_score(y_test, predictions)),
        "mae": float(mean_absolute_error(y_test, predictions)),
    }

    bundle = ModelBundle(model=model, scaler=scaler, feature_columns=FEATURE_COLUMNS)
    save_model(bundle, MODEL_PATH)
    return metrics


if __name__ == "__main__":
    training_metrics = train_and_save_model()
    print("Training completed.")
    print(f"R2: {training_metrics['r2']:.3f}")
    print(f"MAE: {training_metrics['mae']:.3f}")
    print(f"Model saved to: {MODEL_PATH}")
