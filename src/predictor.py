from __future__ import annotations

import pandas as pd

from src.model_handler import ModelBundle


def predict_single(bundle: ModelBundle, input_data: dict[str, float]) -> float:
    """Predict final score for one student record."""
    input_frame = pd.DataFrame([input_data], columns=bundle.feature_columns)
    scaled = bundle.scaler.transform(input_frame)
    prediction = float(bundle.model.predict(scaled)[0])
    return max(0.0, min(100.0, prediction))


def predict_batch(bundle: ModelBundle, input_frame: pd.DataFrame) -> pd.DataFrame:
    """Predict final scores for multiple students and return a new dataframe."""
    batch = input_frame.copy()
    for column in bundle.feature_columns:
        batch[column] = pd.to_numeric(batch[column], errors="coerce")
        batch[column] = batch[column].fillna(batch[column].median())

    scaled = bundle.scaler.transform(batch[bundle.feature_columns])
    predicted_scores = bundle.model.predict(scaled)

    output = input_frame.copy()
    output["predicted_score"] = [max(0.0, min(100.0, float(v))) for v in predicted_scores]
    return output
