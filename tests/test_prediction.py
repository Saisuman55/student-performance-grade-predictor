from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import pandas as pd

from src.model_handler import ModelBundle
from src.predictor import predict_single


def test_predict_single_returns_bounded_score():
    feature_columns = [
        "attendance",
        "assignment_1",
        "assignment_2",
        "assignment_3",
        "internal_exam",
    ]

    x_train = pd.DataFrame(
        [
            [90, 88, 87, 86, 89],
            [70, 65, 68, 66, 64],
            [50, 55, 52, 54, 53],
        ],
        columns=feature_columns,
    )
    y_train = [90, 68, 54]

    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)

    model = LinearRegression()
    model.fit(x_train_scaled, y_train)

    bundle = ModelBundle(model=model, scaler=scaler, feature_columns=feature_columns)

    result = predict_single(
        bundle,
        {
            "attendance": 80,
            "assignment_1": 78,
            "assignment_2": 79,
            "assignment_3": 77,
            "internal_exam": 76,
        },
    )

    assert isinstance(result, float)
    assert 0.0 <= result <= 100.0
