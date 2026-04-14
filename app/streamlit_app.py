from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from src.config import FEATURE_COLUMNS, MODEL_PATH
from src.model_handler import load_model
from src.predictor import predict_batch, predict_single
from src.training import train_and_save_model
from utils.grade_utils import generate_feedback, get_grade


st.set_page_config(page_title="Student Performance & Grade Predictor", page_icon="🎓", layout="wide")


@st.cache_resource
def get_model_bundle():
    """Load model bundle from disk for fast reuse in Streamlit sessions."""
    if not MODEL_PATH.exists():
        train_and_save_model()
    return load_model(MODEL_PATH)


def _show_sidebar() -> str:
    st.sidebar.title("Student Predictor")
    st.sidebar.caption("Portfolio-ready ML mini project")
    return st.sidebar.radio("Go to", ["Single Prediction", "Batch Prediction", "Model Info"])


def _single_prediction_page(bundle) -> None:
    st.title("🎯 Single Student Prediction")
    st.write("Enter student performance details and get score + grade prediction.")

    col1, col2 = st.columns(2)
    with col1:
        attendance = st.slider("Attendance (%)", 0, 100, 80)
        assignment_1 = st.slider("Assignment 1", 0, 100, 75)
        assignment_2 = st.slider("Assignment 2", 0, 100, 75)
        assignment_3 = st.slider("Assignment 3", 0, 100, 75)
        internal_exam = st.slider("Internal Exam", 0, 100, 75)

        if st.button("Predict Score", type="primary"):
            student_record = {
                "attendance": attendance,
                "assignment_1": assignment_1,
                "assignment_2": assignment_2,
                "assignment_3": assignment_3,
                "internal_exam": internal_exam,
            }
            predicted_score = predict_single(bundle, student_record)
            grade = get_grade(predicted_score)

            st.session_state["latest_prediction"] = {
                "score": predicted_score,
                "grade": grade,
                "tips": generate_feedback(
                    attendance=attendance,
                    assignment_avg=(assignment_1 + assignment_2 + assignment_3) / 3,
                    exam_score=internal_exam,
                ),
            }

    with col2:
        if "latest_prediction" not in st.session_state:
            st.info("Run a prediction to view results here.")
            return

        result = st.session_state["latest_prediction"]
        st.metric("Predicted Score", f"{result['score']:.2f}/100")
        st.metric("Predicted Grade", result["grade"])

        st.subheader("Feedback")
        for tip in result["tips"]:
            st.write(f"- {tip}")


def _batch_prediction_page(bundle) -> None:
    st.title("📥 Batch Prediction")
    st.write("Upload a CSV with required feature columns to predict scores in bulk.")

    st.caption(f"Required columns: {', '.join(FEATURE_COLUMNS)}")
    uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

    if uploaded_file is None:
        return

    input_frame = pd.read_csv(uploaded_file)
    missing = [column for column in FEATURE_COLUMNS if column not in input_frame.columns]
    if missing:
        st.error(f"Missing required columns: {', '.join(missing)}")
        return

    output = predict_batch(bundle, input_frame)
    output["predicted_grade"] = output["predicted_score"].apply(get_grade)

    st.success(f"Predictions generated for {len(output)} records.")
    st.dataframe(output, use_container_width=True)

    csv_data = output.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download Predictions (CSV)",
        data=csv_data,
        file_name="student_predictions.csv",
        mime="text/csv",
    )


def _model_info_page() -> None:
    st.title("🧠 Model Information")
    st.write("This app uses a saved Linear Regression model and StandardScaler.")
    st.write(f"Model file: {MODEL_PATH}")
    st.write(f"Model exists: {'Yes' if Path(MODEL_PATH).exists() else 'No'}")

    if st.button("Retrain Model"):
        metrics = train_and_save_model()
        st.success("Model retrained and saved successfully.")
        st.write(f"R2: {metrics['r2']:.3f}")
        st.write(f"MAE: {metrics['mae']:.3f}")


def main() -> None:
    bundle = get_model_bundle()
    page = _show_sidebar()

    if page == "Single Prediction":
        _single_prediction_page(bundle)
    elif page == "Batch Prediction":
        _batch_prediction_page(bundle)
    else:
        _model_info_page()


if __name__ == "__main__":
    main()
