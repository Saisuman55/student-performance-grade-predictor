from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "student_data.csv"
MODEL_PATH = BASE_DIR / "models" / "student_grade_model.pkl"

FEATURE_COLUMNS = [
    "attendance",
    "assignment_1",
    "assignment_2",
    "assignment_3",
    "internal_exam",
]
TARGET_COLUMN = "final_score"
