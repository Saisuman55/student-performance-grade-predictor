from __future__ import annotations


def get_grade(score: float) -> str:
    """Convert numeric score to letter grade."""
    if score >= 85:
        return "A"
    if score >= 70:
        return "B"
    if score >= 55:
        return "C"
    return "D"


def generate_feedback(attendance: float, assignment_avg: float, exam_score: float) -> list[str]:
    """Generate simple coaching tips based on input values."""
    tips: list[str] = []

    if attendance < 75:
        tips.append("Improve attendance to increase your final score consistency.")
    else:
        tips.append("Attendance is good. Keep it consistent.")

    if assignment_avg < 60:
        tips.append("Focus on assignments and submit all tasks on time.")
    elif assignment_avg >= 80:
        tips.append("Assignment performance is strong. Keep this momentum.")

    if exam_score < 60:
        tips.append("Revise core concepts and practice with sample papers.")

    return tips
