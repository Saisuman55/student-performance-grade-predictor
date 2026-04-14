# Student Performance & Grade Predictor

Portfolio-ready ML project that predicts final student scores and grades from attendance, assignments, and internal exam marks.

Tagline: From raw academic inputs to deployable predictions in a clean, industry-style repository.

## Demo

This project provides:
- Single student prediction (interactive Streamlit form)
- Batch prediction using CSV upload
- Grade mapping (A/B/C/D) with simple feedback tips
- Persisted model loading (no retraining on every app run)

This makes it suitable for both academic submission and beginner ML portfolio presentation.

## Features

- Modular codebase with separation of concerns
- Model training and saved model loading via pickle
- Streamlit UI separated from core ML logic
- Batch CSV prediction support
- Simple unit test for prediction function
- Deployment-ready project structure

## Why This Repository Looks Professional

- Clear folder organization used in real projects
- Predictive logic is separated from UI code
- Reproducible setup with requirements and test file
- Ready for GitHub and Streamlit Cloud deployment

## Tech Stack

- Python
- Pandas
- NumPy
- Scikit-learn
- Streamlit
- Pytest

## Professional Folder Structure

```
student-performance-grade-predictor/
├── app/
│   ├── __init__.py
│   └── streamlit_app.py
├── src/
│   ├── __init__.py
│   ├── config.py
│   ├── model_handler.py
│   ├── predictor.py
│   └── training.py
├── utils/
│   ├── __init__.py
│   └── grade_utils.py
├── data/
│   └── student_data.csv
├── models/
│   └── .gitkeep
├── notebooks/
│   └── README.md
├── assets/
│   └── screenshots/
│       └── .gitkeep
├── tests/
│   └── test_prediction.py
├── .gitignore
├── requirements.txt
└── README.md
```

### Folder Purpose (Brief)

- app/: Streamlit UI layer only
- src/: Core ML logic (training, model IO, prediction)
- utils/: Helper functions like grade logic and feedback
- data/: Input dataset files
- models/: Saved ML artifacts (.pkl)
- notebooks/: EDA and experimentation notebooks
- assets/: Screenshots and media for documentation
- tests/: Basic unit tests

## Installation

1. Clone the repository

```bash
git clone <your-repo-url>
cd student-performance-grade-predictor
```

2. Create and activate a virtual environment

```bash
python -m venv .venv
```

Windows:

```bash
.venv\Scripts\activate
```

macOS/Linux:

```bash
source .venv/bin/activate
```

3. Install dependencies

```bash
pip install -r requirements.txt
```

## How To Run Locally

1. Train and save model (one-time or whenever dataset changes)

```bash
python -m src.training
```

2. Start Streamlit app

```bash
streamlit run app/streamlit_app.py
```

## Screenshots

Add your app screenshots in assets/screenshots and update links below.

- Home / Single Prediction Page: [Add Screenshot](assets/screenshots/single_prediction.png)
- Batch Prediction Page: [Add Screenshot](assets/screenshots/batch_prediction.png)
- Model Info Page: [Add Screenshot](assets/screenshots/model_info.png)

## Model Handling

- Model is trained in src/training.py
- Model is saved as models/student_grade_model.pkl
- App loads model from disk using src/model_handler.py
- If model file is missing, app can trigger training automatically

## Basic Testing

Run tests:

```bash
pytest -q
```

Current test:
- tests/test_prediction.py verifies predict_single returns a float in the valid range [0, 100]

## Deployment Instructions

### Streamlit Community Cloud

1. Push this repository to GitHub
2. Go to Streamlit Community Cloud
3. Click New app
4. Select your repository and branch
5. Set main file path to app/streamlit_app.py
6. Deploy

### How To Push To GitHub

```bash
git init
git add .
git commit -m "Initial project setup"
git branch -M main
git remote add origin <your-repo-url>
git push -u origin main
```

## Suggested Commit Strategy

Use small, meaningful commits. Example sequence:

1. Initial project setup
2. Added dataset and training pipeline
3. Added model save/load utilities
4. Integrated Streamlit UI
5. Added batch prediction feature
6. Added tests and documentation
7. Deployment and repository polishing

## Future Improvements

- Add model comparison dashboard
- Add input validation with form constraints
- Add SHAP explainability plots
- Add CI with GitHub Actions for tests
- Add Docker support for one-command deployment

## Author

- Sai Suman
- GitHub: https://github.com/Saisuman55
- LinkedIn: https://www.linkedin.com/in/your-profile

## Extra Repository Assets

- GitHub description, topics, and portfolio summary template: see [GITHUB_SHOWCASE.md](GITHUB_SHOWCASE.md)
