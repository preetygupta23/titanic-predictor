**Titanic Survival Predictor: A QA-Driven ML Pipeline**
📌 Project Overview
This repository contains a production-grade Machine Learning pipeline for predicting Titanic passenger survival. Unlike standard notebook-based approaches, this project implements Modular Software Architecture, Automated CI/CD Testing, and a Formal QA Framework.

🛠 Technical Stack
Language: Python 3.9+

Model: Random Forest Classifier

Libraries: Scikit-Learn, Pandas, Joblib, PyTest

Automation: GitHub Actions (CI/CD)

Deployment: Streamlit Cloud

🧪 Quality Assurance & Automation
This project follows a "Shift-Left" testing philosophy, where quality is checked at every stage of the pipeline.

1. Automated Testing Suite (test_suite.py)
We use pytest to verify system integrity before any deployment.

Data Integrity: Validates 0% null values after preprocessing.

Model Functionality: Ensures the .pkl artifact loads correctly and generates predictions.

Boundary Value Analysis (BVA): Tests extreme age inputs (0.42 to 80 years) to ensure UI stability.

2. CI/CD Pipeline
Every code "push" triggers a GitHub Action that:

Sets up a clean Python environment.

Installs dependencies from requirements.txt.

Executes the Automated QA Suite.

Blocks deployment if any test fails, ensuring the live app remains stable.

📂 Repository Structure
Plaintext

├── .github/workflows/   # CI/CD Automation (GitHub Actions)
├── data/                # Raw and processed datasets
├── models/              # Serialized .pkl files
├── app.py               # Streamlit Web Interface
├── preprocessor.py      # Feature engineering & cleaning logic
├── test_suite.py        # Automated QA tests
├── requirements.txt     # Dependency management
└── QA_Report.md         # Full evaluation metrics and test cases
📊 Model Evaluation
Metric	Result
Accuracy	83%
Precision	79%
Recall	71%
CV Stability	+/- 0.02


🚀 How to Run Locally
Clone the repo: git clone https://github.com/YOUR_USERNAME/YOUR_REPO.git

Install dependencies: pip install -r requirements.txt

Run the QA Suite: pytest test_suite.py

Launch the App: streamlit run app.py