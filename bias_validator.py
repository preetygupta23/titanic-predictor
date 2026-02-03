"""
**Purpose**: Model Fairness/Unit Testing.
- **Slicing**: Breaks down performance metrics by `Sex` and `Pclass`.
- **Validation**: Ensures that the model is actually learning survival nuances for men and 3rd class passengers rather than just relying on majority-class statistics.
"""

import pandas as pd
import joblib
from sklearn.metrics import accuracy_score, recall_score
from preprocessor import clean_data
from feature_engineering import run_feature_engineering  # <--- Add this import


def run_bias_audit(data_path, model_path):
    # 1. Load Model and Data
    model = joblib.load(model_path)
    raw_df = pd.read_csv(data_path)

    # 2. Prepare Data
    # FIRST: Create the new features (FamilySize, etc.)
    engineered_df = run_feature_engineering(raw_df.copy())

    # SECOND: Clean and Encode
    cleaned_df = clean_data(engineered_df)

    # THIRD: Align columns with what the model expects
    X = cleaned_df.drop('Survived', axis=1)

    # 3. Get Predictions
    # We use the engineered/cleaned X to get predictions
    raw_df['Predictions'] = model.predict(X)

    print("\n=== MODEL BIAS AUDIT ===")

    # 4. Audit by Gender
    for gender in ['male', 'female']:
        subset = raw_df[raw_df['Sex'] == gender]
        acc = accuracy_score(subset['Survived'], subset['Predictions'])
        rec = recall_score(subset['Survived'], subset['Predictions'])
        print(f"Gender: {gender:6} | Accuracy: {acc:.2f} | Recall (Survival): {rec:.2f}")

    print("-" * 35)

    # 5. Audit by Class
    for pclass in sorted(raw_df['Pclass'].unique()):
        subset = raw_df[raw_df['Pclass'] == pclass]
        acc = accuracy_score(subset['Survived'], subset['Predictions'])
        print(f"Class: {pclass}        | Accuracy: {acc:.2f}")


if __name__ == "__main__":
    run_bias_audit('train.csv', 'titanic_model.pkl')