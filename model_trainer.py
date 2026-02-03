"""
**Purpose**: Model selection and training.
- **Algorithm**: Random Forest Classifier (set to `max_depth=5` to prevent overfitting).
- **`get_feature_importance(model, feature_names)`**: Visualizes which columns influenced the decision-making process.
- **Artifacts**: Saves `titanic_model.pkl` (the weights) and `model_columns.pkl` (the feature order).
"""

"""
Module: model_trainer.py
Purpose: Trains the Random Forest and validates stability using Cross-Validation.
"""

import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report
from data_loader import load_titanic_data
from feature_engineering import run_feature_engineering
from preprocessor import clean_data


def train_titanic_model(data_path):
    # 1. Pipeline: Load -> Engineer -> Clean
    raw_df = load_titanic_data(data_path)
    engineered_df = run_feature_engineering(raw_df)
    df_cleaned = clean_data(engineered_df)

    # 2. Split Features and Target
    X = df_cleaned.drop('Survived', axis=1)
    y = df_cleaned['Survived']

    # 3. Initialize Model
    model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)

    # 4. CROSS-VALIDATION (The Stability Test)
    # We split the data into 5 'folds' and test the model 5 times
    print("🔄 Running 5-Fold Cross-Validation...")
    cv_scores = cross_val_score(model, X, y, cv=5)

    print(f"Mean CV Accuracy: {cv_scores.mean():.2f}")
    print(f"Accuracy Deviation: +/- {cv_scores.std():.2f}")
    print("-" * 30)

    # 5. Final Training on full Train set
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    model.fit(X_train, y_train)

    # 6. Save Artifacts
    joblib.dump(model, 'titanic_model.pkl')
    joblib.dump(X.columns.tolist(), 'model_columns.pkl')

    return model


if __name__ == "__main__":
    train_titanic_model('train.csv')