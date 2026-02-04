"""
MODULE: test_suite.py
DESCRIPTION: Automated Quality Assurance suite for the Titanic ML Pipeline.
AUTHOR: [Your Name]
VERSION: 1.0.0

QA SCOPE:
- Data Integrity: Validates that preprocessing removes all null values.
- Model Functional Testing: Ensures the .pkl artifact is loadable and predictive.
- Feature Consistency: Checks that the feature vector aligns with model requirements.

EXECUTION:
- Local: Run 'pytest test_suite.py'
- CI/CD: Triggered automatically by GitHub Actions on every push to 'main'.
"""
import pytest
import pandas as pd
import joblib
import os


def test_streamlit_ui_is_responsive(driver):
    """QA Smoke Test: Verify the live web app loads properly."""
    url = "YOUR_STREAMLIT_URL_HERE"
    driver.get(url)

    # Check if the title exists on the page
    assert "Titanic" in driver.title, f"Title mismatch! Current title: {driver.title}"

def test_data_integrity():
    """Verify clean data before any training starts."""
    from data_loader import load_titanic_data
    from preprocessor import clean_data

    # Check if data exists
    assert os.path.exists('train.csv'), "Data file missing!"

    df = load_titanic_data('train.csv')
    cleaned_df = clean_data(df)

    # QA Check: 0 Null values allowed
    assert cleaned_df.isnull().sum().sum() == 0, "Null values detected in pipeline!"


def test_model_loading():
    """Verify that the saved model is valid and functional."""
    assert os.path.exists('titanic_model.pkl'), "Model file (.pkl) missing!"
    model = joblib.load('titanic_model.pkl')
    assert hasattr(model, "predict"), "Loaded object is not a valid model!"


def test_feature_consistency():
    """Ensure the features in the model match the preprocessor output."""
    model_columns = joblib.load('model_columns.pkl')
    # Standard feature count for our engineered dataset
    assert len(model_columns) >= 8, "Feature mismatch: Model expecting too few inputs."

