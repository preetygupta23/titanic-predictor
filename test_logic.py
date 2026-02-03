"""
Module: test_logic.py
Purpose: Automated Unit Testing for data integrity.
Logic: Uses 'assert' statements to force failure if data quality standards aren't met.
"""

import pandas as pd
from preprocessor import clean_data
from feature_engineering import run_feature_engineering


def test_preprocessing_integrity():
    """
    Automated test to ensure the cleaning pipeline leaves no nulls
    and produces the correct structure.
    """
    print("🔍 Running Data Integrity Unit Tests...")

    # Load a sample of the raw data
    df = pd.read_csv('train.csv')

    # Process the data
    df_engineered = run_feature_engineering(df)
    df_final = clean_data(df_engineered)

    # 1. TEST: Null Values
    null_count = df_final.isnull().sum().sum()
    assert null_count == 0, f"❌ Test Failed: Found {null_count} missing values after preprocessing!"
    print("✅ Pass: No missing values found.")

    # 2. TEST: Leakage Columns
    forbidden_cols = ['Name', 'Ticket', 'PassengerId', 'Cabin']
    for col in forbidden_cols:
        assert col not in df_final.columns, f"❌ Test Failed: Data leakage! {col} is still in the dataset."
    print("✅ Pass: All leakage columns removed.")

    # 3. TEST: Data Types
    # Ensure all columns are numeric (Random Forest requires numbers, not strings)
    non_numeric = df_final.select_dtypes(exclude=['number', 'bool']).columns.tolist()
    assert len(non_numeric) == 0, f"❌ Test Failed: Non-numeric columns found: {non_numeric}"
    print("✅ Pass: All features are model-ready (numeric).")

    print("\n🏆 ALL LOGIC TESTS PASSED!")


if __name__ == "__main__":
    test_preprocessing_integrity()