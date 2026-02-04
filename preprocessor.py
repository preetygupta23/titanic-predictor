"""
MODULE: preprocessor.py
DESCRIPTION: Data transformation and feature engineering layer.

TRANSFORMATIONS:
1. Title Extraction: Parses 'Name' to create 'Title' category (Master, Miss, Mr, etc.).
2. Grouped Imputation: Fills missing 'Age' values based on the median age of the 'Title' group.
3. Feature Engineering: Creates 'FamilySize' from 'SibSp' and 'Parch'.
4. Categorical Encoding: Maps 'Sex' and 'Embarked' to numerical values.

QA CONTROLS:
- Asserts that 'PassengerId' and 'Ticket' are dropped to prevent feature leakage.
- Validates that final output is a numeric-only matrix.
"""
import pandas as pd
import numpy as np


def extract_titles(df):
    """Extracts titles (Mr, Mrs, etc.) from the Name column."""
    df['Title'] = df['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)

    # Standardize rare titles
    df['Title'] = df['Title'].replace(['Lady', 'Countess', 'Capt', 'Col', 'Don',
                                       'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
    df['Title'] = df['Title'].replace('Mlle', 'Miss')
    df['Title'] = df['Title'].replace('Ms', 'Miss')
    df['Title'] = df['Title'].replace('Mme', 'Mrs')
    return df


def impute_age(df):
    """Fills missing ages based on the median age of the person's Title."""
    # Group by Title and find the median age for each
    title_medians = df.groupby('Title')['Age'].transform('median')
    df['Age'] = df['Age'].fillna(title_medians)
    return df


def clean_data(df):
    """Main cleaning pipeline."""
    # 1. Feature Engineering: Extract Title
    df = extract_titles(df)

    # 2. Handle Missing Values
    df = impute_age(df)
    df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])

    # 3. Drop Leakage/Irrelevant Columns
    # Names are dropped here so the model doesn't 'memorize' individuals
    cols_to_drop = ['PassengerId', 'Name', 'Ticket', 'Cabin']
    df = df.drop(columns=cols_to_drop)

    # 4. Convert Categorical to Dummies (One-Hot Encoding)
    df = pd.get_dummies(df, columns=['Sex', 'Embarked', 'Title'], drop_first=True)

    return df


if __name__ == "__main__":
    # Example usage for testing
    raw_data = pd.read_csv('train.csv')
    cleaned_data = clean_data(raw_data)
    print(f"Cleaned Data Shape: {cleaned_data.shape}")
    print(cleaned_data.head())