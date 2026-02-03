"""
Module: test_distribution.py
Purpose: Statistical validation of the Age imputation logic.
Author: Preety Gupta
Date: 2026

"""
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from preprocessor import extract_titles, impute_age


def verify_age_distribution(file_path):
    # Load raw data
    df = pd.read_csv(file_path)

    # Capture distribution BEFORE imputation (dropping NaNs for the plot)
    before = df['Age'].dropna()

    # Apply our imputation logic
    df = extract_titles(df)
    df_after = impute_age(df)
    after = df_after['Age']

    # Plotting
    plt.figure(figsize=(10, 6))
    sns.kdeplot(before, label='Original (Non-missing)', fill=True, color="skyblue")
    sns.kdeplot(after, label='After Title-Based Imputation', fill=True, color="orange", alpha=0.4)

    plt.title('Age Distribution: Original vs. Imputed')
    plt.xlabel('Age')
    plt.ylabel('Density')
    plt.legend()
    plt.show()

    # Quantitative Check
    print(f"Original Mean: {before.mean():.2f} | Imputed Mean: {after.mean():.2f}")
    print(f"Original Std: {before.std():.2f} | Imputed Std: {after.std():.2f}")


if __name__ == "__main__":
    verify_age_distribution('train.csv')