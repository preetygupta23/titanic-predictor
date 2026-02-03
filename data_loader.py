"""**Purpose**: Handles data ingestion and initial health checks.
- **`load_titanic_data(file_path)`**: Safely reads CSV files and returns a DataFrame or None.
- **`check_class_balance(df)`**: Performs an audit of the target variable (`Survived`).
    - *Metric*: Returns the percentage split.
    - *Goal*: Alert the user if the dataset is too skewed to train effectively.
"""

import pandas as pd

def load_titanic_data(file_path):
    """Loads data and performs initial integrity checks."""
    try:
        df = pd.read_csv(file_path)
        print(f"✅ Successfully loaded {file_path}")
        return df
    except FileNotFoundError:
        print(f"❌ Error: {file_path} not found.")
        return None


def check_class_balance(df):
    """Verifies the ratio of Survived vs Not Survived."""
    if 'Survived' not in df.columns:
        print("ℹ️ No target column found (likely test data).")
        return

    counts = df['Survived'].value_counts(normalize=True) * 100
    print("--- Class Balance ---")
    print(f"Not Survived (0): {counts[0]:.2f}%")
    print(f"Survived (1):     {counts[1]:.2f}%")

    # If the balance is worse than 80/20, we might need resampling.
    # Titanic is usually ~60/40, which is healthy.
    if counts.min() < 20:
        print("⚠️ Warning: Significant class imbalance detected.")
    else:
        print("✅ Class balance is healthy for training.")


if __name__ == "__main__":
    data = load_titanic_data('train.csv')
    if data is not None:
        check_class_balance(data)