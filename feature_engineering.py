"""**Purpose**: Generates new signals from existing data.
- **`create_family_features(df)`**: Calculates `FamilySize` (SibSp + Parch + 1) and creates a binary `IsAlone` flag.
- **`bin_fare(df)`**: Uses quantiles to group the `Fare` column into four categories, reducing the impact of outliers.
- **`run_feature_engineering(df)`**: Orchestrates the order of feature creation.
"""

import pandas as pd


def create_family_features(df):
    """
    Combines SibSp (siblings/spouses) and Parch (parents/children)
    to see if traveling alone impacted survival.
    """
    # FamilySize = individual + siblings + parents
    df['FamilySize'] = df['SibSp'] + df['Parch'] + 1

    # IsAlone: 1 if traveling alone, 0 otherwise
    df['IsAlone'] = 0
    df.loc[df['FamilySize'] == 1, 'IsAlone'] = 1

    return df


def bin_fare(df):
    """Groups Fare into 4 logical quartiles to reduce noise."""
    df['FareBin'] = pd.qcut(df['Fare'], 4, labels=[0, 1, 2, 3])
    return df


def run_feature_engineering(df):
    """Applies all engineering transformations."""
    df = create_family_features(df)
    df = bin_fare(df)

    # Drop the intermediate columns that are now redundant
    # We keep Pclass because it's highly predictive
    return df


if __name__ == "__main__":
    # Test on raw data
    df_raw = pd.read_csv('train.csv')
    df_engineered = run_feature_engineering(df_raw)
    print("New Columns Created:", [col for col in df_engineered.columns if col not in df_raw.columns])
    print(df_engineered[['FamilySize', 'IsAlone', 'FareBin']].head())
