"""
**Purpose**: Production-ready inference.
- **Feature Alignment**: Uses `reindex` to ensure that the `test.csv` has the exact same column structure as the training set (even if some Titles are missing in the test set).
- **Output**: Generates `submission.csv` in the standard Kaggle format.

"""
import pandas as pd
import joblib
from preprocessor import clean_data


def generate_predictions(test_data_path, model_path, columns_path):
    # 1. Load the unseen data and the saved model assets
    test_df = pd.read_csv(test_data_path)
    passenger_ids = test_df['PassengerId']  # Save for the final CSV

    model = joblib.load(model_path)
    model_columns = joblib.load(columns_path)

    # 2. Preprocess the test data
    # Note: We must ensure the test columns match the training columns exactly
    X_test = clean_data(test_df)

    # Handle the specific case of a missing Fare in the test set
    if 'Fare' in X_test.columns:
        X_test['Fare'] = X_test['Fare'].fillna(X_test['Fare'].median())

    # 3. Align Columns
    # This ensures that if a 'Title' exists in train but not test,
    # the model still gets the correct input shape.
    X_test = X_test.reindex(columns=model_columns, fill_value=0)

    # 4. Predict
    predictions = model.predict(X_test)

    # 5. Create Submission DataFrame
    submission = pd.DataFrame({
        "PassengerId": passenger_ids,
        "Survived": predictions
    })

    submission.to_csv('submission.csv', index=False)
    print("Success! 'submission.csv' has been created.")


if __name__ == "__main__":
    # Ensure you have run model_trainer.py first to generate the .pkl files
    generate_predictions('test.csv', 'titanic_model.pkl', 'model_columns.pkl')