"""
**Purpose**: The Orchestrator.
- **Workflow**: Manages the dependency chain between all other scripts. It serves as the single entry point for the user.

"""

import os
import sys
from model_trainer import train_titanic_model
from bias_validator import run_bias_audit
from predict import generate_predictions

def run_pipeline():
    # Define file paths
    TRAIN_DATA = 'train.csv'
    TEST_DATA = 'test.csv'
    MODEL_PATH = 'titanic_model.pkl'
    COLUMNS_PATH = 'model_columns.pkl'

    # Check if data exists
    if not os.path.exists(TRAIN_DATA):
        print(f"Error: {TRAIN_DATA} not found. Please add the dataset to the folder.")
        return

    print("🚀 Starting Titanic Survival Pipeline...\n")


    # Step 1: Training
    print("Step 1: Training Model...")
    train_titanic_model(TRAIN_DATA)
    print("✅ Training Complete.\n")

    # Step 2: Bias Validation
    print("Step 2: Running Bias Audit...")
    run_bias_audit(TRAIN_DATA, MODEL_PATH)
    print("✅ Audit Complete.\n")

    # Step 3: Prediction
    if os.path.exists(TEST_DATA):
        print("Step 3: Generating Final Predictions...")
        generate_predictions(TEST_DATA, MODEL_PATH, COLUMNS_PATH)
        print("✅ submission.csv created.")
    else:
        print("Step 3: Skip (test.csv not found).")

    print("\n🎉 Pipeline execution finished successfully.")

if __name__ == "__main__":
    run_pipeline()