# 🚢 Titanic Survival Prediction Pipeline

## Project Overview
This project implements a modular Machine Learning pipeline to predict passenger survival on the Titanic. It moves beyond simple notebooks by using specialized scripts for data loading, feature engineering, bias auditing, and model inference.

## 🏗 Project Architecture
The project follows a "Pipeline as Code" philosophy:
1. **Extraction**: Loading raw data and checking class balance.
2. **Engineering**: Creating socio-demographic features (Family Size, Titles).
3. **Preprocessing**: Handling missing data via Title-based imputation.
4. **Validation**: Auditing for gender and class bias before final prediction.

## 🚀 How to Run
1. Ensure you have the dataset files (`train.csv`, `test.csv`) in the root folder.
2. Install dependencies: `pip install pandas scikit-learn seaborn matplotlib joblib`
3. Execute the entire pipeline:
   ```bash
   python main.py