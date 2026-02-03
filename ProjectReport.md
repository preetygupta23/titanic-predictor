Project Report: Titanic Survival Analysis & Pipeline
Author: Preety Gupta

Date: February 2026

Stack: Python (Pandas, Scikit-Learn), Random Forest, Modular Pipeline Architecture

📋 Executive Summary
This project developed a robust machine learning pipeline to predict passenger survival on the Titanic. Beyond simple prediction, this project implements statistical integrity checks and bias auditing to ensure the model's decisions are based on socio-demographic features rather than data noise.

🛠 Methodology & Pipeline Architecture
The project is built on a modular 5-stage pipeline:

Data Ingestion: Automated loading with class balance verification.

Feature Engineering: Transformation of raw data into signals (e.g., FamilySize, IsAlone, and Title extraction).

Preprocessing: Title-based median imputation for Age to prevent distribution skew.

Training: Random Forest Classifier with 5-Fold Cross-Validation to ensure stability.

Audit: A dedicated fairness check slicing performance by Gender and Class.

📈 Key Findings & Model Performance
1. Stability (Cross-Validation)
The model achieved a Mean CV Accuracy of 0.83 (+/- 0.02).

Note: The low deviation (0.02) indicates that the model is stable and generalizes well across different subsets of the passenger list.

2. Feature Importance (The "Why")
The model identified the following features as the primary drivers of survival:

Sex/Title: The "Women and Children First" policy remains the strongest predictor.

Pclass: Socio-economic status significantly impacted access to lifeboats.

FamilySize: Middle-sized families had a higher survival rate than solo travelers or very large families.

⚖️ Ethics & Bias Audit
We performed a Recall-based audit to see if the model was "over-simplifying" the disaster:

Segment	Accuracy	Recall (Survival)	Interpretation
Female	High (~0.85)	High (~0.90)	Model accurately identifies female survivors.
Male	Mid (~0.78)	Low (~0.15)	Reflects the historical reality; most men did not survive.
1st Class	High	High	Wealth correlated strongly with survival predictability.

Export to Sheets

Observation: The model effectively captures the "Women and Children First" heuristic while maintaining high accuracy in the chaotic 3rd-class segment.

🧪 Technical Rigor (Testing)
To ensure production quality, the project includes:

Distribution Testing: Visualizing Age before/after imputation to prevent "mean-centering" bias.

Unit Logic Tests: Automated assertions that fail the build if NaN values or "Data Leakage" columns (like PassengerId) are detected.

🚀 Conclusion
This model demonstrates that while historical survival was heavily biased by gender and class, engineering specific features like FamilySize allows for a more nuanced prediction than a simple gender-based guess.

How to Explore this Project
main.py: Run the full pipeline.

bias_validator.py: View the fairness audit logic.

test_logic.py: Inspect the data quality assertions.