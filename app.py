import streamlit as st
import pandas as pd
import joblib

# Load the trained model and the column names
model = joblib.load('titanic_model.pkl')
model_columns = joblib.load('model_columns.pkl')

st.title("🚢 Titanic Survival Predictor")
st.write("Enter passenger details to see if they would have survived the sinking.")

# User Inputs
pclass = st.selectbox("Ticket Class (1 = 1st, 2 = 2nd, 3 = 3rd)", [1, 2, 3])
sex = st.selectbox("Gender", ["male", "female"])
age = st.slider("Age", 0, 100, 30)
sibsp = st.number_input("Siblings/Spouses Aboard", 0, 10, 0)
parch = st.number_input("Parents/Children Aboard", 0, 10, 0)
fare = st.number_input("Fare Paid", 0.0, 500.0, 32.0)
embarked = st.selectbox("Port of Embarkation", ["S", "C", "Q"])

# Prediction Logic
if st.button("Predict Survival"):
    # Create a dataframe for the input
    input_data = pd.DataFrame([[pclass, sex, age, sibsp, parch, fare, embarked]],
                              columns=['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked'])

    # Simple manual preprocessing to match our pipeline
    input_data['Sex_male'] = 1 if sex == 'male' else 0
    # (Note: For a full app, you'd import your preprocessor.py functions here)

    # Align columns with model
    input_data = input_data.reindex(columns=model_columns, fill_value=0)

    prediction = model.predict(input_data)

    if prediction[0] == 1:
        st.success("✨ This passenger would likely have SURVIVED.")
    else:
        st.error("💀 This passenger would likely NOT have survived.")