import streamlit as st
import pandas as pd
import joblib

# Load model and columns
model = joblib.load('credit_risk_rf_model.pkl')
model_columns = joblib.load('model_columns.pkl')

st.title("Credit Risk Prediction App")

# Collect user input
credit_amount = st.number_input("Credit Amount", min_value=0)
duration_months = st.number_input("Duration (months)", min_value=1)
age = st.number_input("Age", min_value=18, max_value=100)
checking_account_status = st.selectbox("Checking Account Status", ['A11', 'A12', 'A13', 'A14'])
purpose = st.selectbox("Purpose", [
    'A40', 'A41', 'A42', 'A43', 'A44', 'A45', 'A46', 'A47', 'A48', 'A49', 'A410'
])

if st.button("Predict Credit Risk"):
    input_dict = {
        'credit_amount': [credit_amount],
        'duration_months': [duration_months],
        'age': [age],
        'checking_account_status': [checking_account_status],
        'purpose': [purpose],
        # Add default values for all other features your model expects
        'credit_history': ['A34'],
        'savings_account_bonds': ['A65'],
        'employment_since': ['A75'],
        'installment_rate': [2],
        'personal_status_sex': ['A93'],
        'other_debtors': ['A101'],
        'residence_since': [2],
        'Property': ['A124'],
        'other_installment_plans': ['A143'],
        'housing': ['A152'],
        'existing_credits': [1],
        'job': ['A173'],
        'people_liable': [1],
        'telephone': ['A192'],
        'foreign_worker': ['A202']
    }
    df = pd.DataFrame(input_dict)
    df = pd.get_dummies(df)
    df = df.reindex(columns=model_columns, fill_value=0)
    risk_score = model.predict_proba(df)[:, 1][0]
    label = "High Risk" if risk_score > 0.5 else "Low Risk"
    st.write(f"Risk Score: {risk_score:.2f} ({label})")