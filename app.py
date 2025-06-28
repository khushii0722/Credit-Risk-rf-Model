import streamlit as st
import pandas as pd
import joblib

# 1. Define the mapping dictionary FIRST
checking_account_options = {
    "< 0 DM": "A11",
    "0 <= ... < 200 DM": "A12",
    ">= 200 DM or salary assignments": "A13",
    "No checking account": "A14"
}

# 2. Then use it in your selectbox
checking_account_status_label = st.selectbox("Checking Account Status", list(checking_account_options.keys()))
checking_account_status = checking_account_options[checking_account_status_label]
import streamlit as st
import pandas as pd
import joblib

# Step 1: Mapping dictionaries (put here)
checking_account_options = { ... }
purpose_options = { ... }
# (add other mappings as needed)

# Load model and columns
model = joblib.load('credit_risk_rf_model.pkl')
model_columns = joblib.load('model_columns.pkl')

st.title("Credit Risk Prediction App")

# Step 2: User-friendly dropdowns
checking_account_status_label = st.selectbox("Checking Account Status", list(checking_account_options.keys()))
checking_account_status = checking_account_options[checking_account_status_label]

purpose_label = st.selectbox("Purpose", list(purpose_options.keys()))
purpose = purpose_options[purpose_label]

# (collect other inputs...)

if st.button("Predict Credit Risk"):
    # Step 3: Use the codes in your input dictionary
    input_dict = {
        'credit_amount': [credit_amount],
        'duration_months': [duration_months],
        'age': [age],
        'checking_account_status': [checking_account_status],
        'purpose': [purpose],
        # ...other features...
    }
    # (rest of your prediction code)
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
    # Checking Account Status
checking_account_options = {
    "< 0 DM": "A11",
    "0 <= ... < 200 DM": "A12",
    ">= 200 DM or salary assignments": "A13",
    "No checking account": "A14"
}

# Purpose
purpose_options = {
    "Car (new)": "A40",
    "Car (used)": "A41",
    "Furniture/equipment": "A42",
    "Radio/TV": "A43",
    "Domestic appliance": "A44",
    "Repairs": "A45",
    "Education": "A46",
    "Vacation": "A47",
    "Retraining": "A48",
    "Business": "A49",
    "Others": "A410"
}

# Credit History
credit_history_options = {
    "No credits taken/All paid back duly": "A30",
    "All credits at this bank paid back duly": "A31",
    "Existing credits paid back duly till now": "A32",
    "Delay in paying off in the past": "A33",
    "Critical account/other credits existing": "A34"
}

# Savings Account/Bonds
savings_account_options = {
    "< 100 DM": "A61",
    "100 <= ... < 500 DM": "A62",
    "500 <= ... < 1000 DM": "A63",
    ">= 1000 DM": "A64",
    "Unknown/No savings account": "A65"
}

# Employment Since
employment_since_options = {
    "Unemployed": "A71",
    "< 1 year": "A72",
    "1 <= ... < 4 years": "A73",
    "4 <= ... < 7 years": "A74",
    ">= 7 years": "A75"
}

# Personal Status/Sex
personal_status_options = {
    "Male, single": "A91",
    "Male, married/widowed": "A92",
    "Male, divorced/separated": "A93",
    "Female, divorced/separated/married": "A94"
}

# Other Debtors
other_debtors_options = {
    "None": "A101",
    "Co-applicant": "A102",
    "Guarantor": "A103"
}

# Property
property_options = {
    "Real estate": "A121",
    "Building society savings/life insurance": "A122",
    "Car or other": "A123",
    "Unknown/No property": "A124"
}

# Other Installment Plans
other_installment_options = {
    "Bank": "A141",
    "Stores": "A142",
    "None": "A143"
}

# Housing
housing_options = {
    "Own": "A151",
    "For free": "A152",
    "Rent": "A153"
}

# Job
job_options = {
    "Unemployed/unskilled - non-resident": "A171",
    "Unskilled - resident": "A172",
    "Skilled employee/official": "A173",
    "Management/self-employed/highly qualified": "A174"
}

# Telephone
telephone_options = {
    "None": "A191",
    "Yes, registered under the customer’s name": "A192"
}

# Foreign Worker
foreign_worker_options = {
    "Yes": "A201",
    "No": "A202"
}
