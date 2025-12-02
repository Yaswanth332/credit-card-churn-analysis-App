import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Load pipeline & threshold
pipeline = joblib.load('churn_pipeline.pkl')
best_threshold = joblib.load('best_threshold.pkl')

st.title("Bank Customer Churn Prediction 🏦")
st.write("Predict whether a customer is likely to churn based on their profile and activity.")

# ---- Input form ----
st.sidebar.header("Customer Information")

# Basic inputs
customer_age = st.sidebar.number_input("Customer Age", min_value=18, max_value=100, value=45)
gender = st.sidebar.selectbox("Gender", ["M", "F"])
dependent_count = st.sidebar.number_input("Dependent Count", min_value=0, max_value=10, value=2)

education_level = st.sidebar.selectbox("Education Level", [
    "Uneducated", "High School", "College", "Graduate",
    "Post-Graduate", "Doctorate", "Unknown"
])

marital_status = st.sidebar.selectbox("Marital Status", [
    "Single", "Married", "Divorced", "Unknown"
])

card_category = st.sidebar.selectbox("Card Category", [
    "Blue", "Silver", "Gold", "Platinum"
])

income_category = st.sidebar.selectbox("Income Category", [
    "Less than $40K", "$40K - $60K", "$60K - $80K",
    "$80K - $120K", "$120K +"
])

months_on_book = st.sidebar.number_input("Months on Book (Tenure)", min_value=6, max_value=100, value=39)
total_relationship_count = st.sidebar.number_input("Total Relationship Count", min_value=1, max_value=10, value=5)
months_inactive_12_mon = st.sidebar.number_input("Months Inactive (Last 12 Months)", min_value=0, max_value=12, value=1)
contacts_count_12_mon = st.sidebar.number_input("Contacts Count (Last 12 Months)", min_value=0, max_value=20, value=3)

credit_limit = st.sidebar.number_input("Credit Limit", min_value=100.0, max_value=100000.0, value=12691.0)
total_revolving_bal = st.sidebar.number_input("Total Revolving Balance", min_value=0.0, max_value=100000.0, value=777.0)
avg_open_to_buy = st.sidebar.number_input("Avg Open To Buy", min_value=0.0, max_value=100000.0, value=11914.0)

total_amt_chng_q4_q1 = st.sidebar.number_input("Total Amount Change (Q4/Q1)", min_value=0.0, max_value=10.0, value=1.33)
total_trans_amt = st.sidebar.number_input("Total Transaction Amount (Last 12 Months)", min_value=0.0, max_value=100000.0, value=1144.0)
total_trans_ct = st.sidebar.number_input("Total Transaction Count (Last 12 Months)", min_value=0, max_value=300, value=42)
total_ct_chng_q4_q1 = st.sidebar.number_input("Total Transaction Count Change (Q4/Q1)", min_value=0.0, max_value=10.0, value=1.62)
avg_utilization_ratio = st.sidebar.number_input("Avg Utilization Ratio", min_value=0.0, max_value=1.0, value=0.06)

# Create a single-row DataFrame in the SAME schema as training
input_dict = {
    "Customer_Age": [customer_age],
    "Gender": [gender],
    "Dependent_count": [dependent_count],
    "Education_Level": [education_level],
    "Marital_Status": [marital_status],
    "Card_Category": [card_category],
    "Months_on_book": [months_on_book],
    "Total_Relationship_Count": [total_relationship_count],
    "Months_Inactive_12_mon": [months_inactive_12_mon],
    "Contacts_Count_12_mon": [contacts_count_12_mon],
    "Credit_Limit": [credit_limit],
    "Total_Revolving_Bal": [total_revolving_bal],
    "Avg_Open_To_Buy": [avg_open_to_buy],
    "Total_Amt_Chng_Q4_Q1": [total_amt_chng_q4_q1],
    "Total_Trans_Amt": [total_trans_amt],
    "Total_Trans_Ct": [total_trans_ct],
    "Total_Ct_Chng_Q4_Q1": [total_ct_chng_q4_q1],
    "Avg_Utilization_Ratio": [avg_utilization_ratio],
    "Income_Category": [income_category]
}

input_df = pd.DataFrame(input_dict)

st.subheader("Input Data Preview")
st.write(input_df)

if st.button("Predict Churn"):
    # Get probability of churn (class 1)
    prob = pipeline.predict_proba(input_df)[:, 1][0]
    pred = int(prob >= best_threshold)

    st.subheader("Prediction Result")
    if pred == 1:
        st.error(f"⚠️ This customer is **LIKELY TO CHURN**.\n\nChurn probability: **{prob:.2%}**")
    else:
        st.success(f"✅ This customer is **UNLIKELY TO CHURN**.\n\nChurn probability: **{prob:.2%}**")

    st.caption(f"Decision threshold used: {best_threshold:.3f}")
