import streamlit as st
import pickle
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
import pandas as pd
import numpy as np

# Load trained model
model = tf.keras.models.load_model("model.h5")

# Load StandardScaler, LabelEncoder, OneHotEncoder

with open("label_encoder_gender.pkl", "rb") as file:
    label_encoder_gender = pickle.load(file)

with open("ohe_geo.pkl", "rb") as file:
    ohe_geo = pickle.load(file)

with open("scaler.pkl", "rb") as file:
    scaler = pickle.load(file)

# Streamlit app

st.set_page_config(page_icon="🏛️", page_title="Customer Churn Prediction")
st.title("Customer Churn Prediction")
st.subheader("Predict and Analyze Customer Retention")

geography =st.selectbox("Geography", ohe_geo.categories_[0])
gender = st.selectbox("Gender", label_encoder_gender.classes_)
age = st.slider("Age", 18, 92,value=56)
balance = st.number_input("Balance ($)")
credit_score = st.number_input("Credit Score", min_value=300, max_value=900)
estimated_salary = st.number_input("Estimated Salary")
tenure = st.slider("Tenure", 0, 10, value=5)
num_of_products = st.slider("Number of Products", 1,4, value=2)
has_cr_card = st.selectbox("Has Credit Card?", [0,1])
is_active_member = st.selectbox("Is Active Member", [0,1])

# Prepare the input data
input_data = pd.DataFrame({
    'CreditScore': [credit_score],
    'Gender': [label_encoder_gender.transform([gender])[0]],
    'Age': [age],
    'Tenure': [tenure],
    'Balance': [balance],
    'NumOfProducts': [num_of_products],
    'HasCrCard': [has_cr_card],
    'IsActiveMember': [is_active_member],
    'EstimatedSalary': [estimated_salary]
})

# One-hot encode "Geography"
geo_encode = ohe_geo.transform([[geography]]).toarray()
geo_encoded_columns = [f"Geography_{category}" for category in ohe_geo.categories_[0]]
geo_encoded_df = pd.DataFrame(geo_encode, columns=geo_encoded_columns)

# Combine OHE columns with input data
input_data = pd.concat([input_data.reset_index(drop=True), geo_encoded_df], axis=1)
# Scale the input data
input_data_scaled = scaler.transform(input_data)

# Prediction
prediction = model.predict(input_data_scaled)
prediction_proba = prediction[0][0]

if st.button("Submit"):
    if prediction_proba>0.5:
        st.warning("The Customer is Likely to Churn", icon="⚠️")
        st.warning(f"Churn Probability: {prediction_proba * 100:.2f}%")
    else:
        st.success("The Customer is not Likely to Churn", icon="✅")
        st.success(f"Churn Probability: {prediction_proba * 100:.2f}%")

st.divider()
st.link_button(label="Project Creator", url="https://github.com/amansherjada")