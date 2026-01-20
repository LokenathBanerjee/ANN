import streamlit as st
import pandas as pd
import pickle
import tensorflow as tf

# -------------------------------------------------
# Page Config
# -------------------------------------------------
st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📊",
    layout="centered"
)
st.image('pic.jpg', use_container_width=True)
# -------------------------------------------------
# Title
# -------------------------------------------------
st.title(":blue[Customer Churn Prediction]")
st.caption(
    "Predict whether a customer is likely to churn using a trained ANN model."
)

st.markdown("---")

# -------------------------------------------------
# Load Model & Preprocessor
# -------------------------------------------------
@st.cache_resource
def load_assets():
    model = tf.keras.models.load_model("Saved_Model/churn_model.h5")
    with open("Saved_Model/preprocessor.pkl", "rb") as f:
        preprocessor = pickle.load(f)
    return model, preprocessor

model, preprocessor = load_assets()

# -------------------------------------------------
# Prediction Function
# -------------------------------------------------
def predict_churn(input_df):
    processed = preprocessor.transform(input_df)
    prob = model.predict(processed)[0][0]

    st.markdown("---")
    st.subheader("🔍 Prediction Result")
    st.metric("Churn Probability", f"{prob:.2f}")

    if prob > 0.5:
        st.error("⚠️ Customer is likely to churn.")
    else:
        st.success("✅ Customer is not likely to churn.")

# =================================================
# 🧍 PERSONAL DETAILS
# =================================================
st.subheader("🧍 Personal Details")

credit_score = st.number_input(
    "Credit Score",
    min_value=300,
    max_value=900,
    value=650
)

st.subheader("Age")

# Initialize session state (important)
if "age" not in st.session_state:
    st.session_state.age = 30

col1, col2 = st.columns([3, 1])

with col1:
    st.session_state.age = st.slider(
        "Age",
        min_value=18,
        max_value=92,
        value=st.session_state.age,
        key="age_slider"
    )

with col2:
    st.session_state.age = st.number_input(
        "Type Age",
        min_value=18,
        max_value=92,
        value=st.session_state.age,
        step=1,
        key="age_input"
    )

age = st.session_state.age


gender = st.selectbox(
    "Gender",
    ["Male", "Female"]
)

geography = st.selectbox(
    "Country",
    ["France", "Germany", "Spain"]
)

st.markdown("---")

# =================================================
# 🏦 ACCOUNT DETAILS
# =================================================
st.subheader("🏦 Account Details")

tenure = st.number_input(
    "Tenure (Years)",
    min_value=0,
    max_value=10,
    value=3,
    step=1
)


num_of_products = st.number_input(
    "Number of Products",
    min_value=1,
    max_value=4,
    value=1,
    step=1
)

has_cr_card = st.radio(
    "Has Credit Card?",
    ["Yes", "No"]
)

is_active_member = st.radio(
    "Is Active Member?",
    ["Yes", "No"]
)

st.markdown("---")

# =================================================
# 💰 FINANCIAL INFORMATION
# =================================================
st.subheader("💰 Financial Information")

balance = st.number_input(
    "Account Balance",
    value=50000.0
)

estimated_salary = st.number_input(
    "Estimated Salary",
    value=60000.0
)

st.markdown("---")

# -------------------------------------------------
# Button
# -------------------------------------------------
if st.button("🚀 Predict Churn", use_container_width=True):

    input_data = pd.DataFrame({
        "CreditScore": [credit_score],
        "Geography": [geography],
        "Gender": [gender],
        "Age": [age],
        "Tenure": [tenure],
        "Balance": [balance],
        "NumOfProducts": [num_of_products],
        "HasCrCard": [1 if has_cr_card == "Yes" else 0],
        "IsActiveMember": [1 if is_active_member == "Yes" else 0],
        "EstimatedSalary": [estimated_salary]
    })

    predict_churn(input_data)


# -------------------------------------------------
# Footer
# -------------------------------------------------
st.markdown("---")
st.caption(" 2026 Customer Churn Prediction | ANN • TensorFlow • Streamlit")
