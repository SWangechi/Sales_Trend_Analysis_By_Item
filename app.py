import streamlit as st
import pandas as pd
import joblib

# Page config
st.set_page_config(page_title="Cafe Sales Item Predictor", layout="centered")
st.title("☕ Cafe Sales Item Predictor")
st.markdown("Predict which **Item** was sold based on transaction details.")

# Load assets
@st.cache_resource
def load_model():
    return joblib.load("final_rf_model.pkl")

@st.cache_resource
def load_scaler():
    return joblib.load("scaler.pkl")

@st.cache_resource
def load_encoders():
    return (
        joblib.load("payment_encoder.pkl"),
        joblib.load("location_encoder.pkl"),
        joblib.load("item_label_encoder.pkl")
    )

model = load_model()
scaler = load_scaler()
payment_encoder, location_encoder, item_encoder = load_encoders()

# Input
st.sidebar.header("🧾 Transaction Details")
quantity = st.sidebar.number_input("Quantity", min_value=1, value=1)
price_per_unit = st.sidebar.number_input("Price Per Unit", min_value=0.0, value=10.0)
payment_method = st.sidebar.selectbox("Payment Method", list(payment_encoder.classes_))
location = st.sidebar.selectbox("Location", list(location_encoder.classes_))

# Encode inputs
payment_encoded = payment_encoder.transform([payment_method])[0]
location_encoded = location_encoder.transform([location])[0]

input_df = pd.DataFrame({
    'Quantity': [quantity],
    'Price Per Unit': [price_per_unit],
    'Payment Method': [payment_encoded],
    'Location': [location_encoded]
})

# Scale input
input_scaled = scaler.transform(input_df)

# Predict
prediction_encoded = model.predict(input_scaled)[0]

# Check if predicted label is in encoder
if prediction_encoded in range(len(item_encoder.classes_)):
    predicted_item = item_encoder.inverse_transform([prediction_encoded])[0]
else:
    predicted_item = f"Unknown class: {prediction_encoded}"

# Output
st.subheader("🔍 Predicted Item")
st.success(f"🍽️ The predicted item sold is: **{predicted_item}**")

st.subheader("📊 Input Summary")
st.write(input_df)
