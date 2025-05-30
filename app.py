import streamlit as st
import pandas as pd
import pickle
import shap
import matplotlib.pyplot as plt

# Load model and data
@st.cache_data
def load_model():
    with open('best_model.pkl', 'rb') as f:
        model = pickle.load(f)
    return model

@st.cache_data
def load_data():
    return pd.read_csv('encoded_dataset.csv')

model = load_model()
data = load_data()

st.set_page_config(page_title="Sales Predictor", layout="wide")
st.title("📈 Sales Trend Predictor")
st.markdown("Use the inputs below to predict sales for a specific item.")

# Sidebar for inputs
st.sidebar.header("🛠️ Input Parameters")

exclude_cols = [
    'Transaction ID', 'Transaction Date', 'Total Spent',
    'target', 'sales', 'item_id', 'item_name'
]

input_features = [col for col in data.columns if col not in exclude_cols]

input_data = {}
for col in input_features:
    if data[col].nunique() <= 10:
        options = sorted(data[col].dropna().unique().tolist())
        input_data[col] = st.sidebar.selectbox(f"{col}", options)
    else:
        min_val = float(data[col].min())
        max_val = float(data[col].max())
        default_val = float(data[col].mean())
        input_data[col] = st.sidebar.slider(f"{col}", min_val, max_val, default_val)

# Create DataFrame from input
input_df = pd.DataFrame([input_data])

# Make Prediction
try:
    prediction = model.predict(input_df)[0]
    st.subheader("🔍 Predicted Sales:")
    st.success(f"💰 Estimated sales: **{prediction:.2f} units**")

    # Show input summary
    st.subheader("📊 Input Summary")
    st.dataframe(input_df.T.rename(columns={0: "Value"}))

    # SHAP interpretation
    st.subheader("🔍 Feature Contribution (SHAP)")
    explainer = shap.Explainer(model)
    shap_values = explainer(input_df)

    # Display SHAP bar chart (for single prediction)
    st.pyplot(shap.plots.bar(shap_values[0], show=False))

    # Optional: SHAP force plot
    with st.expander("🌈 Show SHAP Force Plot"):
        st.set_option('deprecation.showPyplotGlobalUse', False)
        shap.initjs()
        fig = shap.plots.force(shap_values[0], matplotlib=True, show=False)
        st.pyplot(fig)

    # 🔽 Download Prediction Summary
    st.subheader("⬇️ Download Prediction Summary")
    summary_df = input_df.copy()
    summary_df["Predicted Sales"] = prediction

    csv = summary_df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Download Prediction Summary as CSV",
        data=csv,
        file_name='sales_prediction_summary.csv',
        mime='text/csv'
    )

except Exception as e:
    st.error("🚨 Prediction or explanation failed.")
    st.exception(e)

# Show sample data
if st.checkbox("Show Sample of Training Data"):
    st.dataframe(data.sample(10))
