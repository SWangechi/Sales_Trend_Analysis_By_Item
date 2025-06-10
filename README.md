# ☕ Cafe Sales Item Prediction App

This project is a **machine learning-powered web application** that predicts the most likely item sold at a café based on transaction details, including quantity, price, payment method, and location.

Built using `scikit-learn` for model training and `Streamlit` for a simple, interactive interface, the app demonstrates how to deploy predictive models in a real-world business scenario.

---

## 🎯 Objective

To accurately **predict the specific item sold** in a sales transaction using structured input data. This can help automate transaction analysis, customer behaviour prediction, and sales pattern recognition.

---

## 🧠 Machine Learning Workflow

1. **Data Preprocessing**

   * Cleaned and transformed raw café sales data.
   * Encoded categorical features (`Payment Method`, `Location`, `Item`) using `LabelEncoder`.
   * Applied `StandardScaler` to numerical features.

2. **Model Training**

   * Used `RandomForestClassifier` to predict the `Item` based on transaction inputs.
   * Trained on features: `Quantity`, `Price Per Unit`, `Payment Method`, and `Location`.

3. **Model Saving**

   * Trained model and encoders saved using `joblib`:

     * `final_rf_model.pkl`
     * `scaler.pkl`
     * `payment_encoder.pkl`
     * `location_encoder.pkl`
     * `item_label_encoder.pkl`

---

## 🚀 App Features

* Built with [**Streamlit**](https://streamlit.io/)
* Sidebar input for:

  * Quantity
  * Price per Unit
  * Payment Method
  * Location
* Real-time prediction of the item sold
* Shows input summary and results
* Fully offline and local inference

---

## 🛠️ Project Structure

```
Sales_Trend_Analysis_By_Item/
├── app.py                              # Streamlit app for prediction
├── final_rf_model.pkl                  # Trained Random Forest model
├── scaler.pkl                          # Fitted StandardScaler
├── payment_encoder.pkl                 # LabelEncoder for Payment Method
├── location_encoder.pkl                # LabelEncoder for Location
├── item_label_encoder.pkl              # LabelEncoder for Item
├── enhanced_cafe_sales_analysis.ipynb  # Notebook to preprocess, train & save model
├── dirty_cafe_sales.csv                # Input sales dataset (replace with your actual file)
```

---

## ▶️ How to Run the App

1. **Install Requirements**

   ```bash
   pip install streamlit pandas scikit-learn joblib
   ```

2. **Train the Model (Optional but Recommended for First Use)**
   Open and run `train_model.ipynb` to:

   * Fit and encode the data
   * Save all required files (`*.pkl`)

3. **Start the App**

   ```bash
   streamlit run app.py
   ```

4. **Use the Web UI** to make predictions based on sales inputs.

---

## 📌 Example Prediction

Given:

* Quantity: `2`
* Price per Unit: `10.0`
* Payment Method: `Credit Card`
* Location: `Takeaway`

**Output**:

```
🍽️ The predicted item sold is: Cappuccino
```

---

## ✅ Future Enhancements

* Add visual insights in the app (e.g., item popularity)
* Support time-series forecasting (Prophet, ARIMA)
* Add user authentication for internal use

---
## Deployed app link:
https://salestrendanalysisbyitem.streamlit.app/
---

## 👨‍💻 Author

**\Stella Gituire** – Data Scientist & ML Engineer
GitHub: [@Stella Gituire](https://github.com/SWangechi)

---
