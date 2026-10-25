from pathlib import Path
import joblib
import pandas as pd
import streamlit as st

MODEL_PATH = Path(__file__).resolve().parent / "SuperKart_prediction_model_v1_0.joblib"

st.set_page_config(page_title="SuperKart Sales Predictor", page_icon="🛒")
st.title("🛒 SuperKart Sales Predictor")
st.write("Enter the product and store attributes to estimate total sales.")

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

model = load_model()

st.subheader("Single prediction")
with st.form("prediction_form"):
    product_weight = st.number_input("Product Weight", min_value=0.0, value=12.0)
    sugar = st.selectbox("Product Sugar Content", ["Low Sugar", "Regular", "No Sugar"])
    allocated_area = st.number_input("Product Allocated Area", min_value=0.0, value=0.05)
    mrp = st.number_input("Product MRP", min_value=0.0, value=150.0)
    product_type = st.text_input("Product Type", "Snack Foods")
    product_id = st.text_input("Product ID", "FD1")
    store_id = st.text_input("Store ID", "OUT049")
    establishment_year = st.number_input("Store Establishment Year", min_value=1900, max_value=2100, value=2000)
    store_size = st.selectbox("Store Size", ["Low", "Medium", "High"])
    city_type = st.selectbox("Store Location City Type", ["Tier 1", "Tier 2", "Tier 3"])
    store_type = st.selectbox("Store Type", ["Departmental Store", "Supermarket Type1", "Supermarket Type2", "Food Mart"])
    submitted = st.form_submit_button("Predict sales")

if submitted:
    row = pd.DataFrame([{
        "Product_Weight": product_weight,
        "Product_Sugar_Content": sugar,
        "Product_Allocated_Area": allocated_area,
        "Product_Type": product_type,
        "Product_MRP": mrp,
        "Product_Id": product_id,
        "Store_Id": store_id,
        "Store_Establishment_Year": establishment_year,
        "Store_Size": store_size,
        "Store_Location_City_Type": city_type,
        "Store_Type": store_type,
    }])
    try:
        prediction = float(model.predict(row)[0])
        st.success(f"Predicted total sales: {prediction:,.2f}")
    except Exception as exc:
        st.error(f"Prediction failed. Check that the input fields match the training dataset. Details: {exc}")

st.subheader("Batch predictions")
uploaded = st.file_uploader("Upload a CSV containing the model feature columns", type=["csv"])
if uploaded is not None:
    batch = pd.read_csv(uploaded)
    st.write("Input preview", batch.head())
    if st.button("Predict batch"):
        try:
            batch["Predicted_Product_Store_Sales_Total"] = model.predict(batch)
            st.dataframe(batch)
            st.download_button("Download predictions", batch.to_csv(index=False), "superkart_predictions.csv", "text/csv")
        except Exception as exc:
            st.error(f"Batch prediction failed. Ensure the CSV has the same feature columns as the training data. Details: {exc}")
