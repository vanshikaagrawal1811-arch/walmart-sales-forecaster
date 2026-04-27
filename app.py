import streamlit as st
import pickle
import gzip
import numpy as np

# Load compressed model
with gzip.open('walmart_model_compressed.pkl', 'rb') as f:
    model = pickle.load(f)

# Load scaler
scaler = pickle.load(open('walmart_scaler.pkl', 'rb'))

st.set_page_config(page_title="Walmart Sales Forecaster", page_icon="🛒")

st.title("🛒 Walmart Weekly Sales Forecaster")
st.write("Predict weekly sales for any Walmart store based on key factors")

st.sidebar.header("Input Parameters")

store = st.sidebar.slider("Store Number", 1, 45, 1)
holiday = st.sidebar.selectbox("Holiday Week?", [0, 1],
                               format_func=lambda x: "Yes" if x==1 else "No")
temperature = st.sidebar.slider("Temperature (°F)", -10, 110, 60)
fuel_price = st.sidebar.slider("Fuel Price ($)", 2.0, 5.0, 3.5)
cpi = st.sidebar.slider("CPI", 120.0, 230.0, 180.0)
unemployment = st.sidebar.slider("Unemployment (%)", 3.0, 15.0, 7.0)
year = st.sidebar.selectbox("Year", [2010, 2011, 2012])
month = st.sidebar.selectbox("Month", range(1, 13))
week = st.sidebar.selectbox("Week of Year", range(1, 53))
day = st.sidebar.slider("Day of Month", 1, 31, 15)

if st.button("Predict Weekly Sales 🚀"):
    features = np.array([[store, holiday, temperature,
                          fuel_price, cpi, unemployment,
                          year, month, week, day]])
    features_scaled = scaler.transform(features)
    prediction = model.predict(features_scaled)[0]

    st.success(f"💰 Predicted Weekly Sales: ${prediction:,.2f}")

    if prediction > 1200000:
        st.info("📈 High sales week expected — prepare for higher customer footfall")
    elif prediction > 700000:
        st.info("📊 Average sales week expected — normal operations recommended")
    else:
        st.warning("📉 Lower sales week expected — consider running promotions to boost sales")
