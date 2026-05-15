import streamlit as st
import pickle
import gzip
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt

# ============================================================
# CONFIG
# ============================================================
st.set_page_config(
    page_title="Walmart Sales Forecaster",
    page_icon="🛒",
    layout="wide"
)

# ============================================================
# LOAD MODEL
# ============================================================
@st.cache_resource
def load_model():
    with gzip.open('walmart_model_compressed.pkl.gz', 'rb') as f:
        model = pickle.load(f)
    return model

model     = load_model()
explainer = shap.TreeExplainer(model)

# ============================================================
# SIDEBAR INPUTS
# ============================================================
st.sidebar.header("Input Parameters")

store        = st.sidebar.slider("Store Number", 1, 45, 1)
holiday      = st.sidebar.selectbox("Holiday Week?", [0, 1],
                   format_func=lambda x: "Yes" if x==1 else "No")
temperature  = st.sidebar.slider("Temperature (°F)", -10, 110, 60)
fuel_price   = st.sidebar.slider("Fuel Price ($)", 2.0, 5.0, 3.5)
cpi          = st.sidebar.slider("CPI", 120.0, 230.0, 180.0)
unemployment = st.sidebar.slider("Unemployment (%)", 3.0, 15.0, 7.0)
year         = st.sidebar.selectbox("Year", [2010, 2011, 2012])
month        = st.sidebar.selectbox("Month", range(1, 13))
week         = st.sidebar.selectbox("Week of Year", range(1, 53))
day          = st.sidebar.slider("Day of Month", 1, 31, 15)

# ============================================================
# MAIN PAGE
# ============================================================
st.title("🛒 Walmart Weekly Sales Forecaster")
st.write("Predict weekly sales for any Walmart store based on key factors.")

if st.button("Predict Weekly Sales 🚀"):

    input_df = pd.DataFrame([{
        'Store':        store,
        'Holiday_Flag': holiday,
        'Temperature':  temperature,
        'Fuel_Price':   fuel_price,
        'CPI':          cpi,
        'Unemployment': unemployment,
        'Year':         year,
        'Month':        month,
        'Week':         week,
        'Day':          day
    }])

    prediction = model.predict(input_df)[0]

    # ── Prediction metrics
    st.markdown("---")
    col1, col2, col3 = st.columns(3)
    col1.metric("Predicted Weekly Sales", f"${prediction:,.2f}")
    col2.metric("Store",                  f"#{store}")
    col3.metric("Holiday Week",           "Yes" if holiday else "No")

    # ── Sales level message
    if prediction > 1_200_000:
        st.success("📈 High sales week — prepare for higher customer footfall.")
    elif prediction > 700_000:
        st.info("📊 Average sales week — normal operations recommended.")
    else:
        st.warning("📉 Lower sales week — consider running promotions.")

    # ── SHAP bar chart
    st.markdown("---")
    st.subheader("🔍 Why this prediction?")
    st.caption("Positive values pushed sales up, negative values pushed sales down.")

    shap_values = explainer.shap_values(input_df)

    shap_df = pd.DataFrame({
        'Feature': [
            f"Store #{store}",
            f"Holiday {'Yes' if holiday else 'No'}",
            f"Temperature {temperature}°F",
            f"Fuel Price ${fuel_price}",
            f"CPI {cpi}",
            f"Unemployment {unemployment}%",
            f"Year {year}",
            f"Month {month}",
            f"Week {week}",
            f"Day {day}"
        ],
        'Impact': shap_values[0]
    }).sort_values('Impact')

    colors = ['#E8413B' if x > 0 else '#4C9BE8' for x in shap_df['Impact']]

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.barh(shap_df['Feature'], shap_df['Impact'], color=colors)

    ax.axvline(x=0, color='black', linewidth=0.8)
    ax.set_xlabel("Impact on Predicted Sales ($)", fontsize=11)
    ax.set_title("Feature Impact on This Prediction", fontsize=13, pad=12)
    ax.tick_params(axis='y', labelsize=11)
    ax.tick_params(axis='x', labelsize=9)

    for bar, val in zip(bars, shap_df['Impact']):
        ax.text(
            val + (max(shap_df['Impact']) * 0.01),
            bar.get_y() + bar.get_height() / 2,
            f"${val:+,.0f}",
            va='center', fontsize=9
        )

    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.clf()

    # ── Holiday comparison
    st.markdown("---")
    st.subheader("🗓️ Holiday Impact for This Store")

    input_regular                 = input_df.copy()
    input_regular['Holiday_Flag'] = 0
    input_holiday                 = input_df.copy()
    input_holiday['Holiday_Flag'] = 1

    regular_pred = model.predict(input_regular)[0]
    holiday_pred = model.predict(input_holiday)[0]
    boost        = round((holiday_pred / regular_pred - 1) * 100, 1)

    c1, c2, c3 = st.columns(3)
    c1.metric("Regular Week",  f"${regular_pred:,.2f}")
    c2.metric("Holiday Week",  f"${holiday_pred:,.2f}")
    c3.metric("Holiday Boost", f"+{boost}%")
