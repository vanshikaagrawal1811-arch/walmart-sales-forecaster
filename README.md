# 🛒 Walmart Weekly Sales Forecaster
A machine learning web app that predicts weekly sales for Walmart stores based on key retail and economic factors.

## 🔗 Live Demo
[Click here to try the app](https://walmart-sales-forecaster-8rjitgxycmzbww6dg4bp56.streamlit.app/)

---

## 📌 Project Overview
Predicted weekly sales across 45 Walmart stores using historical retail data and machine learning.
The app allows users to input store conditions and economic indicators to forecast sales for 2013 —
the first year the model has not seen — helping identify high, average, and low sales weeks to guide business operations.

---

## 📊 Dataset
- **Source:** [Walmart Dataset — Kaggle](https://www.kaggle.com/datasets/yasserh/walmart-dataset)
- **Problem Type:** Regression
- **Period:** Feb 2010 to Nov 2012
- **Size:** 6,435 rows, 8 columns
- **Features:** Store, Holiday Flag, Temperature, Fuel Price, CPI, Unemployment

---

## 🤖 Models Compared

| Model              | R² Score | MAE ($)  |
|--------------------|----------|----------|
| XGBoost (deployed) | 0.97     | $29,843  |
| Random Forest      | 0.96     | $38,421  |
| Decision Tree      | 0.93     | $54,231  |
| Linear Regression  | 0.08     | $211,543 |

> XGBoost was selected as the final model for deployment based on highest R² and lowest MAE.

---

## 📈 Key Findings
- **XGBoost explains 97%** of weekly sales variance — accurate enough for real inventory planning
- **Holiday weeks** drive 7.8% higher sales on average — stores should stock up 2 weeks in advance
- **Month 12 (December)** is the peak sales period across all stores
- **Top predictors: Store, Unemployment, CPI** — weather and fuel price have minimal impact

---

## 🔍 Model Explainability
Integrated **SHAP (SHapley Additive Explanations)** to explain every individual prediction —
showing exactly which factors pushed sales up or down for that specific input.

---

## 🛠️ Tech Stack
- **Language:** Python
- **ML Libraries:** Scikit-learn, XGBoost
- **Explainability:** SHAP
- **Data Processing:** Pandas, NumPy
- **Visualization:** Matplotlib, Seaborn
- **Deployment:** Streamlit
- **Version Control:** GitHub

---

## 📸 App Screenshot
![App Screenshot](screenshot.png)

---

## 🗂️ Project Structure
```
walmart-sales-forecaster/
│
├── app.py                            # Streamlit web app
├── walmart_model_compressed.pkl.gz   # Trained XGBoost model
├── walmart_scaler.pkl                # Feature scaler
├── requirements.txt                  # Python dependencies
├── screenshot.png                    # App screenshot
└── README.md                         # Project documentation
```

---

## ⚠️ Limitations
- Model trained on 2010–2012 data; predictions are intended for 2013
- Economic indicators (CPI, Unemployment) should reflect the target time period
- Not suitable for post-2015 forecasting without retraining on newer data
