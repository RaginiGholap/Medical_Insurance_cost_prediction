import streamlit as st
import pickle
import pandas as pd
import numpy as np

# Load trained model
with open("insurance_model.pkl", "rb") as f:
    model = pickle.load(f)

st.set_page_config(page_title="Insurance Charges Predictor", layout="centered")

st.title("💊 Medical Insurance Charges Prediction App")
st.write("Predict insurance charges and understand *why* the prediction is high or low.")

# -------- Sidebar Inputs --------
st.sidebar.header("🧾 Enter Customer Details")

age = st.sidebar.slider("Age", 18, 100, 30)
bmi = st.sidebar.slider("BMI", 10.0, 50.0, 25.0)
children = st.sidebar.selectbox("Number of Children", [0, 1, 2, 3, 4, 5])

sex = st.sidebar.radio("Sex", ["female", "male"])
smoker = st.sidebar.radio("Smoker", ["no", "yes"])
region = st.sidebar.selectbox(
    "Region",
    ["northeast", "northwest", "southeast", "southwest"]
)

# -------- Encoding --------
sex_male = 1 if sex == "male" else 0
smoker_yes = 1 if smoker == "yes" else 0

region_northwest = 1 if region == "northwest" else 0
region_southeast = 1 if region == "southeast" else 0
region_southwest = 1 if region == "southwest" else 0

# -------- Input Data --------
input_data = pd.DataFrame([[
    age,
    bmi,
    children,
    sex_male,
    smoker_yes,
    region_northwest,
    region_southeast,
    region_southwest
]], columns=[
    "age",
    "bmi",
    "children",
    "sex_male",
    "smoker_yes",
    "region_northwest",
    "region_southeast",
    "region_southwest"
])

# -------- Prediction --------
if st.button("🔍 Predict Insurance Charges"):
    prediction = model.predict(input_data)[0]

    # Confidence range (+/- 10%)
    lower = prediction * 0.9
    upper = prediction * 1.1

    st.success(f"💰 **Predicted Insurance Charges:** ₹ {prediction:,.2f}")
    st.info(f"📊 **Expected Range:** ₹ {lower:,.2f} – ₹ {upper:,.2f}")

    # -------- Explainable AI (Feature Impact) --------
    coeffs = model.coef_
    feature_names = input_data.columns

    impact_df = pd.DataFrame({
        "Feature": feature_names,
        "Impact": coeffs * input_data.iloc[0]
    }).sort_values(by="Impact", ascending=False)

    st.subheader("📌 Feature Impact on Prediction")
    st.bar_chart(impact_df.set_index("Feature"))

    # -------- Smart Explanation --------
    top_feature = impact_df.iloc[0]["Feature"]

    explanation_map = {
        "smoker_yes": "Smoking significantly increases insurance charges.",
        "bmi": "Higher BMI leads to increased medical risk.",
        "age": "Insurance cost rises with increasing age.",
        "children": "More dependents slightly increase charges."
    }

    st.subheader("🧠 Insight")
    st.write(explanation_map.get(top_feature, "Multiple factors influence the prediction."))

    # -------- What-If Analysis --------
    if smoker == "yes":
        input_data_no_smoker = input_data.copy()
        input_data_no_smoker["smoker_yes"] = 0
        reduced_charge = model.predict(input_data_no_smoker)[0]

        st.warning(
            f"🚭 If the customer were **not a smoker**, charges could reduce to "
            f"₹ {reduced_charge:,.2f}"
        )
