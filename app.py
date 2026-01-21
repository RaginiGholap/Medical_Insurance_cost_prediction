import streamlit as st
import pickle
import pandas as pd

# Load trained model
with open("insurance_model.pkl", "rb") as f:
    model = pickle.load(f)

st.title("Insurance Charges Prediction App")

st.write("Enter customer details to predict insurance charges")

# -------- User Inputs --------
age = st.number_input("Age", min_value=0)
bmi = st.number_input("BMI")
children = st.number_input("Number of Children", min_value=0, step=1)

sex = st.selectbox("Sex", ["female", "male"])
smoker = st.selectbox("Smoker", ["no", "yes"])
region = st.selectbox(
    "Region",
    ["northeast", "northwest", "southeast", "southwest"]
)

# -------- Encoding Inputs --------
sex_male = 1 if sex == "male" else 0
smoker_yes = 1 if smoker == "yes" else 0

region_northwest = 1 if region == "northwest" else 0
region_southeast = 1 if region == "southeast" else 0
region_southwest = 1 if region == "southwest" else 0

# -------- Prediction --------
if st.button("Predict Insurance Charges"):
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

    prediction = model.predict(input_data)

    st.success(f"Predicted Insurance Charges: ₹ {prediction[0]:.2f}")
