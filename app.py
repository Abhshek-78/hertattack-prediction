import streamlit as st
import pandas as pd
import joblib


st.set_page_config(
    page_title="Heart Disease Prediction",
    page_icon="❤️",
    layout="wide"
)


model = joblib.load("knn_heart.pkl")
scaler = joblib.load("Sscaler.pkl")
expected_columns = joblib.load("columns.pkl")


st.title("❤️ Heart Disease Prediction Dashboard")
st.markdown("AI powered clinical decision support system")


st.sidebar.header("Patient Information")

age = st.sidebar.slider("Age", 18, 100, 40)
sex = st.sidebar.selectbox("Sex", ["Male", "Female"])
fastinbs = st.sidebar.selectbox("Fasting Blood Sugar >120", [0, 1])


col1, col2 = st.columns(2)

with col1:
    st.subheader("🫀 Cardiac Parameters")

    chest_pain = st.selectbox("Chest Pain Type", ["ATA", "NAP", "TA", "ASY"])
    resting_BP = st.number_input("Resting Blood Pressure", 80, 200, 120)
    cholesterol = st.number_input("Cholesterol (mg/dl)", 100, 600, 200)

    resting_ecg = st.selectbox("Resting ECG", ["Normal", "ST", "LVH"])

with col2:
    st.subheader("🏃 Exercise Parameters")

    max_hr = st.slider("Max Heart Rate", 60, 220, 150)
    exercise_angina = st.selectbox("Exercise-Induced Angina", ["Y", "N"])
    oldpeak = st.slider("Oldpeak (ST Depression)", 0.0, 6.0, 1.0)
    st_slope = st.selectbox("ST Slope", ["Up", "Flat", "Down"])

st.divider()


if st.button("🔍 Predict Heart Disease Risk"):

    raw_input = {
        'Age': age,
        'Sex': sex,
        'ChestPainType': chest_pain,
        'RestingBP': resting_BP,
        'Cholesterol': cholesterol,
        'FastingBS': fastinbs,
        'RestingECG': resting_ecg,
        'MaxHR': max_hr,
        'ExerciseAngina': exercise_angina,
        'Oldpeak': oldpeak,
        'ST_Slope': st_slope
    }

    input_df = pd.DataFrame([raw_input])

    for col in expected_columns:
        if col not in input_df.columns:
            input_df[col] = 0

    input_df = input_df[expected_columns]

    scaled_input = scaler.transform(input_df)

    prediction = model.predict(scaled_input)[0]

    st.subheader("🧾 Prediction Result")

    if prediction == 1:
        st.error("⚠️ High Risk of Heart Disease")
    else:
        st.success("✅ Low Risk of Heart Disease")