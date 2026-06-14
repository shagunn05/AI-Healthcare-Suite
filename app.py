import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model

st.title("ANN Test Page")

model = load_model("Models/ann_model.keras")

glucose = st.number_input("Glucose")
bp = st.number_input("Blood Pressure")
bmi = st.number_input("BMI")

if st.button("Predict"):
    data = np.array([[glucose, bp, bmi]])
    pred = model.predict(data)
    st.success(pred)