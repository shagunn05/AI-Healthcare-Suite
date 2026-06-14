import streamlit as st
import numpy as np
import os
from tensorflow.keras.models import load_model
from PIL import Image

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="AI Healthcare Suite",
    page_icon="🏥",
    layout="wide"
)

# ---------------- BASE PATH (IMPORTANT FIX) ----------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "Models")

# ---------------- MODEL LOADING (CACHE FOR SPEED + NO RELOAD ERROR) ----------------
@st.cache_resource
def load_ann():
    return load_model(os.path.join(MODEL_DIR, "ann_model.keras"))

@st.cache_resource
def load_cnn():
    return load_model(os.path.join(MODEL_DIR, "cnn_pneumonia.h5"))

@st.cache_resource
def load_lstm():
    return load_model(os.path.join(MODEL_DIR, "lstm_model.keras"))


ann_model = load_ann()
cnn_model = load_cnn()
lstm_model = load_lstm()

# ---------------- SIDEBAR MENU ----------------
menu = st.sidebar.radio(
    "Choose Model",
    ["Home", "ANN Diabetes", "CNN Pneumonia", "RNN/LSTM/GRU Time Series"],
    key="main_menu"
)

# ---------------- HOME ----------------
if menu == "Home":
    st.title("🏥 AI Healthcare Suite")
    st.write("ANN + CNN + RNN/LSTM/GRU integrated system")

# ---------------- ANN ----------------
elif menu == "ANN Diabetes":
    st.header("🩺 Diabetes Prediction (ANN)")

    col1, col2 = st.columns(2)

    with col1:
        pregnancies = st.number_input("Pregnancies", 0, 20, key="p1")
        glucose = st.number_input("Glucose", 0, 300, key="p2")
        bp = st.number_input("Blood Pressure", 0, 200, key="p3")
        skin = st.number_input("Skin Thickness", 0, 100, key="p4")

    with col2:
        insulin = st.number_input("Insulin", 0, 900, key="p5")
        bmi = st.number_input("BMI", 0.0, 70.0, key="p6")
        dpf = st.number_input("Diabetes Pedigree", 0.0, 3.0, key="p7")
        age = st.number_input("Age", 0, 120, key="p8")

    if st.button("Predict Diabetes"):
        input_data = np.array([[pregnancies, glucose, bp, skin,
                                 insulin, bmi, dpf, age]])

        prediction = ann_model.predict(input_data)[0][0]

        if prediction > 0.5:
            st.error("⚠️ High Chance of Diabetes")
        else:
            st.success("✅ Low Risk of Diabetes")

# ---------------- CNN ----------------
elif menu == "CNN Pneumonia":
    st.header("🫁 Pneumonia Detection (CNN)")

    file = st.file_uploader("Upload Chest X-ray Image", type=["jpg", "png", "jpeg"])

    if file is not None:
        img = Image.open(file)
        st.image(img, caption="Uploaded Image", use_container_width=True)

        img = img.resize((224, 224))
        img = np.array(img) / 255.0
        img = np.expand_dims(img, axis=0)

        prediction = cnn_model.predict(img)[0][0]

        if prediction > 0.5:
            st.error("🫁 Pneumonia Detected")
        else:
            st.success("😊 Normal Lung")

# ---------------- RNN / LSTM / GRU ----------------
elif menu == "RNN/LSTM/GRU Time Series":
    st.header("📈 Time Series Prediction (LSTM/GRU)")

    values = st.text_area("Enter comma separated values", "10,20,30,40,50")

    if st.button("Predict Next Value"):
        try:
            data = np.array([float(i) for i in values.split(",")])
            data = data.reshape(1, len(data), 1)

            prediction = lstm_model.predict(data)[0][0]

            st.success(f"📊 Next Value Prediction: {prediction:.2f}")

        except Exception as e:
            st.error(f"Input Error: {e}")