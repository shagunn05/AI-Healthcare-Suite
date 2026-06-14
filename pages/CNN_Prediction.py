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

# ---------------- BASE PATH ----------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "Models")

# ---------------- SAFETY CHECK ----------------
if not os.path.exists(MODEL_DIR):
    st.error("Models folder not found!")
    st.stop()

# ---------------- MODEL LOADING (SAFE + FAST) ----------------
@st.cache_resource
def load_ann():
    return load_model(os.path.join(MODEL_DIR, "ann_model.keras"))

@st.cache_resource
def load_cnn():
    return load_model(os.path.join(MODEL_DIR, "cnn_pneumonia.h5"))

@st.cache_resource
def load_lstm():
    return load_model(os.path.join(MODEL_DIR, "lstm_model.keras"))

try:
    ann_model = load_ann()
    cnn_model = load_cnn()
    lstm_model = load_lstm()
except Exception as e:
    st.error(f"Model loading failed: {e}")
    st.stop()

# ---------------- SIDEBAR MENU ----------------
menu = st.sidebar.radio(
    "Choose Model",
    ["Home", "ANN Diabetes", "CNN Pneumonia", "RNN/LSTM/GRU Time Series"],
    key="main_menu"
)

# ---------------- HOME ----------------
if menu == "Home":
    st.title("🏥 AI Healthcare Suite")
    st.write("ANN + CNN + RNN/LSTM/GRU Integrated System")

# ---------------- ANN ----------------
elif menu == "ANN Diabetes":

    st.header("🩺 Diabetes Prediction (ANN)")

    pregnancies = st.number_input("Pregnancies", 0, 20, key="p1")
    glucose = st.number_input("Glucose", 0, 300, key="p2")
    bp = st.number_input("Blood Pressure", 0, 200, key="p3")
    skin = st.number_input("Skin Thickness", 0, 100, key="p4")
    insulin = st.number_input("Insulin", 0, 900, key="p5")
    bmi = st.number_input("BMI", 0.0, 70.0, key="p6")
    dpf = st.number_input("Diabetes Pedigree", 0.0, 3.0, key="p7")
    age = st.number_input("Age", 0, 120, key="p8")

    if st.button("Predict Diabetes"):
        try:
            input_data = np.array([[pregnancies, glucose, bp, skin,
                                     insulin, bmi, dpf, age]])

            prediction = ann_model.predict(input_data)[0][0]

            if prediction > 0.5:
                st.error("⚠️ High Risk of Diabetes")
            else:
                st.success("✅ Low Risk of Diabetes")

        except Exception as e:
            st.error(f"Prediction error: {e}")

# ---------------- CNN ----------------
elif menu == "CNN Pneumonia":

    st.header("🫁 Pneumonia Detection (CNN)")

    file = st.file_uploader("Upload Chest X-ray", type=["jpg", "png", "jpeg"])

    if file is not None:
        try:
            img = Image.open(file).convert("RGB")
            st.image(img, caption="Uploaded Image", use_container_width=True)

            img = img.resize((224, 224))
            img = np.array(img) / 255.0
            img = np.expand_dims(img, axis=0)

            prediction = cnn_model.predict(img)[0][0]

            if prediction > 0.5:
                st.error("🫁 Pneumonia Detected")
            else:
                st.success("😊 Normal Lung")

        except Exception as e:
            st.error(f"Image processing error: {e}")

# ---------------- LSTM / RNN ----------------
elif menu == "RNN/LSTM/GRU Time Series":

    st.header("📈 Time Series Prediction (LSTM/GRU)")

    values = st.text_area("Enter comma-separated values", "10,20,30,40,50")

    if st.button("Predict Next Value"):
        try:
            data = [float(i.strip()) for i in values.split(",")]

            data = np.array(data).reshape(1, len(data), 1)

            prediction = lstm_model.predict(data)[0][0]

            st.success(f"📊 Next Value: {prediction:.2f}")

        except Exception as e:
            st.error(f"Input error: {e}")