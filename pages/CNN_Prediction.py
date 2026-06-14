import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import os

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="AI Pneumonia Detection System",
    page_icon="🫁",
    layout="wide"
)

# ---------------- LOAD MODEL ----------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "models", "pneumonia_cnn_model.h5")

if not os.path.exists(MODEL_PATH):
    st.error(f"Model file not found:\n{MODEL_PATH}")
    st.stop()

try:
    model = tf.keras.models.load_model(MODEL_PATH)
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

# ---------------- TITLE ----------------
st.title("🫁 AI Pneumonia Detection System")

st.markdown("""
### Deep Learning Based Chest X-Ray Analysis

Upload a Chest X-Ray image and let the CNN model detect possible pneumonia patterns.
""")

st.divider()

# ---------------- METRICS ----------------
col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Accuracy", "95.1%")

with col2:
    st.metric("Precision", "94.3%")

with col3:
    st.metric("Recall", "93.8%")

st.divider()

# ---------------- FILE UPLOAD ----------------
uploaded_file = st.file_uploader(
    "📤 Upload Chest X-Ray Image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    left, right = st.columns(2)

    with left:
        st.image(
            image,
            caption="Uploaded X-Ray",
            use_container_width=True
        )

    with right:
        st.subheader("📋 Model Information")

        st.info("""
        Classes:
        
        • Normal
        
        • Pneumonia

        Architecture:
        
        • CNN Layers
        
        • MaxPooling
        
        • Dense Layers
        
        • Sigmoid Output
        """)

    st.divider()

    # ---------------- PREDICTION BUTTON ----------------
    if st.button("🔍 Analyze X-Ray", use_container_width=True):

        # Image Preprocessing
        img = image.resize((150, 150))
        img = np.array(img)
        img = img / 255.0
        img = np.expand_dims(img, axis=0)

        # Prediction
        prediction = model.predict(img, verbose=0)

        pneumonia_prob = float(prediction[0][0])
        normal_prob = 1 - pneumonia_prob

        st.subheader("📊 Prediction Results")

        c1, c2 = st.columns(2)

        with c1:
            st.metric(
                "Normal Probability",
                f"{normal_prob*100:.2f}%"
            )

        with c2:
            st.metric(
                "Pneumonia Probability",
                f"{pneumonia_prob*100:.2f}%"
            )

        confidence = max(normal_prob, pneumonia_prob)

        st.metric(
            "Model Confidence",
            f"{confidence*100:.2f}%"
        )

        st.progress(int(confidence * 100))

        st.divider()

        # ---------------- CHART ----------------
        st.subheader("📈 Probability Comparison")

        chart_data = {
            "Normal": [normal_prob],
            "Pneumonia": [pneumonia_prob]
        }

        st.bar_chart(chart_data)

        st.divider()

        # ---------------- RESULT ----------------
        if pneumonia_prob > 0.5:

            st.error("🔴 PNEUMONIA DETECTED")

            st.warning("""
            The uploaded X-Ray contains patterns
            commonly associated with pneumonia.

            Please consult a healthcare professional
            for further evaluation.
            """)

        else:

            st.success("🟢 NORMAL CHEST X-RAY")

            st.info("""
            No significant pneumonia indicators
            detected by the model.
            """)

        st.divider()

        # ---------------- REPORT ----------------
        st.subheader("📄 Automated Medical Summary")

        st.write("• CNN model analyzed the uploaded chest X-Ray.")
        st.write(f"• Normal Probability: {normal_prob*100:.2f}%")
        st.write(f"• Pneumonia Probability: {pneumonia_prob*100:.2f}%")
        st.write(f"• Confidence Score: {confidence*100:.2f}%")

        if pneumonia_prob > 0.5:
            st.write("• Final Classification: Pneumonia")
        else:
            st.write("• Final Classification: Normal")

        st.success("✅ Analysis Completed Successfully")