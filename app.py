import streamlit as st
import numpy as np

st.title("🏥 AI Healthcare - Heart Disease Prediction")

# User Inputs
glucose = st.number_input("Glucose", min_value=0.0, max_value=300.0, value=120.0)
bp = st.number_input("Blood Pressure", min_value=0.0, max_value=200.0, value=80.0)
bmi = st.number_input("BMI", min_value=0.0, max_value=60.0, value=25.0)

# Feature Vector
data = np.array([glucose, bp, bmi])

# --- PURE MATH NEURAL NETWORK INFRASTRUCTURE (No TensorFlow Needed) ---
# NOTE: Replace these placeholder values with your model's actual trained weights if available,
# or use this clean structure to instantly process the network logic.
W1 = np.array([[0.5, -0.2], [0.1, 0.8], [-0.3, 0.4]]) # Input to Hidden Layer Weights
b1 = np.array([0.1, -0.2])                            # Hidden Layer Biases
W2 = np.array([[0.7], [-0.5]])                         # Hidden to Output Layer Weights
b2 = np.array([0.05])                                  # Output Layer Bias

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def relu(x):
    return np.maximum(0, x)

if st.button("Predict"):
    # Forward Pass
    hidden_layer = relu(np.dot(data, W1) + b1)
    output_layer = sigmoid(np.dot(hidden_layer, W2) + b2)
    prediction_prob = output_layer[0]
    
    # Visual Output Dashboard
    st.subheader("Diagnostic Results")
    if prediction_prob > 0.5:
        st.error(f"⚠️ High Risk Detected! (Probability: {prediction_prob*100:.2f}%)")
    else:
        st.success(f"✅ Low Risk. Everything looks stable. (Probability: {prediction_prob*100:.2f}%)")