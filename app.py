import streamlit as st

st.set_page_config(
    page_title="AI Healthcare Suite",
    page_icon="🧠",
    layout="wide"
)

st.title("🧠 AI Healthcare Suite")

st.markdown("""
### Welcome 👋  
This system includes:
- ANN for Diabetes Prediction  
- CNN for Pneumonia Detection  
- LSTM/GRU for Time Series Analysis  
""")

st.sidebar.success("Select a module from pages 👈")