import streamlit as st

st.set_page_config(
    page_title="AI Healthcare Suite",
    page_icon="🏥",
    layout="centered"
)

st.title("🏥 AI Healthcare Suite")
st.write("Multi AI Models: ANN + CNN + RNN/LSTM/GRU")

st.success("👈 Sidebar se model select karo")

st.sidebar.title("Select Model")
choice = st.sidebar.selectbox(
    "Choose Prediction Type",
    ["Home", "ANN Diabetes", "CNN Pneumonia", "RNN Time Series"]
)

if choice == "Home":
    st.info("This system includes 3 AI models:")
    st.markdown("""
    - 🧠 ANN → Diabetes Prediction  
    - 🫁 CNN → Chest X-ray Pneumonia Detection  
    - 📈 RNN/LSTM/GRU → Time Series Prediction  
    """)

elif choice == "ANN Diabetes":
    st.switch_page("pages/1_ANN_Diabetes.py")

elif choice == "CNN Pneumonia":
    st.switch_page("pages/2_CNN_Pneumonia.py")

elif choice == "RNN Time Series":
    st.switch_page("pages/3_RNN_TimeSeries.py")