import os
from tensorflow.keras.models import load_model

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def load_ann():
    path = os.path.join(BASE_DIR, "Models", "ann_model.keras")
    return load_model(path)

def load_cnn():
    path = os.path.join(BASE_DIR, "Models", "cnn_pneumonia_model.h5")
    return load_model(path)

def load_lstm():
    path = os.path.join(BASE_DIR, "Models", "lstm_model.keras")
    return load_model(path)