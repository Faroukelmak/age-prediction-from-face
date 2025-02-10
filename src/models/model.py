from ultralytics import YOLO
import tensorflow as tf
import streamlit as st



@st.cache_resource
def load_yolo_model(yolo_path=r"models\best.pt"):
    """Load the YOLO model for face detection."""
    from ultralytics import YOLO
    return YOLO(yolo_path)

# def load_cnn_model(cnn_path="C:/Users/Farouk/Desktop/age-prediction/models/CNN_V5_v1.h5"):
#     """Load the CNN model for age prediction."""
#     return tf.keras.models.load_model(cnn_path)



from tensorflow.keras.losses import MeanSquaredError

def load_cnn_model():
    cnn_path = r'models\tinyvgg_utkface.h5'  # Update with your actual model path
    # Loading the model and passing custom_objects
    cnn_model = tf.keras.models.load_model(cnn_path, custom_objects={'mse': MeanSquaredError()}, compile=False)
    return cnn_model
