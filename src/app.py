import streamlit as st
from PIL import Image
from models.model import load_yolo_model, load_cnn_model
from models.predict import predict_age
from models.detection_prediction import detect_and_annotate_faces
import numpy as np

# Load models
yolo_model = load_yolo_model()
cnn_model = load_cnn_model()

st.title("Face Age Prediction App 🧑‍🔬📷")

# Upload image
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    image = np.array(image)

    # Detect faces and annotate with ages
    annotated_image = detect_and_annotate_faces(image, yolo_model, cnn_model)

    # Display the annotated image
    st.image(annotated_image, caption="Detected Faces with Predicted Ages", use_container_width=True)
