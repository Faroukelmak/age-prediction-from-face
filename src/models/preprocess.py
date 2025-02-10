import numpy as np
import cv2
from tensorflow.keras.preprocessing.image import img_to_array

def preprocess_face(face):
    """Preprocess the cropped face for CNN input."""
    face = cv2.resize(face, (224, 224))  # Resize to match CNN input size
    face = face.astype("float") / 255.0  # Normalize
    face = img_to_array(face)
    face = np.expand_dims(face, axis=0)  # Add batch dimension
    return face