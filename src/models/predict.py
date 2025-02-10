
from models.preprocess import preprocess_face

def predict_age(face, model):
    """Predict age from the cropped face using CNN."""
    processed_face = preprocess_face(face)
    prediction = model.predict(processed_face)
    return int(prediction[0][0])  # Assuming regression output