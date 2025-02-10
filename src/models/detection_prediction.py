import cv2
import numpy as np
from models.predict import predict_age


def detect_and_annotate_faces(image, model, cnn_model):
    try:
        
        results = model(image)[0]  # YOLO detection
        annotated_image = image.copy()

        print(f"Number of boxes detected: {len(results.boxes)}")

        for box in results.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            print(f"Box coordinates: {x1}, {y1}, {x2}, {y2}")

            # Ensure coordinates are valid
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(image.shape[1] - 1, x2), min(image.shape[0] - 1, y2)

            face = image[y1:y2, x1:x2]
            if face.size == 0:
                print("Skipping invalid face crop.")
                continue

            age = predict_age(face, cnn_model)

            
            # Draw bounding box (thinner)
            cv2.rectangle(annotated_image, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Thickness = 1

            # Add age label (smaller font, thinner text)
            label = f" {age} "
            cv2.putText(annotated_image, label, (x1, max(y1 - 5, 0)), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 1)  # Font size = 0.4, Thickness = 1



        return annotated_image

    except Exception as e:
        print(f"Error during face detection and annotation: {e}")
        return image
