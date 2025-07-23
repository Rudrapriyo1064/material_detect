import tensorflow as tf
import cv2
import numpy as np
from pathlib import Path
from config import MATERIAL_MODEL_PATH, MATERIAL_LABELS_PATH

class MaterialClassifier:
    def __init__(self):
        self.model = tf.keras.models.load_model(MATERIAL_MODEL_PATH)
        with open(MATERIAL_LABELS_PATH, 'r') as f:
            self.labels = [line.strip() for line in f.readlines()]
    
    def preprocess_image(self, image):
        """Preprocess image for MobileNetV2"""
        image = cv2.resize(image, (224, 224))
        image = image / 255.0  # Normalize to [0,1]
        return np.expand_dims(image, axis=0)
    
    def classify(self, image):
        """Classify material of a cropped object image"""
        processed_img = self.preprocess_image(image)
        predictions = self.model.predict(processed_img)
        class_id = np.argmax(predictions)
        confidence = predictions[0][class_id]
        return self.labels[class_id], float(confidence)