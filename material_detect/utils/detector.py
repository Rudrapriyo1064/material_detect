from ultralytics import YOLO
import cv2
import numpy as np
from pathlib import Path
from config import YOLO_MODEL_PATH, YOLO_CONFIDENCE_THRESHOLD, YOLO_IOU_THRESHOLD

class ObjectDetector:
    def __init__(self):
        self.model = YOLO(YOLO_MODEL_PATH)
        self.model.fuse()
        
    def detect(self, image):
        """Detect objects in an image and return bounding boxes and class IDs"""
        results = self.model.predict(
            image,
            conf=YOLO_CONFIDENCE_THRESHOLD,
            iou=YOLO_IOU_THRESHOLD,
            device='cpu'  # Use '0' for GPU if available
        )
        
        detections = []
        for result in results:
            for box in result.boxes:
                detections.append({
                    'bbox': box.xyxy[0].tolist(),
                    'confidence': box.conf.item(),
                    'class_id': int(box.cls.item()),
                    'class_name': result.names[int(box.cls.item())]
                })
        
        return detections, results[0].orig_img
    
    def draw_detections(self, image, detections):
        """Draw bounding boxes and labels on image"""
        for det in detections:
            x1, y1, x2, y2 = map(int, det['bbox'])
            label = f"{det['class_name']} {det['confidence']:.2f}"
            
            # Draw rectangle
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Draw label background
            (text_width, text_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(image, (x1, y1 - text_height - 10), (x1 + text_width, y1), (0, 255, 0), -1)
            
            # Draw text
            cv2.putText(image, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        
        return image