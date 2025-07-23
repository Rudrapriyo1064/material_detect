import os
from pathlib import Path

# Base directory
BASE_DIR = Path(__file__).parent

# Model paths
YOLO_MODEL_PATH = BASE_DIR / "models" / "yolov8n.pt"
MATERIAL_MODEL_PATH = BASE_DIR / "models" / "material_classifier" / "model.h5"
MATERIAL_LABELS_PATH = BASE_DIR / "models" / "material_classifier" / "labels.txt"

# Data paths
SAMPLE_IMAGES_DIR = BASE_DIR / "data" / "samples"
OUTPUT_DIR = BASE_DIR / "data" / "outputs"

# Ensure directories exist
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(YOLO_MODEL_PATH.parent, exist_ok=True)
os.makedirs(MATERIAL_MODEL_PATH.parent, exist_ok=True)

# YOLO configuration
YOLO_CONFIDENCE_THRESHOLD = 0.5
YOLO_IOU_THRESHOLD = 0.45