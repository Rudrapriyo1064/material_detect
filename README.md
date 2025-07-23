# 🧠 Material Detection System

This project performs **object detection** using YOLOv8 and **material classification** using a custom-trained TensorFlow model. It processes images to identify objects and predict the material of each detected item.

---

---

## 🚀 Features

- ⚡ Detects objects in images using **YOLOv8n**
- 🔍 Classifies **materials** of the detected objects
- 🖼️ Saves annotated output images
- 🧩 Modular and extensible codebase

---

## 📦 Requirements

Create a virtual environment and install dependencies:

```bash
python -m venv .materialvenv
.\.materialvenv\Scripts\activate     # On Windows
# source .materialvenv/bin/activate  # On Linux/macOS

pip install -r requirements.txt
requirements.txt
shell
Copy
Edit
ultralytics>=8.0.0
tensorflow
opencv-python
numpy
matplotlib
absl-py
🧪 How to Use
Activate your environment:

bash
Copy
Edit
.\.materialvenv\Scripts\activate
Run the detection pipeline:

bash
Copy
Edit
python main.py
If no image path is specified, the script processes sample images inside data/samples/.

View results in data/outputs/ with bounding boxes and material labels.

💡 Notes
yolov8n.pt is a lightweight YOLOv8 model suitable for fast inference.

The material classifier (model.h5) is a TensorFlow model trained on labeled material data.

The labels.txt contains material categories used during classification.

📌 Future Work
Integrate live webcam or video feed

Add confidence threshold tuning for material classification

Build a GUI or web interface using Streamlit or Gradio

🧑‍💻 Author
Rudrapriyo Dutta
