import cv2
import argparse
from pathlib import Path
from config import SAMPLE_IMAGES_DIR, OUTPUT_DIR
from utils.detector import ObjectDetector
from utils.classifier import MaterialClassifier

def process_image(image_path, detector, classifier, output_dir):
    """Process a single image for object and material detection"""
    # Read image
    image = cv2.imread(str(image_path))
    if image is None:
        print(f"Error: Could not read image {image_path}")
        return
    
    # Detect objects
    detections, orig_image = detector.detect(image)
    
    # Process each detection
    for det in detections:
        # Crop object
        x1, y1, x2, y2 = map(int, det['bbox'])
        cropped = image[y1:y2, x1:x2]
        
        # Classify material
        material, confidence = classifier.classify(cropped)
        
        # Update detection info
        det['material'] = material
        det['material_confidence'] = confidence
    
    # Draw results
    result_image = orig_image.copy()
    for det in detections:
        x1, y1, x2, y2 = map(int, det['bbox'])
        label = f"{det['class_name']} | {det['material']} ({det['material_confidence']:.2f})"
        
        # Draw bounding box
        cv2.rectangle(result_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # Draw label background
        (text_width, text_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(result_image, (x1, y1 - text_height - 10), (x1 + text_width, y1), (0, 255, 0), -1)
        
        # Draw text
        cv2.putText(result_image, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    
    # Save results
    output_path = output_dir / f"result_{image_path.name}"
    cv2.imwrite(str(output_path), result_image)
    print(f"Results saved to {output_path}")
    
    return detections

def main():
    # Parse arguments
    parser = argparse.ArgumentParser(description='Object Material Detection')
    parser.add_argument('--image', type=str, help='Path to input image', default=None)
    args = parser.parse_args()
    
    # Initialize models
    print("Loading models...")
    detector = ObjectDetector()
    classifier = MaterialClassifier()
    print("Models loaded successfully.")
    
    # Process single image or sample images
    if args.image:
        image_path = Path(args.image)
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {args.image}")
        process_image(image_path, detector, classifier, OUTPUT_DIR)
    else:
        print(f"No image specified, processing sample images from {SAMPLE_IMAGES_DIR}...")
        for image_path in SAMPLE_IMAGES_DIR.glob('*'):
            if image_path.suffix.lower() in ['.jpg', '.jpeg', '.png']:
                print(f"\nProcessing {image_path.name}...")
                process_image(image_path, detector, classifier, OUTPUT_DIR)

if __name__ == "__main__":
    main()