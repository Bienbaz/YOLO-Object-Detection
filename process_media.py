from detector import ObjectDetector
import json
import os

def process_image(image_path: str, conf_threshold: float = 0.5):
    detector = ObjectDetector()
    results = detector.detect(image_path, conf=conf_threshold)
    
    detections = []
    for result in results:
        for box in result.boxes:
            detections.append({
                "class_id": int(box.cls.item()),
                "class_name": result.names[int(box.cls.item())],
                "confidence": float(box.conf.item()),
                "bbox": box.xyxy[0].tolist()
            })
    
    # Save JSON
    output = {"image": image_path, "detections": detections}
    json_path = image_path.replace(".jpg", "_detections.json").replace(".png", "_detections.json")
    with open(json_path, "w") as f:
        json.dump(output, f, indent=2)
    
    return detections, results
