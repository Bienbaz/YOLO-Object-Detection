from detector import ObjectDetector
import json
import cv2
import time

def process_image(image_path: str, conf_threshold: float = 0.5):
    detector = ObjectDetector("yolov8n.pt")
    results = detector.detect(image_path, conf=conf_threshold)
    # ... (same as before)
    return detections, results

def process_video(video_path: str, conf_threshold: float = 0.5, output_path="output_video.mp4"):
    """Process video file with real-time detection."""
    detector = ObjectDetector("yolov8n.pt")
    cap = cv2.VideoCapture(video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
    
    frame_count = 0
    start_time = time.time()
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1
        results = detector.detect(frame, conf=conf_threshold, save=False)
        # Annotate frame (simplified)
        annotated = results[0].plot()
        out.write(annotated)
    
    cap.release()
    out.release()
    print(f"✅ Processed {frame_count} frames. Output: {output_path}")
    print(f"Avg FPS: {frame_count / (time.time() - start_time):.2f}")
