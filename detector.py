from ultralytics import YOLO
import torch

class ObjectDetector:
    def __init__(self, model_name="yolov8n.pt"):
        self.model = YOLO(model_name)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"✅ Model loaded on {self.device}")

    def detect(self, source, conf=0.5, save=False):
        """Run detection on image/video source."""
        results = self.model(source, conf=conf, save=save, device=self.device)
        return results
