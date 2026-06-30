# YOLOv8 Real-Time Object Detection Pipeline

## Performance Comparison
| Model    | Avg Inference Time (ms) | FPS (approx) | Use Case              |
|----------|-------------------------|--------------|-----------------------|
| yolov8n  | ~15-25                  | 40+          | Real-time / Edge      |
| yolov8s  | ~30-45                  | 25+          | Balanced accuracy     |

**Threshold Tuning Results** (on test images):
- conf=0.3: More detections (higher recall, more FPs)
- conf=0.5: Balanced (recommended)
- conf=0.7: Fewer but precise detections

## Business Use Case: Retail Foot Traffic Analysis
Detect and count `person` class in store cameras to analyze customer flow, optimize staffing, and improve layout. Relevant COCO classes: `person`, `backpack`, `handbag`.

## Optimizations Recommended
- Use `yolov8n` for speed on CPU
- Fine-tune on retail-specific data
- Lower confidence threshold during peak hours

**Sample Output Images**: Included in repo (annotated detections)
