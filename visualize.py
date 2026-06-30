import cv2
from ultralytics.utils.plotting import Annotator

def annotate_image(image_path: str, results):
    """Draw bounding boxes and labels."""
    img = cv2.imread(image_path)
    annotator = Annotator(img)
    
    for result in results:
        for box in result.boxes:
            b = box.xyxy[0].tolist()
            c = int(box.cls.item())
            annotator.box_label(b, f"{result.names[c]} {box.conf.item():.2f}")
    
    output_path = image_path.replace(".", "_annotated.")
    cv2.imwrite(output_path, annotator.result())
    print(f"✅ Annotated image saved: {output_path}")
    return output_path
