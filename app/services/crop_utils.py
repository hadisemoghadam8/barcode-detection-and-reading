import cv2
import numpy as np

def get_crops(image, results):
    """
    استخراج برش‌های هر شناسایی‌شده از خروجی YOLO.
    ورودی:
        image: تصویر اصلی (np.ndarray)
        results: خروجی مدل YOLO از ultralytics
    خروجی:
        لیست از برش‌ها (np.ndarray)
    """
    crops = []
    if not results or len(results[0].boxes) == 0:
        return crops

    boxes = results[0].boxes.xyxy.cpu().numpy().astype(int)
    for (x1, y1, x2, y2) in boxes:
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(image.shape[1], x2), min(image.shape[0], y2)
        crop = image[y1:y2, x1:x2].copy()
        crops.append(crop)

    return crops
