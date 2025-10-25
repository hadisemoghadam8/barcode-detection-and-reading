# ==========================================================
# 📦 File: app/services/crop_utils.py
# ✂️ وظیفه: برش نواحی شناسایی‌شده از تصویر بر اساس خروجی YOLO
# ✅ بدون استفاده از cv2
# ==========================================================

import numpy as np
from PIL import Image

def get_crops(image: Image.Image, results):
    """
    استخراج برش‌های هر شناسایی‌شده از خروجی YOLO.
    ورودی:
        image: تصویر اصلی (PIL.Image)
        results: خروجی مدل YOLO از ultralytics
    خروجی:
        لیست از برش‌ها (np.ndarray)
    """
    crops = []
    if not results or len(results[0].boxes) == 0:
        return crops

    boxes = results[0].boxes.xyxy.cpu().numpy().astype(int)

    for (x1, y1, x2, y2) in boxes:
        # اطمینان از داخل محدوده بودن
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(image.width, x2), min(image.height, y2)

        # برش با PIL
        crop = image.crop((x1, y1, x2, y2))
        crops.append(np.array(crop))  # تبدیل به NumPy برای OCR یا پردازش بعدی

    return crops
