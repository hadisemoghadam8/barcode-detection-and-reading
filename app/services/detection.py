# ===============================================================
# 📦 File: app/services/detection.py
# 🧠 وظیفه: خواندن بارکد و Batch Code از تصویر
# ✅ نسخه‌ی نهایی پایدار با EasyOCR + pyzbar + adaptive threshold + OCR دوحالته
# ===============================================================

import io
import re
import numpy as np
from PIL import Image
import easyocr
from pyzbar.pyzbar import decode
import cv2

# --- EasyOCR فقط یک بار لود می‌شود ---
reader = easyocr.Reader(['en'], gpu=False)

# --- Regex جامع برای batch code‌ها ---
BATCH_REGEX = re.compile(
    r"((S[O0]P\d*M*R*\d{5,})|(SR\d{4,10})|(SHV\d+[A-Z]*\d+)|(LOT[-\s]?\d{4,10})|(BATCH[-\s]?\d{4,10})|(MR\d{5,}))",
    re.IGNORECASE,
)


# --- پیش‌پردازش تصویر برای OCR ---
def preprocess_for_ocr(img_pil: Image.Image) -> np.ndarray:
    """بهبود کیفیت برای OCR: حذف نویز، افزایش کنتراست و آستانه تطبیقی"""
    img = np.array(img_pil)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    gray = cv2.bilateralFilter(gray, 5, 75, 75)  # حفظ لبه‌ها + حذف نویز
    gray = cv2.convertScaleAbs(gray, alpha=1.6, beta=35)  # افزایش کنتراست
    gray = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 21, 11
    )
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)


# --- پاک‌سازی متن OCR ---
def clean_ocr_text(text: str) -> str:
    """تصحیح اشتباهات متداول OCR"""
    text = text.upper()
    replacements = {
        "O": "0",
        "S0P": "SOP",
        "SOPP": "SOP",
        "‘": "",
        "’": "",
        "`": "",
        "'": "",
        "*": "",
        "|": "",
        " ": "",
        "I": "1",
        "L": "1",
    }
    for k, v in replacements.items():
        text = text.replace(k, v)
    return re.sub(r"[^A-Z0-9]", "", text)


# --- OCR در دو حالت (عادی و وارونه) ---
def dual_ocr(np_img: np.ndarray) -> str:
    """OCR با تصویر معمولی و نگاتیو برای دقت بیشتر"""
    results_normal = reader.readtext(np_img)
    text_normal = " ".join([r[1] for r in results_normal])

    inverted = cv2.bitwise_not(np_img)
    results_inv = reader.readtext(inverted)
    text_inv = " ".join([r[1] for r in results_inv])

    return f"{text_normal} {text_inv}"


# --- تابع اصلی ---
def read_barcode_and_batch(image_bytes: bytes, bbox: tuple | None = None):
    """
    خواندن داده بارکد و متن Batch از تصویر
    ورودی:
        image_bytes: تصویر به صورت bytes
        bbox: مختصات (x1, y1, x2, y2) در صورت موجود بودن (از YOLO)
    """
    try:
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        img.thumbnail((1600, 1600))

        # --- مرحله ۱: خواندن بارکد با pyzbar ---
        decoded = decode(img)
        if decoded:
            d = decoded[0]
            barcode_data = d.data.decode("utf-8", errors="ignore")
            barcode_type = d.type
        else:
            barcode_data, barcode_type = None, None

        # --- مرحله ۲: OCR ---
        np_img = preprocess_for_ocr(img)
        h, w = np_img.shape[:2]

        rois = []
        if bbox:  # اگر مختصات از YOLO آمده
            x1, y1, x2, y2 = map(int, bbox)
            margin = 60
            top_y1 = max(0, y1 - margin)
            rois.append(np_img[top_y1:y1, x1:x2])
        else:
            # در غیر این صورت، چند ناحیه عمومی تست می‌شود
            rois = [
                np_img[0:int(h * 0.3), :],
                np_img[int(h * 0.3):int(h * 0.6), :],
                np_img,
            ]

        full_text = ""
        for roi in rois:
            full_text += " " + dual_ocr(roi)

        clean_text = clean_ocr_text(full_text)

        # --- مرحله ۳: یافتن batch code ---
        match = BATCH_REGEX.search(clean_text)
        if match:
            barcode_text = match.group(1).upper()
        else:
            # فیلتر رشته‌های پر از نویز
            if len(clean_text) < 6 or clean_text.count("1") > len(clean_text) / 2:
                barcode_text = "No batch text detected"
            else:
                barcode_text = clean_text[:30]

# --- مرحله ۴: تصحیح خودکار بر اساس شباهت با barcode_data ---
        if barcode_data and barcode_text and barcode_text[:5] in barcode_data:
            barcode_text = barcode_data

        return {
            "barcode_data": barcode_data or "No barcode detected",
            "barcode_type": barcode_type or "Unknown",
            "barcode_text": barcode_text,
        }

    except Exception as e:
        print(f"[ERROR] read_barcode_and_batch failed: {e}")
        return {
            "barcode_data": None,
            "barcode_type": None,
            "barcode_text": None,
        }                