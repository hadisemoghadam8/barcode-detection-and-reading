# ===============================================================
# 📦 File: app/services/detection.py
# 🧠 وظیفه: خواندن بارکد و Batch Code از تصویر
# 🚀 نسخه 3.0 (پایدار و دقیق با multi-ROI OCR)
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

# --- Regex قدرتمند برای انواع batch code ---
BATCH_REGEX = re.compile(
    r"((S[O0]P\d*M*R*\d{5,})|(SR\d{4,10})|(LOT[-\s]?\d{4,10})|(BATCH[-\s]?\d{4,10})|(MR\d{5,})|(MFG\d{4,10})|(EXP\d{4,10}))",
    re.IGNORECASE,
)


# --- مرحله ۱: پیش‌پردازش برای OCR ---
def preprocess_for_ocr(img_pil: Image.Image) -> np.ndarray:
    img = np.array(img_pil)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # افزایش وضوح و حذف نویز
    gray = cv2.bilateralFilter(gray, 5, 75, 75)
    gray = cv2.convertScaleAbs(gray, alpha=1.8, beta=35)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)

    # آستانه‌گذاری تطبیقی
    thresh = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 25, 15
    )

    return cv2.cvtColor(thresh, cv2.COLOR_GRAY2RGB)


# --- مرحله ۲: پاک‌سازی متن OCR ---
def clean_ocr_text(text: str) -> str:
    text = text.upper()
    replacements = {
        "O": "0",
        "I": "1",
        "L": "1",
        "S0P": "SOP",
        "SOPP": "SOP",
        "S0": "SO",
        "BATCHC": "BATCH",
        "*": "",
        "'": "",
        " ": "",
        "`": "",
        "’": "",
        "‘": "",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    # حذف نویزهای OCR
    text = re.sub(r"[^A-Z0-9]", "", text)
    return text


# --- مرحله ۳: OCR دوگانه (EasyOCR و ترکیب چند بخش) ---
def dual_ocr(np_img: np.ndarray) -> str:
    """خواندن متن از تصویر با EasyOCR"""
    results = reader.readtext(np_img)
    return " ".join([r[1] for r in results])


# --- مرحله ۴: خواندن بارکد و batch ---
def read_barcode_and_batch(image_bytes: bytes):
    try:
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        img.thumbnail((1600, 1600))

        # --- بارکدخوانی با pyzbar ---
        decoded = decode(img)
        if decoded:
            d = decoded[0]
            barcode_data = d.data.decode("utf-8", errors="ignore").strip()
            barcode_type = d.type
        else:
            barcode_data, barcode_type = None, None

        # --- OCR چندبخشی ---
        np_img = preprocess_for_ocr(img)
        h, w = np_img.shape[:2]

        rois = [
            np_img[0:int(h * 0.4), :],           # بالا
            np_img[int(h * 0.4):int(h * 0.7), :],# وسط
            np_img,                              # کل تصویر
        ]

        full_text = ""
        for roi in rois:
            roi_text = dual_ocr(roi)
            full_text += " " + roi_text

        clean_text = clean_ocr_text(full_text)

        # --- یافتن batch code ---
        match = BATCH_REGEX.search(clean_text)
        if match:
            barcode_text = match.group(1).upper()
        else:
            barcode_text = clean_text[:30] or "No batch text detected"

        # --- مرحله تصحیح تطبیقی ---
        if barcode_data and barcode_text:
            if barcode_text[:5] in barcode_data or barcode_data[:5] in barcode_text:
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
