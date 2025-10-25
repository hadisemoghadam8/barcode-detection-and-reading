# ===============================================================
# 📦 File: app/services/detection.py
# 🧠 وظیفه: خواندن بارکد و Batch Code از تصویر
# ✅ نسخه‌ی بهینه‌شده با EasyOCR + pyzbar + preprocess
# ===============================================================

import io
import re
import numpy as np
from PIL import Image
import easyocr
from pyzbar.pyzbar import decode
import cv2  # فقط برای preprocess در OCR

# --- EasyOCR فقط یک‌بار لود می‌شود ---
reader = easyocr.Reader(['en'], gpu=False)

# --- Regex انعطاف‌پذیر برای batch code ---
# (مثل S0P4MR17702892 یا SR21092200)
BATCH_REGEX = re.compile(r"(S[O0]P\d*M*R*\d{5,}|SR\d{4,10})", re.IGNORECASE)


# --- مرحله ۱: بهبود تصویر برای OCR ---
def preprocess_for_ocr(img_pil: Image.Image) -> np.ndarray:
    """افزایش کیفیت OCR با حذف نویز و افزایش کنتراست"""
    img = np.array(img_pil)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    gray = cv2.convertScaleAbs(gray, alpha=1.6, beta=25)  # کنتراست بیشتر
    gray = cv2.GaussianBlur(gray, (3, 3), 0)              # حذف نویز
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return cv2.cvtColor(thresh, cv2.COLOR_GRAY2RGB)


# --- مرحله ۲: پاک‌سازی متن OCR ---
def clean_ocr_text(text: str) -> str:
    """اصلاح اشتباهات متداول OCR"""
    text = text.upper()
    fixes = {
        "O": "0",
        "S0P": "SOP",
        "SOPP": "SOP",
        "*": "",
        "'": "",
        " ": "",
        "`": "",
        "’": "",
        "‘": "",
    }
    for k, v in fixes.items():
        text = text.replace(k, v)
    text = re.sub(r"[^A-Z0-9]", "", text)
    return text


# --- مرحله ۳: OCR + Barcode Detection ---
def read_barcode_and_batch(image_bytes: bytes):
    """
    ورودی: تصویر (bytes)
    خروجی: dict شامل داده‌ی بارکد و متن بالای آن
    """
    try:
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        img.thumbnail((1600, 1600))  # برای سرعت بهتر

        # --- بارکدخوانی با pyzbar ---
        decoded = decode(img)
        if decoded:
            d = decoded[0]
            barcode_data = d.data.decode("utf-8", errors="ignore")
            barcode_type = d.type
        else:
            barcode_data, barcode_type = None, None

        # --- OCR با پیش‌پردازش ---
        np_img = preprocess_for_ocr(img)
        h, w = np_img.shape[:2]

        # 🎯 اول: فقط بخش بالا (متن احتمالی)
        roi_top = np_img[0:int(h * 0.4), :]
        results_top = reader.readtext(roi_top)
        raw_text_top = " ".join([r[1] for r in results_top])

        # 🧠 دوم: اگر چیزی پیدا نشد، کل تصویر را امتحان کن
        if not raw_text_top.strip():
            results_full = reader.readtext(np_img)
            raw_text_full = " ".join([r[1] for r in results_full])
            raw_text = raw_text_full
        else:
            raw_text = raw_text_top

        # پاک‌سازی متن
        clean_text = clean_ocr_text(raw_text)

        # --- یافتن batch code معتبر ---
        match = BATCH_REGEX.search(clean_text)
        if match:
            barcode_text = match.group(1).upper()
        elif clean_text:
            barcode_text = clean_text[:30]
        else:
            barcode_text = None

        # --- بازگرداندن خروجی ---
        return {
            "barcode_data": barcode_data or "No barcode detected",
            "barcode_type": barcode_type or "Unknown",
            "barcode_text": barcode_text or "No batch text detected",
        }

    except Exception as e:
        print(f"[ERROR] read_barcode_and_batch failed: {e}")
        return {
            "barcode_data": None,
            "barcode_type": None,
            "barcode_text": None,
        }
