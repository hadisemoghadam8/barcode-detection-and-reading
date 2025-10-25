# ===============================================================
# 📦 File: app/services/detection.py
# 🧠 وظیفه: خواندن بارکد و Batch Code از تصویر
# ✅ نسخه‌ی نهایی پایدار با EasyOCR + pyzbar + فیلتر هوشمند
# ===============================================================

import io
import re
import numpy as np
from PIL import Image
import easyocr
from pyzbar.pyzbar import decode
import cv2

reader = easyocr.Reader(['en'], gpu=False)

BATCH_REGEX = re.compile(r"(S[O0]P\d*M*R*\d{5,}|SR\d{4,10})", re.IGNORECASE)


def preprocess_for_ocr(img_pil: Image.Image) -> np.ndarray:
    """افزایش کیفیت OCR با حذف نویز و افزایش کنتراست"""
    img = np.array(img_pil)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    gray = cv2.convertScaleAbs(gray, alpha=1.5, beta=30)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return cv2.cvtColor(thresh, cv2.COLOR_GRAY2RGB)


def clean_ocr_text(text: str) -> str:
    """پاکسازی و تصحیح OCR"""
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
        "|": "",
        "I": "1",
        "L": "1",
    }
    for k, v in fixes.items():
        text = text.replace(k, v)
    text = re.sub(r"[^A-Z0-9]", "", text)
    return text


def dual_ocr(np_img: np.ndarray) -> str:
    """OCR با دو حالت: عادی و وارونه"""
    results_normal = reader.readtext(np_img)
    text_normal = " ".join([r[1] for r in results_normal])
    inverted = cv2.bitwise_not(np_img)
    results_inv = reader.readtext(inverted)
    text_inv = " ".join([r[1] for r in results_inv])
    return text_normal + " " + text_inv


def read_barcode_and_batch(image_bytes: bytes):
    """خواندن داده بارکد و متن Batch از تصویر"""
    try:
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        img.thumbnail((1600, 1600))

        # --- مرحله ۱: بارکدخوانی ---
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
        rois = [
            np_img[0:int(h * 0.4), :],              # بالا
            np_img[int(h * 0.4):int(h * 0.7), :],  # وسط
            np_img,                                # کل تصویر
        ]

        full_text = ""
        for roi in rois:
            full_text += " " + dual_ocr(roi)

        clean_text = clean_ocr_text(full_text)

        # --- مرحله ۳: فیلتر و یافتن batch code ---
        match = BATCH_REGEX.search(clean_text)
        if match:
            barcode_text = match.group(1).upper()
        else:
            # فیلتر رشته‌های تصادفی غیرمعتبر
            if len(clean_text) < 6 or clean_text.count("1") > len(clean_text) / 2:
                barcode_text = "No batch text detected"
            else:
                barcode_text = clean_text[:30]

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
