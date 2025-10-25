# ===============================================================
# 📦 File: app/services/detection.py
# 🧠 وظیفه: خواندن بارکد و Batch Code از تصویر
# ✅ نسخه‌ی Dual OCR (EasyOCR + Tesseract + Preprocess + pyzbar)
# ===============================================================

import io
import re
import numpy as np
from PIL import Image
import cv2
import easyocr
import pytesseract
from pyzbar.pyzbar import decode

# اگر در ویندوز هستی و مسیر تسرکت شناخته نمی‌شود، مسیر زیر را تنظیم کن:
# pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# --- EasyOCR فقط یک‌بار لود می‌شود ---
reader = easyocr.Reader(['en'], gpu=False)

# --- Regex جامع برای batch code / MFG / EXP ---
BATCH_REGEX = re.compile(
    r"(S[O0]P\d*M*R*\d{5,}|SR\d{4,10}|BATCH\d{2,10}|BTCH\d{2,10}|MFG\d{2,10}|EXP\d{2,10})",
    re.IGNORECASE,
)

# ===============================================================
# 🧩 مرحله ۱: پیش‌پردازش تصویر برای OCR
# ===============================================================
def preprocess_for_ocr(img_pil: Image.Image) -> np.ndarray:
    """بهبود تصویر برای OCR"""
    img = np.array(img_pil)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    gray = cv2.convertScaleAbs(gray, alpha=1.7, beta=25)
    gray = cv2.bilateralFilter(gray, 7, 75, 75)
    thresh = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 31, 10
    )
    return cv2.cvtColor(thresh, cv2.COLOR_GRAY2RGB)


# ===============================================================
# 🧩 مرحله ۲: پاک‌سازی و یکسان‌سازی متن OCR
# ===============================================================
def clean_ocr_text(text: str) -> str:
    """اصلاح خطاهای متداول OCR"""
    text = text.upper()
    fixes = {
        "O": "0",
        "I": "1",
        "L": "1",
        "BATCH": "BTCH",
        "S0P": "SOP",
        "SOPP": "SOP",
        "*": "",
        "'": "",
        "`": "",
        " ": "",
        "’": "",
        "‘": "",
        ":": "",
        "|": "",
    }
    for k, v in fixes.items():
        text = text.replace(k, v)
    text = re.sub(r"[^A-Z0-9]", "", text)
    return text


# ===============================================================
# 🧩 مرحله ۳: OCR با EasyOCR و Tesseract
# ===============================================================
def dual_ocr(np_img: np.ndarray) -> str:
    """خواندن متن با هر دو موتور OCR"""
    # EasyOCR
    easy_texts = [r[1] for r in reader.readtext(np_img)]
    easy_text = " ".join(easy_texts)

    # Tesseract
    tess_text = pytesseract.image_to_string(
        cv2.cvtColor(np_img, cv2.COLOR_RGB2BGR),
        config="--psm 6",
    )

    combined = easy_text + " " + tess_text
    return combined.strip()


# ===============================================================
# 🧩 مرحله ۴: OCR + Barcode Detection
# ===============================================================
def read_barcode_and_batch(image_bytes: bytes):
    """
    ورودی: تصویر (bytes)
    خروجی: dict شامل داده‌ی بارکد و متن بالای آن
    """
    try:
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        img.thumbnail((1600, 1600))

        # --- بارکدخوانی ---
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

        # OCR در سه ناحیه برای دقت بالاتر
        rois = [
            np_img[0:int(h * 0.4), :],        # بالا
            np_img[int(h * 0.4):int(h * 0.7), :],  # وسط
            np_img,                            # کل تصویر
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
