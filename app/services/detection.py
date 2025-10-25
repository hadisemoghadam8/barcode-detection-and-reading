# app/services/detection_pil.py
from pyzbar.pyzbar import decode
from PIL import Image, ImageEnhance, ImageFilter
import pytesseract
import re
import io
import pytesseract
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# ========================
# 🔹 خواندن بارکد با pyzbar
# ========================
def read_barcode_pil(image: Image.Image):
    """
    تلاش برای خواندن بارکد از تصویر با استفاده از pyzbar.
    خروجی:
        {'barcode': str, 'barcode_type': str}
        یا None اگر بارکدی پیدا نشد.
    """
    try:
        decoded = decode(image)
        if not decoded:
            return None
        d = decoded[0]
        return {
            "barcode": d.data.decode("utf-8", errors="ignore"),
            "barcode_type": d.type
        }
    except Exception as e:
        print(f"[WARN] Barcode read error: {e}")
        return None


# ========================
# 🔹 تابع ترکیبی نهایی
# ========================
def read_barcode_and_batch(image_bytes: bytes):

    try:
        img = Image.open(io.BytesIO(image_bytes))

        # خواندن بارکد
        barcode_info = read_barcode_pil(img)


        return {
            "barcode_data": barcode_info["barcode"] if barcode_info else "No barcode detected",
            "barcode_type": barcode_info["barcode_type"] if barcode_info else None
        }

    except Exception as e:
        print(f"[ERROR] read_barcode_and_batch failed: {e}")
        return {
            "barcode_data": None,
            "barcode_type": None
        }
