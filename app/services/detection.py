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
    """
    خواندن خطوط بارکد (با pyzbar) + متن بالای بارکد (با pytesseract)
    خروجی: {'barcode_data': str, 'barcode_type': str, 'barcode_text': str}
    """

    try:
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")

        # ===== (1) خواندن بارکد =====
        barcode_info = read_barcode_pil(img)

        # ===== (2) برش ناحیه بالای تصویر برای OCR =====
        # فرض می‌کنیم متن بالای بارکد در حدود 25% بالایی تصویر است.
        w, h = img.size
        text_region = img.crop((0, 0, w, int(h * 0.25)))

        # کمی وضوح تصویر را برای OCR افزایش می‌دهیم
        text_region = text_region.filter(ImageFilter.SHARPEN)
        text_region = ImageEnhance.Contrast(text_region).enhance(1.5)

        # ===== (3) اجرای OCR =====
        try:
            raw_text = pytesseract.image_to_string(text_region, config="--psm 6").strip()
            # حذف نویز (کاراکترهای غیرحرفی/عددی اضافه)
            cleaned_text = re.sub(r"[^A-Za-z0-9\-_/]", "", raw_text)
        except Exception:
            cleaned_text = None

        # ===== (4) بازگرداندن خروجی =====
        return {
            "barcode_data": barcode_info["barcode"] if barcode_info else "No barcode detected",
            "barcode_type": barcode_info["barcode_type"] if barcode_info else None,
            "barcode_text": cleaned_text if cleaned_text else "No text detected"
        }

    except Exception as e:
        print(f"[ERROR] read_barcode_and_batch failed: {e}")
        return {
            "barcode_data": None,
            "barcode_type": None,
            "barcode_text": None
        }
