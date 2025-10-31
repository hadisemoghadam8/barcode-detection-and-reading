# ===============================================================
# 📦 File: app/services/detection.py
# 🧠 خواندن بارکد و متن زیر آن + شمارش تکراری‌ها + ترسیم روی تصویر
# ===============================================================

import io
import re
import numpy as np
from PIL import Image
import easyocr
from pyzbar.pyzbar import decode
import cv2
from collections import Counter

reader = easyocr.Reader(['en'], gpu=False)

BARCODE_TEXT_REGEX = re.compile(
    r"((S[O0]P\d*M*R*\d{5,})|(SR\d{4,10})|(SHV\d+[A-Z]*\d+)|(LOT[-\s]?\d{4,10})|(BATCH[-\s]?\d{4,10})|(MR\d{5,}))",
    re.IGNORECASE,
)


def preprocess_for_ocr(img_pil: Image.Image) -> np.ndarray:
    img = np.array(img_pil)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    gray = cv2.bilateralFilter(gray, 5, 75, 75)
    gray = cv2.convertScaleAbs(gray, alpha=1.7, beta=35)
    gray = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 21, 11
    )
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)


def clean_ocr_text(text: str) -> str:
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


def dual_ocr(np_img: np.ndarray) -> str:
    results_normal = reader.readtext(np_img)
    text_normal = " ".join([r[1] for r in results_normal])
    inverted = cv2.bitwise_not(np_img)
    results_inv = reader.readtext(inverted)
    text_inv = " ".join([r[1] for r in results_inv])
    return f"{text_normal} {text_inv}"


def draw_results_on_image(np_img, decoded_barcodes, duplicates_count):
    """رسم کادر سبز یا قرمز روی تصویر با تعداد تکرارها"""
    for d in decoded_barcodes:
        (x, y, w, h) = d.rect
        data = d.data.decode("utf-8", errors="ignore")
        count = duplicates_count.get(data, 1)

        if data:
            color = (0, 255, 0)  # سبز
            label = f"x{count}"
        else:
            color = (0, 0, 255)  # قرمز
            label = "No barcode"

        cv2.rectangle(np_img, (x, y), (x + w, y + h), color, 3)
        cv2.putText(np_img, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

    return np_img


def read_barcode_and_batch(image_bytes: bytes):
    try:
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        np_img = np.array(img)

        decoded = decode(img)
        all_results = []

        for d in decoded:
            data = d.data.decode("utf-8", errors="ignore")
            btype = d.type

            # ناحیه زیر بارکد برای OCR
            x, y, w, h = d.rect
            y2 = y + h
            bottom_y2 = min(np_img.shape[0], y2 + 50)
            roi = np_img[y2:bottom_y2, x:x + w]
            ocr_text = dual_ocr(roi)
            clean_text = clean_ocr_text(ocr_text)

            match = BARCODE_TEXT_REGEX.search(clean_text)
            barcode_text = match.group(1).upper() if match else clean_text[:30]

            all_results.append({
                "barcode_data": data or "No barcode detected",
                "barcode_type": btype or "Unknown",
                "barcode_text": barcode_text or "No text",
            })

        # شمارش تکراری‌ها
        barcode_data_list = [r["barcode_data"] for r in all_results if r["barcode_data"] != "No barcode detected"]
        duplicates_count = dict(Counter(barcode_data_list))

        # اضافه کردن شمارش به JSON
        for r in all_results:
            r["count"] = duplicates_count.get(r["barcode_data"], 1)

        # رسم کادرها روی عکس
        np_img_drawn = draw_results_on_image(np_img.copy(), decoded, duplicates_count)

        # ذخیره تصویر خروجی برای بررسی (اختیاری)
        cv2.imwrite("output_detected.jpg", cv2.cvtColor(np_img_drawn, cv2.COLOR_RGB2BGR))

        # نتیجه نهایی JSON
        result = {
            "total_barcodes_detected": len(all_results),
            "results": all_results
        }

        return result

    except Exception as e:
        print(f"[ERROR] read_barcode_and_batch failed: {e}")
        return {"error": str(e)}
