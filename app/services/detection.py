#  File: app/services/detection.py

import io
import re
import numpy as np
from PIL import Image
import easyocr
import cv2
from pyzbar.pyzbar import decode as zbar_decode
from collections import Counter

reader = easyocr.Reader(['en'], gpu=False)

BARCODE_TEXT_REGEX = re.compile(
    r"((S[O0]P\d*M*R*\d{5,})|(SR\d{4,10})|(SHV\d+[A-Z]*\d+)|(LOT[-\s]?\d{4,10})|(BATCH[-\s]?\d{4,10})|(MR\d{5,}))",
    re.IGNORECASE,
)

#  Preprocess image for OCR
def preprocess_for_ocr(img_pil: Image.Image) -> np.ndarray:
    img = np.array(img_pil)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    gray = cv2.bilateralFilter(gray, 5, 75, 75)
    gray = cv2.convertScaleAbs(gray, alpha=1.6, beta=30)
    gray = cv2.adaptiveThreshold(gray, 255,
                                 cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                 cv2.THRESH_BINARY, 21, 11)
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)


#  Clean OCR text
def clean_ocr_text(text: str) -> str:
    text = text.upper()
    for k, v in {
        "O": "0", "S0P": "SOP", "SOPP": "SOP", "I": "1", "L": "1",
        "‘": "", "’": "", "`": "", "'": "", "*": "", "|": "", " ": ""
    }.items():
        text = text.replace(k, v)
    return re.sub(r"[^A-Z0-9]", "", text)


#  OCR in normal and inverted mode
def dual_ocr(np_img: np.ndarray) -> str:
    normal = reader.readtext(np_img)
    inverted = reader.readtext(cv2.bitwise_not(np_img))
    return " ".join([r[1] for r in normal + inverted])


# -----------------------------------------------
#  Fallback OpenCV barcode detector
# -----------------------------------------------
def opencv_decode(np_img):
    try:
        detector = cv2.barcode_BarcodeDetector()
        retval, decoded_info, _, _ = detector.detectAndDecode(np_img)
        if retval and decoded_info:
            return [{"data": d, "type": "OpenCV"} for d in decoded_info if d]
    except Exception:
        pass
    return []


#  Main function
def read_barcode_and_batch(image_bytes: bytes):
    try:
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        np_img = np.array(img)

        decoded = zbar_decode(img)
        if not decoded:
            decoded = opencv_decode(np_img)

        if not decoded:
            return {
                "barcode_data": None,
                "barcode_type": None,
                "barcode_text": None
            }

        d = decoded[0]
        if isinstance(d, dict):
            data, btype = d["data"], d["type"]
        else:
            data, btype = d.data.decode("utf-8", errors="ignore"), d.type

        # فقط داده‌ی واقعی بارکد
        return {
            "barcode_data": data or None,
            "barcode_type": btype or None,
            "barcode_text": None
        }

    except Exception as e:
        print(f"[ERROR] read_barcode_and_batch failed: {e}")
        return {
            "barcode_data": None,
            "barcode_type": None,
            "barcode_text": None
        }
