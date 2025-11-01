#  app/main.py

from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from ultralytics import YOLO
from PIL import Image, ImageDraw
from io import BytesIO
from app.services.detection import read_barcode_and_batch
from app.core.config import settings
from app.services.barcode_rules import ikco, generic, saipa

import numpy as np
import zipfile
import json
import os
from collections import Counter
from enum import Enum

class Manufacturer(str, Enum):
    ikco = "ikco"
    saipa = "saipa" #ب عنوان مثالللللل
    generic = "generic"


# ------------------------- راه‌اندازی FastAPI -------------------------
app = FastAPI(title="📦 Barcode OCR API", debug=True)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

# ------------------------- بارگذاری مدل YOLO -------------------------
model = YOLO(settings.MODEL_PATH)


#  Endpoint: /predict/
@app.post("/predict/")
async def predict_image(
    file: UploadFile = File(...),
    threshold: float = Form(0.5),
    download_zip: bool = Form(False),
    factory: Manufacturer = Form(Manufacturer.generic)
):

    try:
        contents = await file.read()
        image = Image.open(BytesIO(contents)).convert("RGB")
        np_img = np.array(image)

        # --- اجرای YOLO ---
        results = model(np_img, conf=threshold)
        boxes = results[0].boxes.xyxy.cpu().numpy() if len(results[0].boxes) > 0 else []

        response_data = []
        crops = []

        # اگر هیچ باکسی پیدا نشد
        if len(boxes) == 0:
            empty_detection = [
                {
                    "crop_index": i,
                    "barcode_data": None,
                    "barcode_type": None,
                    "barcode_text": None,
                    "count": 0,
                    "is_duplicate": False
                }
                for i in range(7)
            ]
            return JSONResponse({
                "message": "⚠️ No barcodes detected.",
                "total_barcodes_detected": 0,
                "unique_barcodes": 0,
                "duplicate_barcodes": 0,
                "duplicates": {},
                "detections": empty_detection
            })

        # --- پردازش هر باکس ---
        for i, box in enumerate(boxes):
            x1, y1, x2, y2 = map(int, box[:4])
            crop = image.crop((x1, y1, x2, y2))
            crops.append(crop)

            buffer = BytesIO()
            crop.save(buffer, format="JPEG")

            result = read_barcode_and_batch(buffer.getvalue())

            barcode_data = result.get("barcode_data")

            if factory == Manufacturer.ikco:
                parsed = ikco.parse_barcode(barcode_data or "")
            elif factory == Manufacturer.saipa:
                parsed = saipa.parse_barcode(barcode_data or "")
            else:
                parsed = generic.parse_barcode(barcode_data or "")

            response_data.append({
                "crop_index": i,
                "barcode_data": barcode_data,
                "barcode_type": result.get("barcode_type"),
                "barcode_text": result.get("barcode_text"),
                "part_code": parsed.get("part_code"),
                "manufacturer": parsed.get("manufacturer"),
                "serial": parsed.get("serial"),
                "print_repeat_count": parsed.get("part_info", {}).get("PrintRepeatCount", 1)
            })


        # --- شمارش بارکدها ---
        all_barcodes = [
            d["barcode_data"]
            for d in response_data
            if d["barcode_data"] and d["barcode_data"] != "No barcode detected"
        ]
        barcode_counts = Counter(all_barcodes)
        total_barcodes = len(all_barcodes)
        unique_barcodes = len(barcode_counts)
        duplicate_barcodes = sum(1 for c in barcode_counts.values() if c > 1)
        duplicates = {code: c for code, c in barcode_counts.items() if c > 1}

        # --- افزودن شمارش و وضعیت تکرار ---
        for item in response_data:
            data = item.get("barcode_data")
            count = barcode_counts.get(data, 0)
            item["count"] = count
            item["is_duplicate"] = count > 1

        # --- تولید تصویر نهایی با برچسب‌ها ---
        labeled = image.copy()
        draw = ImageDraw.Draw(labeled)

        for i, box in enumerate(boxes):
            x1, y1, x2, y2 = map(int, box[:4])
            item = response_data[i]
            data = item.get("barcode_data", "")
            count = item.get("count", 0)

            # 👇 اضافه‌شده: گرفتن تعداد مجاز چاپ از اطلاعات پارت
            allowed = int(item.get("print_repeat_count", 1))


            # 👇 رنگ و برچسب براساس مجاز بودن چاپ
            if not data or data == "No barcode detected":
                color = "red"
                label_text = "x0"
            elif count > allowed:
                color = "red"
                label_text = f"x{count} DUPLICATE"
            else:
                color = "green"
                label_text = f"x{count}"

            draw.rectangle((x1, y1, x2, y2), outline=color, width=3)
            draw.text((x1, max(0, y1 - 18)), label_text, fill=color)


        # استخراج اطلاعات عمومی پارت از اولین بارکد معتبر
        first_valid = next((item for item in response_data if item.get("part_code")), None)

        # همیشه مقداردهی اولیه انجام بده تا خطا نده
        if 'part_info' not in locals():
            part_info = {
                "part_code": None,
                "manufacturer": None,
                "serial_prefix": None
            }

        if first_valid:
            part_info["part_code"] = first_valid.get("part_code")
            part_info["manufacturer"] = first_valid.get("manufacturer")
            serial = first_valid.get("serial")
            part_info["serial_prefix"] = serial[:4] if serial else None

        #بخش اخر که جیسون ساخته میشه . خطوط پایانی بعد از ترسیم تصویر
        stats = {
            "message": "✅ Barcode detection completed.",
            "threshold": threshold,
            "part_info": part_info,
            "total_barcodes_detected": total_barcodes,
            "unique_barcodes": unique_barcodes,
            "duplicate_barcodes": duplicate_barcodes,
            "duplicates": duplicates,
            "detections": [
                {
                    "crop_index": d["crop_index"],
                    "barcode_data": d["barcode_data"],
                    "barcode_text": d["barcode_text"],
                    "count": d["count"],
                    "is_duplicate": d["is_duplicate"]
                }
                for d in response_data
            ]
        }

        #  حالت JSON (پیش‌فرض)
        if not download_zip:
            return JSONResponse(stats)

        #  ساخت ZIP (اختیاری)
        zip_buffer = BytesIO()
        with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zipf:
            labeled_bytes = BytesIO()
            labeled.save(labeled_bytes, format="JPEG")
            labeled_bytes.seek(0)
            zipf.writestr("labeled_image.jpg", labeled_bytes.read())

            for i, crop in enumerate(crops):
                crop_bytes = BytesIO()
                crop.save(crop_bytes, format="JPEG")
                crop_bytes.seek(0)
                zipf.writestr(f"crop_{i}.jpg", crop_bytes.read())

            zipf.writestr("barcodes.json", json.dumps(stats, indent=4, ensure_ascii=False))

        zip_buffer.seek(0)
        zip_name = f"{os.path.splitext(file.filename)[0]}_barcode_prediction.zip"

        return StreamingResponse(
            zip_buffer,
            media_type="application/zip",
            headers={"Content-Disposition": f"attachment; filename={zip_name}"}
        )

    except Exception as e:
        print(f"[ERROR /predict/]: {e}")
        return JSONResponse({"error": str(e)}, status_code=500)


#  مسیر استاتیک برای نمایش فایل‌ها

app.mount("/static", StaticFiles(directory="app/static"), name="static")
