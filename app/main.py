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

import numpy as np
import zipfile
import json
import os
from collections import Counter


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
    download_zip: bool = Form(False)
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

            response_data.append({
                "crop_index": i,
                "barcode_data": result.get("barcode_data"),
                "barcode_type": result.get("barcode_type"),
                "barcode_text": result.get("barcode_text"),
            })

        # --- شمارش تکراری‌ها ---
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

        # افزودن شمارش به خروجی
        for item in response_data:
            data = item.get("barcode_data")
            count = barcode_counts.get(data, 0)
            item["count"] = count
            item["is_duplicate"] = count > 1

        # تولید تصویر نهایی با برچسب‌ها
        labeled = image.copy()
        draw = ImageDraw.Draw(labeled)

        for i, box in enumerate(boxes):
            x1, y1, x2, y2 = map(int, box[:4])
            item = response_data[i]
            data = item.get("barcode_data", "")
            count = item.get("count", 0)

            # رنگ‌گذاری بر اساس وضعیت
            if not data or data == "No barcode detected":
                color = "red"
                label_text = "x0"
            else:
                color = "green"
                label_text = f"x{count}"

            draw.rectangle((x1, y1, x2, y2), outline=color, width=3)
            draw.text((x1, max(0, y1 - 18)), label_text, fill=color)

        #  آمار کلی
        stats = {
            "message": "✅ Barcode detection completed.",
            "threshold": threshold,
            "total_barcodes_detected": total_barcodes,
            "unique_barcodes": unique_barcodes,
            "duplicate_barcodes": duplicate_barcodes,
            "duplicates": duplicates,
            "detections": response_data
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
