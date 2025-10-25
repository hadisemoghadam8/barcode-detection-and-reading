# ==========================================================
# 📦 File: app/api/routes.py
# 📋 وظیفه: API اصلی برای شناسایی بارکد و Batch Code
# ✅ نسخه بدون OpenCV و pytesseract (استفاده از Pillow + EasyOCR + pyzbar)
# ==========================================================

from fastapi import APIRouter, File, UploadFile, Form
from fastapi.responses import JSONResponse, StreamingResponse
from ultralytics import YOLO
from app.services.crop_utils import get_crops
from app.services.detection import read_barcode_and_batch
from PIL import Image, ImageDraw
from io import BytesIO
import os, zipfile, json, numpy as np


router = APIRouter()
model = YOLO("./app/model/weights/best.pt")


@router.post("/predict/")
async def predict_image(
    file: UploadFile = File(...),
    threshold: float = Form(0.5),
    download_zip: bool = Form(False)
):
    try:
        # 📥 خواندن تصویر
        contents = await file.read()
        img = Image.open(BytesIO(contents)).convert("RGB")
        img_np = np.array(img)

        # 🚀 اجرای YOLO
        results = model(img_np, conf=threshold)
        crops = get_crops(img, results)

        response_data = []
        labeled_image = img.copy()
        draw = ImageDraw.Draw(labeled_image)

        # ✂️ پردازش هر crop و رسم مستطیل
        for i, crop in enumerate(crops):
            buffer = BytesIO()
            crop.save(buffer, format="JPEG")
            result = read_barcode_and_batch(buffer.getvalue())

            # 📊 ذخیره نتیجه در خروجی JSON
            response_data.append({
                "crop_number": i,
                "barcode_data": result.get("barcode_data"),
                "barcode_type": result.get("barcode_type"),
                "barcode_text": result.get("barcode_text"),
            })

            # 🎨 رسم مستطیل با رنگ‌بندی بر اساس تطبیق داده‌ها
            box = results[0].boxes.xyxy[i].cpu().numpy().astype(int)
            x1, y1, x2, y2 = box

            data = result.get("barcode_data", "")
            text = result.get("barcode_text", "")

            if not data or not text:
                color = "yellow"      # یکی از داده‌ها شناسایی نشده
            elif data.strip() == text.strip():
                color = "green"       # تطابق کامل
            else:
                color = "red"         # عدم تطابق

            draw.rectangle((x1, y1, x2, y2), outline=color, width=4)
            draw.text((x1, max(0, y1 - 14)), f"{i+1}", fill=color)

        # ⚙️ اگر ZIP خواسته نشده → فقط JSON برگردون
        if not download_zip:
            return JSONResponse({
                "message": "✅ Barcode detection completed.",
                "threshold": threshold,
                "detections": response_data
            })

        # 📦 ساخت ZIP در حافظه
        zip_buffer = BytesIO()
        with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zipf:
            # تصویر نهایی برچسب‌دار
            labeled_bytes = BytesIO()
            labeled_image.save(labeled_bytes, format="JPEG")
            labeled_bytes.seek(0)
            zipf.writestr("labeled_image.jpg", labeled_bytes.read())

            # برش‌ها
            for i, crop in enumerate(crops):
                crop_bytes = BytesIO()
                crop.save(crop_bytes, format="JPEG")
                crop_bytes.seek(0)
                zipf.writestr(f"crop_{i}.jpg", crop_bytes.read())

            # فایل JSON
            zipf.writestr("barcodes.json", json.dumps(response_data, indent=4))

        zip_buffer.seek(0)

        zip_name = f"{os.path.splitext(file.filename)[0]}_barcode_prediction.zip"
        return StreamingResponse(
            zip_buffer,
            media_type="application/zip",
            headers={"Content-Disposition": f"attachment; filename={zip_name}"}
        )

    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)
