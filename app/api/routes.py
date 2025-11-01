#  app/api/routes.py


from fastapi import APIRouter, File, UploadFile, Form
from fastapi.responses import JSONResponse, StreamingResponse
from ultralytics import YOLO
from app.services.crop_utils import get_crops
from app.services.detection import read_barcode_and_batch
from PIL import Image, ImageDraw
from io import BytesIO
import os, zipfile, json, numpy as np
from collections import Counter
import re
from urllib.parse import quote

router = APIRouter()
model = YOLO("./app/model/weights/best.pt")


@router.post("/predict/")
async def predict_image(
    file: UploadFile = File(...),
    threshold: float = Form(0.5),
    download_zip: bool = Form(False)
):
    try:
        #  خواندن تصویر ورودی
        contents = await file.read()
        img = Image.open(BytesIO(contents)).convert("RGB")
        img_np = np.array(img)

        #  اجرای YOLO برای شناسایی بارکدها
        results = model(img_np, conf=threshold)
        boxes = results[0].boxes.xyxy.cpu().numpy() if len(results[0].boxes) > 0 else []
        crops = get_crops(img, results)

        response_data = []
        labeled_image = img.copy()
        draw = ImageDraw.Draw(labeled_image)

        #  OCR و Barcode روی هر crop
        for i, crop in enumerate(crops):
            buffer = BytesIO()
            crop.save(buffer, format="JPEG")

            result = read_barcode_and_batch(buffer.getvalue())

            response_data.append({
                "crop_index": i,
                "barcode_data": result.get("barcode_data"),
                "barcode_type": result.get("barcode_type"),
                "barcode_text": result.get("barcode_text"),
            })

        #  شمارش بارکدها و تکراری‌ها
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

        #  افزودن شمارش تکرار به هر رکورد
        for item in response_data:
            data = item.get("barcode_data")
            count = barcode_counts.get(data, 0)
            item["count"] = count
            item["is_duplicate"] = count > 1

        #  رسم کادرها
        for i, item in enumerate(response_data):
            try:
                if i < len(boxes):
                    x1, y1, x2, y2 = map(int, boxes[i][:4])
                else:
                    continue

                data = item.get("barcode_data", "")
                count = item.get("count", 0)

                color = "green" if data and data != "No barcode detected" else "red"
                label_text = f"x{count}"

                draw.rectangle((x1, y1, x2, y2), outline=color, width=4)
                draw.text((x1, max(0, y1 - 18)), label_text, fill=color)
            except Exception:
                continue

        #  آمار کلی
        stats = {
            "message": "✅ Barcode detection completed.",
            "threshold": threshold,
            "total_barcodes_detected": total_barcodes,
            "unique_barcodes": unique_barcodes,
            "duplicate_barcodes": duplicate_barcodes,
            "duplicates": duplicates,
            "detections": response_data,
        }

        #  درصورت عدم درخواست ZIP فقط JSON برگردون
        if not download_zip:
            return JSONResponse(stats)

        #  ساخت فایل ZIP خروجی
        zip_buffer = BytesIO()
        with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zipf:
            # تصویر نهایی با برچسب‌ها
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
            zipf.writestr("barcodes.json", json.dumps(stats, indent=4, ensure_ascii=False))

        zip_buffer.seek(0)

        # پشتیبانی از نام فارسی
        orig_name = os.path.splitext(file.filename)[0]
        zip_filename = f"{orig_name}_barcode_prediction.zip"
        ascii_fallback = re.sub(r'[^\x00-\x7F]+', '_', zip_filename)
        quoted = quote(zip_filename, safe='')
        content_disposition = f'attachment; filename="{ascii_fallback}"; filename*=UTF-8\'\'{quoted}'

        return StreamingResponse(
            zip_buffer,
            media_type="application/zip",
            headers={"Content-Disposition": content_disposition}
        )

    except Exception as e:
        print(f"[ERROR /predict/]: {e}")
        return JSONResponse({"error": str(e)}, status_code=500)
