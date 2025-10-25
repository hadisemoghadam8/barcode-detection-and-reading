#نسخه‌ی RAM-Based (بدون ذخیره در دیسک)

# app/main.py
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from ultralytics import YOLO
from io import BytesIO
from PIL import Image
from app.services.crop_utils import get_crops
from app.services.detection import read_barcode_and_batch
import io
from app.core.config import settings
import os, zipfile, json, cv2, numpy as np


app = FastAPI(debug=True)

# ✅ فعال‌سازی CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

# ✅ بارگذاری مدل YOLO
model = YOLO(settings.MODEL_PATH)

@app.post("/predict/")
async def predict_image(
    file: UploadFile = File(...),
    threshold: float = Form(0.5),
    download_zip: bool = Form(False)
):
    try:
        # 📥 خواندن تصویر
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        # 🚀 اجرای YOLO
        results = model(img, conf=threshold)
        crops = get_crops(img, results)

        response_data = []
        for i, crop in enumerate(crops):
            buffer = io.BytesIO()
            Image.fromarray(crop).save(buffer, format="JPEG")
            result = read_barcode_and_batch(buffer.getvalue())

            response_data.append({
                "crop_index": i,
                **result
            })

        # اگر ZIP خواسته نشده، فقط JSON بده
        if not download_zip:
            return JSONResponse({
                "message": "✅ Barcode detection completed.",
                "threshold": threshold,
                "detections": response_data
            })

        # 🖼 ساخت تصویر برچسب‌دار
        labeled_image_array = results[0].plot()
        if labeled_image_array.shape[2] == 3:
            labeled_image_array = cv2.cvtColor(labeled_image_array, cv2.COLOR_BGR2RGB)
        labeled_image = Image.fromarray(labeled_image_array)

        # 📦 ساخت ZIP در حافظه
        zip_buffer = BytesIO()
        with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zipf:
            # تصویر برچسب‌دار
            labeled_bytes = BytesIO()
            labeled_image.save(labeled_bytes, format="JPEG")
            labeled_bytes.seek(0)
            zipf.writestr("labeled_image.jpg", labeled_bytes.read())

            # برش‌ها
            for i, crop in enumerate(crops):
                _, buffer = cv2.imencode(".jpg", crop)
                zipf.writestr(f"crop_{i}.jpg", buffer.tobytes())

            # JSON
            zipf.writestr("barcodes.json", json.dumps(response_data, indent=4))

        zip_buffer.seek(0)

        # ✅ Swagger فایل ZIP را به‌صورت خودکار دانلود می‌کند
        zip_name = f"{os.path.splitext(file.filename)[0]}_barcode_prediction.zip"
        return StreamingResponse(
            zip_buffer,
            media_type="application/zip",
            headers={"Content-Disposition": f"attachment; filename={zip_name}"}
        )

    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


# 🌐 فقط برای نمایش فایل‌های استاتیک (اختیاری)
app.mount("/static", StaticFiles(directory="app/static"), name="static")
