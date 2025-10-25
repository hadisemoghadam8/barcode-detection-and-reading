# ==========================================================
# 📦 File: app/main.py  (RAM-based + No OpenCV)
# 🧠 هدف: تشخیص بارکد و متن روی تصاویر با YOLO + OCR (بدون ذخیره روی دیسک)
# ==========================================================

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
import zipfile, json, os


# ------------------------- راه‌اندازی FastAPI -------------------------
app = FastAPI(
    title="📦 Barcode & Batch OCR API",
    debug=True
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)


# ------------------------- بارگذاری مدل YOLO -------------------------
model = YOLO(settings.MODEL_PATH)


# ==========================================================
# 📤 Endpoint: /predict/
# ==========================================================
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

        results = model(np_img, conf=threshold)
        boxes = results[0].boxes.xyxy.cpu().numpy()

        response_data = []
        crops = []

        # --- پردازش هر باکس ---
        for i, box in enumerate(boxes):
            x1, y1, x2, y2 = map(int, box[:4])
            crop = image.crop((x1, y1, x2, y2))
            crops.append(crop)

            buffer = BytesIO()
            crop.save(buffer, format="JPEG")
            result = read_barcode_and_batch(buffer.getvalue())
            result["crop_index"] = i
            response_data.append(result)

        # --- فقط JSON (بدون ZIP) ---
        if not download_zip:
            return JSONResponse({
                "message": "✅ Barcode detection completed.",
                "threshold": threshold,
                "detections": response_data
            })

        # ==========================================================
        # 📸 تولید تصویر با رنگ‌بندی بر اساس تطبیق داده‌ها
        # ==========================================================
        labeled = image.copy()
        draw = ImageDraw.Draw(labeled)

        for i, box in enumerate(boxes):
            x1, y1, x2, y2 = map(int, box[:4])
            result = response_data[i]

            data = result.get("barcode_data", "")
            text = result.get("barcode_text", "")

            # 🎨 انتخاب رنگ بر اساس وضعیت تطبیق
            if not data or not text:
                color = "yellow"      # بارکد یا متن شناسایی نشده
            elif data.strip() == text.strip():
                color = "green"       # تطابق کامل
            else:
                color = "red"         # عدم تطابق

            # رسم مستطیل و شماره‌ی باکس
            draw.rectangle((x1, y1, x2, y2), outline=color, width=4)
            draw.text((x1, max(0, y1 - 14)), f"{i+1}", fill=color)

        # ذخیره در حافظه
        labeled_bytes = BytesIO()
        labeled.save(labeled_bytes, format="JPEG")
        labeled_bytes.seek(0)

        # ==========================================================
        # 📦 ساخت ZIP شامل تصویر رنگی + برش‌ها + JSON
        # ==========================================================
        zip_buffer = BytesIO()
        with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zipf:
            zipf.writestr("labeled_image.jpg", labeled_bytes.read())

            for i, crop in enumerate(crops):
                crop_bytes = BytesIO()
                crop.save(crop_bytes, format="JPEG")
                crop_bytes.seek(0)
                zipf.writestr(f"crop_{i}.jpg", crop_bytes.read())

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


# ==========================================================
# 🌐 مسیر استاتیک (برای فایل‌های فرانت‌اند در صورت نیاز)
# ==========================================================
app.mount("/static", StaticFiles(directory="app/static"), name="static")

# 🚫 هیچ endpoint برای "/" تعریف نشده
# در نتیجه، هر درخواست GET به ریشه → خطای 404 برمی‌گردد.
