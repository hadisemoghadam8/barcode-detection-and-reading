# app/api/routes.py
from fastapi import APIRouter, File, UploadFile, Form
from fastapi.responses import FileResponse, JSONResponse
from ultralytics import YOLO
from app.services.crop_utils import get_crops
from app.services.detection import read_barcode_and_batch
from PIL import Image
from io import BytesIO
import os, zipfile, json, tempfile, cv2, numpy as np

router = APIRouter()
model = YOLO("./app/model/weights/best.pt")


@router.post("/predict/")
async def predict_image(
    file: UploadFile = File(...),
    threshold: float = Form(0.5),
    download_zip: bool = Form(False)
):
    try:
        # خواندن تصویر ارسالی
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        # اجرای YOLO برای شناسایی بارکدها
        results = model(img, conf=threshold)
        crops = get_crops(img, results)

        response_data = []
        temp_dir = tempfile.mkdtemp() if download_zip else None

        # پردازش هر کادر شناسایی‌شده
        for i, crop in enumerate(crops):
            buffer = BytesIO()
            Image.fromarray(crop).save(buffer, format="JPEG")

            # استخراج اطلاعات بارکد و کد بچ
            barcode = read_barcode_and_batch(buffer.getvalue())

            response_data.append({
                "crop_number": i,
                "barcode_data": barcode["barcode_data"] if barcode else "No barcode detected",
                "barcode_type": barcode["barcode_type"] if barcode else None,
            })

            # ذخیره تصویر در پوشه موقت در صورت نیاز
            if download_zip:
                crop_path = os.path.join(temp_dir, f"crop_{i}.jpg")
                cv2.imwrite(crop_path, crop)

        # در صورت درخواست ZIP خروجی
        if download_zip:
            labeled_image_array = results[0].plot()
            labeled_image_array = cv2.cvtColor(labeled_image_array, cv2.COLOR_BGR2RGB)
            labeled_image = Image.fromarray(labeled_image_array)
            labeled_image_path = os.path.join(temp_dir, "labeled_image.jpg")
            labeled_image.save(labeled_image_path, format="JPEG")

            json_path = os.path.join(temp_dir, "barcodes.json")
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(response_data, f, ensure_ascii=False, indent=4)

            zip_name = f"{os.path.splitext(file.filename)[0]}_barcode_prediction.zip"
            zip_path = os.path.join(temp_dir, zip_name)
            with zipfile.ZipFile(zip_path, "w") as zipf:
                zipf.write(labeled_image_path, "labeled_image.jpg")
                for i in range(len(crops)):
                    zipf.write(os.path.join(temp_dir, f"crop_{i}.jpg"), f"crop_{i}.jpg")
                zipf.write(json_path, "barcodes.json")

            return FileResponse(zip_path, media_type="application/zip", filename=zip_name)

        # بازگشت پاسخ JSON در حالت عادی
        return JSONResponse({
            "message": "✅ Barcode detection completed.",
            "threshold": threshold,
            "detections": response_data
        })

    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)
