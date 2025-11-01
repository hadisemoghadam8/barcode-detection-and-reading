# app/services/barcode_rules/ikco.py
import csv
from pathlib import Path

# ---------------------------
# خواندن اطلاعات PartCode از فایل CSV
# ---------------------------
def load_part_data():
    path = Path(__file__).resolve().parent.parent / "parts_config.csv"
    part_data = {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                code = row["PartCode"].strip()
                part_data[code] = {
                    "PartName": row["PartName"],
                    "PartNumber": row["PartNumber"],
                    "PrintRepeatCount": int(row["PrintRepeatCount"]),
                }
    except Exception as e:
        print(f"[WARN] Could not load IKCO part data: {e}")
    return part_data


IKCO_PARTS = load_part_data()


# ---------------------------
# تحلیل بارکد ایران‌خودرو
# ---------------------------
def parse_barcode(barcode: str):
    """
    استخراج اطلاعات از بارکدهای ایران‌خودرو
    فرمت: SHV23SD2140007
    """
    try:
        barcode = barcode.strip()

        if not barcode.startswith("SHV"):
            return {"manufacturer": "Unknown"}

        prefix = barcode[:3]
        part_code = barcode[3:5]
        serial = barcode[5:]

        info = IKCO_PARTS.get(part_code, None)

        return {
            "manufacturer": "IKCO",
            "prefix": prefix,
            "part_code": part_code,
            "serial": serial,
            "part_info": info
        }

    except Exception as e:
        return {"error": f"Failed to parse IKCO barcode: {e}"}
