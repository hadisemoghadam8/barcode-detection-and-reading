#app/services/part_lookup.py

import csv
import os

def get_part_info_from_csv(part_code: str):
    """
    بر اساس PartCode، اطلاعات مربوط به PartName و PartNumber و PrintRepeatCount را از CSV می‌خواند.
    """
    csv_path = os.path.join(os.path.dirname(__file__), "..", "parts_config.csv")
    csv_path = os.path.abspath(csv_path)

    if not os.path.exists(csv_path):
        print(f"⚠️ CSV file not found at {csv_path}")
        return {
            "PartName": None,
            "PartNumber": None,
            "PrintRepeatCount": None
        }

    try:
        with open(csv_path, mode="r", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            for row in reader:
                # مقایسه بدون حساسیت به حروف
                if str(row.get("PartCode")).strip().upper() == str(part_code).strip().upper():
                    return {
                        "PartName": row.get("PartName"),
                        "PartNumber": row.get("PartNumber"),
                        "PrintRepeatCount": row.get("PrintRepeatCount")
                    }
    except Exception as e:
        print(f"❌ Error reading CSV: {e}")

    # اگر پیدا نشد
    return {
        "PartName": None,
        "PartNumber": None,
        "PrintRepeatCount": None
    }
