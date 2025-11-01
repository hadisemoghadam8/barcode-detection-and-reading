# app/services/barcode_rules/saipa.py

def parse_barcode(barcode: str):
    """
    تابع نمونه برای پارس بارکدهای سایپا.
    فعلاً فقط ساختار خالی برمی‌گردونه تا ارور نده.
    """
    if not barcode:
        return {
            "manufacturer": "Saipa",
            "prefix": None,
            "part_code": None,
            "serial": None,
            "part_info": None
        }

    # نمونه ساده: فرض کن بارکد سایپا مثل "SPX45ABC12345" باشه
    return {
        "manufacturer": "Saipa",
        "prefix": barcode[:3],
        "part_code": barcode[3:5],
        "serial": barcode[5:],
        "part_info": {
            "PrintRepeatCount": 1
        }
    }
