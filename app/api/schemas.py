#\app\api\schemas.py
from pydantic import BaseModel

class DetectionResponse(BaseModel):
    barcode_value: str
    confidence: float
    type: str
