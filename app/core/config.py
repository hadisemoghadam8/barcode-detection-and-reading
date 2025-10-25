#app\core\config.py
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    MODEL_PATH: str = "./app/model/weights/best.pt"

    class Config:
        env_file = ".env"

settings = Settings()
