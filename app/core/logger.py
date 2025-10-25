#app\core\logger.py
from loguru import logger

logger.add("logs.log", rotation="1 MB")
