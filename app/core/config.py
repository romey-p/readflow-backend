import os
from dotenv import load_dotenv
from pathlib import Path

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent.parent

class Settings:
    PROJECT_NAME: str = "readflow-server"

    MONGO_URI: str = os.getenv("MONGO_URI", "")
    DATABASE_NAME: str = os.getenv("DATABASE_NAME", "")

    BASE_MODEL: str = "monologg/koelectra-base-v3-discriminator"
    WEIGHTS_DIR: str = os.path.join(BASE_DIR, "weights")
    WEIGHTS_PATH: str = os.path.join(BASE_DIR, "weights", "difficulty_regression_model.pt")

settings = Settings()