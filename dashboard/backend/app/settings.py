from pydantic_settings import BaseSettings
from dotenv import load_dotenv

load_dotenv()

class Settings(BaseSettings):
    MONGODB_URI: str
    MONGODB_DB: str = "karachi_aqi"
    FRONTEND_ORIGIN: str = "http://localhost:5173"

settings = Settings()
