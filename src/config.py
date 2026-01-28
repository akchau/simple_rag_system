from pathlib import Path
from pydantic_settings import BaseSettings
from src.api_clients.mistral_api_client import ModelsEnum


INDEX_DIR = Path("faiss_index")
INDEX_FILE = INDEX_DIR / "index.faiss"
DOCS_FILE = INDEX_DIR / "documents.pkl"

class Settings(BaseSettings):
    API_TOKEN: str
    MODEL: ModelsEnum
    NOTES_DIR: str
    CHUNK_SIZE: int

    class Config:
        env_file = ".env"


settings = Settings() # type: ignore