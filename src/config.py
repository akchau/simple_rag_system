from pathlib import Path
from pydantic_settings import BaseSettings
from src.api_clients.base import LLMChoice
from src.api_clients.mistral_api_client import ModelsEnum
from src.api_clients.ollama_api_client import OllamaModelsEnum
from src.services.retrieval.rag_engine import EmbeddingModel
from src.utils.prompt_manager import PromptTypes


INDEX_DIR = Path("faiss_index")
INDEX_FILE = INDEX_DIR / "index.faiss"
DOCS_FILE = INDEX_DIR / "documents.pkl"


class Settings(BaseSettings):
    MISTRAL_API_TOKEN: str
    MISTRAL_MODEL: ModelsEnum
    NOTES_DIR: str
    CHUNK_SIZE: int
    PROMPT_TYPE: PromptTypes
    LLM_TYPE: LLMChoice
    EMBEDDING_MODEL: EmbeddingModel
    OLLAMA_MODEL: OllamaModelsEnum

    class Config:
        env_file = ".env"


settings = Settings() # type: ignore