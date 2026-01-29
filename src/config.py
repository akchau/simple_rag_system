from pathlib import Path
from typing import Optional
from pydantic_settings import BaseSettings
from src.api_clients.base import LLMChoice
from src.api_clients.mistral_api_client import ModelsEnum
from src.api_clients.ollama_api_client import OllamaModelsEnum
from src.services.retrieval.rag_engine import EmbeddingModel
from src.types_.base_types import ChunkSize
from src.utils.prompt_manager import PromptTypes


INDEX_DIR = Path("faiss_index")
INDEX_FILE = INDEX_DIR / "index.faiss"
DOCS_FILE = INDEX_DIR / "documents.pkl"


class Settings(BaseSettings):
    NOTES_DIR: str
    CHUNK_SIZE: ChunkSize
    OVERLAP: int
    PROMPT_TYPE: PromptTypes

    LLM_TYPE: LLMChoice

    EMBEDDING_MODEL: EmbeddingModel

    OLLAMA_BASE_URL: str
    OLLAMA_MODEL: OllamaModelsEnum
    OLLAMA_NUM_CTX: int = 8192
    OLLAMA_TEMPERATURE: float = 0.0
    OLLAMA_NUM_PREDICT: int = 500

    CHROMA_DIR_PATH: str
    CHROMA_COLLECTION_NAME: str
    CHROMA_DB_HOST: str
    CHROMA_DB_PORT: int
    CHROMA_PERSIST_DIRECTORY: Optional[str]
    CHROMA_ALLOW_RESET: bool = False
    CHROMA_ANONYMIZED_TELEMETRY: bool = False

    RETRIEVAL_K: int

    MISTRAL_API_TOKEN: str
    MISTRAL_MODEL: ModelsEnum

    RAG_ENGINE_TYPE: str

    class Config:
        env_file = ".env"


settings = Settings() # type: ignore