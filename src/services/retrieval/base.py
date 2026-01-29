from abc import ABC, abstractmethod
from enum import Enum
from typing import ClassVar, Generic, TypeVar

from pydantic import BaseModel

from sentence_transformers import SentenceTransformer

from src.services.retrieval.chunk_generator import ChunkGenerator
from src.types_.base_types import UserQuestion


class RAGEngineType(str, Enum):
    FAISS = "faiss"
    CHROMADB = "chromadb"


class BaseRetrievalConfig(BaseModel):
    retrieval_k: int


RAGEngineConfigTypeVar = TypeVar("RAGEngineConfigTypeVar", bound=BaseRetrievalConfig)


class RAGEngineBase(ABC, Generic[RAGEngineConfigTypeVar]):
    """
    Абстрактный базовый класс для RAG‑движков.
    Каждый наследник должен определить атрибут `engine_type`.
    """
    engine_type: ClassVar[RAGEngineType]

    def __init__(self, config: RAGEngineConfigTypeVar, chunk_generator: ChunkGenerator, embedding_model: SentenceTransformer):
        self.chunk_generator = chunk_generator
        self.embedding_model = embedding_model
        self.config = config

    @abstractmethod
    def build_index(self) -> None:
        pass

    @abstractmethod
    def load_index(self) -> bool:
        pass

    @abstractmethod
    def retrieve(self, query: UserQuestion) -> str:
        pass

    def load_or_build_index(self) -> None:
        if not self.load_index():
            self.build_index()
