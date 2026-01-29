from enum import Enum
from pathlib import Path
import pickle
from typing import Type
import chromadb
from chromadb.api import ClientAPI
from chromadb.config import Settings
import faiss
import numpy as np
from src.services.retrieval.base import BaseRetrievalConfig, RAGEngineBase


class EmbeddingModel(str, Enum):
    all_MiniLM_L6_v2 = "sentence-transformers/all-MiniLM-L6-v2"
    multilingual_e5_small = "intfloat/multilingual-e5-small"


class RAGEngineType(str, Enum):
    FAISS = "faiss"
    CHROMADB = "chromadb"


class ChromaRAGConfig(BaseRetrievalConfig):
    db_dir: Path
    collection_name: str = "rag_collection"
    host: str = "localhost"
    port: int = 8000
    persist_directory: Path | None = None  # Для persistent-хранилища
    allow_reset: bool = False
    anonymized_telemetry: bool = False


class FAISSRAGConfig(BaseRetrievalConfig):
    index_dir: Path
    index_file: Path
    docs_file: Path
    retrieval_k: int = 4


class FAISSRAGEngine(RAGEngineBase[FAISSRAGConfig]):
    engine_type = RAGEngineType.FAISS

    def __init__(self, config: FAISSRAGConfig):
        super().__init__(config)
        self.index = None
        self.documents: list[dict[str, str]] = []

    def build_index(self) -> None:
        chunks = self.chunk_generator.get_chunks()
        if not chunks:
            print("Нет чанков для индексации.")
            return

        print(f"Генерация эмбеддингов для {len(chunks)} чанков...")
        embeddings = self.embedding_model.encode(chunks, show_progress_bar=True)
        dim = embeddings.shape[1]

        self.index = faiss.IndexFlatL2(dim)
        self.index.add(np.array(embeddings).astype("float32"))

        self.documents = [
            {"text": txt, "source": src}
            for txt, src in zip(chunks, [c["source"] for c in chunks])
        ]

        self.config.index_dir.mkdir(exist_ok=True)
        faiss.write_index(self.index, str(self.config.index_file))
        with open(self.config.docs_file, "wb") as f:
            pickle.dump(self.documents, f)

        print(f"Индекс сохранён. Всего чанков: {len(self.documents)}")

    def load_index(self) -> bool:
        if (not self.config.index_file.exists() or
            not self.config.docs_file.exists()):
            return False

        self.index = faiss.read_index(str(self.config.index_file))
        with open(self.config.docs_file, "rb") as f:
            self.documents = pickle.load(f)

        print(f"Индекс загружен. Чанков: {len(self.documents)}")
        return True

    def retrieve(self, query: str) -> str:
        if self.index is None or len(self.documents) == 0:
            return ""

        query_vec = self.embedding_model.encode([query])
        _, I = self.index.search(
            query_vec.astype("float32"),
            self.config.retrieval_k  # Берём k из конфигурации
        )

        results = []
        for idx in I[0]:
            if idx < len(self.documents):
                doc = self.documents[idx]
                results.append(f"[Источник: {doc['source']}]\n{doc['text']}")

        return "\n\n---\n\n".join(results)


class ChromaRAGEngine(RAGEngineBase[ChromaRAGConfig]):
    engine_type = RAGEngineType.CHROMADB

    @property
    def settings(self) -> Settings:
        return Settings(
            chromedb_http_host=self.config.host,
            chromedb_http_port=self.config.port,
            allow_reset=self.config.allow_reset,
            anonymized_telemetry=self.config.anonymized_telemetry
        )

    @property
    def client(self) -> ClientAPI:
        """Ленивая инициализация клиента ChromaDB"""
        if self.config.persist_directory:
            self._client = chromadb.PersistentClient(
                path=str(self.config.persist_directory),
                settings=self.settings
            )
        else:
            self._client = chromadb.Client(self.settings)
        return self._client


    def _embed_func(self, texts: List[str]) -> List[List[float]]:
        return self.embedding_model.encode(texts).tolist()


    def build_index(self) -> None:
        chunks = self.chunk_generator.get_chunks()
        if not chunks:
            print("Нет чанков для индексации.")
            return

        documents = [chunk["text"] for chunk in chunks]
        metadatas = [{"source": chunk["source"]} for chunk in chunks]
        ids = [str(i) for i in range(len(chunks))]

        print(f"Добавление {len(documents)} чанков в коллекцию...")

        self.collection = self.client.get_or_create_collection(
            name=self.config.collection_name,
            embedding_function=self._embed_func
        )

        self.collection.add(
            documents=documents,
            metadatas=metadatas,
            ids=ids
        )
        print(f"Коллекция обновлена. Всего документов: {len(documents)}")


    def load_index(self) -> bool:
        self.collection = self.client.get_or_create_collection(
            name=self.config.collection_name,
            embedding_function=self._embed_func
        )
        count = self.collection.count()
        print(f"Коллекция загружена. Документов: {count}")
        return count > 0

    def retrieve(self, query: str) -> str:
        if self.collection is None or self.collection.count() == 0:
            return ""

        results = self.collection.query(
            query_texts=[query],
            n_results=self.config.retrieval_k,
            include=["documents", "metadatas"]
        )

        formatted_results = []
        for doc, meta in zip(results["documents"][0], results["metadatas"][0]):
            formatted_results.append(f"[Источник: {meta['source']}]\n{doc}")


        return "\n\n---\n\n".join(formatted_results)


class RagEngineFactory:

    engines: list[Type[RAGEngineBase]] = [
        FAISSRAGEngine,
        ChromaRAGEngine
    ]

    @classmethod
    def get_rag_engine_by_type(cls, target_type: RAGEngineType) -> Type[RAGEngineBase]:
        for engine in cls.engines:
            if engine.engine_type == target_type:
                return engine
        raise RuntimeError("Не удалось получить движок")
