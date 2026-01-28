from enum import Enum
from pathlib import Path
import pickle
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from src.services.retrieval.chunk_generator import ChunkGenerator


class EmbeddingModel(str, Enum):
    all_MiniLM_L6_v2 = "sentence-transformers/all-MiniLM-L6-v2"
    multilingual_e5_small = "intfloat/multilingual-e5-small"


class RAGEngine:
    """ Движок """

    def __init__(self,
                 chunk_generator: ChunkGenerator,
                 embedding_model: SentenceTransformer,
                 index_dir: Path,
                 index_file: Path,
                 docs_file: Path):

        self.documents: list[dict[str, str]] = []
        self.index = None

        self.chunk_generator =chunk_generator
        self.embedding_model = embedding_model
        self.index_file = index_file
        self.index_dir = index_dir
        self.docs_file = docs_file

    def _build_index(self):
        """Создаёт FAISS-индекс на основе заметок."""
        chunks = self.chunk_generator.get_chunks()
        if chunks:
            print(f"Генерация эмбеддингов для {len(chunks)} чанков...")
            embeddings = self.embedding_model.encode(chunks, show_progress_bar=True)
            dim = embeddings.shape[1]
            self.index = faiss.IndexFlatL2(dim)
            self.index.add(np.array(embeddings).astype("float32"))
            self.documents = [{"text": txt, "source": src} for txt, src in zip(chunks, chunk_sources)]
            self.index_dir.mkdir(exist_ok=True)
            faiss.write_index(self.index, str(self.index_file))
            with open(self.docs_file, "wb") as f:
                pickle.dump(self.documents, f)
            print(f"Индекс сохранён. Всего чанков: {len(self.documents)}")

    def _load_index(self):
        """Загружает индекс с диска."""
        if not self.index_file.exists() or not self.docs_file.exists():
            return False
        self.index = faiss.read_index(str(self.index_file))
        with open(self.docs_file, "rb") as f:
            self.documents = pickle.load(f)
        print(f"Индекс загружен. Чанков: {len(self.documents)}")
        return True

    def load_or_build_index(self):
        """Загружает индекс или строит новый."""
        if not self._load_index():
            self._build_index()

    def retrieve(self, query: str, k: int = 4) -> str:
        """Возвращает текст релевантных чанков с указанием источников."""
        if self.index is None or len(self.documents) == 0:
            return ""
        query_vec = self.embedding_model.encode([query])
        _, I = self.index.search(query_vec.astype("float32"), k)
        results = []
        for idx in I[0]:
            if idx < len(self.documents):
                doc = self.documents[idx]
                results.append(f"[Источник: {doc['source']}]\n{doc['text']}")
        return "\n\n---\n\n".join(results)