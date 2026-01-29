
from src.services.local_manger.local_manager import LocalManager
from src.services.retrieval.exc import DocsNotExist
from src.types_.base_types import ChunkSize, Overlap

DocText = str
ChunkText = str

class ChunkGenerator:

    def __init__(self, 
            local_manager: LocalManager, 
            chunk_size: ChunkSize,
            overlap: Overlap
    ):
        self.overlap = overlap
        self.local_manager = local_manager
        self.chunk_size = chunk_size
        
        if chunk_size <= 0:
            raise ValueError("chunk_size должен быть положительным числом")
        if overlap < 0:
            raise ValueError("overlap не может быть отрицательным")
        if overlap >= chunk_size:
            raise ValueError("overlap должен быть меньше chunk_size")

    def _chunk_doc(self, doc_text: DocText) -> list[ChunkText]:
        """Разбивает текст на чанки по символам с перекрытием"""
        chunks: list[str] = []
        text_len = len(doc_text)
        if text_len <= self.chunk_size:
            if doc_text.strip():
                chunks.append(doc_text)
            return chunks

        step = self.chunk_size - self.overlap
        for i in range(0, text_len, step):
            chunk = doc_text[i:i + self.chunk_size]
            if chunk.strip():
                chunks.append(chunk)
        
        return chunks

    def get_chunks(self) -> list[ChunkText]:
        raw_docs = self.local_manager.get_documents_data()
        if not raw_docs:
            raise DocsNotExist

        all_chunks: list[ChunkText] = []
        chunk_sources: list[str] = []

        for doc in raw_docs:
            chunks = self._chunk_doc(doc["text"])
            for chunk in chunks:
                if chunk.strip():
                    all_chunks.append(chunk)
                    chunk_sources.append(doc["source"])

        if not all_chunks:
            print("Нет текста для индексации.")
        
        return all_chunks