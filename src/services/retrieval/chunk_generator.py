
from src.services.local_manger.local_manager import LocalManager
from src.types_.base_types import ChunkSize, Overlap

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

    def _chunk_text(self, text: str) -> list[ChunkText]:
        """ Разбивает текст на чанки """
        words = text.split()
        chunks: list[str] = []
        
        for i in range(0, len(words), self.chunk_size - self.overlap):
            chunk = " ".join(words[i:i + self.chunk_size])
            chunks.append(chunk)
        return chunks

    def get_chunks(self) -> list[ChunkText]:
        raw_docs = self.local_manager.get_documents_data()
        if not raw_docs:
            print("Нет заметок для индексации.")
            return []

        all_chunks: list[ChunkText] = []
        chunk_sources: list[str] = []

        for doc in raw_docs:
            chunks = self._chunk_text(doc["text"])
            for chunk in chunks:
                if chunk.strip():
                    all_chunks.append(chunk)
                    chunk_sources.append(doc["source"])

        if not all_chunks:
            print("Нет текста для индексации.")
        
        return all_chunks