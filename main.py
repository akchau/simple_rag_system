
import time
from typing import List
from mistralai.models.audiochunk import AudioChunk
from mistralai.models.documenturlchunk import DocumentURLChunk
from mistralai.models.filechunk import FileChunk
from mistralai.models.imageurlchunk import ImageURLChunk
from mistralai.models.referencechunk import ReferenceChunk
from mistralai.models.sdkerror import SDKError
from mistralai.models.textchunk import TextChunk

from mistralai.models.thinkchunk import ThinkChunk

from mistralai.types.basemodel import Unset

import numpy as np
from pathlib import Path
from mistralai import Mistral
from unstructured.partition.md import partition_md

from base_types import LocalStoragePath
from config import settings
from enums import ModelsEnum
from sentence_transformers import SentenceTransformer
import faiss
import pickle



class MistralClient:
    """ Клиент для выполнения запросов к БД """
    def __init__(self, api_key: str):
        self.client = Mistral(api_key=api_key)
    
    def send_request(self, text_request: str, model=ModelsEnum.LARGE) -> LocalStoragePath | List[ImageURLChunk | DocumentURLChunk | TextChunk | ReferenceChunk | FileChunk | ThinkChunk | AudioChunk] | None | Unset:
        while True:
            try:
                response = self.client.chat.complete(
                    model=model,
                    messages=[
                        {"role": "user", "content": text_request},
                    ]
                )
                return response.choices[0].message.content
            except SDKError:
                print("Произошла ошибка при запросе. Спросим еще раз через 5 секунд")
                time.sleep(5)



INDEX_DIR = Path("faiss_index")
INDEX_FILE = INDEX_DIR / "index.faiss"
DOCS_FILE = INDEX_DIR / "documents.pkl"
EMBEDDING_MODEL = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")


class RAGEngine:
    """ Движок """

    def __init__(self, db_dir: LocalStoragePath, client: MistralClient):
        self.documents = []
        self.index = None
        self.db_dir_path = Path(db_dir)
        self.client = client
        self._load_documents()
        self._load_or_build_index()

    def _load_documents(self):
        """Загружает все .md файлы из директории."""
        if not self.db_dir_path.exists():
            self.db_dir_path.mkdir()
            print(f"Папка {self.db_dir_path} создана. Добавьте туда свои .md заметки!")

        for md_file in self.db_dir_path.glob("*.md"):
            try:
                elements = partition_md(filename=str(md_file))
                text = "\n".join([str(el) for el in elements])
                if text.strip():
                    self.documents.append({"text": text, "source": md_file.name})
            except Exception as e:
                print(f"Ошибка при обработке {md_file}: {e}")
        print(f"Загружено {len(self.documents)} документов.")

    def _chunk_text(self, text: str, chunk_size: int = 600, overlap: int = 80) -> list[str]:
        """Разбивает текст на чанки."""
        words = text.split()
        chunks = []
        for i in range(0, len(words), chunk_size - overlap):
            chunk = " ".join(words[i:i + chunk_size])
            chunks.append(chunk)
        return chunks

    def _build_index(self):
        """Создаёт FAISS-индекс на основе заметок."""
        if not self.documents:
            print("Нет заметок для индексации.")
            return

        all_chunks = []
        chunk_sources = []

        for doc in self.documents:
            chunks = self._chunk_text(doc["text"])
            for chunk in chunks:
                if chunk.strip():
                    all_chunks.append(chunk)
                    chunk_sources.append(doc["source"])

        if not all_chunks:
            print("Нет текста для индексации.")
            return

        print(f"Генерация эмбеддингов для {len(all_chunks)} чанков...")
        embeddings = EMBEDDING_MODEL.encode(all_chunks, show_progress_bar=True)
        dim = embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dim)
        self.index.add(np.array(embeddings).astype("float32"))
        self.documents = [{"text": txt, "source": src} for txt, src in zip(all_chunks, chunk_sources)]
        INDEX_DIR.mkdir(exist_ok=True)
        faiss.write_index(self.index, str(INDEX_FILE))
        with open(DOCS_FILE, "wb") as f:
            pickle.dump(self.documents, f)

        print(f"Индекс сохранён. Всего чанков: {len(self.documents)}")

    def _load_index(self):
        """Загружает индекс с диска."""
        if not INDEX_FILE.exists() or not DOCS_FILE.exists():
            return False
        self.index = faiss.read_index(str(INDEX_FILE))
        with open(DOCS_FILE, "rb") as f:
            self.documents = pickle.load(f)
        print(f"Индекс загружен. Чанков: {len(self.documents)}")
        return True

    def _load_or_build_index(self):
        """Загружает индекс или строит новый."""
        if not self._load_index():
            self._build_index()

    def retrieve(self, query: str, k: int = 4) -> str:
        """Возвращает текст релевантных чанков с указанием источников."""
        if self.index is None or len(self.documents) == 0:
            return ""
        query_vec = EMBEDDING_MODEL.encode([query])
        D, I = self.index.search(query_vec.astype("float32"), k)
        results = []
        for idx in I[0]:
            if idx < len(self.documents):
                doc = self.documents[idx]
                results.append(f"[Источник: {doc['source']}]\n{doc['text']}")
        return "\n\n---\n\n".join(results)



RAG_PROMPT_TEMPLATE = """
Ты — помощник, отвечающий ТОЛЬКО на основе предоставленных заметок.
Если информации нет — подготовь ее сам."

Заметки:
{context}

Вопрос: {question}
Ответ:
"""


def main():
    llm_client = MistralClient(settings.API_TOKEN)
    engine = RAGEngine(db_dir=settings.NOTES_DIR, client=llm_client)

    try:
        while True:
            question = input("Введите ваш запрос: ").strip()
            if not question:
                continue

            context = engine.retrieve(question)
            if not context:
                print("Нет данных в заметках. Отправляю запрос без контекста...\n")
                final_prompt = question
            else:
                final_prompt = RAG_PROMPT_TEMPLATE.format(context=context, question=question)

            print("\n\n\n----------------------------- Результат --------------------------")
            answer = llm_client.send_request(final_prompt, model=settings.MODEL)
            print(answer)
            print("----------------------------- ------- --------------------------\n\n\n")

    except KeyboardInterrupt:
        print("\nСервис остановлен!")

main()