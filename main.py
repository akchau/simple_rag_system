import time
from typing import List
from mistralai.models.sdkerror import SDKError
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
    def __init__(self, api_key: str):
        self.client = Mistral(api_key=api_key)
    
    def send_request(self, text_request: str, model=ModelsEnum.LARGE) -> str:
        # Проверка на заглушку
        if settings.API_TOKEN == "dummy_key":
            return ">> (ИИ не отвечает, так как нет ключа, но поиск выше сработал!) <<"
            
        while True:
            try:
                response = self.client.chat.complete(
                    model=model,
                    messages=[{"role": "user", "content": text_request}]
                )
                return response.choices[0].message.content
            except SDKError:
                print("Ошибка API. Повтор через 5 сек...")
                time.sleep(5)
            except Exception as e:
                return f"Error: {e}"

INDEX_DIR = Path("faiss_index")
INDEX_FILE = INDEX_DIR / "index.faiss"
DOCS_FILE = INDEX_DIR / "documents.pkl"
EMBEDDING_MODEL = SentenceTransformer("intfloat/multilingual-e5-small")

class RAGEngine:
    def __init__(self, db_dir: LocalStoragePath, client: MistralClient):
        self.documents = []
        self.index = None
        self.db_dir_path = Path(db_dir)
        self.client = client
        self._load_documents()
        self._load_or_build_index()

    def _load_documents(self):
        if not self.db_dir_path.exists():
            self.db_dir_path.mkdir()
            print(f"Папка {self.db_dir_path} создана.")

        # Сортируем файлы для порядка
        for md_file in sorted(self.db_dir_path.glob("*.md")):
            try:
                elements = partition_md(filename=str(md_file))
                text = "\n".join([str(el) for el in elements])
                if text.strip():
                    self.documents.append({"text": text, "source": md_file.name})
            except Exception as e:
                print(f"Ошибка {md_file}: {e}")
        print(f"Загружено {len(self.documents)} документов.")

    def _chunk_text(self, text: str, chunk_size: int = 600, overlap: int = 80) -> list[str]:
        words = text.split()
        chunks = []
        for i in range(0, len(words), chunk_size - overlap):
            chunk = " ".join(words[i:i + chunk_size])
            chunks.append(chunk)
        return chunks

    def _build_index(self):
        if not self.documents:
            print("Нет заметок.")
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
        print("Индекс сохранён.")

    def _load_index(self):
        if not INDEX_FILE.exists() or not DOCS_FILE.exists():
            return False
        self.index = faiss.read_index(str(INDEX_FILE))
        with open(DOCS_FILE, "rb") as f:
            self.documents = pickle.load(f)
        print(f"Индекс загружен ({len(self.documents)} чанков).")
        return True

    def _load_or_build_index(self):
        if not self._load_index():
            self._build_index()

    def retrieve(self, query: str, k: int = 4) -> str:
        if self.index is None or len(self.documents) == 0:
            return ""
        query_vec = EMBEDDING_MODEL.encode([query])
        D, I = self.index.search(query_vec.astype("float32"), k)
        results = []
        for idx in I[0]:
            if idx < len(self.documents):
                doc = self.documents[idx]
                results.append(f"📄 [ФАЙЛ: {doc['source']}]\n{doc['text']}")
        return "\n\n------------------------------------------------\n\n".join(results)

RAG_PROMPT_TEMPLATE = """
Заметки: {context}
Вопрос: {question}
"""

def main():
    llm_client = MistralClient(settings.API_TOKEN)
    engine = RAGEngine(db_dir=settings.NOTES_DIR, client=llm_client)
    try:
        while True:
            q = input("\n🔎 Ваш вопрос: ").strip()
            if not q: continue
            
            print("\n...Ищу информацию в базе знаний...")
            ctx = engine.retrieve(q, k=1)
            
            if ctx:
                print("\n✅ НАЙДЕНЫ СЛЕДУЮЩИЕ ДОКУМЕНТЫ:")
                print("==================================================")
                print(ctx)
                print("==================================================")
            else:
                print("❌ Ничего релевантного не найдено.")

            print("\n🤖 Ответ ИИ:")
            print(llm_client.send_request(RAG_PROMPT_TEMPLATE.format(context=ctx, question=q), model=settings.MODEL))
            
    except KeyboardInterrupt:
        print("\nСтоп.")

if __name__ == "__main__":
    main()
