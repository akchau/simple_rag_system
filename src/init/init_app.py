from dataclasses import dataclass
from pathlib import Path

from sentence_transformers import SentenceTransformer

from src.config import DOCS_FILE, INDEX_DIR, INDEX_FILE, settings

from src.services.retrieval.rag_engine import ChunkGenerator, RAGEngine

from src.api_clients.mistral_api_client import MistralClient, ModelsEnum
from src.services.local_manager import LocalManager


@dataclass
class AppContainer:
    llm_client: MistralClient
    engine: RAGEngine


local_manager = LocalManager(dir_path=Path(settings.NOTES_DIR))
llm_client = MistralClient(settings.API_TOKEN, model=ModelsEnum.LARGE.value)
chunk_generator = ChunkGenerator(
    local_manager=local_manager,
    chunk_size=600
)
embedding_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
engine = RAGEngine(
    chunk_generator=llm_client,
    embedding_model=embedding_model,
    index_file=INDEX_FILE,
    docs_file=DOCS_FILE,
    index_dir=INDEX_DIR
)

def get_app_container() -> AppContainer:
    return AppContainer(
        llm_client=llm_client,
        engine=engine
    )