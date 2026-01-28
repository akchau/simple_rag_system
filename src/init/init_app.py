from dataclasses import dataclass
from pathlib import Path
from typing import Type

from pydantic import BaseModel
from sentence_transformers import SentenceTransformer

from src.api_clients.base import LLMChoice
from src.api_clients.factory import ApiClientFactory
from src.api_clients.ollama_api_client import OllamaInitModel, OllamaModelsEnum
from src.config import DOCS_FILE, INDEX_DIR, INDEX_FILE, settings

from src.services.retrieval.rag_engine import ChunkGenerator, RAGEngine

from src.api_clients.mistral_api_client import MistralClient, MistralInitData, ModelsEnum
from src.services.local_manager import LocalManager
from src.utils.prompt_manager import BasePromptManager, PromptFactory


LLM_CLIENTS_INIT_DATA: dict[LLMChoice, BaseModel] = {
    LLMChoice.MISTRAL: MistralInitData(
        api_key=settings.MISTRAL_API_TOKEN,
        model=settings.MISTRAL_MODEL
    ),
    LLMChoice.OLLAMA: OllamaInitModel(
        model=settings.OLLAMA_MODEL,
        url="http://localhost:11434/api/generate"
    )   
}

def init_llm_client(llm_type: LLMChoice):
    print(f"Запуск с LLM: {llm_type.value}")
    llm_client_class = ApiClientFactory.get_client_by_type(llm_type)
    return llm_client_class(LLM_CLIENTS_INIT_DATA.get(llm_type))


@dataclass
class AppContainer:
    llm_client: MistralClient
    engine: RAGEngine
    prompt_manager_class: Type[BasePromptManager]

prompt_manager_class = PromptFactory.get_prompt_class_by_type(settings.PROMPT_TYPE)
local_manager = LocalManager(dir_path=Path(settings.NOTES_DIR))

chunk_generator = ChunkGenerator(
    local_manager=local_manager,
    chunk_size=600
)
llm_client = init_llm_client(settings.LLM_TYPE)
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
        engine=engine,
        prompt_manager_class=prompt_manager_class
    )