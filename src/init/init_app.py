from dataclasses import dataclass
from pathlib import Path
from typing import Type

from pydantic import BaseModel
from sentence_transformers import SentenceTransformer

from src.api_clients.base import BaseLLMClient, LLMChoice
from src.api_clients.factory import ApiClientFactory
from src.api_clients.ollama_api_client import OllamaInitModel
from src.config import DOCS_FILE, INDEX_DIR, INDEX_FILE, settings

from src.services.retrieval.base import BaseRetrievalConfig, RAGEngineBase, RAGEngineType
from src.services.retrieval.chunk_generator import ChunkGenerator
from src.services.retrieval.rag_engine import ChromaRAGConfig, FAISSRAGConfig, RagEngineFactory

from src.api_clients.mistral_api_client import MistralClient, MistralInitData
from src.services.local_manger.local_manager import LocalManager
from src.utils.prompt_manager import BasePromptManager, PromptFactory


__local_manager = LocalManager(dir_path=Path(settings.NOTES_DIR))
__embedding_model = SentenceTransformer(settings.EMBEDDING_MODEL)
__chunk_generator = ChunkGenerator(
    local_manager=__local_manager,
    chunk_size=settings.CHUNK_SIZE,
    overlap=settings.OVERLAP
)


RAG_ENGINE_INIT_DATA: dict[RAGEngineType, BaseRetrievalConfig] = {
    RAGEngineType.CHROMADB: ChromaRAGConfig(
        db_dir=Path(settings.CHROMA_DIR_PATH),
        collection_name="rag_collection",
        host=settings.CHROMA_DB_HOST,
        port=settings.CHROMA_DB_PORT,
        persist_directory=settings.CHROMA_PERSIST_DIRECTORY,
        allow_reset=settings.CHROMA_ALLOW_RESET,
        anonymized_telemetry=settings.CHROMA_ANONYMIZED_TELEMETRY,
        retrieval_k=settings.RETRIEVAL_K
    ),
    RAGEngineType.FAISS: FAISSRAGConfig(
        index_file=INDEX_FILE,
        docs_file=DOCS_FILE,
        index_dir=INDEX_DIR,
        retrieval_k=settings.RETRIEVAL_K
    )
}

def init_rag_client(rag_type: RAGEngineType) -> RAGEngineBase:
    rag_engine_class = RagEngineFactory.get_rag_engine_by_type(rag_type)
    return rag_engine_class(
        config=RAG_ENGINE_INIT_DATA.get(rag_type),
        chunk_generator=__chunk_generator,
        embedding_model=__embedding_model,
    )



LLM_CLIENTS_INIT_DATA: dict[LLMChoice, BaseModel] = {
    LLMChoice.MISTRAL: MistralInitData(
        api_key=settings.MISTRAL_API_TOKEN,
        model=settings.MISTRAL_MODEL
    ),
    LLMChoice.OLLAMA: OllamaInitModel(
        model=settings.OLLAMA_MODEL,
        base_url=settings.OLLAMA_BASE_URL,
        num_ctx=settings.OLLAMA_NUM_CTX,
        temperature=settings.OLLAMA_TEMPERATURE,
        num_predict=settings.OLLAMA_NUM_PREDICT
    )   
}

def init_llm_client(llm_type: LLMChoice) -> BaseLLMClient:
    print(f"Запуск с LLM: {llm_type.value}")
    llm_client_class = ApiClientFactory.get_client_by_type(llm_type)
    return llm_client_class(LLM_CLIENTS_INIT_DATA.get(llm_type))


@dataclass
class AppContainer:
    llm_client: MistralClient
    engine: RAGEngineBase
    prompt_manager_class: Type[BasePromptManager]

prompt_manager_class = PromptFactory.get_prompt_class_by_type(settings.PROMPT_TYPE)


__llm_client = init_llm_client(settings.LLM_TYPE)

__rag_engine = init_rag_client(settings.RAG_ENGINE_TYPE)

def get_app_container() -> AppContainer:
    return AppContainer(
        llm_client=__llm_client,
        engine=__rag_engine,
        prompt_manager_class=prompt_manager_class
    )