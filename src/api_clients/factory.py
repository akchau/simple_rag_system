from typing import Type
from src.api_clients.base import BaseLLMClient, LLMChoice
from src.api_clients.mistral_api_client import MistralClient
from src.api_clients.ollama_api_client import OllamaApiClient


class ApiClientFactory:
    
    api_client_classes: list[Type[BaseLLMClient]] = [
        OllamaApiClient,
        MistralClient
    ]

    @classmethod
    def get_client_by_type(cls, target_type: LLMChoice) -> type[BaseLLMClient]:
        for client in cls.api_client_classes:
            if client.client_type == target_type:
                print(f"В системе найден класс для работы с {target_type.value}")
                return client
        raise RuntimeError(f"Не удалось подобрать llm-клиента по типу {target_type}")