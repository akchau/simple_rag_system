from enum import Enum
from pydantic import BaseModel
import requests

from src.api_clients.base import BaseLLMClient, LLMChoice, measure_time


class OllamaModelsEnum(str, Enum):
    """ Доступные модели для Mistral API SDK """
    QWEN_2_5 = "qwen2.5"


class OllamaInitModel(BaseModel):
    model: OllamaModelsEnum
    url: str


class OllamaApiClient(BaseLLMClient[OllamaInitModel]):

    client_type = LLMChoice.OLLAMA

    @measure_time
    def send_request(self, text_request: str) -> str:

        model = self._data.model
        url = self._data.url
        try:
            r = requests.post(
                url=url,
                json={
                    "model": model, 
                    "prompt": text_request, 
                    "stream": False, 
                    "options": {
                        "num_ctx": 8192,     # Оптимизация: уменьшили с 16k для скорости
                        "temperature": 0.0, 
                        "num_predict": 500   # Ограничение длины ответа
                    }
                })
            ans = r.json().get("response", "Ошибка модели")
            return ans
        except Exception as e:
            raise RuntimeError(f"Ollama не отвечает. ({e})")