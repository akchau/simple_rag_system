from enum import Enum
from pydantic import BaseModel
import requests

from src.api_clients.base import BaseLLMClient, LLMChoice, measure_time
from src.utils.strip_slash import strip_slash


class OllamaModelsEnum(str, Enum):
    """ Доступные модели для Mistral API SDK """
    QWEN_2_5 = "qwen2.5"


class OllamaInitModel(BaseModel):
    base_url: str
    model: OllamaModelsEnum
    num_ctx: int = 8192
    temperature: float = 0.0
    num_predict: int = 500


class OllamaApiClient(BaseLLMClient[OllamaInitModel]):

    client_type = LLMChoice.OLLAMA


    @measure_time
    def send_request(self, text_request: str) -> str:

        model = self._data.model
        num_ctx = self._data.num_ctx
        temperature = self._data.temperature
        num_predict = self._data.num_predict
        base_url = strip_slash(self._data.base_url)
        url = f"{base_url}/api/generate"
        try:
            r = requests.post(
                url=url,
                json={
                    "model": model, 
                    "prompt": text_request, 
                    "stream": False, 
                    "options": {
                        "num_ctx": num_ctx,
                        "temperature": temperature, 
                        "num_predict": num_predict
                    }
                })
            ans = r.json().get("response", "Ошибка модели")
            return ans
        except Exception as e:
            raise RuntimeError(f"Ollama не отвечает. ({e})")