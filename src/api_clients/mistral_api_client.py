from enum import Enum
import time

from mistralai import Mistral, SDKError
from pydantic import BaseModel

from src.api_clients.base import BaseLLMClient, LLMChoice, measure_time


class ModelsEnum(str, Enum):
    """ Доступные модели для Mistral API SDK """
    MEDIUM = "mistral-medium"
    TINY = "mistral-tiny"
    SMALL = "mistral-small"
    LARGE = "mistral-large-latest"


class MistralInitData(BaseModel):
    api_key: str
    model: ModelsEnum


class MistralClient(BaseLLMClient[MistralInitData]):
    """ Клиент для выполнения запросов к API Mistral """

    client_type = LLMChoice.MISTRAL
    
    @property
    def _client(self) -> Mistral:
        api_key = self._data.api_key
        return Mistral(api_key=api_key)

    @measure_time
    def send_request(self, text_request: str) -> str:
        counter = 0

        model = self._data.model

        while counter < 3:
            try:
                response = self._client.chat.complete(
                    model=model,
                    messages=[
                        {"role": "user", "content": text_request},
                    ]
                )
                return response.choices[0].message.content
            except SDKError:
                print("Произошла ошибка при запросе. Спросим еще раз через 5 секунд")
                counter += 1
                time.sleep(5)
        raise RuntimeError("Не удалось подключиться к Mistral")