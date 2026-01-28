from enum import Enum
import time

from mistralai import Mistral, SDKError


class ModelsEnum(str, Enum):
    """ Доступные модели для Mistral API SDK """
    MEDIUM = "mistral-medium"
    TINY = "mistral-tiny"
    SMALL = "mistral-small"
    LARGE = "mistral-large-latest"


class MistralClient:
    """ Клиент для выполнения запросов к API Mistral """
    def __init__(self, api_key: str, model: str = ModelsEnum.LARGE.value):
        self._client = Mistral(api_key=api_key)
        self._model = model
    
    def send_request(self, text_request: str):
        """ З попытки запрос к Mistral """
        counter = 0
        while counter < 3:
            try:
                response = self._client.chat.complete(
                    model=self._model,
                    messages=[
                        {"role": "user", "content": text_request},
                    ]
                )
                return response.choices[0].message.content
            except SDKError:
                print("Произошла ошибка при запросе. Спросим еще раз через 5 секунд")
                counter += 1
                time.sleep(5)