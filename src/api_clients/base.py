from abc import ABC, abstractmethod
from enum import Enum
from functools import wraps
import time
from typing import Generic, TypeVar

from pydantic import BaseModel


class LLMChoice(str, Enum):
    MISTRAL = "mistral"
    OLLAMA = "ollama"


def measure_time(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        try:
            return func(*args, **kwargs)
        finally:
            end_time = time.perf_counter()
            print(f"Время выполнения запроса: {end_time - start_time:.2f} сек")
    return wrapper


InitDataTypeVar = TypeVar("InitDataTypeVar", bound=BaseModel)


class BaseLLMClient(ABC, Generic[InitDataTypeVar]):
    
    client_type: LLMChoice = None

    def __init__(self, init_data: InitDataTypeVar):
        self._data = init_data

    @abstractmethod
    def send_request(self, text_request: str) -> str:
        ...