from abc import ABC, abstractmethod
from typing import Generic, TypeVar

from src.init import AppContainer


DataTypeVar = TypeVar("DataTypeVar")


class BaseUseCase(ABC, Generic[DataTypeVar]):
    
    def __init__(self, app: AppContainer, data: DataTypeVar):
        self._app = app
        self._data = data

    @abstractmethod
    def execute(self):
        ...