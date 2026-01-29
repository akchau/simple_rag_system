from abc import ABC, abstractmethod
from pathlib import Path


class DocumentParser(ABC):
    
    @abstractmethod
    def parse(self, file: Path) -> list[Element]:
        pass