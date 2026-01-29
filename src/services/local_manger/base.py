from abc import ABC, abstractmethod
from pathlib import Path

from unstructured.documents.elements import Element

class DocumentParser(ABC):
    
    @abstractmethod
    def parse(self, file: Path) -> list[Element]:
        pass