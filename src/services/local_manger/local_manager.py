from pathlib import Path
from typing import Any, Generator

from unstructured.documents.elements import Element

from unstructured.partition.md import partition_md
from unstructured.partition.pdf import partition_pdf
from unstructured.partition.docx import partition_docx
from unstructured.partition.text import partition_text

from src.services.local_manger.base import DocumentParser


class MDParser(DocumentParser):
    def parse(self, file: Path) -> list[Element]:
        return partition_md(filename=str(file))



class PDFParser(DocumentParser):
    def parse(self, file: Path) -> list[Element]:
        return partition_pdf(filename=str(file))



class DOCXParser(DocumentParser):
    def parse(self, file: Path) -> list[Element]:
        return partition_docx(filename=str(file))



class TextParser(DocumentParser):
    def parse(self, file: Path) -> list[Element]:
        return partition_text(filename=str(file))


class DocumentParserFactory:
    _parsers: dict[str, DocumentParser] = {
        ".md": MDParser(),
        ".pdf": PDFParser(),
        ".docx": DOCXParser(),
        ".txt": TextParser(),
    }

    @classmethod
    def get_parser(cls, extension: str) -> DocumentParser:
        ext = extension.lower()
        if ext not in cls._parsers:
            raise ValueError(f"Неподдерживаемый формат: {ext}")
        return cls._parsers[ext]


class LocalManager:

    SUPPORTED_EXTENSIONS = {".md", ".pdf", ".docx", ".txt"}
    
    def __init__(self, dir_path: Path):
        self._dir_path = dir_path
    
    def _load_documents(self) -> Generator[Path, Any, None]:
        if not self._dir_path.exists():
            self._dir_path.mkdir()
            print(f"Папка {self._dir_path} создана. Добавьте туда документы!")
            return

        for file in self._dir_path.iterdir():
            if file.suffix.lower() in self.SUPPORTED_EXTENSIONS:
                yield file

    def _partition_file(self, file: Path) -> list[Element]:
        parser = DocumentParserFactory.get_parser(file.suffix)
        return parser.parse(file)

    def get_documents_data(self) -> list[dict[str, str]]:
        docs: list[dict[str, str]] = []

        for file in self._load_documents():
            try:
                elements = self._partition_file(file)
                text = "\n".join(str(el) for el in elements)

                if text.strip():
                    docs.append({
                        "text": text,
                        "source": file.name,
                    })

            except Exception as e:
                print(f"Ошибка при обработке {file}: {e}")

        print(f"Загружено {len(docs)} документов.")
        return docs