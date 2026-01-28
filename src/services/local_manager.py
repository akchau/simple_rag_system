from pathlib import Path
from typing import Any, Generator

from unstructured.partition.md import partition_md


class LocalManager:
    
    def __init__(self, dir_path: Path):
        self._dir_path = dir_path
    
    def _load_documents(self) -> Generator[Path, Any, None]:
        if not self._dir_path.exists():
            self._dir_path.mkdir()
            print(f"Папка {self._dir_path} создана. Добавьте туда свои .md заметки!")
            return

        for md_file in self._dir_path.glob("*.md"):
            yield md_file

    def get_documents_data(self) -> list[dict[str, str]]:
        docs: list[dict[str, str]] = []
        for md_file in self._load_documents():
            try:
                elements = partition_md(filename=str(md_file))
                text = "\n".join([str(el) for el in elements])
                if text.strip():
                    docs.append({"text": text, "source": md_file.name})
            except Exception as e:
                print(f"Ошибка при обработке {md_file}: {e}")
        print(f"Загружено {len(docs)} документов.")
        return docs