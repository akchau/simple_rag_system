import unittest
from unittest.mock import create_autospec

from src.services.local_manger.local_manager import LocalManager
from src.services.retrieval.chunk_generator import ChunkGenerator


TEST_CHUNK_SIZE = 2
OVERLAP = 1


class TestChunkGenerator(unittest.TestCase):

    def setUp(self):
        self.mock_local_manager = create_autospec(LocalManager, instance=True)

        self.generator = ChunkGenerator(
            local_manager=self.mock_local_manager,
            chunk_size=TEST_CHUNK_SIZE,
            overlap=OVERLAP
        )

    def test_good_case(self):
        doc_text = "x" * 10
        self.mock_local_manager.get_documents_data.return_value = [
            {"source": "file.txt", "text": doc_text}
        ]

        chunks = self.generator.get_chunks()

        expected = [
            {"text": "xx", "source": "file.txt"},
            {"text": "xx", "source": "file.txt"},
            {"text": "xx", "source": "file.txt"},
            {"text": "xx", "source": "file.txt"},
            {"text": "xx", "source": "file.txt"},
            {"text": "xx", "source": "file.txt"},
            {"text": "xx", "source": "file.txt"},
            {"text": "xx", "source": "file.txt"},
            {"text": "xx", "source": "file.txt"},
            {"text": "x",  "source": "file.txt"},
        ]

        self.assertEqual(chunks, expected)

    def test_short_text(self):
        doc_text = "x"
        self.mock_local_manager.get_documents_data.return_value = [
            {"source": "file.txt", "text": doc_text}
        ]

        chunks = self.generator.get_chunks()

        self.assertEqual(
            chunks,
            [{"text": "x", "source": "file.txt"}]
        )

    def test_empty_text(self):
        doc_text = ""
        self.mock_local_manager.get_documents_data.return_value = [
            {"source": "file.txt", "text": doc_text}
        ]

        chunks = self.generator.get_chunks()

        self.assertEqual(chunks, [])