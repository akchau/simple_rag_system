import unittest
from unittest.mock import Mock, create_autospec

from src.services.local_manger.local_manager import LocalManager
from src.services.retrieval.chunk_generator import ChunkGenerator


TEST_CHUNK_SIZE = 2
OVERLAP = 1


class TestChunkText(unittest.IsolatedAsyncioTestCase):
    
    def setUp(self):
        self.mock_local_manager = Mock(return_value=create_autospec(LocalManager, instance=False))
        self.generator = ChunkGenerator(
            local_manager=self.mock_local_manager,
            chunk_size=TEST_CHUNK_SIZE,
            overlap=OVERLAP
        )

    def test_good_case(self):
        doc_text = "x"*10
        self.mock_local_manager.get_documents_data.return_value = [
            {"source": "file.txt", "text": doc_text}
        ]
        chunks = self.generator.get_chunks()
        self.assertEqual(
            chunks,
            ['xx', 'xx', 'xx', 'xx', 'xx', 'xx', 'xx', 'xx', 'xx', 'x']
        )

    def test_short_text(self):
        doc_text = "x"
        self.mock_local_manager.get_documents_data.return_value = [
            {"source": "file.txt", "text": doc_text}
        ]
        chunks = self.generator.get_chunks()
        self.assertEqual(
            chunks,
            ['x']
        )

    def test_empty_text(self):
        doc_text = ""
        self.mock_local_manager.get_documents_data.return_value = [
            {"source": "file.txt", "text": doc_text}
        ]
        chunks = self.generator.get_chunks()
        self.assertEqual(
            chunks,
            []
        )