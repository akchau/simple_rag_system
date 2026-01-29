import unittest
from unittest.mock import Mock, create_autospec

from src.services.local_manger.local_manager import LocalManager
from src.services.retrieval.chunk_generator import ChunkGenerator


TEST_CHUNK_SIZE = 11
OVERLAP = 15


class TestChunkText(unittest.IsolatedAsyncioTestCase):
    
    def setUp(self):
        self.mock_local_manager = Mock(return_value=create_autospec(LocalManager, instance=False))
        self.generator = ChunkGenerator(
            local_manager=self.mock_local_manager,
            chunk_size=TEST_CHUNK_SIZE,
            overlap=OVERLAP
        )

    def test_good_case(self):
        self.mock_local_manager.get_documents_data.return_value = [
            {"source": "file.txt", "text": "x"*100}
        ]
        self.generator.get_chunks()