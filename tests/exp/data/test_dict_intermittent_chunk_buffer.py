from pathlib import Path

import pytest

from exp.data.dict_intermittent_chunk_buffer import DictIntermittentChunkBuffer


class TestDictIntermittentChunkBuffer:
    """Test suite for DictIntermittentChunkBuffer class."""

    @pytest.fixture
    def buffer(self) -> DictIntermittentChunkBuffer[int]:
        """Fixture providing a standard IntermittentBuffer for tests."""
        return DictIntermittentChunkBuffer[int](
            intermittent_key_first_add_steps={"key1": 2, "key2": 3},
            chunk_key_first_add_steps={"key3": 4, "key4": 5},
            get_interval=6,
            max_size=50,
        )

    def test_init(self):
        """Test DictIntermittentChunkBuffer initialization with various
        parameters."""
        # Test with standard parameters
        max_size = 50
        buffer = DictIntermittentChunkBuffer[int](
            intermittent_key_first_add_steps={"key1": 2, "key2": 3},
            chunk_key_first_add_steps={"key3": 4, "key4": 5},
            get_interval=5,
            max_size=max_size,
        )

        assert buffer.max_size == max_size
        assert len(buffer.intermittent_buffers) == 2
        assert len(buffer.chunk_buffers) == 2

    def test_length(self, buffer: DictIntermittentChunkBuffer[int]):
        """Test the length of the buffer."""
        assert len(buffer) == 0
        for i in range(500):
            buffer.add({"key1": i, "key2": i, "key3": i, "key4": i})
            _ = len(buffer)

    def test_keys(self, buffer: DictIntermittentChunkBuffer[int]):
        """Test the keys of the buffer."""
        for i in range(500):
            buffer.add({"key1": i, "key2": i, "key3": i, "key4": i})

        data = buffer.get_data()
        assert set(data.keys()) == {"key1", "key2", "key3", "key4"}
