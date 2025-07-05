from pathlib import Path

import pytest

from exp.data.chunk_buffer import ChunkBuffer


class TestChunkBuffer:
    """Test suite for IntermittentBuffer."""

    @pytest.fixture
    def buffer(self) -> ChunkBuffer[int]:
        """Fixture providing a standard ChunkBuffer for tests."""
        return ChunkBuffer[int](
            max_size=50, get_interval=5, first_add_steps=2, first_store_steps=10
        )

    def test_init(self):
        """Test ChunkBuffer initialization with various parameters."""
        # Test with standard parameters
        max_size = 50
        buffer = ChunkBuffer[int](
            max_size, get_interval=5, first_add_steps=2, first_store_steps=10
        )

        assert buffer.max_size == max_size

    def test_add_and_get_data(self, buffer: ChunkBuffer[int]):
        """Test adding data to the buffer and retrieving it."""
        for i in range(0, 10):
            buffer.add(i)

        data = buffer.get_data()
        assert data == []

        for i in range(10, 20):
            buffer.add(i)

        data = buffer.get_data()
        assert data == [[2, 3, 4, 5, 6], [7, 8, 9, 10, 11]]

    def test_max_size_constraint(self):
        """Test the buffer respects its maximum size constraint."""
        max_size = 2
        buffer = ChunkBuffer[int](
            max_size=2, get_interval=5, first_add_steps=2, first_store_steps=10
        )

        # Add more items than the max size
        for i in range(25):
            buffer.add(i)

        # Check only the most recent max_size items are kept
        data = buffer.get_data()
        assert data == [[7, 8, 9, 10, 11], [12, 13, 14, 15, 16]]
        assert len(data) == max_size

    def test_get_data_returns_copy(self, buffer: ChunkBuffer[int]):
        """Test that get_data returns a copy that doesn't affect the internal
        state."""
        for i in range(0, 20):
            buffer.add(i)

        # Get data and modify it
        data = buffer.get_data()
        data.append([2])

        # Verify internal state is unchanged
        new_data = buffer.get_data()
        assert new_data == [[2, 3, 4, 5, 6], [7, 8, 9, 10, 11]]
        assert len(new_data) == 2

    def test_save_and_load_state(self, buffer: ChunkBuffer[int], tmp_path: Path):
        """Test saving and loading the buffer state."""
        # Add some data to the buffer
        for i in range(0, 100):
            buffer.add(i)

        # Save state
        save_path = tmp_path / "test_buffer"
        buffer.save_state(save_path)

        # Verify file was created with .pkl extension
        assert save_path.with_suffix(".pkl").is_file()

        # Create a new buffer and load state
        new_buffer = ChunkBuffer[int](
            max_size=buffer.max_size,
            get_interval=buffer._get_interval,
            first_add_steps=buffer._first_add_steps,
            first_store_steps=buffer._first_store_steps,
        )
        new_buffer.load_state(save_path)

        # Check that loaded data matches original
        original_data = buffer.get_data()
        loaded_data = new_buffer.get_data()

        assert loaded_data == original_data

    def test_len(self, buffer: ChunkBuffer[int]):
        """Test the __len__ method returns the correct buffer size."""
        for i in range(20):
            buffer.add(i)
        assert len(buffer) == 2

        for i in range(5):
            buffer.add(i)
        assert len(buffer) == 3

        for i in range(5):
            buffer.add(i)
        assert len(buffer) == 4
