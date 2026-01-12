import pickle
from collections import deque
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import override

from pamiq_core.data import DataBuffer

from .chunk_buffer import ChunkBuffer
from .intermittent_buffer import IntermittentBuffer


class DictIntermittentChunkBuffer[T](
    DataBuffer[Mapping[str, T], dict[str, list[T] | list[list[T]]]]
):
    """Implementation of DataBuffer that composites IntermittentBuffer and
    ChunkBuffer."""

    def __init__(
        self,
        intermittent_key_first_add_steps: Mapping[str, int],
        chunk_key_first_add_steps: Mapping[str, int],
        get_interval: int,
        max_size: int,
    ) -> None:
        self._max_size = max_size
        first_store_steps = (
            max(
                max(intermittent_key_first_add_steps.values()),
                max(chunk_key_first_add_steps.values()),
            )
            + get_interval
        )
        self.intermittent_buffers: dict[str, IntermittentBuffer[T]] = {
            key: IntermittentBuffer[T](
                max_size=max_size,
                get_interval=get_interval,
                first_add_steps=intermittent_key_first_add_steps[key],
                first_store_steps=first_store_steps,
            )
            for key in intermittent_key_first_add_steps
        }
        self.chunk_buffers: dict[str, ChunkBuffer[T]] = {
            key: ChunkBuffer[T](
                max_size=max_size,
                get_interval=get_interval,
                first_add_steps=chunk_key_first_add_steps[key],
                first_store_steps=first_store_steps,
            )
            for key in chunk_key_first_add_steps
        }

    @property
    def max_size(self) -> int:
        """Returns the maximum number of data points that can be stored in the
        buffer."""
        return self._max_size

    @override
    def add(self, data: Mapping[str, T]) -> None:
        """Add a new data sample to the buffer.

        The data must contain exactly the keys specified during initialization.
        If the buffer is full, the oldest entry will be removed.

        Args:
            data: Dictionary containing data for each key.

        Raises:
            ValueError: If the data keys don't match the expected keys.
        """
        if set(data.keys()) != set(self.intermittent_buffers.keys()) | set(
            self.chunk_buffers.keys()
        ):
            raise ValueError(
                f"Data keys {set(data.keys())} do not match expected keys "
                f"{set(self.intermittent_buffers.keys()) | set(self.chunk_buffers.keys())}"
            )
        for key, value in data.items():
            if key in self.intermittent_buffers:
                self.intermittent_buffers[key].add(value)
            if key in self.chunk_buffers:
                self.chunk_buffers[key].add(value)

    @override
    def get_data(self) -> dict[str, list[T] | list[list[T]]]:
        """Retrieve all stored data from the buffer.

        Returns:
            Dictionary mapping each key to a list of its stored values.
            The lists maintain the sequential order in which data was added.
        """
        out: dict[str, list[T] | list[list[T]]] = {}
        for key in self.intermittent_buffers:
            out[key] = self.intermittent_buffers[key].get_data()
        for key in self.chunk_buffers:
            out[key] = self.chunk_buffers[key].get_data()
        return out

    @override
    def __len__(self) -> int:
        """Returns the current number of samples in the buffer.

        Returns:
            int: The number of samples currently stored in the buffer.
        """
        lens = []
        for key in self.intermittent_buffers:
            lens.append(len(self.intermittent_buffers[key]))
        for key in self.chunk_buffers:
            lens.append(len(self.chunk_buffers[key]))
        if len(set(lens)) != 1:
            raise ValueError(
                "All buffers must have the same length, but found different lengths."
            )
        return lens[0]

    @override
    def save_state(self, path: Path) -> None:
        """Save the buffer state to the specified path.

        Saves the data queue to a pickle file with .pkl extension.

        Args:
            path: File path where to save the buffer state (without extension)
        """
        for key, buffer in self.intermittent_buffers.items():
            buffer.save_state(path.with_name(f"{key}_intermittent"))
        for key, buffer in self.chunk_buffers.items():
            buffer.save_state(path.with_name(f"{key}_chunk"))

    @override
    def load_state(self, path: Path) -> None:
        """Load the buffer state from the specified path.

        Loads data queue from pickle file with .pkl extension.

        Args:
            path: File path from where to load the buffer state (without extension)
        """
        for key in self.intermittent_buffers:
            self.intermittent_buffers[key].load_state(
                path.with_name(f"{key}_intermittent")
            )
        for key in self.chunk_buffers:
            self.chunk_buffers[key].load_state(path.with_name(f"{key}_chunk"))
