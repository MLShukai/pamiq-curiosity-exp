import pickle
from collections import deque
from pathlib import Path
from typing import override

from pamiq_core.data import DataBuffer


class IntermittentBuffer[T](DataBuffer[T, list[T]]):
    """Implementation of DataBuffer that maintains data in intermittent order.

    This buffer stores collected data points in an ordered queue,
    preserving the insertion order with a maximum size limit.
    """

    @override
    def __init__(
        self,
        max_size: int,
        get_interval: int,
        first_add_steps: int,
        first_store_steps: int,
    ):
        """Initialize a new IntermittentBuffer.

        Args:
            max_size: Maximum number of data points to store.
            get_interval: Interval at which data is retrieved.
            first_get_steps: Number of steps to retrieve data initially.
            first_store_steps: Number of steps to store data initially.
        """
        super().__init__(max_size)

        assert first_add_steps < first_store_steps
        self._get_interval = get_interval
        self._first_add_steps = first_add_steps
        self._first_store_steps = first_store_steps
        self._temporary_queue: deque[T] = deque(
            maxlen=(first_store_steps - first_add_steps) // get_interval + 1
        )
        self._queue: deque[T] = deque(maxlen=max_size)
        self._counter = 0
        self._is_wakeup = True
        self._current_size = 0
        self._max_size = max_size

    @property
    def max_size(self) -> int:
        """Returns the maximum number of data points that can be stored in the
        buffer."""
        return self._max_size

    @override
    def add(self, data: T) -> None:
        """Add a new data sample to the buffer.

        Args:
            data: Data element to add to the buffer.
        """
        if self._is_wakeup:
            if (self._counter - self._first_add_steps) % self._get_interval == 0:
                self._temporary_queue.append(data)

            if self._counter == self._first_store_steps:
                self._counter = 0
                self._is_wakeup = False
        else:
            self._counter = self._counter % self._get_interval
            if (
                self._counter + self._first_store_steps - self._first_add_steps
            ) % self._get_interval == 0:
                self._temporary_queue.append(data)
        if self._counter == 0 and not self._is_wakeup:
            self._queue.append(self._temporary_queue.popleft())
            if self._current_size < self._max_size:
                self._current_size += 1
        self._counter += 1

    @override
    def get_data(self) -> list[T]:
        """Retrieve all stored data from the buffer.

        Returns:
            List of all stored data elements preserving the original insertion order.
        """
        return list(self._queue)

    @override
    def __len__(self) -> int:
        """Returns the current number of samples in the buffer.

        Returns:
            int: The number of samples currently stored in the buffer.
        """
        return self._current_size

    @override
    def save_state(self, path: Path) -> None:
        """Save the buffer state to the specified path.

        Saves the data queue to a pickle file with .pkl extension.

        Args:
            path: File path where to save the buffer state (without extension)
        """
        with open(path.with_suffix(".pkl"), "wb") as f:
            pickle.dump(
                {
                    "queue": self._queue,
                    "temporary_queue": self._temporary_queue,
                    "counter": self._counter,
                    "is_wakeup": self._is_wakeup,
                },
                f,
            )

    @override
    def load_state(self, path: Path) -> None:
        """Load the buffer state from the specified path.

        Loads data queue from pickle file with .pkl extension.

        Args:
            path: File path from where to load the buffer state (without extension)
        """
        with open(path.with_suffix(".pkl"), "rb") as f:
            state = pickle.load(f)
            self._queue = deque(state["queue"], maxlen=self._max_size)
            self._temporary_queue = deque(
                state["temporary_queue"],
                maxlen=(self._first_store_steps - self._first_add_steps)
                // self._get_interval
                + 1,
            )
            self._counter = state["counter"]
            self._is_wakeup = state["is_wakeup"]
            self._current_size = len(self._queue)
