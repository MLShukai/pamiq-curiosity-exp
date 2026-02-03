from typing import override

from pamiq_core import Agent
from torch import Tensor

from exp.data import BufferName, DataKey
from exp.models import ModelName


class NoCollectAgent(Agent[Tensor, Tensor]):
    """Agent that encodes observations into embeddings using a pre-trained
    encoder model without collecting any data."""

    @override
    def __init__(
        self,
        model_name: str | None = None,
    ) -> None:
        """Initialize the IdentityAgent.

        Args:
            model_name: Name of the encoder model to use
            data_collector_name: Name of the data collector to store observations
        """
        super().__init__()
        self.model_name = model_name

    @override
    def on_inference_models_attached(self) -> None:
        """Set up the encoder model when inference models are attached.

        This method is called automatically by the PAMIQ framework when
        inference models are attached to the agent.
        """
        super().on_inference_models_attached()
        if self.model_name is not None:
            self.encoder = self.get_inference_model(self.model_name)

    @override
    def step(self, observation: Tensor) -> Tensor:
        """Process an observation and return its encoded representation.

        The method also collects the original observation for potential training.

        Args:
            observation: The input observation tensor (e.g., an image or audio)

        Returns:
            The encoded representation of the observation
        """
        return self.encoder(observation) if self.model_name is not None else observation
