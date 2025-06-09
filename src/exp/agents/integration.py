from typing import override

from pamiq_core import Agent
from torch import Tensor

from .unimodal_encoding import UnimodalEncodingAgent


class IntegratedCuriosityFramework(Agent[Tensor, Tensor]):
    """Integrated framework combining unimodal encoding agent and curiosity
    agent."""

    def __init__(
        self,
        curiosity: Agent[Tensor, Tensor],
        unimodal_encoding: UnimodalEncodingAgent = UnimodalEncodingAgent(),
    ) -> None:
        """Initialize the integrated curiosity framework.

        Args:
            curiosity: Agent that selects actions based on intrinsic motivation
            unimodal_encoding: Unimodal encoding agents to encode observation.
        """
        super().__init__(
            agents={"curiosity": curiosity, "unimodal_encoding": unimodal_encoding}
        )

        self.curiosity = curiosity
        self.encoding = unimodal_encoding

    @override
    def step(self, observation: Tensor) -> Tensor:
        """Processltimodal observation and produce actions.

        Args:
            observation: Tensor to encode.

        Returns:
            Action tensor selected by the curiosity agent
        """
        encoded = self.encoding.step(observation)
        action = self.curiosity.step(encoded)

        return action
