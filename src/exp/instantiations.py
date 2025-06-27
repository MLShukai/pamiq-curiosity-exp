import logging
from typing import Any

import hydra
from omegaconf import DictConfig
from pamiq_core import Interaction

logger = logging.getLogger(__name__)


def instantiate_interaction(cfg: DictConfig) -> Interaction[Any, Any]:
    logger.info("Instantiating Interaction...")
    return hydra.utils.instantiate(cfg)
