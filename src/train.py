import logging

import hydra
import mlflow
import rootutils
from omegaconf import DictConfig, OmegaConf

from exp.instantiations import (
    instantiate_buffers,
    instantiate_interaction,
    instantiate_models,
    instantiate_trainers,
)
from exp.oc_resolvers import register_custom_resolvers

# Register OmegaConf custom resolvers.
register_custom_resolvers()

# find root directory
rootutils.setup_root(__file__, indicator="pyproject.toml")

logger = logging.getLogger(__name__)


@hydra.main("./configs", "train", version_base="1.3")
def main(cfg: DictConfig) -> None:
    cfg_view = cfg.copy()
    OmegaConf.resolve(cfg_view)
    print(OmegaConf.to_yaml(cfg_view))
    # Convert device and dtype string object to pytorch object.

    shared_cfg = cfg.shared
    shared_cfg.device = f"${{torch.device:{shared_cfg.device}}}"
    shared_cfg.dtype = f"${{torch.dtype:{shared_cfg.dtype}}}"

    mlflow.set_tracking_uri(cfg.paths.mlflow_dir)

    instantiate_interaction(cfg)

    instantiate_models(cfg)

    instantiate_trainers(cfg)

    instantiate_buffers(cfg)


if __name__ == "__main__":
    main()
