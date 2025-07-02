import logging

import hydra
import mlflow
import rootutils
from omegaconf import DictConfig, OmegaConf
from pamiq_core import LaunchConfig, launch

from exp.instantiations import (
    instantiate_buffers,
    instantiate_interaction,
    instantiate_models,
    instantiate_trainers,
)
from exp.mlflow import flatten_config
from exp.oc_resolvers import register_custom_resolvers

# Register OmegaConf custom resolvers.
register_custom_resolvers()

# find root directory
rootutils.setup_root(__file__, indicator="pyproject.toml")

logger = logging.getLogger(__name__)

logging.captureWarnings(True)


@hydra.main("./configs", "train", version_base="1.3")
def main(cfg: DictConfig) -> None:
    cfg_view = cfg.copy()
    OmegaConf.resolve(cfg_view)
    logger.info(f"Loaded configuration:\n{OmegaConf.to_yaml(cfg_view)}")

    # Convert device and dtype string object to pytorch object.
    shared_cfg = cfg.shared
    shared_cfg.device = f"${{torch.device:{shared_cfg.device}}}"
    shared_cfg.dtype = f"${{torch.dtype:{shared_cfg.dtype}}}"

    mlflow.set_tracking_uri(cfg.paths.mlflow_dir)

    with mlflow.start_run(tags=cfg.tags, log_system_metrics=True):
        log_config(cfg_view)

        launch(
            interaction=instantiate_interaction(cfg),
            models=instantiate_models(cfg),
            data=instantiate_buffers(cfg),
            trainers=instantiate_trainers(cfg),
            config=LaunchConfig(
                **cfg.launch,
            ),
        )


def log_config(cfg: DictConfig) -> None:
    mlflow.log_text(OmegaConf.to_yaml(cfg), "config.yaml")

    log_targets = ["interaction", "models", "trainers", "buffers"]
    mlflow.log_params(flatten_config({key: cfg[key] for key in log_targets}))


if __name__ == "__main__":
    main()
