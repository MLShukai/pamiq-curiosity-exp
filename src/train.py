import hydra
import rootutils
from omegaconf import DictConfig, OmegaConf

from exp.oc_resolvers import register_custom_resolvers

# Register OmegaConf custom resolvers.
register_custom_resolvers()

# find root directory
rootutils.setup_root(__file__, indicator="pyproject.toml")


@hydra.main("./configs", "train", version_base="1.3")
def main(cfg: DictConfig) -> None:
    cfg_view = cfg.copy()
    OmegaConf.resolve(cfg_view)
    print(OmegaConf.to_yaml(cfg_view))
    # Convert device and dtype string object to pytorch object.

    shared_cfg = cfg.shared
    shared_cfg.device = f"${{torch.device:{shared_cfg.device}}}"
    shared_cfg.dtype = f"${{torch.dtype:{shared_cfg.dtype}}}"


if __name__ == "__main__":
    main()
