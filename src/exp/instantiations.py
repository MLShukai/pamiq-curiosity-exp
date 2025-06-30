import logging
from typing import Any

import hydra
from omegaconf import DictConfig, open_dict
from pamiq_core import Interaction
from pamiq_core.torch import TorchTrainer, TorchTrainingModel

from exp.models import ModelName

logger = logging.getLogger(__name__)

# Fixed parameters
PATCH_SIZE = 12
JEPA_BATCH_SIZE = 32


def instantiate_interaction(cfg: DictConfig) -> Interaction[Any, Any]:
    logger.info("Instantiating Interaction...")
    return hydra.utils.instantiate(cfg.interaction)


def instantiate_models(cfg: DictConfig) -> dict[ModelName, TorchTrainingModel[Any]]:
    logger.info("Instantiating Models...")
    model_cfg = cfg.models
    device, dtype = cfg.shared.device, cfg.shared.dtype

    from exp.envs.vrchat import OSC_ACTION_CHOICES
    from exp.models.jepa import create_image_jepa
    from exp.models.utils import ActionInfo, ObsInfo

    logger.info("Instantiating JEPA models...")
    context_encoder, target_encoder, predictor, avg_pool_infer = create_image_jepa(
        image_size=(cfg.shared.image.height, cfg.shared.image.width),
        patch_size=PATCH_SIZE,
        in_channels=(cfg.shared.image.channels),
        hidden_dim=432,
        embed_dim=128,
        depth=6,
        num_heads=3,
        output_downsample=3,
    )

    # Setup config
    obs_info = ObsInfo(
        dim=target_encoder.embed_dim,
        dim_hidden=1024,
        num_tokens=avg_pool_infer.output_patch_count,
    )
    action_info = ActionInfo(list(OSC_ACTION_CHOICES), dim=8)
    with open_dict(model_cfg) as opened_cfg:
        piv_cfg = opened_cfg[ModelName.POLICY_VALUE]
        piv_cfg.obs_info = obs_info
        piv_cfg.action_choices = OSC_ACTION_CHOICES

        fd_cfg = opened_cfg[ModelName.FORWARD_DYNAMICS]
        fd_cfg.obs_info = obs_info
        fd_cfg.action_info = action_info

    # Instantiate Policy Value model.
    logger.info("Instantiating Policy...")
    policy_value = hydra.utils.instantiate(piv_cfg)

    # Instantiate Forward Dynamics model.
    logger.info("Instantiating Forward Dynamics...")
    forward_dynamics = hydra.utils.instantiate(fd_cfg)

    return {
        ModelName.IMAGE_JEPA_CONTEXT_ENCODER: TorchTrainingModel(
            context_encoder,
            has_inference_model=False,
            device=device,
            dtype=dtype,
        ),
        ModelName.IMAGE_JEPA_TARGET_ENCODER: TorchTrainingModel(
            target_encoder,
            has_inference_model=True,
            device=device,
            dtype=dtype,
            inference_procedure=avg_pool_infer,
        ),
        ModelName.IMAGE_JEPA_PREDICTOR: TorchTrainingModel(
            predictor,
            has_inference_model=False,
            device=device,
            dtype=dtype,
        ),
        ModelName.FORWARD_DYNAMICS: TorchTrainingModel(
            forward_dynamics,
            has_inference_model=True,
            device=device,
            dtype=dtype,
        ),
        ModelName.POLICY_VALUE: TorchTrainingModel(
            policy_value, has_inference_model=True, device=device, dtype=dtype
        ),
    }


def instantiate_trainers(cfg: DictConfig) -> dict[str, TorchTrainer]:
    logger.info("Instantiating Trainers...")
    from functools import partial

    from torch.optim import AdamW

    # JEPA Trainer
    logger.info("Instantiating JEPA Trainer...")
    from exp.models.components.image_patchifier import ImagePatchifier
    from exp.trainers.jepa import JEPATrainer, MultiBlockMaskCollator2d

    jepa = JEPATrainer(
        partial(AdamW, lr=1e-4),
        collate_fn=MultiBlockMaskCollator2d(
            num_patches=ImagePatchifier.compute_num_patches(
                image_size=(
                    cfg.shared.image.height,
                    cfg.shared.image.width,
                ),
                patch_size=PATCH_SIZE,
            ),
            mask_scale=(0.025, 0.125),  # 2.5%-12.5% of patches masked per mask.
            n_masks=4,
            min_keep=7,
        ),
        batch_size=JEPA_BATCH_SIZE,
        min_new_data_count=128,
    )

    trainer_cfg = cfg.trainers

    logger.info("Instantiating Forward Dynamics Trainer...")
    forward_dynamics = hydra.utils.instantiate(trainer_cfg.forward_dynamics)

    logger.info("Instantiating Policy Trainer...")
    policy_value = hydra.utils.instantiate(trainer_cfg.policy)

    return {"jepa": jepa, "forward_dynamics": forward_dynamics, "policy": policy_value}
