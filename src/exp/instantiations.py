import logging
from collections.abc import Mapping
from typing import Any

import hydra
from omegaconf import DictConfig
from pamiq_core import DataBuffer, Interaction
from pamiq_core.torch import TorchTrainer, TorchTrainingModel

from exp.data import BufferName, DataKey
from exp.models import ModelName

logger = logging.getLogger(__name__)

# Fixed parameters
PATCH_SIZE = 12
JEPA_BATCH_SIZE = 32


def instantiate_interaction(cfg: DictConfig) -> Interaction[Any, Any]:
    logger.info("Instantiating Interaction...")
    return hydra.utils.instantiate(cfg.interaction)


def instantiate_models(cfg: DictConfig) -> dict[str, TorchTrainingModel[Any]]:
    logger.info("Instantiating Models...")
    device, dtype = cfg.shared.device, cfg.shared.dtype

    from exp.models.jepa import create_image_jepa

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

    models = {
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
    }

    # Instantiating
    for name, model_cfg in cfg.models.items():
        logger.info(f"Instantiating model: '{name}' ...")

        models[ModelName(name)] = hydra.utils.instantiate(model_cfg)

    return models  # pyright: ignore


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

    trainers_dict: dict[str, TorchTrainer] = {"jepa": jepa}
    for name, trainer_cfg in cfg.trainers.items():
        logger.info(f"Instantiating Trainer: '{name}' ...")
        trainers_dict[name] = hydra.utils.instantiate(trainer_cfg)

    return trainers_dict


def instantiate_buffers(cfg: DictConfig) -> Mapping[str, DataBuffer[Any]]:
    logger.info("Instantiating DataBuffers...")
    from exp.trainers.jepa import JEPATrainer

    buffers_dict = {
        BufferName.IMAGE: JEPATrainer.create_buffer(
            batch_size=JEPA_BATCH_SIZE,
            iteration_count=16,
            expected_survival_length=10 * 60 * 60,  # 10fps 1hour.
        )
    }

    for name, buffer_cfg in cfg.buffers.items():
        logger.info(f"Instantiating DataBuffer: '{name}'")
        buffers_dict[BufferName(name)] = hydra.utils.instantiate(buffer_cfg)
    return buffers_dict  # pyright: ignore
