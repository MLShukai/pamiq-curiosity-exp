import logging
from collections.abc import Mapping
from typing import Any

import hydra
from omegaconf import DictConfig
from pamiq_core import DataBuffer, Interaction
from pamiq_core.torch import TorchTrainer, TorchTrainingModel

from exp.data import BufferName
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

    models_dict: dict[str, TorchTrainingModel[Any]] = {
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

        model: TorchTrainingModel[Any] | list[TorchTrainingModel[Any]] = (
            hydra.utils.instantiate(model_cfg)
        )
        if isinstance(model, list):
            logger.info(f"Instantiated {len(model)} models.")
            for i, m in enumerate(model):
                models_dict[str(name) + str(i)] = m
        else:
            models_dict[str(name)] = model
    return models_dict


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
        trainer: TorchTrainer | list[TorchTrainer] = hydra.utils.instantiate(
            trainer_cfg
        )
        if isinstance(trainer, list):
            logger.info(f"Instantiated {len(trainer)} trainers.")
            for i, t in enumerate(trainer):
                trainers_dict[str(name) + str(i)] = t
        else:
            trainers_dict[str(name)] = trainer

    return trainers_dict


def instantiate_buffers(cfg: DictConfig) -> Mapping[str, DataBuffer[Any, Any]]:
    logger.info("Instantiating DataBuffers...")
    from exp.trainers.jepa import JEPATrainer

    buffers_dict: dict[str, DataBuffer[Any, Any]] = {
        BufferName.IMAGE: JEPATrainer.create_buffer(
            batch_size=JEPA_BATCH_SIZE,
            iteration_count=16,
            expected_survival_length=10 * 60 * 60,  # 10fps 1hour.
        )
    }

    for name, buffer_cfg in cfg.buffers.items():
        logger.info(f"Instantiating DataBuffer: '{name}'")
        buffer: DataBuffer[Any, Any] | list[DataBuffer[Any, Any]] = (
            hydra.utils.instantiate(buffer_cfg)
        )
        if isinstance(buffer, list):
            logger.info(f"Instantiated {len(buffer)} buffers.")
            for i, buf in enumerate(buffer):
                buffers_dict[str(name) + str(i)] = buf
        else:
            buffers_dict[str(name)] = buffer
    return buffers_dict
