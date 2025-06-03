from typing import Final

import torch
from pamiq_core.interaction.modular_env import ModularEnvironment
from pamiq_core.interaction.wrappers import LambdaWrapper
from pamiq_vrchat.actuators import OscAction, OscAxes, OscButtons, SmoothOscActuator
from pamiq_vrchat.sensors import ImageSensor
from torch import Tensor
from torchvision.transforms.v2 import ToImage, ToPureTensor

from . import transforms

OSC_ACTION_CHOICES: Final[tuple[int, ...]] = (3, 3, 3, 2, 2)


class OscTransform:
    """Transforms discrete action indices into OscAction format.

    Converts a tensor of discrete action indices into a structured
    OscAction dictionary that can be used by the OscActuator. Maps
    avatar movement directions and button states based on action
    indices.
    """

    def __init__(
        self,
        *,
        vertical_velocity: float = 1.0,
        horizontal_velocity: float = 1.0,
        look_horizontal_velocity: float = 1.0,
    ) -> None:
        """Initialize the OSC transform.

        Args:
            vertical_velocity: Velocity multiplier for vertical movement (forward/backward).
            horizontal_velocity: Velocity multiplier for horizontal movement (left/right).
            look_horizontal_velocity: Velocity multiplier for horizontal look movement.
        """
        self.vertical_velocity = vertical_velocity
        self.horizontal_velocity = horizontal_velocity
        self.look_horizontal_velocity = look_horizontal_velocity

        # Mapping from discrete actions (0,1,2) to directional values (0, +1, -1)
        self._velocity_map: dict[int, float] = {0: 0.0, 1: 1.0, 2: -1.0}

    def __call__(self, action: Tensor) -> OscAction:
        """Transform discrete action tensor into OscAction format.

        Args:
            action: Tensor of discrete action indices with shape (5,), where:
                - index 0: vertical movement (0=stop, 1=forward, 2=backward)
                - index 1: horizontal movement (0=stop, 1=right, 2=left)
                - index 2: look horizontal movement (0=stop, 1=right, 2=left)
                - index 3: jump button (0=release, 1=press)
                - index 4: run button (0=release, 1=press)

        Returns:
            OscAction dictionary with axes and buttons fields.

        Raises:
            ValueError: If action tensor shape is invalid or contains unsupported values.
        """
        if action.ndim != 1:
            raise ValueError("Action tensor must be 1-dimensional")
        if action.numel() != len(OSC_ACTION_CHOICES):
            raise ValueError(
                f"Action tensor must have {len(OSC_ACTION_CHOICES)} elements"
            )

        action_list: list[int] = action.detach().cpu().long().tolist()

        if (vertical := self._velocity_map.get(action_list[0])) is None:
            raise ValueError(f"Invalid vertical movement action: {action_list[0]}")
        if (horizontal := self._velocity_map.get(action_list[1])) is None:
            raise ValueError(f"Invalid horizontal movement action: {action_list[1]}")
        if (look_horizontal := self._velocity_map.get(action_list[2])) is None:
            raise ValueError(
                f"Invalid look horizontal movement action: {action_list[2]}"
            )

        return OscAction(
            axes={
                OscAxes.Vertical: vertical * self.vertical_velocity,
                OscAxes.Horizontal: horizontal * self.horizontal_velocity,
                OscAxes.LookHorizontal: look_horizontal * self.look_horizontal_velocity,
            },
            buttons={
                OscButtons.Jump: bool(action_list[3]),
                OscButtons.Run: bool(action_list[4]),
            },
        )


def create_env(
    image_size: tuple[int, int] = (144, 144),
    osc_host: str = "127.0.0.1",
    osc_port: int = 9000,
    actuator_delta_time: float = 0.1,
    vertical_velocity: float = 1.0,
    horizontal_velocity: float = 1.0,
    look_horizontal_velocity: float = 0.7,
    device: torch.device | None = None,
    dtype: torch.dtype | None = None,
) -> ModularEnvironment[Tensor, Tensor]:
    """Create a VRChat environment with image sensor and OSC actuator.

    Args:
        image_size: Target size for image resizing as (height, width).
        osc_host: VRChat OSC hostname or IP address for avatar control.
        osc_port: VRChat OSC port number for avatar control.
        actuator_delta_time: Time interval between actuator updates in seconds.
        vertical_velocity: Velocity multiplier for forward/backward movement.
        horizontal_velocity: Velocity multiplier for left/right movement.
        look_horizontal_velocity: Velocity multiplier for horizontal look movement.
        device: Target device for tensor operations. If None, uses default device.
        dtype: Target data type for tensors. If None, uses default dtype.

    Returns:
        A ModularEnvironment instance configured for VRChat interaction.
    """
    return ModularEnvironment(
        sensor=LambdaWrapper(
            transforms.compose(
                ToImage(),
                transforms.ToDevice(device),
                transforms.image.ResizeAndCenterCrop(image_size),
                transforms.ToDtype(dtype),
                transforms.Standardize(),
                ToPureTensor(),
            ),
        ).wrap_sensor(ImageSensor()),
        actuator=LambdaWrapper(
            OscTransform(
                vertical_velocity=vertical_velocity,
                horizontal_velocity=horizontal_velocity,
                look_horizontal_velocity=look_horizontal_velocity,
            )
        ).wrap_actuator(
            SmoothOscActuator(osc_host, osc_port, delta_time=actuator_delta_time)
        ),
    )
