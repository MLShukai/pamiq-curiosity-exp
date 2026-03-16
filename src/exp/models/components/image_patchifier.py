from typing import override

import torch
import torch.nn as nn

from exp.utils import size_2d, size_2d_to_int_tuple

from ..utils import init_weights


class ImageIdentityPatchifier(nn.Module):
    """A placeholder patchifier that returns the input image as-is."""

    @override
    def __init__(
        self,
        patch_size: size_2d = 16,
        in_channels: int = 3,
    ) -> None:
        """Initializes the ImageIdentityPatchifier.

        Args:
            patch_size: Pixel size per a patch.
            in_channels: Num of input images channels.
        """
        super().__init__()
        self.patch_size = patch_size
        self.in_channels = in_channels

    @override
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Embed image to patch.

        Args:
            x: Input images(shape: [batch, channels, height, width]).

        Returns:
            patch embeddings (shape: [batch, n_patches, patch_size * patch_size * in_channels]).
        """
        batch_size, channels, height, width = x.shape
        patch_height, patch_width = size_2d_to_int_tuple(self.patch_size)

        if channels != self.in_channels:
            raise ValueError(
                f"Expected input with {self.in_channels} channels, but got {channels} channels."
            )
        if height % patch_height != 0 or width % patch_width != 0:
            raise ValueError(
                f"Input image size ({height}, {width}) is not divisible by patch size "
                f"({patch_height}, {patch_width})."
            )

        n_patches_height = height // patch_height
        n_patches_width = width // patch_width
        n_patches = n_patches_height * n_patches_width

        # Reshape and permute to get patches
        x = x.unfold(2, patch_height, patch_height).unfold(3, patch_width, patch_width)
        x = x.contiguous().view(
            batch_size, channels, n_patches, patch_height, patch_width
        )
        x = x.permute(0, 2, 1, 3, 4).contiguous()
        x = x.view(batch_size, n_patches, -1)  # Flatten each patch

        return x

    @staticmethod
    def compute_num_patches_height_width(
        image_size: size_2d, patch_size: size_2d
    ) -> tuple[int, int]:
        """Compute the number of patches in each dimension for given image and
        patch sizes.

        Args:
            image_size: Size of input image as (height, width) or single int.
            patch_size: Size of each patch as (height, width) or single int.

        Returns:
            Number of patches as (num_patches_height, num_patches_width).

        Raises:
            ValueError: If image_size is smaller than patch_size in any dimension.
        """
        image_size = size_2d_to_int_tuple(image_size)
        patch_size = size_2d_to_int_tuple(patch_size)

        def compute(size: int, patch: int) -> int:
            return int((size - patch) / patch + 1)

        out = (
            compute(image_size[0], patch_size[0]),
            compute(image_size[1], patch_size[1]),
        )
        for i, o in enumerate(out):
            if o <= 0:
                dim_name = "height" if i == 0 else "width"
                raise ValueError(
                    f"Image {dim_name} {image_size[i]} is too small for patch {dim_name} "
                    f"{patch_size[i]}. Resulting number of patches would be {o}."
                )
        return out

    @staticmethod
    def compute_num_patches(image_size: size_2d, patch_size: size_2d) -> int:
        """Compute the total number of patches for given image and patch sizes.

        Args:
            image_size: Size of input image as (height, width) or single int.
            patch_size: Size of each patch as (height, width) or single int.
        Returns:
            Total number of patches.
        """
        height_patches, width_patches = (
            ImageIdentityPatchifier.compute_num_patches_height_width(
                image_size, patch_size
            )
        )
        return height_patches * width_patches


class AddAndRemoveBatchDim(nn.Module):
    @override
    def __init__(self) -> None:
        """Initializes the AddAndRemoveBatchDim module."""
        super().__init__()

    @override
    def __call__(self, encoder: nn.Module, data: torch.Tensor) -> torch.Tensor:
        data = data.unsqueeze(0)  # Add batch dimension
        data = encoder(data)
        data = data.squeeze(0)  # Remove batch dimension
        return data


class ImagePatchifier(nn.Module):
    """Convert input images into patch embeddings."""

    @override
    def __init__(
        self,
        patch_size: size_2d = 16,
        in_channels: int = 3,
        embed_dim: int = 768,
        init_std: float = 0.02,
    ) -> None:
        """Initializes the ImagePatchifier.

        Args:
            patch_size: Pixel size per a patch.
            in_channels: Num of input images channels.
            embed_dim: Num of embed dimensions per a patch
            init_std: Standard deviation for weight initialization.
        """
        super().__init__()
        self.proj = nn.Conv2d(
            in_channels=in_channels,
            out_channels=embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
        )
        init_weights(self.proj, init_std)

    @override
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Embed image to patch.

        Args:
            x: Input images(shape: [batch, channels, height, width]).

        Returns:
            patch embeddings (shape: [batch, n_patches, embed_dim]).
        """
        x = self.proj(x).flatten(-2).transpose(-2, -1)
        return x

    @staticmethod
    def compute_num_patches(
        image_size: size_2d, patch_size: size_2d
    ) -> tuple[int, int]:
        """Compute the number of patches in each dimension for given image and
        patch sizes.

        Args:
            image_size: Size of input image as (height, width) or single int.
            patch_size: Size of each patch as (height, width) or single int.

        Returns:
            Number of patches as (num_patches_height, num_patches_width).

        Raises:
            ValueError: If image_size is smaller than patch_size in any dimension.
        """
        image_size = size_2d_to_int_tuple(image_size)
        patch_size = size_2d_to_int_tuple(patch_size)

        def compute(size: int, patch: int) -> int:
            return int((size - patch) / patch + 1)

        out = (
            compute(image_size[0], patch_size[0]),
            compute(image_size[1], patch_size[1]),
        )
        for i, o in enumerate(out):
            if o <= 0:
                dim_name = "height" if i == 0 else "width"
                raise ValueError(
                    f"Image {dim_name} {image_size[i]} is too small for patch {dim_name} "
                    f"{patch_size[i]}. Resulting number of patches would be {o}."
                )
        return out
