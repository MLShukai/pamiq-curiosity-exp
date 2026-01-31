import pytest
import torch

from exp.models.components.image_patchifier import (
    ImageIdentityPatchifier,
    ImagePatchifier,
)
from exp.utils import size_2d, size_2d_to_int_tuple


class TestImageIdentityPatchifier:
    @pytest.mark.parametrize("batch_size", [1, 4])
    @pytest.mark.parametrize("img_size", [224, 512])
    @pytest.mark.parametrize("patch_size", [16])
    def test_forward(self, batch_size, img_size, patch_size):
        layer = ImageIdentityPatchifier(patch_size, 3)

        image = torch.randn(batch_size, 3, img_size, img_size)
        out = layer(image)
        assert isinstance(out, torch.Tensor)
        assert out.shape == (
            batch_size,
            (img_size // patch_size) ** 2,
            patch_size * patch_size * 3,
        )

    @pytest.mark.parametrize(
        "image_size,patch_size,batch_size",
        [
            (64, 8, 1),  # Small image
            (224, 16, 2),  # Standard case
            (256, 32, 3),  # Larger patches
            ((128, 256), (16, 32), 1),  # Rectangular image and patches
            ((96, 96), (12, 12), 4),  # Square with different params
        ],
    )
    def test_compute_num_patches_consistency_with_forward(
        self, image_size, patch_size, batch_size: int
    ):
        """Test that compute_num_patches matches actual forward pass output."""
        # Compute expected number of patches
        expected_patches_2d = ImageIdentityPatchifier.compute_num_patches(
            image_size, patch_size
        )
        expected_total_patches = expected_patches_2d[0] * expected_patches_2d[1]

        # Create patchifier and input
        patchifier = ImageIdentityPatchifier(patch_size=patch_size)

        if isinstance(image_size, tuple):
            height, width = image_size
        else:
            height = width = image_size

        input_image = torch.randn(batch_size, 3, height, width)
        output = patchifier(input_image)

        patch_height, patch_width = size_2d_to_int_tuple(patch_size)

        # Check consistency
        assert output.shape[1] == expected_total_patches
        assert output.shape == (
            batch_size,
            expected_total_patches,
            patch_height * patch_width * 3,
        )


class TestImagePatchifier:
    @pytest.mark.parametrize("batch_size", [1, 4])
    @pytest.mark.parametrize("img_size", [224, 512])
    @pytest.mark.parametrize("patch_size", [16])
    @pytest.mark.parametrize("embed_dim", [768])
    def test_forward(self, batch_size, img_size, patch_size, embed_dim):
        layer = ImagePatchifier(patch_size, 3, embed_dim)

        image = torch.randn(batch_size, 3, img_size, img_size)
        out = layer(image)
        assert isinstance(out, torch.Tensor)
        assert out.shape == (batch_size, (img_size // patch_size) ** 2, embed_dim)

    @pytest.mark.parametrize(
        "image_size,patch_size,embed_dim,batch_size",
        [
            (64, 8, 768, 1),  # Small image
            (224, 16, 512, 2),  # Standard case
            (256, 32, 384, 3),  # Larger patches
            ((128, 256), (16, 32), 768, 1),  # Rectangular image and patches
            ((96, 96), (12, 12), 256, 4),  # Square with different params
        ],
    )
    def test_compute_num_patches_consistency_with_forward(
        self, image_size, patch_size, embed_dim: int, batch_size: int
    ):
        """Test that compute_num_patches matches actual forward pass output."""
        # Compute expected number of patches
        expected_patches_2d = ImagePatchifier.compute_num_patches(
            image_size, patch_size
        )
        expected_total_patches = expected_patches_2d[0] * expected_patches_2d[1]

        # Create patchifier and input
        patchifier = ImagePatchifier(patch_size=patch_size, embed_dim=embed_dim)

        if isinstance(image_size, tuple):
            height, width = image_size
        else:
            height = width = image_size

        input_image = torch.randn(batch_size, 3, height, width)
        output = patchifier(input_image)

        # Check consistency
        assert output.shape[1] == expected_total_patches
        assert output.shape == (batch_size, expected_total_patches, embed_dim)
