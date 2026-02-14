from typing import override

import torch
import torch.nn as nn


class CNNEncoder(nn.Module):
    @override
    def __init__(
        self, input_channels: int, input_height: int, input_width: int, output_dim: int
    ):
        """Initialize a simple CNN encoder.

        Args:
            input_channels: Number of input channels (e.g., 3 for RGB images).
            input_height: Height of the input images.
            input_width: Width of the input images.
            output_dim: Dimension of the output feature vector.
        """
        super().__init__()
        conv_output_height = input_height
        conv_output_width = input_width

        self.norm_in = nn.InstanceNorm2d(3)

        self.conv1 = nn.Conv2d(input_channels, 4, kernel_size=5, padding=2)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        conv_output_height = conv_output_height // 2
        conv_output_width = conv_output_width // 2

        self.conv2 = nn.Conv2d(4, 16, kernel_size=5, padding=2)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        conv_output_height = conv_output_height // 2
        conv_output_width = conv_output_width // 2

        self.conv3 = nn.Conv2d(16, 64, kernel_size=5, padding=2)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        conv_output_height = conv_output_height // 2
        conv_output_width = conv_output_width // 2

        self.flatten = nn.Flatten()

        # Calculate the size of the feature map after convolutions
        conv_output_size = 64 * conv_output_height * conv_output_width
        self.fc = nn.Linear(conv_output_size, output_dim)
        self.norm_out = nn.LayerNorm(output_dim)

        self.act = nn.SiLU()

    @override
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the CNN encoder.

        Args:
            x: Input tensor of shape (batch_size, input_channels, input_height, input_width).

        Returns:
            Encoded feature tensor of shape (batch_size, output_dim).
        """
        no_batch = x.ndim == 3
        if no_batch:
            x = x.unsqueeze(0)  # Add batch dimension if missing
        x = self.norm_in(x)
        x = self.pool1(self.act(self.conv1(x)))
        x = self.pool2(self.act(self.conv2(x)))
        x = self.pool3(self.act(self.conv3(x)))
        x = self.flatten(x)
        x = self.fc(x)
        x = self.norm_out(x)
        if no_batch:
            x = x.squeeze(0)
        return x
