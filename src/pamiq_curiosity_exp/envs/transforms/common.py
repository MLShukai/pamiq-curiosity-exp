import torch


class Standardize:
    """Standardize input by removing mean and dividing by standard deviation.

    This module performs standardization (zero mean, unit variance) on
    the input tensor.
    """

    def __init__(self, eps: float = 1e-8) -> None:
        """Initialize the Standardize transform.

        Args:
            eps: Small value to add to standard deviation to avoid division by zero.
        """
        super().__init__()
        self.eps = eps

    def __call__(self, input: torch.Tensor) -> torch.Tensor:
        """Apply standardization to the input tensor.

        Args:
            input: Input tensor to standardize.

        Returns:
            Standardized tensor with zero mean and unit variance.
        """
        if input.numel() <= 1:  # Can not compute std
            return torch.zeros_like(input)

        return (input - input.mean()) / (input.std() + self.eps)


class ToDtype:
    """Convert tensor to specified data type.

    This transform converts input tensors to the specified dtype. If no
    dtype is provided, it uses PyTorch's default dtype.
    """

    def __init__(self, dtype: torch.dtype | None = None) -> None:
        """Initialize the ToDtype transform.

        Args:
            dtype: Target data type. If None, uses PyTorch's default dtype.
        """
        if dtype is None:
            dtype = torch.get_default_dtype()
        self.dtype = dtype

    def __call__(self, input: torch.Tensor) -> torch.Tensor:
        """Convert input tensor to the target data type.

        Args:
            input: Input tensor to convert.

        Returns:
            Tensor converted to the target dtype.
        """
        return input.type(self.dtype)


class ToDevice:
    """Move tensor to specified device.

    This transform moves input tensors to the specified device. If no
    device is provided, it uses PyTorch's default device.
    """

    def __init__(self, device: torch.device | None = None) -> None:
        """Initialize the ToDevice transform.

        Args:
            device: Target device. If None, uses PyTorch's default device.
        """
        if device is None:
            device = torch.get_default_device()
        self.device = device

    def __call__(self, input: torch.Tensor) -> torch.Tensor:
        """Move input tensor to the target device.

        Args:
            input: Input tensor to move.

        Returns:
            Tensor moved to the target device.
        """
        return input.to(self.device)
