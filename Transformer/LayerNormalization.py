from imports import *

class LayerNormalization(nn.Module):
    def __init__(self, 
                eps: float = 1e-6
                ) -> None:
        """
        Initialize the LayerNormalization module.

        Args:
            eps (float): A small value to prevent division by zero for numerical stability. Default is 1e-6.
        """
        super().__init__()  # Initialize the parent class (nn.Module)
        self.eps = eps  # Save the epsilon value for numerical stability
        self.alpha = nn.Parameter(torch.ones(1))  # Learnable scale parameter, initialized to 1
        self.bias = nn.Parameter(torch.zeros(1))  # Learnable shift parameter, initialized to 0

    def forward(self, 
                x: torch.Tensor
                ) -> torch.Tensor:
        """
        Forward pass through the LayerNormalization module.

        Args:
            x (torch.Tensor): Input tensor to be normalized.

        Returns:
            torch.Tensor: Normalized tensor with the same shape as input.
        """
        mean = x.mean(-1, keepdim=True)  # Compute the mean of the input tensor along the last dimension
        std = x.std(-1, keepdim=True)  # Compute the standard deviation of the input tensor along the last dimension
        # Normalize the input tensor, scale by alpha, and shift by bias
        return self.alpha * (x - mean) / (std + self.eps) + self.bias
