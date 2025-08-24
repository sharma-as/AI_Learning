"""
Basic implementation of scaled dot-product self-attention using PyTorch.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SelfAttention(nn.Module):
    """Simple self-attention mechanism."""

    def __init__(self, embed_dim: int) -> None:
        """Initialize the linear layers used to compute attention.

        Args:
            embed_dim: Dimensionality of the input embeddings.
        """
        super().__init__()
        self.query = nn.Linear(embed_dim, embed_dim)
        self.key = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)
        self.scale = embed_dim ** 0.5

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply self-attention to the input tensor.

        Args:
            x: Input tensor of shape (batch, seq_len, embed_dim).

        Returns:
            A tuple containing the output tensor and the attention weights.
        """
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)

        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / self.scale
        attn_weights = F.softmax(attn_scores, dim=-1)
        output = torch.matmul(attn_weights, v)
        return output, attn_weights


if __name__ == "__main__":
    torch.manual_seed(0)
    sample = torch.randn(1, 4, 8)  # (batch, seq_len, embed_dim)
    attention = SelfAttention(embed_dim=8)
    out, weights = attention(sample)
    print("Output shape:", out.shape)
    print("Attention weights shape:", weights.shape)
