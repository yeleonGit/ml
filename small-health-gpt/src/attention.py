import torch
import torch.nn as nn
import torch.nn.functional as F
from config import BLOCK_SIZE, DROPOUT


class SingleHead(nn.Module):
    """
    One attention head.
    Learns one type of relationship between tokens.
    """

    def __init__(self, head_size: int, n_embd: int):
        """
        Args:
            head_size: size of Q, K, V projections for this head
            n_embd: embedding dimension from the full model
        """
        super().__init__()
        # Linear projections — no bias needed
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.key   = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)

        # Causal mask: lower-triangular matrix of 1s
        # Prevents token at position i from attending to position j > i
        # This is what makes it an autoregressive (left-to-right) model
        self.register_buffer(
            "mask", torch.tril(torch.ones(BLOCK_SIZE, BLOCK_SIZE))
        )

        self.dropout = nn.Dropout(DROPOUT)

    def forward(self, x):
        """
        Args:
            x: (Batch, Time, Channels) — the token embeddings

        Returns:
            out: (Batch, Time, head_size) — attended output
        """
        B, T, C = x.shape

        q = self.query(x)    # (B, T, head_size)
        k = self.key(x)      # (B, T, head_size)
        v = self.value(x)    # (B, T, head_size)

        # Compute raw attention scores
        # (B, T, head_size) @ (B, head_size, T) → (B, T, T)
        scale = k.shape[-1] ** -0.5    # Scale to prevent softmax saturation
        scores = q @ k.transpose(-2, -1) * scale

        # Apply causal mask: set future positions to -inf so softmax → 0
        scores = scores.masked_fill(self.mask[:T, :T] == 0, float("-inf"))

        # Softmax over last dim → attention weights (sum to 1 per row)
        weights = F.softmax(scores, dim=-1)
        weights = self.dropout(weights)

        # Weighted sum of values
        out = weights @ v    # (B, T, T) @ (B, T, head_size) → (B, T, head_size)
        return out


class MultiHeadAttention(nn.Module):
    """
    Multiple attention heads running in parallel.
    Each head specializes in different token relationships.
    Their outputs are concatenated and projected back to n_embd.
    """

    def __init__(self, n_head: int, n_embd: int):
        super().__init__()
        head_size = n_embd // n_head   # Each head gets an equal share of the embedding
        self.heads = nn.ModuleList([
            SingleHead(head_size, n_embd) for _ in range(n_head)
        ])
        # Final linear projection after concatenation
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(DROPOUT)

    def forward(self, x):
        # Run all heads, concatenate along the channel dimension
        out = torch.cat([head(x) for head in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out
