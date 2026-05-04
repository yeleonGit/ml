# A transformer is built from stacked identical "blocks."
# Each block has two sub-layers:
#   1. Multi-Head Attention — tokens communicate with each other
#   2. FeedForward Network — each token processes its own information independently
#
# Both sub-layers use:
#   - Residual connections: output = x + sublayer(x)
#     This helps gradients flow during training (avoids vanishing gradients)
#   - Layer Normalization: stabilizes training by normalizing activations
#
# The full model:
#   Token Embedding → Position Embedding → N × Block → LayerNorm → Linear Head
#
# Token Embedding:  integer IDs → dense vectors
# Position Embedding: adds positional information (order matters in language!)
# Linear Head:      projects back to vocab_size for next-char prediction

import torch
import torch.nn as nn
import torch.nn.functional as F
from attention import MultiHeadAttention
from config import BLOCK_SIZE, N_EMBD, N_HEAD, N_LAYER, DROPOUT, DEVICE


class FeedForward(nn.Module):
    """
    A simple 2-layer MLP applied independently to each token.
    Gives the model capacity to process information after attention.

    Hidden dim is 4x n_embd — following the original "Attention is All You Need" paper.
    """
    def __init__(self, n_embd: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),   # Expand
            nn.ReLU(),                         # Non-linearity
            nn.Linear(4 * n_embd, n_embd),   # Contract back
            nn.Dropout(DROPOUT),
        )

    def forward(self, x):
        return self.net(x)


class TransformerBlock(nn.Module):
    """
    One transformer block = Attention + FeedForward, both with residual + LayerNorm.

    Note: We use "Pre-LN" style (LayerNorm before the sublayer),
    which is more stable to train than the original "Post-LN" design.
    """
    def __init__(self, n_embd: int, n_head: int):
        super().__init__()
        self.attention = MultiHeadAttention(n_head, n_embd)
        self.ffwd      = FeedForward(n_embd)
        self.ln1       = nn.LayerNorm(n_embd)   # Applied before attention
        self.ln2       = nn.LayerNorm(n_embd)   # Applied before feedforward

    def forward(self, x):
        # Residual connection around attention
        x = x + self.attention(self.ln1(x))
        # Residual connection around feedforward
        x = x + self.ffwd(self.ln2(x))
        return x


class TinyGPT(nn.Module):
    """
    The full language model.
    Input: (Batch, Time) integer token IDs
    Output: (Batch, Time, vocab_size) logits — raw scores for each possible next token
    """
    def __init__(self, vocab_size: int):
        super().__init__()

        # Embedding tables
        self.token_embedding    = nn.Embedding(vocab_size, N_EMBD)
        self.position_embedding = nn.Embedding(BLOCK_SIZE, N_EMBD)

        # Stack of N_LAYER transformer blocks
        self.blocks = nn.Sequential(
            *[TransformerBlock(N_EMBD, N_HEAD) for _ in range(N_LAYER)]
        )

        # Final layer norm before the output head
        self.ln_final = nn.LayerNorm(N_EMBD)

        # Projects from embedding space → vocabulary (for next-token prediction)
        self.lm_head = nn.Linear(N_EMBD, vocab_size)

    def forward(self, idx, targets=None):
        """
        Args:
            idx:     (B, T) integer token IDs
            targets: (B, T) integer token IDs shifted by 1 (for loss calculation)

        Returns:
            logits: (B, T, vocab_size)
            loss:   scalar cross-entropy loss (None if targets not provided)
        """
        B, T = idx.shape

        # Get token + position embeddings and add them
        tok_emb = self.token_embedding(idx)                          # (B, T, N_EMBD)
        pos_emb = self.position_embedding(torch.arange(T, device=DEVICE))  # (T, N_EMBD)
        x = tok_emb + pos_emb                                        # (B, T, N_EMBD)

        # Pass through all transformer blocks
        x = self.blocks(x)
        x = self.ln_final(x)

        # Project to vocabulary size
        logits = self.lm_head(x)    # (B, T, vocab_size)

        # Calculate loss if targets are provided
        loss = None
        if targets is not None:
            # Reshape for cross_entropy: expects (N, C) and (N,)
            B, T, C = logits.shape
            loss = F.cross_entropy(
                logits.view(B * T, C),
                targets.view(B * T)
            )

        return logits, loss

    def count_parameters(self):
        """Print total number of trainable parameters."""
        total = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"Model parameters: {total:,} ({total/1e6:.1f}M)")
        return total
