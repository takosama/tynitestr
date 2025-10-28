"""
Hyena-based lightweight language model compatible with the training loop.

- Causal depthwise convolutions for long-range mixing
- Gated convolutional blocks + MLP
- Forward returns either [B, T, V] (default) or [B, V] via `last_only`

Note: LoRA in this repo targets nn.Linear layers; depthwise convs are left
untouched, which is fine in practice.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as cp

from config import D_MODEL, N_LAYER
from model import MLP, RMSNorm  # reuse existing lightweight implementations

# Low-rank head bottleneck width
HEAD_RANK = 128


class DepthwiseCausalConv1d(nn.Module):
    """Depthwise 1D convolution with causal (left-only) padding."""

    def __init__(self, channels: int, kernel_size: int, dilation: int = 1):
        super().__init__()
        assert kernel_size >= 1
        self.left_pad = dilation * (kernel_size - 1)
        self.conv = nn.Conv1d(
            channels,
            channels,
            kernel_size=kernel_size,
            groups=channels,
            dilation=dilation,
            padding=0,
            bias=False,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, C] -> [B, C, T]
        x_bcT = x.transpose(1, 2).contiguous()
        if self.left_pad:
            x_bcT = F.pad(x_bcT, (self.left_pad, 0))
        y = self.conv(x_bcT)
        return y.transpose(1, 2)


class HyenaBlock(nn.Module):
    """Gated depthwise causal conv + MLP block."""

    def __init__(
        self,
        d_model: int,
        kernel_size: int = 17,
        dilation: int = 1,
        mlp_mult: int = 4,
        t_chunk_size: int = 0,
    ):
        super().__init__()
        self.ln1 = RMSNorm(d_model)
        self.gate = nn.Linear(d_model, d_model, bias=False)
        self.dwconv = DepthwiseCausalConv1d(
            d_model, kernel_size=kernel_size, dilation=dilation
        )
        self.proj = nn.Linear(d_model, d_model, bias=False)
        self.ln2 = RMSNorm(d_model)
        self.mlp = MLP(d_model, mult=mlp_mult, t_chunk_size=t_chunk_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.ln1(x)
        g = torch.sigmoid(self.gate(h))
        h = self.dwconv(h)
        h = self.proj(g * h)
        x = x + h
        x = x + self.mlp(self.ln2(x))
        return x


class HyenaLM(nn.Module):
    """Hyena-based language model.

    Args:
        vocab_size: tokenizer vocab size
        d_model: hidden width (from config)
        n_layer: number of Hyena blocks (from config)
        block_size: maximum sequence length (context length)
        dropout: dropout probability (default 0.0)
        kernel_size: kernel size for depthwise causal conv

    Notes:
        - If `last_only=True`, returns [B, V] for the last step only.
        - If `last_only=False` (default), returns full sequence logits [B, T, V].
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int = D_MODEL,
        n_layer: int = N_LAYER,
        block_size: int = 32,
        dropout: float = 0.0,
        kernel_size: int = 17,
    ):
        super().__init__()
        self.wte = nn.Embedding(vocab_size, d_model)
        # ty with generate.py
        self.wpe = nn.Embedding(block_size, d_model)
        self.drop = nn.Dropout(dropout)

        # Conservative chunk size to reduce peak activation memory in MLPs
        mlp_t_chunk = max(1, min(block_size // 2, 64)) if block_size >= 32 else 0

        # Stagger dilations to increase receptive field
        dilations = [1, 2, 1, 2]
        self.blocks = nn.ModuleList(
            [
                HyenaBlock(
                    d_model,
                    kernel_size=kernel_size,
                    dilation=dilations[i % len(dilations)],
                    mlp_mult=4,
                    t_chunk_size=mlp_t_chunk,
                )
                for i in range(n_layer)
            ]
        )
        self.ln_f = nn.LayerNorm(d_model)

        # Low-rank factorized output head: [D -> R] then [R -> V]
        self.lm_head = nn.Sequential(
            nn.Linear(d_model, HEAD_RANK, bias=False),
            nn.Linear(HEAD_RANK, vocab_size, bias=False),
        )

        # Cache position ids; not persisted to checkpoints
        self.register_buffer("_pos_ids", torch.arange(block_size), persistent=False)

        # Optional per-block activation checkpointing (set externally)
        self.checkpoint_blocks: bool = True

    ###### last_only: bool = Trueをデフォルトに　このコメントに従ってクラスのいらない部分を変更
    def forward(self, x: torch.Tensor, last_only: bool = True) -> torch.Tensor:
        """Forward pass.

        Args:
            x: token ids [B, T]
            last_only: if True, return only last-step logits [B, V]

        Returns:
            [B, V] when last_only else [B, T, V].
        """
        B, T = x.size()
        pos = self._pos_ids[:T].to(x.device)

        h = self.wte(x) + self.wpe(pos)[None, :, :]
        h = self.drop(h)

        if self.checkpoint_blocks and self.training:
            for blk in self.blocks:
                h = cp.checkpoint(blk, h)
        else:
            for blk in self.blocks:
                h = blk(h)

        h = self.ln_f(h)

        if last_only:
            return self.lm_head(h[:, -1, :])  # [B, V]
        return self.lm_head(h)  # [B, T, V]
