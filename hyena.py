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
    """Depthwise 1D 'conv' via FFT (causal: left-only padding)."""

    def __init__(self, channels: int, kernel_size: int, dilation: int = 1):
        super().__init__()
        assert kernel_size >= 1
        self.channels = channels
        self.kernel_size = kernel_size
        self.dilation = dilation
        # 有効カーネル長（空挿入後）
        self.L_eff = dilation * (kernel_size - 1) + 1
        # 参照用：左側に入れるパディング量（因果）
        self.left_pad = self.L_eff - 1

        # depthwise 用にチャネルごとに独立したカーネル（bias なし）
        # 形状: [C, K]（生のカーネル）; dilation は forward で展開
        self.weight = nn.Parameter(torch.empty(channels, kernel_size))
        nn.init.kaiming_uniform_(self.weight, a=5**0.5)

    @staticmethod
    def _next_pow2(n: int) -> int:
        # FFT 長は T + L_eff - 1 以上の 2 の冪に
        p = 1
        while p < n:
            p <<= 1
        return p

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, C] -> [B, C, T]
        x_bct = x.transpose(1, 2).contiguous()
        B, C, T = x_bct.shape
        assert C == self.channels, f"channels mismatch: {C} != {self.channels}"

        device = x_bct.device
        dtype = x_bct.dtype

        # ダイレーション展開: [C, L_eff] に埋め込み（0 埋め）
        # 例: dilation=2, K=3 -> 位置 [0,2,4] に w を配置
        k_eff = torch.zeros(C, self.L_eff, device=device, dtype=dtype)
        k_eff[:, :: self.dilation] = self.weight

        # conv1d(相関) = 畳み込み(k を時間反転)なので flip する
        k_rev = torch.flip(k_eff, dims=[-1])  # [C, L_eff]

        # 入力を因果パディング（左側のみに L_eff-1）
        x_pad = F.pad(x_bct, (self.left_pad, 0))  # [B, C, T+left_pad]

        # 線形畳み込み長
        target_len = x_pad.shape[-1] + k_rev.shape[-1] - 1
        n_fft = self._next_pow2(target_len)

        # ゼロパディングして FFT
        x_pad = F.pad(x_pad, (0, n_fft - x_pad.shape[-1]))         # [B, C, n_fft]
        k_pad = F.pad(k_rev, (0, n_fft - k_rev.shape[-1]))         # [C, n_fft]

        Xf = torch.fft.rfft(x_pad, n=n_fft)                        # [B, C, n_rfft]
        Kf = torch.fft.rfft(k_pad, n=n_fft)                        # [C, n_rfft]
        Yf = Xf * Kf                                               # broadcast on C

        y_full = torch.fft.irfft(Yf, n=n_fft)                      # [B, C, n_fft]

        # ちょうど長さ T 分だけ取り出す。
        # 検証済み：参照実装(F.conv1d + 左パッド)と一致
        y = y_full[..., self.left_pad : self.left_pad + T]         # [B, C, T]

        return y.transpose(1, 2)  # -> [B, T, C]



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
            nn.Linear(d_model,vocab_size, bias=False),
        )

        # Cache position ids; not persisted to checkpoints
        self.register_buffer("_pos_ids", torch.arange(block_size), persistent=False)

        # Optional per-block activation checkpointing (set externally)
        self.checkpoint_blocks: bool = True

    def forward(self, x: torch.Tensor, last_only: bool = False) -> torch.Tensor:
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
