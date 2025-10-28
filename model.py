import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as cp

from config import D_MODEL, N_HEAD, N_LAYER

# Enable faster kernels when available (no checkpoint format changes)
try:
    # Prefer Flash/SDPA fast paths on CUDA
    if torch.cuda.is_available():
        if hasattr(torch.backends.cuda, "enable_flash_sdp"):
            torch.backends.cuda.enable_flash_sdp(True)
            torch.backends.cuda.enable_mem_efficient_sdp(True)
        elif hasattr(torch.backends.cuda, "sdp_kernel"):
            # PyTorch <=2.1 style context; set global defaults
            torch.backends.cuda.sdp_kernel(
                enable_flash=True, enable_mem_efficient=True, enable_math=False
            )
        # Allow TF32 where beneficial
        if hasattr(torch, "set_float32_matmul_precision"):
            torch.set_float32_matmul_precision("high")
except Exception:
    # Be conservative: any failure here should not break training
    pass


class CausalSelfAttention(nn.Module):
    """Causal self-attention with single qkv projection and optional query chunking.
    Chunking over the time axis reduces peak activation memory while keeping K/V full.
    """

    def __init__(self, d_model: int, n_head: int, q_chunk_size_t: int = 0):
        super().__init__()
        assert d_model % n_head == 0, "d_model must be divisible by n_head"
        self.n_head = n_head
        self.head_dim = d_model // n_head
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.proj = nn.Linear(d_model, d_model, bias=False)
        self.q_chunk_size_t = int(max(0, q_chunk_size_t))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.size()
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        if self.q_chunk_size_t and T > self.q_chunk_size_t and self.training:
            y_buf = torch.empty_like(q)
            for start in range(0, T, self.q_chunk_size_t):
                end = min(T, start + self.q_chunk_size_t)
                q_chunk = q[:, :, start:end, :]
                y_chunk = F.scaled_dot_product_attention(q_chunk, k, v, is_causal=True)
                y_buf[:, :, start:end, :] = y_chunk
            y = y_buf
        else:
            y = F.scaled_dot_product_attention(q, k, v, is_causal=True)

        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.proj(y)


class MLP(nn.Module):
    def __init__(self, d_model: int, mult: int = 4, t_chunk_size: int = 0):
        super().__init__()
        self.fc1 = nn.Linear(d_model, mult * d_model, bias=False)
        # approximate GELU is a little faster with nearly identical behavior
        self.act = nn.GELU(approximate="tanh")
        self.fc2 = nn.Linear(mult * d_model, d_model, bias=False)
        self.t_chunk_size = int(max(0, t_chunk_size))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Optionally process along time in chunks to lower peak activation memory
        if self.t_chunk_size and x.size(1) > self.t_chunk_size and self.training:
            B, T, C = x.shape
            out = torch.empty(B, T, C, dtype=x.dtype, device=x.device)
            for start in range(0, T, self.t_chunk_size):
                end = min(T, start + self.t_chunk_size)
                h = self.fc1(x[:, start:end, :])
                h = self.act(h)
                h = self.fc2(h)
                out[:, start:end, :] = h
            return out
        return self.fc2(self.act(self.fc1(x)))


class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = x.norm(2, dim=-1, keepdim=True)
        rms = norm * (1.0 / (x.shape[-1] ** 0.5))
        x_norm = x / (rms + self.eps)
        return x_norm * self.weight


class TransformerBlock(nn.Module):
    def __init__(
        self, d_model: int, n_head: int, attn_q_chunk_t: int = 0, mlp_t_chunk: int = 0
    ):
        super().__init__()
        self.ln1 = RMSNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_head, q_chunk_size_t=attn_q_chunk_t)
        self.ln2 = RMSNorm(d_model)
        self.mlp = MLP(d_model, mult=4, t_chunk_size=mlp_t_chunk)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class TinyGPT2(nn.Module):
    """Concise GPT-2 style language model.

    Args:
        vocab_size: tokenizer vocab size
        d_model: hidden width (defaults from config)
        n_layer: number of transformer blocks (from config)
        n_head: number of attention heads (from config)
        block_size: maximum sequence length (context length)
        dropout: dropout probability (default 0.0)
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int = D_MODEL,
        n_layer: int = N_LAYER,
        n_head: int = N_HEAD,
        block_size: int = 512,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.wte = nn.Embedding(vocab_size, d_model)
        self.wpe = nn.Embedding(block_size, d_model)
        self.drop = nn.Dropout(dropout)
        # choose conservative chunk sizes to reduce peak memory
        attn_q_chunk = max(1, block_size // 2) if block_size >= 32 else 0
        mlp_t_chunk = max(1, block_size // 2) if block_size >= 32 else 0
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    d_model,
                    n_head,
                    attn_q_chunk_t=attn_q_chunk,
                    mlp_t_chunk=mlp_t_chunk,
                )
                for _ in range(n_layer)
            ]
        )
        self.ln_f = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        # Cache position ids to avoid realloc each forward; not saved in ckpt
        self.register_buffer("_pos_ids", torch.arange(block_size), persistent=False)
        # Enable per-block activation checkpointing by default for lower peak memory
        self.checkpoint_blocks: bool = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T] (int64 token ids)
        B, T = x.size()
        pos = self._pos_ids[:T].to(x.device)
        h = self.drop(self.wte(x) + self.wpe(pos)[None, :, :])
        if self.checkpoint_blocks and self.training:
            for blk in self.blocks:
                h = cp.checkpoint(blk, h)
        else:
            for blk in self.blocks:
                h = blk(h)
        h = self.ln_f(h)
        return self.lm_head(h)  # [B, T, vocab]
