"""
Windows-friendly HyenaLM trainer with CPU DataLoader and GPU model.

- CPU DataLoader (optional pinned memory)
- Automatic DataParallel on multi-GPU
- Last-token-only loss for lower memory, AMP on CUDA
- Robust to 2D/3D logits and includes CUDA allocator tuning
"""

import math
import os
from hashlib import sha256
from pathlib import Path
from typing import Optional

# Disable TorchDynamo/torch.compile globally to avoid internal SyntaxError on some setups
os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")
os.environ.setdefault("TORCH_COMPILE_DISABLE", "1")
os.environ.setdefault(
    "PYTORCH_CUDA_ALLOC_CONF", "max_split_size_mb:128,expandable_segments:True"
)

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from checkpoint import save_checkpoint, try_resume
from config import (
    ACCUM_STEPS,
    BATCH_SIZE,
    BETAS,
    CORPUS,
    EPOCHS,
    FORCE_RETRAIN_TOKENIZER,
    GRAD_CHECKPOINT,
    LORA_ALPHA,
    LORA_DROPOUT,
    LORA_R,
    LORA_TARGET_LM_HEAD,
    LR,
    MODEL_SIZE,
    NUM_WORKERS,
    PIN_MEMORY,
    RUN_ID,
    TOK_BIN,
    TOKENIZER_JSON,
    USE_LORA,
    VOCAB_SIZE,
    WEIGHT_DECAY,
    WINDOW,
)
from data import build_memmap_tokens, preprocess_corpus
from data_fast import FastMemmapNexTokDataset, fast_collate_long
from generate import generate_text
from hyena import HyenaLM
from lora import (
    _apply_lora_to_model,
    _collect_lora_params,
    _mark_only_lora_trainable,
)
from optimizer import Lion
from tokenizer import ByteBPETokenizer, load_corpus_text, train_bpe_from_text

# Best-effort: relax Dynamo error handling if imported
try:
    import torch._dynamo as _dynamo  # type: ignore

    _dynamo.config.suppress_errors = True  # swallow internal graph breaks
except Exception:
    pass


def _file_sha256(p: Path) -> str:
    try:
        return sha256(p.read_bytes()).hexdigest()[:16]
    except Exception:
        return "na"


def _ce_last_light(
    logits_last: torch.Tensor, targets_last: torch.Tensor, label_smoothing: float = 0.0
) -> torch.Tensor:
    """Memory-friendly CE on the final time step only."""
    lse = torch.logsumexp(logits_last, dim=-1)
    z_t = logits_last.gather(1, targets_last.unsqueeze(1)).squeeze(1)
    if label_smoothing and label_smoothing > 0.0:
        mean_z = logits_last.mean(dim=-1)
        nll = lse - ((1.0 - label_smoothing) * z_t + label_smoothing * mean_z)
    else:
        nll = lse - z_t
    return nll.mean()


def _last_logits(logits: torch.Tensor) -> torch.Tensor:
    """Return last-token logits with shape [B, V] regardless of input rank."""
    if logits.dim() == 3:
        return logits[:, -1, :]
    if logits.dim() == 2:
        return logits
    raise RuntimeError(f"Unexpected logits dim: {tuple(logits.shape)}")


def seed_worker(_wid: int):
    import numpy as _np

    s = torch.initial_seed() % (2**32)
    _np.random.seed(s)


def _make_dataloader() -> DataLoader:
    ds = FastMemmapNexTokDataset(TOK_BIN, WINDOW)

    extra = {}
    if NUM_WORKERS and NUM_WORKERS > 0:
        extra.update(dict(persistent_workers=True, prefetch_factor=2))

    return DataLoader(
        ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        pin_memory=bool(PIN_MEMORY),
        num_workers=NUM_WORKERS,
        collate_fn=fast_collate_long,
        worker_init_fn=(seed_worker if (NUM_WORKERS and NUM_WORKERS > 0) else None),
        **extra,
    )


def _device_setup(require_cuda: bool = True) -> torch.device:
    """Select device and optionally enforce CUDA. Prints basic GPU info."""
    if not torch.cuda.is_available():
        if require_cuda:
            raise RuntimeError(
                "CUDA GPU not available; install torch with CUDA (e.g., cu121/cu124)."
            )
        return torch.device("cpu")
    try:
        torch.backends.cudnn.benchmark = True  # type: ignore[attr-defined]
        if hasattr(torch, "set_float32_matmul_precision"):
            torch.set_float32_matmul_precision("high")
    except Exception:
        pass
    try:
        n_gpu = torch.cuda.device_count()
        name0 = torch.cuda.get_device_name(0)
        print(f"CUDA GPUs: {n_gpu} | [0]: {name0}")
    except Exception:
        pass
    return torch.device("cuda", 0)


def _prepare_tokenizer_and_memmap() -> tuple[ByteBPETokenizer, int]:
    """Train/load tokenizer, preprocess corpus, build memmap, and return vocab size."""
    if FORCE_RETRAIN_TOKENIZER or not TOKENIZER_JSON.exists():
        _tmp_text = load_corpus_text(CORPUS)
        train_bpe_from_text(_tmp_text, VOCAB_SIZE)
    tokenizer = ByteBPETokenizer(str(TOKENIZER_JSON))

    cleaned = preprocess_corpus(CORPUS)
    mx_id = max(tokenizer.vocab.values())
    if getattr(tokenizer, "special", None):
        mx_id = max(mx_id, *tokenizer.special.values())
    vocab_size = mx_id + 1
    build_memmap_tokens(cleaned, tokenizer)
    return tokenizer, vocab_size


def _build_model(vocab_size: int) -> nn.Module:
    """Construct HyenaLM and apply optional LoRA and checkpointing flag."""
    model = HyenaLM(vocab_size=vocab_size, block_size=WINDOW)
    if not USE_LORA:
        print("[MODE] Plain HyenaLM (no LoRA)")
        for p in model.parameters():
            p.requires_grad = True
    else:
        print("[MODE] HyenaLM + LoRA)")
        _apply_lora_to_model(
            model,
            r=LORA_R,
            alpha=LORA_ALPHA,
            dropout=LORA_DROPOUT,
            target_lm_head=LORA_TARGET_LM_HEAD,
        )
        _mark_only_lora_trainable(model)
    try:
        model.checkpoint_blocks = bool(GRAD_CHECKPOINT)
    except Exception:
        pass
    return model


def main() -> None:
    device = _device_setup(require_cuda=True)
    print("Using device:", device, torch.__version__)

    # Tokenizer + dataset/memmap + dataloader
    tokenizer, vocab_size = _prepare_tokenizer_and_memmap()
    dl = _make_dataloader()

    # Model (Hyena-based LM)
    model = _build_model(vocab_size)

    # DataParallel wrap for multi-GPU
    if device.type == "cuda" and torch.cuda.device_count() > 1:
        print(f"[parallel] DataParallel enabled on {torch.cuda.device_count()} GPUs")
        model = nn.DataParallel(model)
    model = model.to(device)

    # Optimizer
    if USE_LORA:
        lora_params = _collect_lora_params(model)
        opt = Lion(
            [{"params": lora_params, "weight_decay": 0.0}],
            lr=LR,
            betas=BETAS,
            weight_decay=0.0,
        )
    else:
        opt = Lion(model.parameters(), lr=LR, betas=BETAS, weight_decay=WEIGHT_DECAY)

    # Sanity against memmap
    arr = np.memmap(str(TOK_BIN), dtype=np.uint32, mode="r")
    mx_seen = int(arr.max() if arr.size else 0)
    print("[sanity] vocab_size =", vocab_size)
    print("[sanity] TOK_BIN max id =", mx_seen)
    assert (
        mx_seen < vocab_size
    ), f"memmap token id {mx_seen} >= vocab_size {vocab_size} — tokenizer/memmap mismatch"

    LABEL_SMOOTH = 0.03
    scaler: Optional[torch.cuda.amp.GradScaler]
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))

    # Meta
    tokenizer_hash = _file_sha256(TOKENIZER_JSON)
    meta = {
        "run_id": RUN_ID,
        "tokenizer_sha256": tokenizer_hash,
        "vocab_size": vocab_size,
        "model_size": MODEL_SIZE,
        "best_metric": None,
    }

    global_step, start_epoch, best_metric = try_resume(model, opt, scaler)
    best_metric = best_metric or float("inf")

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model ready — vocab={vocab_size} params={n_params:,}")

    # LR schedule
    total_steps_est = max(1, len(dl)) * max(1, EPOCHS)
    WARMUP_STEPS = 100

    def lr_schedule(step: int) -> float:
        if step < WARMUP_STEPS:
            return (step + 1) / WARMUP_STEPS
        t = (step - WARMUP_STEPS) / max(1, (total_steps_est - WARMUP_STEPS))
        return 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * min(1.0, t)))

    # Train loop
    opt_step = global_step
    ema: Optional[float] = None
    ema_beta = 0.98

    model.train()
    for ep in range(start_epoch, EPOCHS + 1):
        total_loss = 0.0
        total_count = 0
        pbar = tqdm(dl, desc=f"Epoch {ep}/{EPOCHS}", leave=False)
        opt.zero_grad(set_to_none=True)

        is_dp = isinstance(model, nn.DataParallel)
        for it, (xb, yb) in enumerate(pbar):
            if is_dp:
                # Let DataParallel scatter from CPU
                xb = xb.long()
                yb = yb.long()
            else:
                xb = xb.long().to(device, non_blocking=True)
                yb = yb.long().to(device, non_blocking=True)

            if opt_step < 8:
                xm, xM = int(xb.min()), int(xb.max())
                ym, yM = int(yb.min()), int(yb.max())
                assert 0 <= xm and xM < vocab_size, f"x in [{xm},{xM}] OOB"
                assert 0 <= ym and yM < vocab_size, f"y in [{ym},{yM}] OOB"

            scale = lr_schedule(opt_step)
            base_lr = LR * scale
            for g in opt.param_groups:
                g["lr"] = base_lr
            scheduler_last_lr = base_lr

            # Forward + last-token loss (robust to 2D/3D logits)
            if scaler.is_enabled():
                bf16_ok = bool(
                    getattr(torch.cuda, "is_bf16_supported", lambda: False)()
                )
                cm = (
                    torch.cuda.amp.autocast(dtype=torch.bfloat16)
                    if bf16_ok
                    else torch.cuda.amp.autocast()
                )
                with cm:
                    logits = model(xb, last_only=True)
                    last_logits = _last_logits(logits)
                    last_targets = yb[:, -1]
                    if last_targets.device != last_logits.device:
                        last_targets = last_targets.to(
                            last_logits.device, non_blocking=True
                        )
                    loss = (
                        _ce_last_light(
                            last_logits, last_targets, label_smoothing=LABEL_SMOOTH
                        )
                        / ACCUM_STEPS
                    )
            else:
                logits = model(xb, last_only=True)
                last_logits = _last_logits(logits)
                last_targets = yb[:, -1]
                if last_targets.device != last_logits.device:
                    last_targets = last_targets.to(
                        last_logits.device, non_blocking=True
                    )
                loss = (
                    _ce_last_light(
                        last_logits, last_targets, label_smoothing=LABEL_SMOOTH
                    )
                    / ACCUM_STEPS
                )

            if scaler.is_enabled():
                scaler.scale(loss).backward()
            else:
                loss.backward()
            del logits

            with torch.no_grad():
                loss_item = float(loss.item() * ACCUM_STEPS)
                bs = xb.size(0)
                total_loss += loss_item * bs
                total_count += bs

            do_step = (it + 1) % ACCUM_STEPS == 0
            if do_step:
                params = (
                    p
                    for p in model.parameters()
                    if p.requires_grad and p.grad is not None
                )
                if scaler.is_enabled():
                    scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                if scaler.is_enabled():
                    scaler.step(opt)
                    scaler.update()
                else:
                    opt.step()
                opt.zero_grad(set_to_none=True)

                ema = (
                    loss_item
                    if ema is None
                    else (ema_beta * ema + (1 - ema_beta) * loss_item)
                )
                opt_step += 1

                if opt_step % 250 == 0:
                    meta["best_metric"] = best_metric
                    save_checkpoint("latest", model, opt, scaler, opt_step, ep, meta)
                    if ema < best_metric:
                        best_metric = ema
                        meta["best_metric"] = best_metric
                        save_checkpoint("best", model, opt, scaler, opt_step, ep, meta)
                        print(f"\n[best] ema={best_metric:.4f} @ step {opt_step}")

                if opt_step % 100 == 0:
                    model.eval()
                    with torch.inference_mode():
                        sample = generate_text(
                            model,
                            tokenizer,
                            seed_text="こんにちは",
                            max_new_tokens=60,
                            temperature=0.8,
                            top_k=60,
                            top_p=0.9,
                        )
                    print("[preview]", sample[:240].replace("\n", " "))
                    if device.type == "cuda":
                        try:
                            torch.cuda.empty_cache()
                        except Exception:
                            pass
                    model.train()

            avg_loss = total_loss / max(1, total_count)
            pbar.set_postfix(
                loss=float(avg_loss),
                ema=(float(ema) if ema else None),
                lr=float(scheduler_last_lr),
            )

    # Final sample
    model.eval()
    with torch.no_grad():
        seed = tokenizer.encode("こんにちは")[:WINDOW]
        x = torch.tensor(seed, dtype=torch.long, device=device).unsqueeze(0)
        logits = model(x)
        last_logits = _last_logits(logits)
        next_id = int(torch.argmax(last_logits, dim=-1).item())
        print("[final]", tokenizer.decode(seed + [next_id]))
    print("done.")


if __name__ == "__main__":
    main()
