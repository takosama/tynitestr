"""
main2.py — Windows-safe trainer with fast DataLoader.

Changes vs original:
- Wraps entrypoint in `if __name__ == "__main__":` to satisfy Windows spawn.
- Uses FastMemmapNexTokDataset + fast_collate_long.
- Sets DataLoader shuffle=False (dataset already samples randomly).
"""
from hashlib import sha256
import math
from pathlib import Path
from typing import Optional
import os

# Force CPU-only execution by hiding CUDA devices from PyTorch
os.environ["CUDA_VISIBLE_DEVICES"] = ""

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
    CSV_SEP,
    CSV_TEXT_COL,
    EPOCHS,
    FORCE_RETRAIN_TOKENIZER,
    LORA_ALPHA,
    LORA_DROPOUT,
    LORA_R,
    LORA_TARGET_LM_HEAD,
    LR,
    MODEL_SIZE,
    NUM_WORKERS,
    RUN_ID,
    TOK_BIN,
    TOKENIZER_JSON,
    USE_LORA,
    VOCAB_SIZE,
    WEIGHT_DECAY,
    WINDOW,
    CORPUS,
)
from data import build_memmap_tokens, preprocess_corpus
from data_fast import FastMemmapNexTokDataset, fast_collate_long
from generate import generate_text
from lora import (
    _apply_lora_to_model,
    _collect_lora_params,
    _mark_only_lora_trainable,
)
from model import TinyGPT2
from optimizer import Lion
from tokenizer import ByteBPETokenizer, load_corpus_text, train_bpe_from_text


def _file_sha256(p: Path) -> str:
    try:
        return sha256(p.read_bytes()).hexdigest()[:16]
    except Exception:
        return "na"


def seed_worker(_wid: int):
    import os as _os
    import numpy as _np

    # Worker seeding (CPU-only)
    s = torch.initial_seed() % (2**32)
    _np.random.seed(s)


def _make_dataloader() -> DataLoader:
    ds = FastMemmapNexTokDataset(TOK_BIN, WINDOW)

    extra = {}
    if NUM_WORKERS and NUM_WORKERS > 0:
        extra.update(dict(persistent_workers=True, prefetch_factor=2))

    # Use DistributedSampler under DDP
    sampler = None
    if False and dist.is_available() and dist.is_initialized():
        sampler = DistributedSampler(ds, shuffle=True)

    return DataLoader(
        ds,
        batch_size=BATCH_SIZE,
        shuffle=(sampler is None),  # samplerがあるときはshuffle不可
        sampler=sampler,
        pin_memory=False,
        num_workers=NUM_WORKERS,
        collate_fn=fast_collate_long,
        worker_init_fn=(seed_worker if (NUM_WORKERS and NUM_WORKERS > 0) else None),
        **extra,
    )

def _ddp_setup_if_needed() -> tuple[torch.device, int, int]:
    # CPU-only mode: do not initialize DDP/CUDA; always run single-process on CPU
    return torch.device("cpu"), 0, 1


def _is_main_process() -> bool:
    # CPU-only, single process
    return True


def main() -> None:
    device, rank, world_size = _ddp_setup_if_needed()
    if _is_main_process():
        print("Using device:", device, torch.__version__, "world_size=", world_size)

    # Tokenizer
    if FORCE_RETRAIN_TOKENIZER or not TOKENIZER_JSON.exists():
        _tmp_text = load_corpus_text(CORPUS)
        train_bpe_from_text(_tmp_text, VOCAB_SIZE)
    tokenizer = ByteBPETokenizer(str(TOKENIZER_JSON))

    # Dataset / memmap
    cleaned = preprocess_corpus(CORPUS)
    mx_id = max(tokenizer.vocab.values())
    if getattr(tokenizer, "special", None):
        mx_id = max(mx_id, *tokenizer.special.values())
    vocab_size = mx_id + 1
    build_memmap_tokens(cleaned, tokenizer)

    dl = _make_dataloader()

    # Model
    model = TinyGPT2(vocab_size=vocab_size, block_size=WINDOW)
    if not USE_LORA:
        print("[MODE] Plain GPT (no LoRA)")
        for p in model.parameters():
            p.requires_grad = True
    else:
        print("[MODE] GPT + LoRA")
        _apply_lora_to_model(
            model,
            r=LORA_R,
            alpha=LORA_ALPHA,
            dropout=LORA_DROPOUT,
            target_lm_head=LORA_TARGET_LM_HEAD,
        )
        _mark_only_lora_trainable(model)

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
    if _is_main_process():
        print("[sanity] vocab_size =", vocab_size)
        print("[sanity] TOK_BIN max id =", mx_seen)
    assert mx_seen < vocab_size, (
        f"memmap token id {mx_seen} >= vocab_size {vocab_size} — tokenizer/memmap mismatch"
    )

    crit = nn.CrossEntropyLoss(label_smoothing=0.03)
    scaler = None  # CPU only

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
    PREVIEW_INTERVAL = 200
    opt_step = global_step
    ema: Optional[float] = None
    ema_beta = 0.98

    import torch.nn.functional as F

    model.train()
    for ep in range(start_epoch, EPOCHS + 1):
        total_loss = 0.0
        total_count = 0
        # Single-process CPU training; no sampler epoch needed
        pbar = tqdm(dl, desc=f"Epoch {ep}/{EPOCHS}", leave=False) if _is_main_process() else dl
        opt.zero_grad(set_to_none=True)

        for it, (xb, yb) in enumerate(pbar):
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

            logits = model(xb)  # [B, T, V]
            # Use channel-first CE to reduce memory vs large flatten
            loss = F.cross_entropy(
                logits.transpose(1, 2), yb, reduction="mean"
            ) / ACCUM_STEPS

            loss.backward()
            del logits

            with torch.no_grad():
                loss_item = float(loss.item() * ACCUM_STEPS)
                bs = xb.size(0)
                total_loss += loss_item * bs
                total_count += bs

            do_step = ((it + 1) % ACCUM_STEPS == 0)
            if do_step:
                params = (
                    p for p in model.parameters() if p.requires_grad and p.grad is not None
                )
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                opt.step()
                opt.zero_grad(set_to_none=True)

                ema = loss_item if ema is None else (ema_beta * ema + (1 - ema_beta) * loss_item)
                opt_step += 1

                if opt_step % 1000 == 0 and _is_main_process():
                    meta["best_metric"] = best_metric
                    save_checkpoint("latest", model, opt, scaler, opt_step, ep, meta)
                    if ema < best_metric:
                        best_metric = ema
                        meta["best_metric"] = best_metric
                        save_checkpoint("best", model, opt, scaler, opt_step, ep, meta)
                        print(f"\n[best] ema={best_metric:.4f} @ step {opt_step}")

                if opt_step % PREVIEW_INTERVAL == 0 and _is_main_process():
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
                    model.train()

            avg_loss = total_loss / max(1, total_count)
            pbar.set_postfix(loss=float(avg_loss), ema=(float(ema) if ema else None), lr=float(scheduler_last_lr))

    # Final sample
    if _is_main_process():
        model.eval()
        with torch.no_grad():
            seed = tokenizer.encode("こんにちは")[:WINDOW]
            x = torch.tensor(seed, dtype=torch.long, device=device).unsqueeze(0)
            logits = model(x)
            next_id = int(torch.argmax(logits[:, -1, :], dim=-1).item())
            print("[final]", tokenizer.decode(seed + [next_id]))
        print("done.")
    # No DDP teardown needed in CPU-only mode


if __name__ == "__main__":
    # For Windows and frozen executables (PyInstaller, etc.)
    import multiprocessing as mp

    mp.freeze_support()
    main()
