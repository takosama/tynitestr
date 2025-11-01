from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch import nn

from config import CKPT_DIR, KEEP_LAST, RUN_ID, USE_LORA
from lora import _remap_old_linear_keys_to_lora


def _model_state(m: nn.Module) -> dict:
    """Return state_dict, handling optional DataParallel wrappers."""
    return m.module.state_dict() if hasattr(m, "module") else m.state_dict()


def _load_model_state(m: nn.Module, sd: dict, strict: bool = True) -> None:
    (m.module if hasattr(m, "module") else m).load_state_dict(sd, strict=strict)


def save_checkpoint(
    tag: str,
    model: nn.Module,
    opt: Any | None,
    scaler: Any | None,
    global_step: int,
    epoch: int,
    meta: dict,
) -> Path:
    """Save a training checkpoint.

    Parameters
    - tag: Label for the checkpoint (e.g., "latest", "best").
    - model: Model whose parameters are saved.
    - opt: Optimizer to save (optional).
    - scaler: AMP GradScaler to save (optional).
    - global_step: Global training step.
    - epoch: Current epoch.
    - meta: Arbitrary metadata to persist.

    Returns
    - Path to the saved checkpoint file.
    """
    ckpt = {
        "tag": tag,
        "global_step": global_step,
        "epoch": epoch,
        "model": _model_state(model),
        "optimizer": opt.state_dict() if opt is not None else None,
        "scaler": scaler.state_dict() if scaler is not None else None,
        "meta": meta,
    }
    path = CKPT_DIR / f"{RUN_ID}_{tag}_{global_step:09d}.pt"
    torch.save(ckpt, path)

    # latest 系のみ保持数を制限
    if tag == "latest":
        latest = sorted(CKPT_DIR.glob(f"{RUN_ID}_latest_*.pt"))
        for p in latest[:-KEEP_LAST]:
            p.unlink(missing_ok=True)
    return path


def _debug_report_load(ret) -> None:
    """Print a brief report about state_dict load mismatches."""
    try:
        print(
            f"▶ load_state: missing={len(ret.missing_keys)} "
            f"unexpected={len(ret.unexpected_keys)}"
        )
        if ret.missing_keys[:5]:
            print("  missing(head):", ret.missing_keys[:5])
        if ret.unexpected_keys[:5]:
            print("  unexpected(head):", ret.unexpected_keys[:5])
    except Exception:
        pass


def _strip_module_prefix(sd: dict) -> dict:
    """Remove a leading 'module.' prefix from keys saved via DataParallel."""
    if any(k.startswith("module.") for k in sd.keys()):
        return {k.replace("module.", "", 1): v for k, v in sd.items()}
    return sd


def try_resume(
    model: nn.Module, opt: Any | None, scaler: Any | None
) -> tuple[int, int, Any | None]:
    """Resume training from the most recent 'latest' checkpoint if present.

    Returns a tuple of (global_step, start_epoch, best_metric).
    """
    cands = sorted(CKPT_DIR.glob("*_latest_*.pt"))
    if not cands:
        return 0, 1, None

    path = cands[-1]
    ckpt = torch.load(path, map_location="cpu")
    sd = _strip_module_prefix(ckpt["model"])  # ここで DataParallel の接頭辞を除去

    # LoRA の有無を自動吸収
    has_lora_in_ckpt = any(
        ".lora_A." in k or ".lora_B." in k or ".base." in k for k in sd.keys()
    )
    if USE_LORA and not has_lora_in_ckpt:
        # 旧(非LoRA) → 新(LoRA) へのキー名移し替え
        sd = _remap_old_linear_keys_to_lora(sd, model)
        _load_model_state(model, sd, strict=False)
    elif (not USE_LORA) and has_lora_in_ckpt:
        # 新(LoRA) ckpt を 非LoRA モデルに読み込む（余剰キーは無視）
        _load_model_state(model, sd, strict=False)
    else:
        # 同一フォーマット同士
        _load_model_state(model, sd, strict=False)

    # try_resume の load_model_state 呼び出し後に:
    ret = (model.module if hasattr(model, "module") else model).load_state_dict(
        sd, strict=False
    )
    _debug_report_load(ret)

    if (opt is not None) and (ckpt.get("optimizer") is not None):
        try:
            opt.load_state_dict(ckpt["optimizer"])
        except Exception as e:
            print(f"⚠ optimizer state load skipped: {e}")

    if (scaler is not None) and (ckpt.get("scaler") is not None):
        try:
            scaler.load_state_dict(ckpt["scaler"])
        except Exception as e:
            print(f"⚠ scaler state load skipped: {e}")

    print(
        f"🔁 Resumed from: {path.name} | step={ckpt['global_step']} "
        f"epoch={ckpt['epoch']}"
    )
    return (
        int(ckpt["global_step"]),
        int(ckpt["epoch"] + 1),
        ckpt["meta"].get("best_metric"),
    )
