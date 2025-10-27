from torch import nn
import torch

from config import CKPT_DIR, KEEP_LAST, RUN_ID, USE_LORA
from lora import _remap_old_linear_keys_to_lora

def _model_state(m: nn.Module):
    return m.module.state_dict() if hasattr(m, "module") else m.state_dict()
def _load_model_state(m: nn.Module, sd, strict: bool = True):
    (m.module if hasattr(m, "module") else m).load_state_dict(sd, strict=strict)

def save_checkpoint(tag:str, model, opt, scaler, global_step:int, epoch:int, meta:dict):
    ckpt = {
        "tag": tag,
        "global_step": global_step,
        "epoch": epoch,
        "model": _model_state(model),
        "optimizer": opt.state_dict() if opt is not None else None,
        "scaler": (scaler.state_dict() if scaler is not None else None),
        "meta": meta,
    }
    path = CKPT_DIR / f"{RUN_ID}_{tag}_{global_step:09d}.pt"
    torch.save(ckpt, path)
    # 世代整理（latest系のみ対象）
    if tag == "latest":
        latest = sorted(CKPT_DIR.glob(f"{RUN_ID}_latest_*.pt"))
        for p in latest[:-KEEP_LAST]:
            p.unlink(missing_ok=True)
    return path
# 1) state_dict ロードの可視化
def _debug_report_load(ret):
    try:
        print(f"▶ load_state: missing={len(ret.missing_keys)} unexpected={len(ret.unexpected_keys)}")
        if ret.missing_keys[:5]: print("  missing(head):", ret.missing_keys[:5])
        if ret.unexpected_keys[:5]: print("  unexpected(head):", ret.unexpected_keys[:5])
    except Exception:
        pass


def _strip_module_prefix(sd: dict):
    # DataParallel で保存された ckpt の "module." を剥がす
    if any(k.startswith("module.") for k in sd.keys()):
        return {k.replace("module.", "", 1): v for k, v in sd.items()} 
    return sd
def try_resume(model, opt, scaler):
    cands = sorted(CKPT_DIR.glob("*_latest_*.pt"))
    if not cands:
        return 0, 1, None
    path = cands[-1]
    ckpt = torch.load(path, map_location="cpu")
    sd = _strip_module_prefix(ckpt["model"])  # ★ここ追加


    # ---- LoRAの有無を自動吸収 ----
    has_lora_in_ckpt = any(".lora_A." in k or ".lora_B." in k or ".base." in k for k in sd.keys())
    if USE_LORA and not has_lora_in_ckpt:
        # 旧(非LoRA) → 新(LoRA) へキー名を移し替え
        sd = _remap_old_linear_keys_to_lora(sd, model)
        _load_model_state(model, sd, strict=False)
    elif (not USE_LORA) and has_lora_in_ckpt:
        # 新(LoRA) ckpt を 非LoRAモデルに読む → 余剰キーを無視
        _load_model_state(model, sd, strict=False)
    else:
        # 同一フォーマット同士
        _load_model_state(model, sd, strict=False)
# try_resumeの load_model_state 呼び出し後に:
    ret = (model.module if hasattr(model, "module") else model).load_state_dict(sd, strict=False)
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

    print(f"🔁 Resumed from: {path.name} | step={ckpt['global_step']} epoch={ckpt['epoch']}")
    return int(ckpt["global_step"]), int(ckpt["epoch"] + 1), ckpt["meta"].get("best_metric")
