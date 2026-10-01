from pathlib import Path

from config import CKPT_DIR, KEEP_LAST, RUN_ID
from model_io import (
    FORMAT_VERSION,
    atomic_save,
    model_config,
    read_bundle,
    unwrap_model,
)


def save_checkpoint(tag, model, opt, scaler, global_step, epoch, meta) -> Path:
    model = unwrap_model(model)
    bundle = dict(
        format_version=FORMAT_VERSION,
        model_config=model_config(model),
        tag=tag,
        global_step=global_step,
        epoch=epoch,
        model=model.state_dict(),
        optimizer=opt.state_dict() if opt else None,
        scaler=scaler.state_dict() if scaler else None,
        meta=dict(meta),
    )
    path = CKPT_DIR / f"{RUN_ID}_{tag}_{global_step:09d}.pt"
    atomic_save(bundle, path)
    # Only prune successfully replaced checkpoints from this run.
    if tag == "latest" and KEEP_LAST > 0:
        for old in sorted(CKPT_DIR.glob(f"{RUN_ID}_latest_*.pt"))[:-KEEP_LAST]:
            old.unlink(missing_ok=True)
    return path


def try_resume(model, opt, scaler, tokenizer_sha256=None):
    candidates = sorted(CKPT_DIR.glob("*_latest_*.pt"))
    if not candidates:
        return 0, 1, None
    path = candidates[-1]
    bundle = read_bundle(path)
    model = unwrap_model(model)
    if bundle["model_config"] != model_config(model):
        raise ValueError(
            "Checkpoint architecture/LoRA configuration differs from this run"
        )
    if (
        tokenizer_sha256 is not None
        and bundle.get("meta", {}).get("tokenizer_sha256") != tokenizer_sha256
    ):
        raise ValueError("Checkpoint tokenizer differs from this run")
    model.load_state_dict(bundle["model"], strict=True)
    if opt is not None and bundle.get("optimizer") is not None:
        opt.load_state_dict(bundle["optimizer"])
    if scaler is not None and bundle.get("scaler") is not None:
        scaler.load_state_dict(bundle["scaler"])
    epoch = int(bundle["epoch"]) + bool(
        bundle.get("meta", {}).get("epoch_complete", False)
    )
    print(f'Resumed from {path.name} at step {bundle["global_step"]}')
    return int(bundle["global_step"]), epoch, bundle.get("meta", {}).get("best_metric")
