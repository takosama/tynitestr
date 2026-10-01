"""Versioned state-dict bundles. Never unpickle executable model objects."""

import hashlib
import os
import tempfile
from pathlib import Path

import torch

FORMAT_VERSION = 1


def unwrap_model(model):
    while hasattr(model, "module") or hasattr(model, "_orig_mod"):
        model = model.module if hasattr(model, "module") else model._orig_mod
    return model


def model_config(model):
    from lora import LoRALinear

    model = unwrap_model(model)
    config = dict(model.model_config)
    adapters = [
        (name, layer)
        for name, layer in model.named_modules()
        if isinstance(layer, LoRALinear)
    ]
    if adapters:
        first = adapters[0][1]
        if not first.enabled or any(
            not m.enabled
            or m.r != first.r
            or m.scaling != first.scaling
            or getattr(m.dropout, "p", 0.0) != getattr(first.dropout, "p", 0.0)
            for _, m in adapters
        ):
            raise ValueError("Mixed/merged LoRA adapters need explicit export first")
        config["lora"] = dict(
            r=first.r,
            alpha=first.scaling * first.r,
            dropout=getattr(first.dropout, "p", 0.0),
            target_lm_head=any(n.startswith("lm_head") for n, _ in adapters),
        )
    return config


def _validate_config(config):
    if not isinstance(config, dict):
        raise ValueError("model_config must be a dictionary")
    architecture = config.get("architecture")
    if architecture not in {"TinyGPT2", "HyenaLM"}:
        raise ValueError("unsupported model architecture")
    common = {
        "architecture",
        "vocab_size",
        "d_model",
        "n_layer",
        "block_size",
        "dropout",
        "lora",
    }
    allowed = common | ({"n_head"} if architecture == "TinyGPT2" else {"kernel_size"})
    required = allowed - {"lora"}
    if not required <= config.keys() or not config.keys() <= allowed:
        raise ValueError("missing or unexpected model configuration fields")
    for key, upper, lower in [
        ("vocab_size", 1_000_000, 1),
        ("d_model", 16384, 1),
        ("n_layer", 128, 0),
        ("block_size", 32768, 1),
        ("n_head", 256, 1),
        ("kernel_size", 32768, 1),
    ]:
        if key in config and (
            type(config[key]) is not int or not lower <= config[key] <= upper
        ):
            raise ValueError(f"invalid {key}")
    if (
        not isinstance(config["dropout"], (int, float))
        or not 0 <= config["dropout"] < 1
    ):
        raise ValueError("invalid dropout")
    if architecture == "TinyGPT2" and config["d_model"] % config["n_head"]:
        raise ValueError("d_model must be divisible by n_head")
    if "lora" in config:
        opts = config["lora"]
        if not isinstance(opts, dict) or set(opts) != {
            "r",
            "alpha",
            "dropout",
            "target_lm_head",
        }:
            raise ValueError("invalid LoRA configuration")
        if type(opts["r"]) is not int or not 1 <= opts["r"] <= 4096:
            raise ValueError("invalid LoRA rank")
        if type(opts["target_lm_head"]) is not bool:
            raise ValueError("invalid LoRA head flag")
        if (
            not isinstance(opts["alpha"], (int, float))
            or not 0 < opts["alpha"] <= 65536
        ):
            raise ValueError("invalid LoRA alpha")
        if (
            not isinstance(opts["dropout"], (int, float))
            or not 0 <= opts["dropout"] < 1
        ):
            raise ValueError("invalid LoRA dropout")
    return config


def _validate_primitives(value, depth=0):
    if depth > 32:
        raise ValueError("checkpoint structure too deep")
    if (
        value is None
        or type(value) in (str, int, float, bool)
        or isinstance(value, torch.Tensor)
    ):
        return
    if isinstance(value, dict):
        for key, item in value.items():
            if type(key) not in (str, int):
                raise ValueError("unsupported checkpoint dictionary key")
            _validate_primitives(item, depth + 1)
    elif isinstance(value, (list, tuple)):
        for item in value:
            _validate_primitives(item, depth + 1)
    else:
        raise ValueError("checkpoint may contain only tensors and primitive data")


def atomic_save(bundle, path):
    _validate_primitives(bundle)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            torch.save(bundle, stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp, path)
    finally:
        if os.path.exists(temp):
            os.unlink(temp)


def read_bundle(path):
    # Explicit True on every load; no retry with False and no custom safe globals.
    bundle = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(bundle, dict) or bundle.get("format_version") != FORMAT_VERSION:
        raise ValueError(
            "Use a versioned state-dict bundle; legacy full-model pickle is unsupported"
        )
    _validate_primitives(bundle)
    _validate_config(bundle.get("model_config"))
    state = bundle.get("model")
    if (
        not isinstance(state, dict)
        or not state
        or not all(
            isinstance(k, str)
            and isinstance(v, torch.Tensor)
            and v.layout == torch.strided
            and v.is_floating_point()
            for k, v in state.items()
        )
    ):
        raise ValueError("model must be a tensor state_dict")
    if not isinstance(bundle.get("meta", {}), dict):
        raise ValueError("invalid metadata")
    for tensor in state.values():
        if not torch.isfinite(tensor).all():
            raise ValueError("non-finite model weights")
    return bundle


def load_model_bundle(path, device="cpu", tokenizer_path=None):
    from hyena import HyenaLM
    from lora import _apply_lora_to_model
    from model import TinyGPT2

    bundle = read_bundle(path)
    if tokenizer_path is not None:
        expected = bundle.get("meta", {}).get("tokenizer_sha256")
        actual = hashlib.sha256(Path(tokenizer_path).read_bytes()).hexdigest()
        if not isinstance(expected, str) or expected != actual:
            raise ValueError(
                "tokenizer hash missing or mismatched; export with the matching tokenizer"
            )
    config = dict(bundle["model_config"])
    architecture = config.pop("architecture")
    lora = config.pop("lora", None)
    # Validate key/shape compatibility without allocating a second full model.
    with torch.device("meta"):
        model = {"TinyGPT2": TinyGPT2, "HyenaLM": HyenaLM}[architecture](**config)
        if lora:
            _apply_lora_to_model(model, **lora)
    model.load_state_dict(bundle["model"], strict=True, assign=True)
    model._pos_ids = torch.arange(config["block_size"])
    return model.to(device).eval()
