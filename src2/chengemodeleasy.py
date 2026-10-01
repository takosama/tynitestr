"""Export an inference bundle; never save or load executable model pickles."""

import argparse
import hashlib
import json
from pathlib import Path

import torch
from model_io import FORMAT_VERSION, _validate_config, atomic_save, load_model_bundle


def convert_checkpoint_to_executable_model(
    pt_path,
    out_path=None,
    force_vocab_size=None,
    *,
    model_config=None,
    tokenizer_path=None
):
    """Compatibility entry point returning a safe state-dict bundle path.

    Legacy tensor dictionaries require an explicit architecture config. Full-model
    pickles are rejected by weights_only=True; there is no unsafe fallback.
    """
    if force_vocab_size is not None:
        raise ValueError(
            "Partial vocab copying is unsupported; tokenizer and weights must match"
        )
    checkpoint = torch.load(pt_path, map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, dict) or not isinstance(
        checkpoint.get("model"), dict
    ):
        raise ValueError("Expected a tensor checkpoint dictionary, not a model object")
    config = model_config or checkpoint.get("model_config")
    _validate_config(config)
    state = checkpoint["model"]
    # DataParallel/compile prefixes from older tensor-only checkpoints.
    normalized = {}
    for key, tensor in state.items():
        while key.startswith(("module.", "_orig_mod.")):
            key = key.split(".", 1)[1]
        if key in normalized:
            raise ValueError("duplicate normalized state key")
        normalized[key] = tensor
    meta = dict(checkpoint.get("meta", {}))
    if tokenizer_path is not None:
        meta["tokenizer_sha256"] = hashlib.sha256(
            Path(tokenizer_path).read_bytes()
        ).hexdigest()
    bundle = dict(
        format_version=FORMAT_VERSION, model_config=config, model=normalized, meta=meta
    )
    output = Path(out_path) if out_path else Path(pt_path).with_suffix(".inference.pt")
    if output.resolve() == Path(pt_path).resolve():
        raise ValueError("Export must not overwrite its input checkpoint")
    # Validate in an owned temporary file before replacing any destination.
    import tempfile

    with tempfile.TemporaryDirectory() as directory:
        candidate = Path(directory) / "check.pt"
        atomic_save(bundle, candidate)
        load_model_bundle(candidate, tokenizer_path=tokenizer_path)
    atomic_save(bundle, output)
    return str(output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint")
    parser.add_argument("--output")
    parser.add_argument(
        "--config", help="JSON model_config for legacy tensor dictionaries"
    )
    parser.add_argument("--tokenizer", required=True)
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text()) if args.config else None
    print(
        convert_checkpoint_to_executable_model(
            args.checkpoint,
            args.output,
            model_config=config,
            tokenizer_path=args.tokenizer,
        )
    )
