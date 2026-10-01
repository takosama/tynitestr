import pickle
from pathlib import Path

import checkpoint
import pytest
import torch
from generate import generate_text, preview_text
from hyena import HyenaLM
from model import TinyGPT2
from model_io import atomic_save, load_model_bundle, model_config, read_bundle
from tokenizer import ByteBPETokenizer, train_bpe_from_text


class DummyTokenizer:
    bos_id = 1

    def encode(self, text):
        return [1, 2] if text else []

    def decode(self, ids):
        return ",".join(map(str, ids))


def tiny(kind=TinyGPT2):
    kwargs = dict(vocab_size=8, d_model=8, n_layer=1, block_size=8, dropout=0.0)
    kwargs.update(n_head=2) if kind is TinyGPT2 else kwargs.update(kernel_size=3)
    return kind(**kwargs)


@pytest.mark.parametrize("kind", [TinyGPT2, HyenaLM])
@pytest.mark.parametrize("temperature", [0.0, 1.0])
def test_generation_contract(kind, temperature):
    model = tiny(kind).eval()
    x = torch.tensor([[1, 2], [3, 4]])
    with torch.no_grad():
        full = model(x)
        last = model(x, last_only=True)
    assert full.shape == (2, 2, 8) and last.shape == (2, 8)
    torch.testing.assert_close(full[:, -1], last)
    result = generate_text(
        model,
        DummyTokenizer(),
        seed_text="x",
        max_new_tokens=1,
        temperature=temperature,
        top_k=3,
        top_p=0.9,
    )
    assert len(result.split(",")) == 3


@pytest.mark.parametrize("kind", [TinyGPT2, HyenaLM])
def test_empty_and_long_seed(kind):
    model = tiny(kind).eval()
    assert (
        len(
            generate_text(
                model, DummyTokenizer(), seed_text="", max_new_tokens=1
            ).split(",")
        )
        == 2
    )
    assert (
        len(
            generate_text(
                model, DummyTokenizer(), seed_ids=[1] * 20, max_new_tokens=2
            ).split(",")
        )
        == 10
    )


def test_preview_failure_restores_training(monkeypatch):
    model = tiny().train()
    monkeypatch.setattr(
        model, "forward", lambda *a, **kw: (_ for _ in ()).throw(TypeError("fixture"))
    )
    with pytest.warns(RuntimeWarning, match="Preview skipped"):
        assert (
            preview_text(model, DummyTokenizer(), seed_text="x", max_new_tokens=1)
            is None
        )
    assert model.training


@pytest.mark.parametrize("kind", [TinyGPT2, HyenaLM])
@pytest.mark.parametrize("use_lora", [False, True])
def test_safe_round_trip(tmp_path, kind, use_lora):
    from lora import _apply_lora_to_model

    model = tiny(kind).eval()
    if use_lora:
        _apply_lora_to_model(model, r=2, alpha=2, dropout=0, target_lm_head=True)
    path = tmp_path / "m.pt"
    atomic_save(
        dict(
            format_version=1,
            model_config=model_config(model),
            model=model.state_dict(),
            meta={},
        ),
        path,
    )
    loaded = load_model_bundle(path)
    x = torch.tensor([[1, 2]])
    torch.testing.assert_close(model(x), loaded(x))
    assert model_config(loaded) == model_config(model)


def test_weights_only_explicit_and_legacy_rejected(tmp_path, monkeypatch):
    original = torch.load
    calls = []

    def spy(*a, **kw):
        calls.append(kw.get("weights_only"))
        return original(*a, **kw)

    monkeypatch.setattr(torch, "load", spy)
    path = tmp_path / "old.pt"
    torch.save({"model": tiny().state_dict()}, path)
    with pytest.raises(ValueError, match="versioned"):
        read_bundle(path)
    assert calls == [True]
    # A benign built-in model object is refused; never execute an attack pickle.
    torch.save(torch.nn.Linear(1, 1), path)
    with pytest.raises(pickle.UnpicklingError):
        read_bundle(path)
    assert calls == [True, True]


def test_shape_and_tokenizer_mismatch_fail(tmp_path):
    m = tiny()
    path = tmp_path / "m.pt"
    tok = tmp_path / "tokenizer.json"
    tok.write_text("{}")
    b = dict(
        format_version=1,
        model_config=model_config(m),
        model=m.state_dict(),
        meta={"tokenizer_sha256": "wrong"},
    )
    atomic_save(b, path)
    with pytest.raises(ValueError, match="tokenizer hash"):
        load_model_bundle(path, tokenizer_path=tok)
    b["model"]["wte.weight"] = torch.zeros(1, 8)
    atomic_save(b, path)
    with pytest.raises(RuntimeError):
        load_model_bundle(path)


def test_atomic_failure_keeps_previous_checkpoint(tmp_path, monkeypatch):
    p = tmp_path / "m.pt"
    p.write_bytes(b"old fixture")

    def fail(obj, stream):
        stream.write(b"partial")
        raise OSError("fixture disk full")

    monkeypatch.setattr(torch, "save", fail)
    with pytest.raises(OSError):
        atomic_save({"model": torch.zeros(1)}, p)
    assert p.read_bytes() == b"old fixture"
    assert list(tmp_path.iterdir()) == [p]


def test_checkpoint_and_resume(tmp_path, monkeypatch):
    monkeypatch.setattr(checkpoint, "CKPT_DIR", tmp_path)
    model = tiny()
    opt = torch.optim.AdamW(model.parameters())
    path = checkpoint.save_checkpoint(
        "latest", model, opt, None, 2, 1, {"tokenizer_sha256": "fixture"}
    )
    assert path.exists()
    other = tiny()
    other_opt = torch.optim.AdamW(other.parameters())
    assert checkpoint.try_resume(other, other_opt, None, "fixture") == (2, 1, None)
    for a, b in zip(model.parameters(), other.parameters()):
        torch.testing.assert_close(a, b)


def test_committed_tokenizer_and_training_roundtrip(tmp_path):
    committed = Path(__file__).resolve().parents[1] / "tokenizer.json"
    tok = ByteBPETokenizer(committed)
    text = "こんにちは、猫です 🐈\nHello!"
    ids = tok.encode(text)
    assert tok.special["<eos>"] not in ids
    assert tok.decode(ids) == text
    path = train_bpe_from_text(text, 300, tmp_path / "tokenizer.json")
    trained = ByteBPETokenizer(path)
    assert trained.decode(trained.encode(text)) == text


def test_legacy_tensor_bundle_explicit_export(tmp_path):
    from chengemodeleasy import convert_checkpoint_to_executable_model

    model = tiny()
    source = tmp_path / "old.pt"
    torch.save({"model": model.state_dict(), "meta": {}}, source)
    with pytest.raises(ValueError):
        convert_checkpoint_to_executable_model(source)
    path = convert_checkpoint_to_executable_model(
        source, model_config=model_config(model)
    )
    loaded = load_model_bundle(path)
    torch.testing.assert_close(
        model.eval()(torch.tensor([[1]])), loaded(torch.tensor([[1]]))
    )


@pytest.mark.parametrize("interrupted", [False, True])
def test_short_trainer_saves_before_final_preview_and_on_interrupt(
    tmp_path, monkeypatch, interrupted
):
    import generate
    import main as trainer
    import numpy as np

    tok_path = train_bpe_from_text("こんにちは abc", 300, tmp_path / "tokenizer.json")
    tokens = tmp_path / "tokens.u32"
    np.array([1, 2, 3, 1], dtype=np.uint32).tofile(tokens)
    monkeypatch.setattr(trainer, "TOKENIZER_JSON", tok_path)
    monkeypatch.setattr(trainer, "TOK_BIN", tokens)
    monkeypatch.setattr(trainer, "FORCE_RETRAIN_TOKENIZER", False)
    monkeypatch.setattr(trainer, "preprocess_corpus", lambda x: x)
    monkeypatch.setattr(trainer, "build_memmap_tokens", lambda *a, **kw: None)
    monkeypatch.setattr(trainer, "WINDOW", 8)
    monkeypatch.setattr(trainer, "EPOCHS", 1)
    monkeypatch.setattr(trainer, "ACCUM_STEPS", 1)
    monkeypatch.setattr(trainer, "GRAD_CHECKPOINT", False)
    monkeypatch.setattr(trainer, "USE_LORA", False)
    monkeypatch.setattr(
        trainer, "TinyGPT2", lambda **kw: TinyGPT2(d_model=8, n_layer=1, n_head=2, **kw)
    )
    monkeypatch.setattr(trainer, "try_resume", lambda *a, **kw: (199, 1, None))
    monkeypatch.setattr(checkpoint, "CKPT_DIR", tmp_path / "checkpoints")
    batch = (torch.tensor([[1, 2]]), torch.tensor([[2, 3]]))

    class Loader:
        def __len__(self):
            return 2

        def __iter__(self):
            yield batch
            if interrupted:
                raise KeyboardInterrupt()

    monkeypatch.setattr(trainer, "_make_dataloader", lambda: Loader())
    monkeypatch.setattr(
        generate,
        "generate_text",
        lambda *a, **kw: (_ for _ in ()).throw(ValueError("preview fixture")),
    )
    with pytest.warns(RuntimeWarning, match="Preview skipped"):
        if interrupted:
            with pytest.raises(KeyboardInterrupt):
                trainer.main()
        else:
            trainer.main()
    path = next((tmp_path / "checkpoints").glob("*latest*.pt"))
    bundle = read_bundle(path)
    assert bundle["global_step"] == 200
    assert bundle["meta"]["epoch_complete"] is not interrupted
    assert len(bundle["meta"]["tokenizer_sha256"]) == 64
    loaded = load_model_bundle(path, tokenizer_path=tok_path)
    assert torch.isfinite(loaded(torch.tensor([[1, 2]]))).all()


def test_memmap_refuses_unverified_legacy_cache(tmp_path):
    import data

    path = train_bpe_from_text("abc", 300, tmp_path / "tokenizer.json")
    tok = ByteBPETokenizer(path)
    corpus = tmp_path / "corpus.txt"
    corpus.write_text("abc")
    tokens = tmp_path / "tokens.u32"
    offsets = tmp_path / "offsets.u64"
    tokens.write_bytes(b"legacy")
    with pytest.raises(ValueError, match="Legacy/incomplete"):
        data.build_memmap_tokens(corpus, tok, tokens, offsets)
    assert tokens.read_bytes() == b"legacy"


@pytest.mark.parametrize(
    "change, message",
    [
        ({"architecture": "UntrustedClass"}, "architecture"),
        ({"d_model": -1}, "d_model"),
        ({"n_head": 3}, "divisible"),
        ({"extra": "field"}, "configuration"),
    ],
)
def test_invalid_architecture_config_rejected(tmp_path, change, message):
    m = tiny()
    config = model_config(m)
    config.update(change)
    p = tmp_path / "bad.pt"
    atomic_save(
        {"format_version": 1, "model_config": config, "model": m.state_dict()}, p
    )
    with pytest.raises(ValueError, match=message):
        load_model_bundle(p)


def test_nonfinite_weights_and_custom_metadata_rejected(tmp_path):
    m = tiny()
    state = m.state_dict()
    state["wte.weight"][0, 0] = float("nan")
    p = tmp_path / "bad.pt"
    atomic_save(
        {"format_version": 1, "model_config": model_config(m), "model": state}, p
    )
    with pytest.raises(ValueError, match="non-finite"):
        read_bundle(p)
    with pytest.raises(ValueError, match="primitive"):
        atomic_save({"bad": object()}, p)


def test_ime_loads_cpu_bundle_with_matching_tokenizer(tmp_path):
    import ime

    tokenizer_path = train_bpe_from_text("こんにちは", 300, tmp_path / "tokenizer.json")
    tok = ByteBPETokenizer(tokenizer_path)
    model = TinyGPT2(
        vocab_size=max(tok.vocab.values()) + 1,
        d_model=8,
        n_head=2,
        n_layer=1,
        block_size=8,
    )
    import hashlib

    bundle = {
        "format_version": 1,
        "model_config": model_config(model),
        "model": model.state_dict(),
        "meta": {
            "tokenizer_sha256": hashlib.sha256(tokenizer_path.read_bytes()).hexdigest()
        },
    }
    p = tmp_path / "model.pt"
    atomic_save(bundle, p)
    loaded, tokenizer = ime._load_model_and_tokenizer(p, tokenizer_path, device="cpu")
    assert next(loaded.parameters()).device.type == "cpu"
    assert tokenizer.decode(tokenizer.encode("こんにちは")) == "こんにちは"


def test_corpus_reading_and_configured_tokenizer_output(tmp_path, monkeypatch):
    import config
    from tokenizer import load_corpus_text

    text = tmp_path / "corpus.txt"
    text.write_text("abc\n日本語", encoding="utf-8")
    assert load_corpus_text(text) == "abc\n日本語"
    csv = tmp_path / "corpus.csv"
    csv.write_text("text,other\nhello,x\n日本語,y\n", encoding="utf-8")
    assert load_corpus_text(csv) == "hello\n日本語"
    bad = tmp_path / "bad.csv"
    bad.write_text("other\nmissing\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Missing text column"):
        load_corpus_text(bad)
    output = tmp_path / "tokenizer.json"
    monkeypatch.setattr(config, "TOKENIZER_JSON", output)
    assert train_bpe_from_text("hello 日本語", 300) == output


def test_parallel_and_compile_wrappers_use_plain_state_keys(tmp_path, monkeypatch):
    from model_io import unwrap_model

    class CompileWrapper(torch.nn.Module):
        def __init__(self, model):
            super().__init__()
            self._orig_mod = model

    model = tiny().eval()
    wrapped = CompileWrapper(torch.nn.DataParallel(model))
    assert unwrap_model(wrapped) is model
    monkeypatch.setattr(checkpoint, "CKPT_DIR", tmp_path)
    path = checkpoint.save_checkpoint("latest", wrapped, None, None, 1, 1, {})
    loaded = load_model_bundle(path)
    assert set(read_bundle(path)["model"]) == set(model.state_dict())
    torch.testing.assert_close(model(torch.tensor([[1]])), loaded(torch.tensor([[1]])))
