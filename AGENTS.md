# Repository Guidelines

## Project Structure & Module Organization
- Core modules: `model.py` (TinyGPT2), `Tynigptkarahyena.py` (HyenaLM), `optimizer.py` (training utils), `data.py` (dataset/token prep), `tokenizer.py` (vocab + encode/decode), `lora.py` (LoRA helpers), `checkpoint.py` (save/load), `config.py` (hyperparams).
- Entry points: `main.py` (TinyGPT2 training), `mainh.py` (HyenaLM training, GPU-only), `main2.py` (alt trainer), `createmodel.py` or `chengemodeleasy.py` (model init), `generate.py` (text generation), `ime.py` (IME/demo), `old/` (legacy).
- Artifacts: `checkpoints*/` (model weights), `tokenizer.json`, corpora files (e.g., `tweets_clean.txt`, `corpus_*.u*`). Avoid committing new large binaries.

## Build, Test, and Development Commands
- Create venv (Windows): `python -m venv .venv && .venv\Scripts\Activate.ps1`
- Base deps: `pip install -r requirements.txt`; dev tools: `pip install -r requirements-dev.txt`.
- Torch install:
  - GPU (CUDA 12.1, Python 3.12): `pip install --index-url https://download.pytorch.org/whl/cu121 torch`
  - CPU only: `pip install torch`
- Train TinyGPT2: `python main.py` (reads `config.py`, writes to `checkpoints/`).
- Train HyenaLM (GPU-only): `python mainh.py`.
- Generate: `python generate.py` (loads latest/best checkpoint; edit path in script or `config.py`).
- Tokenizer: `python tokenizer.py` (produces `tokenizer.json`).

## Coding Style & Naming Conventions
- Python: 3.12 recommended for GPU (PyTorch CUDA wheels); 3.13 works for CPU-only.
- Indentation: 4 spaces; line length 88.
- Names: modules `lower_snake_case`, classes `PascalCase`, functions/vars `snake_case`.
- Type hints and docstrings for public functions.
- Tools: format with `black .` and `isort .`; lint with `ruff check .` before pushing.

## Testing Guidelines
- Framework: `pytest`.
- Location: add tests under `tests/`.
- Naming: files `test_*.py`; functions `test_*`.
- Run tests: `pytest -q`.
- Coverage (target ≥80% for changed code): `pytest --cov=. --cov-report=term-missing`.

## Commit & Pull Request Guidelines
- Commits: clear, scoped messages (prefer Conventional Commits, e.g., `feat: add LoRA adapter init`).
- PRs: small, focused; include description, linked issue, and sample logs/metrics (loss/val) or generated snippets when relevant.
- CI hygiene: code formatted/linted; tests passing; no large binaries in diffs.

## Security & Configuration Tips
- Secrets: do not hardcode keys or tokens; use env vars.
- Checkpoints: store locally; if sharing, upload externally and reference paths in `config.py`.
- Repro: persist key training knobs in `config.py`; seed runs for determinism when possible.

## Agent-Specific Notes
- Follow this file’s scope across the repo; keep changes minimal and focused.
- When editing training loops, ensure checkpoint and generate paths remain consistent.
- `mainh.py` assumes CUDA is available and will raise if not. Prefer testing changes on a CUDA-enabled environment or switch to `main.py` for CPU.
