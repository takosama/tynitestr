# Repository Guidelines

This document guides contributors and AI agents working in this repository. Follow these conventions to keep changes safe, minimal, and consistent.

## Scope & Precedence
- Scope: applies to the entire repo unless a more deeply nested AGENTS.md overrides it.
- Precedence: direct task instructions > nested AGENTS.md > this file.
- Intent: keep diffs focused; do not reformat unrelated files or change filenames/APIs unless required.

## Project Structure & Modules
- Core modules: `model.py` (TinyGPT2), `hyena.py` (HyenaLM), `optimizer.py` (training utils), `data.py` (dataset/token prep), `tokenizer.py` (vocab + encode/decode), `lora.py` (LoRA helpers), `checkpoint.py` (save/load), `config.py` (hyperparams).
- Entry points: `main.py` (TinyGPT2 training), `mainh.py` (HyenaLM training, GPU-only), `main2.py` (alt trainer), `createmodel.py` / `chengemodeleasy.py` (model init), `generate.py` (text generation), `ime.py` (IME/demo), `old/` (legacy).
- Artifacts: `checkpoints*/` (model weights), `tokenizer.json`, corpora files (e.g., `tweets_clean.txt`, `corpus_*.u*`). Avoid committing new large binaries.

## Environment & Setup (Windows)
- Python: 3.12 recommended for GPU (CUDA wheels); 3.13 typically CPU-only.
- Create venv: `python -m venv .venv && .venv\Scripts\Activate.ps1`
- Install deps: `pip install -r requirements.txt` and (optional) `pip install -r requirements-dev.txt`
- Install torch:
  - GPU (CUDA 12.1, Python 3.12): `pip install --index-url https://download.pytorch.org/whl/cu121 torch`
  - CPU only: `pip install torch`

## Common Workflows
- Tokenizer: `python tokenizer.py` (produces `tokenizer.json`). Set `FORCE_RETRAIN_TOKENIZER=True` in `config.py` to retrain.
- Train (TinyGPT2): `python main.py` (reads `config.py`, writes to `checkpoints/`).
- Train (HyenaLM, GPU-only): `python mainh.py` (raises if CUDA is unavailable).
- Generate: `python generate.py` (loads latest/best checkpoint; adjust path in script or `config.py`).
- Resume training: automatic via `try_resume` if checkpoints exist (`latest`, `best`).

## Paths & Artifacts
- Outputs default one level above `src` (e.g., `../checkpoints`, `../tokenizer.json`, `../corpus_tokens.u32`). Keep these paths consistent when editing training or generation code.
- Update `config.py` for local corpus location: `CORPUS`, `CSV_TEXT_COL`, `CSV_SEP`.
- Do not commit corpora, memmaps (`corpus_*.u*`), or checkpoints.

## Model & Training Notes
- LoRA: toggle via `USE_LORA` and tune `LORA_*` in `config.py`. LoRA applies to linear layers; Hyena’s conv layers are unaffected. When LoRA is enabled, optimizer excludes weight decay on LoRA params.
- Sequence length: controlled by `WINDOW`. Generation truncates seeds longer than model context.
- Memory controls: use `GRAD_CHECKPOINT=True`, reduce `BATCH_SIZE`, and increase `ACCUM_STEPS` to mitigate CUDA OOM.
- HyenaLM: `mainh.py` sets `PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128,expandable_segments:True` and disables TorchDynamo to avoid instability.

## Coding Style & Naming
- Python 3.12+/Windows-first; indentation 4 spaces; line length target 88.
- Names: modules `lower_snake_case`, classes `PascalCase`, functions/vars `snake_case`.
- Public functions should include type hints and short docstrings.
- Keep changes minimal; avoid broad refactors unless explicitly requested.

## Testing & Tooling
- Tests: `pytest -q` (add under `tests/`, name `test_*.py`).
- Coverage target (changed code): `pytest --cov=. --cov-report=term-missing` (≥80% where practical).
- Formatting/Linting: `black .`, `isort .`, `ruff check .` before pushing.

## Commit & PR Hygiene
- Commits: clear, scoped (prefer Conventional Commits, e.g., `feat: add LoRA adapter init`).
- PRs: small, focused; include description, linked issue, sample logs/metrics or generated snippets as relevant.
- No large binaries or private datasets in diffs.

## Security & Configuration
- Do not hardcode secrets; use environment variables.
- Persist key knobs in `config.py` for reproducibility; seed runs when practical.
- Keep checkpoint and generate paths consistent when editing training loops.

## Agent-Specific Notes
- Planning: use explicit plans for multi-step or ambiguous tasks; group related edits.
- Preambles: briefly describe upcoming grouped actions before running commands.
- File reads: prefer `rg` for search; read files in ≤250-line chunks.
- Patches: make surgical changes; do not reformat unrelated code; avoid renames unless necessary.
- Validation: where feasible, run the narrowest tests that exercise changed code; for GPU-only flows prefer CPU-friendly validation via `main.py` if CUDA is unavailable.
- HyenaLM is GPU-only: do not attempt to force-run `mainh.py` on CPU; switch to `main.py` for CPU.

## Troubleshooting
- CUDA OOM: lower `BATCH_SIZE`/`WINDOW`, keep `GRAD_CHECKPOINT=True`, increase `ACCUM_STEPS`.
- Token id mismatch: ensure `tokenizer.json` matches memmaps; retrain tokenizer (`FORCE_RETRAIN_TOKENIZER=True`) and rebuild memmaps.
- Slow data: increase `NUM_WORKERS` and `prefetch_factor` cautiously; Windows supports pinned memory via `PIN_MEMORY=True`.

---
Follow this guide’s scope across the repo and keep changes minimal and focused. When in doubt, prefer clarity, small diffs, and consistency with existing patterns.
