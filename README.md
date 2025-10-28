# TinyNiteSTR — Tiny GPT/HyenaLM Trainer (EN/日本語)

A lightweight, hackable training and generation playground featuring a tiny GPT‑style Transformer and a Hyena‑style language model, optional LoRA adapters, simple BPE tokenizer training, memory‑mapped datasets, checkpointing, and text generation.

---

## English

### Features
- TinyGPT2 (Transformer) model (`model.py`)
- HyenaLM (depthwise causal conv + MLP) (`hyena.py`) — GPU‑only trainer in `mainh.py`
- LoRA for efficient finetuning on linear layers (`lora.py`)
- Byte‑BPE tokenizer training and use (`tokenizer.py`)
- Fast data path via memmaps (`data.py` + `data_fast.py`)
- Checkpoint save/load + best/latest tracking (`checkpoint.py`)

### Requirements
- Windows 11, Python 3.12+ (3.12 recommended for GPU; 3.13 typically CPU‑only for PyTorch)
- For GPU training: NVIDIA GPU + recent driver (CUDA 12.x compatible)

### Setup
```powershell
python -m venv .venv
. .venv\Scripts\Activate.ps1
pip install -r requirements.txt         # base deps (install torch separately)
pip install -r requirements-dev.txt     # optional dev tools
```

Install PyTorch (choose one):
- GPU (CUDA 12.1, Python 3.12):
```powershell
pip install --index-url https://download.pytorch.org/whl/cu121 torch
```
- CPU only:
```powershell
pip install torch
```

### Tokenizer
```powershell
python tokenizer.py  # writes tokenizer.json
```

### Train
Edit `config.py` (paths, hyperparameters), especially `CORPUS` and `WINDOW`.
```powershell
# TinyGPT2 (CPU/GPU)
python main.py

# HyenaLM (GPU‑only)
python mainh.py
```
Checkpoints are written to `checkpoints/` with rotating latest and best.

### Generate
```powershell
python generate.py  # uses latest/best checkpoint; adjust paths in script or config.py
```

### Project Structure
- Core: `model.py` (TinyGPT2), `hyena.py` (HyenaLM), `optimizer.py`, `data.py`/`data_fast.py`, `tokenizer.py`, `lora.py`, `checkpoint.py`, `config.py`
- Entry: `main.py` (TinyGPT2), `mainh.py` (HyenaLM, GPU‑only), `main2.py` (alt), `createmodel.py` / `chengemodeleasy.py` (init), `generate.py`, `ime.py`
- Artifacts: `checkpoints*/` (weights), `tokenizer.json`, corpora files (e.g., `corpus_*.u*`)

### Notes
- Hyena trainer (`mainh.py`) requires CUDA; it will raise if no GPU is available. It also sets `PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128,expandable_segments:True` to reduce fragmentation.
- LoRA applies to linear layers. Convolution layers in Hyena remain unaffected.
- If you encounter CUDA OOM, reduce `BATCH_SIZE` and/or `WINDOW`, keep gradient checkpointing on, and increase `ACCUM_STEPS` to maintain effective batch size.

### Development
- Style: 4‑space indent; keep functions/vars `snake_case`, classes `PascalCase`.
- Format/lint: `black .`, `isort .`, `ruff check .`
- Tests: `pytest -q` or with coverage `pytest --cov=. --cov-report=term-missing`
- Do not commit large binaries. Keep checkpoint/generate paths consistent with `config.py`.

---

## 日本語

### 特徴
- TinyGPT2（Transformer）モデル（`model.py`）
- HyenaLM（因果的な深さ方向畳み込み + MLP）（`hyena.py`）— 学習エントリは `mainh.py`（GPU 専用）
- 線形層に対する軽量な微調整 LoRA（`lora.py`）
- Byte‑BPE トークナイザの学習と利用（`tokenizer.py`）
- メモリマップによる高速データ処理（`data.py` + `data_fast.py`）
- チェックポイント保存/復元（最新/ベストの管理）（`checkpoint.py`）

### 要件
- Windows 11, Python 3.12 以上（GPU を使う場合は 3.12 推奨。3.13 は多くの環境で CPU のみ）
- GPU 学習には NVIDIA GPU と CUDA 12.x 互換のドライバ

### セットアップ
```powershell
python -m venv .venv
. .venv\Scripts\Activate.ps1
pip install -r requirements.txt         # 基本依存（torch は別途）
pip install -r requirements-dev.txt     # 開発用ツール（任意）
```

PyTorch のインストール（いずれか）:
- GPU（CUDA 12.1 / Python 3.12）
```powershell
pip install --index-url https://download.pytorch.org/whl/cu121 torch
```
- CPU のみ
```powershell
pip install torch
```

### トークナイザ
```powershell
python tokenizer.py  # tokenizer.json を作成
```

### 学習
`config.py` のパスやハイパーパラメータ（特に `CORPUS`, `WINDOW`）を調整してから実行します。
```powershell
# TinyGPT2（CPU/GPU）
python main.py

# HyenaLM（GPU 専用）
python mainh.py
```
チェックポイントは `checkpoints/` に保存され、最新/ベストがローテーション管理されます。

### 生成
```powershell
python generate.py  # 最新/ベストのチェックポイントを使用（必要に応じてパスを調整）
```

### プロジェクト構成
- コア: `model.py`（TinyGPT2）, `hyena.py`（HyenaLM）, `optimizer.py`, `data.py`/`data_fast.py`, `tokenizer.py`, `lora.py`, `checkpoint.py`, `config.py`
- エントリ: `main.py`（TinyGPT2）, `mainh.py`（HyenaLM, GPU 専用）, `main2.py`（代替）, `createmodel.py` / `chengemodeleasy.py`（初期化）, `generate.py`, `ime.py`
- 生成物: `checkpoints*/`（重み）, `tokenizer.json`, コーパス関連ファイル（例: `corpus_*.u*`）

### 補足
- `mainh.py` は CUDA 前提です（GPU が無い場合はエラー）。`PYTORCH_CUDA_ALLOC_CONF` を設定してメモリ断片化を抑えます。
- LoRA は線形層にのみ適用されます（Hyena の畳み込み層は対象外）。
- CUDA のメモリ不足（OOM）の場合は `BATCH_SIZE` や `WINDOW` を下げ、勾配チェックポイントを有効のまま、`ACCUM_STEPS` を増やして実効バッチを保ちます。

### 開発メモ
- コーディング規約: インデント 4 スペース、`snake_case` / `PascalCase` を遵守。
- 整形/静的解析: `black .`、`isort .`、`ruff check .`
- テスト: `pytest -q`、カバレッジは `pytest --cov=. --cov-report=term-missing`
- 大きなバイナリのコミットは避け、`config.py` と生成スクリプトのパス整合性を維持してください。

