# TinyNiteSTR — Tiny GPT/HyenaLM Trainer (EN/日本語)

A lightweight, hackable training and generation playground featuring a tiny GPT‑2–style Transformer and a Hyena‑style language model, optional LoRA adapters, simple BPE tokenizer training, memory‑mapped datasets, checkpointing, and text generation.

---

## English

### Features
- TinyGPT2 (Transformer) model (`model.py`)
- HyenaLM (depthwise causal conv + MLP) (`Tynigptkarahyena.py`) — GPU‑only trainer in `mainh.py`
- LoRA for efficient finetuning on linear layers (`lora.py`)
- Byte‑BPE tokenizer training and use (`tokenizer.py`)
- Fast data path via memmaps (`data.py` + `data_fast.py`)
- Checkpoint save/load + best/latest tracking (`checkpoint.py`)

### Requirements
- Windows 11, Python 3.12+ (3.12 recommended for GPU; 3.13 is CPU‑only for PyTorch as of now)
- For GPU training: NVIDIA GPU + recent driver (CUDA 12.x compatible)

### Setup
```powershell
python -m venv .venv
. .venv\Scripts\Activate.ps1
pip install -r requirements.txt  # base deps (torch installed separately)
pip install -r requirements-dev.txt  # optional tools
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

Optional helper script for GPU setup (Python 3.12 required):
```powershell
powershell -ExecutionPolicy Bypass -File scripts\setup_cuda121.ps1 -Py312Path "C:\\Python312\\python.exe"
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
- `model.py` — TinyGPT2 (Transformer)
- `Tynigptkarahyena.py` — HyenaLM (causal depthwise conv + MLP)
- `optimizer.py` — optimizers (e.g., Lion)
- `data.py`, `data_fast.py` — preprocessing, memmap datasets, collate
- `tokenizer.py` — Byte‑BPE training + encode/decode
- `lora.py` — LoRA modules and mapping helpers
- `checkpoint.py` — save/resume utilities
- `main.py`, `main2.py`, `mainh.py` — training entry points (HyenaLM is GPU‑only)
- `generate.py` — text generation
- `checkpoints*/` — saved weights; `tokenizer.json` — tokenizer

### Notes
- Hyena trainer (`mainh.py`) requires CUDA; it will raise if no GPU is available. It also sets `PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128,expandable_segments:True` to reduce fragmentation.
- LoRA applies to linear layers. Convolution layers in Hyena remain unaffected, which is fine.
- If you encounter CUDA OOM, reduce `BATCH_SIZE` and/or `WINDOW`, enable gradient checkpointing (already enabled by default), and increase `ACCUM_STEPS` to maintain effective batch size.

---

## 日本語

### 特長
- TinyGPT2（Transformer）: `model.py`
- HyenaLM（因果 depthwise 畳み込み + MLP）: `Tynigptkarahyena.py`（`mainh.py` は GPU 専用）
- LoRA による軽量微調整: `lora.py`
- Byte‑BPE トークナイザの学習と利用: `tokenizer.py`
- メモリマップによる高速データパス: `data.py` / `data_fast.py`
- チェックポイント保存/復元（最新/ベスト）: `checkpoint.py`

### 必要環境
- Windows 11, Python 3.12 以上（GPU 利用は 3.12 推奨。3.13 は現状 CPU のみ）
- GPU 学習には NVIDIA GPU と最新ドライバ（CUDA 12.x 互換）が必要

### セットアップ
```powershell
python -m venv .venv
. .venv\Scripts\Activate.ps1
pip install -r requirements.txt  # ベース依存（torch は別途）
pip install -r requirements-dev.txt  # 任意
```

PyTorch のインストール（いずれか）
- GPU（CUDA 12.1 / Python 3.12）
```powershell
pip install --index-url https://download.pytorch.org/whl/cu121 torch
```
- CPU のみ
```powershell
pip install torch
```

スクリプトで GPU 環境を準備（Python 3.12 必須）
```powershell
powershell -ExecutionPolicy Bypass -File scripts\setup_cuda121.ps1 -Py312Path "C:\\Python312\\python.exe"
```

### トークナイザ
```powershell
python tokenizer.py  # tokenizer.json を生成
```

### 学習
`config.py`（特に `CORPUS`, `WINDOW`）を調整してから実行します。
```powershell
# TinyGPT2（CPU/GPU）
python main.py

# HyenaLM（GPU 専用）
python mainh.py
```

### 生成
```powershell
python generate.py  # 最新/ベストのチェックポイントを使用（必要ならパス調整）
```

### 構成
- `model.py` / `Tynigptkarahyena.py`（Hyena） / `lora.py`
- `data.py` / `data_fast.py`（前処理・メモリマップ Dataset）
- `checkpoint.py` / `optimizer.py`
- `main.py` / `main2.py` / `mainh.py`（Hyena は GPU 専用） / `generate.py`
- `checkpoints*/`（保存先）/ `tokenizer.json`（トークナイザ）

### メモ
- `mainh.py` は CUDA 前提です（GPU 未検出時はエラー）。`PYTORCH_CUDA_ALLOC_CONF` を自動設定して断片化を軽減します。
- CUDA OOM の場合は `BATCH_SIZE`/`WINDOW` を下げ、`ACCUM_STEPS` を増やしてください。

