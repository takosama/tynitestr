# TinyNiteSTR — Tiny GPT-2 Style Trainer (EN/日本語)

A lightweight, hackable training and generation playground for a tiny GPT‑2–style model with optional LoRA adapters. Includes simple BPE tokenizer training, dataset preprocessing to memory‑mapped arrays, checkpointing, and text generation utilities.

---

## English

### Features
- Tiny GPT‑2–style model with attention layers (`model.py`)
- LoRA support for efficient finetuning (`lora.py`)
- BPE tokenizer training and use (`tokenizer.py`)
- Data preprocessing to memmaps for fast training (`data.py`)
- Checkpoint save/load and best/latest tracking (`checkpoint.py`)

### Quickstart
1) Environment
```powershell
python -m venv .venv
. .venv\Scripts\Activate.ps1
pip install -r requirements.txt
pip install -r requirements-dev.txt  # optional
```

2) Tokenizer (if needed)
```powershell
python tokenizer.py  # writes tokenizer.json
```

3) Prepare data and train
```powershell
# Edit config in config.py (paths, hyperparameters)
python main.py  # or python main2.py
```
Checkpoints are created under `checkpoints*/` with best/latest artifacts.

4) Generate text
```powershell
python generate.py  # uses latest/best checkpoint; adjust paths if needed
```

### Quick Example
```python
import torch
from model import TinyGPT2
from tokenizer import ByteBPETokenizer
from checkpoint import try_resume
from config import VOCAB_SIZE
from generate import generate_text

device = "cuda" if torch.cuda.is_available() else "cpu"
tok = ByteBPETokenizer("tokenizer.json")
model = TinyGPT2(vocab_size=VOCAB_SIZE).to(device).eval()
try_resume(model, opt=None, scaler=None)  # load latest checkpoint if present

out = generate_text(
    model, tok, seed_text="Hello, tiny model!", max_new_tokens=80,
    temperature=0.8, top_p=0.9,
)
print(out)
```

### Project Structure
- `model.py` — Tiny GPT‑2 model
- `optimizer.py` — custom optimizers (e.g., Lion)
- `data.py` — corpus preprocessing and `Dataset`
- `tokenizer.py` — ByteBPE training and encode/decode
- `lora.py` — LoRA modules and mapping helpers
- `checkpoint.py` — save/resume utilities
- `main.py`/`main2.py` — training entry points
- `generate.py` — text generation
- `checkpoints*/` — saved weights
- `old/` — legacy experiments

### Tips
- Configure everything in `config.py` (run IDs, dirs, hyperparams).
- For CUDA-specific Torch builds, follow PyTorch install docs and pin `torch` accordingly.

---

## 日本語

### 概要
小さな GPT‑2 系モデルを手軽に学習・生成できる実験用リポジトリです。LoRA による軽量微調整、BPE トークナイザ学習、メモリマップ化したデータ前処理、チェックポイント管理、テキスト生成を備えています。

### はじめ方
1) 環境構築
```powershell
python -m venv .venv
. .venv\Scripts\Activate.ps1
pip install -r requirements.txt
pip install -r requirements-dev.txt  # 任意
```

2) トークナイザ（必要な場合）
```powershell
python tokenizer.py  # tokenizer.json を出力
```

3) 前処理と学習
```powershell
# config.py を編集（パスやハイパーパラメータ）
python main.py  # もしくは python main2.py
```
チェックポイントは `checkpoints*/` に best/latest として保存されます。

4) 生成
```powershell
python generate.py  # 最新/ベストのチェックポイントを使用（必要に応じてパス調整）
```

### 簡単な例
```python
import torch
from model import TinyGPT2
from tokenizer import ByteBPETokenizer
from checkpoint import try_resume
from config import VOCAB_SIZE
from generate import generate_text

device = "cuda" if torch.cuda.is_available() else "cpu"
tok = ByteBPETokenizer("tokenizer.json")
model = TinyGPT2(vocab_size=VOCAB_SIZE).to(device).eval()
try_resume(model, opt=None, scaler=None)  # 可能なら最新チェックポイントを読込

out = generate_text(
    model, tok, seed_text="こんにちは、ちいさなモデル！", max_new_tokens=80,
    temperature=0.8, top_p=0.9,
)
print(out)
```

### 構成
- `model.py`（モデル）/ `lora.py`（LoRA）
- `data.py`（前処理・Dataset）/ `tokenizer.py`（BPE 学習）
- `checkpoint.py`（保存/復元）/ `optimizer.py`
- `main.py`・`main2.py`（学習）/ `generate.py`（生成）
- `checkpoints*/`（学習結果）/ `old/`（過去の実験）

### メモ
- 主要設定は `config.py` で管理します。
- CUDA 環境では PyTorch の公式手順に従い `torch` のビルドを調整してください。
