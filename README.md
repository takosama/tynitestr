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


## 安全性修正・移行手順 (2026-10)

### 新しいcheckpoint

保存はモデルのPythonオブジェクトではなく、`format_version=1`、許可された `model_config`、tensor `state_dict`、primitive metadataの辞書です。読込みは必ず `torch.load(..., weights_only=True, map_location="cpu")` を使い、危険な設定へ戻すfallbackはありません。最新のセキュリティ更新済みPyTorchを使ってください。安全形式でも未知の大きなモデルを無制限に読み込んでよいわけではありません。

- TinyGPT2 / HyenaLM と同一構成のLoRAを復元できます。shape/keyはstrict検証し、構成/語彙/LoRA設定の暗黙の変更や部分コピーをしません。
- IMEはCPU/CUDAを選択でき、重みに保存したtokenizer SHA-256の全64桁と使用するJSONを照合します。 旧checkpointの16桁省略hashはそのまま受理せず、正しいJSONを `--tokenizer` に指定して明示的にexportし直します。
- 保存は一時ファイルを完成させてから置換します。保存失敗時は以前の完成したcheckpointを残します。
- 学習プレビューが失敗しても警告を出して続行し、training/eval状態を復元します。短い学習の通常終了とCtrl-C時にも保存します。SIGKILL、電源断、ディスク故障時まで保存を保証しません。
- 再開では途中保存したepochを最初から再実行します。厳密なバッチ/RNG位置の再開は未対応です。

旧 `.model.pt`（モデル本体pickle）は読み込まないでください。元のtensor辞書checkpointまたは再学習から移行します。既存tensor辞書は `src2/chengemodeleasy.py` で明示的に変換できます。旧データからアーキテクチャを推測しないため、旧checkpointには正しい構成JSONが必要です。下記は構成の例であり、実モデルの値へ合わせてください。

```json
{"architecture":"TinyGPT2","vocab_size":30000,"d_model":768,"n_layer":24,"n_head":16,"block_size":64,"dropout":0.0}
```

```sh
python src2/chengemodeleasy.py path/to/tensor_checkpoint.pt --config model_config.json --tokenizer tokenizer.json --output model.inference.pt
```

生成物はモデルオブジェクトではなく安全なbundleです。既存コードで `torch.load(path)` をそのままモデルとして使う代わりに `model_io.load_model_bundle(...)` を使用してください。元のcheckpointは上書きしません。LoRAでは `lora` の `r`, `alpha`, `dropout`, `target_lm_head` も明示してください。旧全モデルpickleしか残っていない場合、自動変換はできません。

### tokenizerとキャッシュ

欠落していた `src2/tokenizer.py` を、既存のHugging Face ByteLevel BPE JSONに対応するadapterとして追加しました。`encode` はBOS/EOSを自動追加せず、データ作成側がEOSを追加します。既存JSONにdecoderがない場合、ファイルを改変せずメモリ内でByteLevel decoderを補います。日本語・改行・絵文字を往復確認しました。

以前の欠落した独自tokenizer実装と完全に同じencode挙動だったことは確認できません。既存モデルへの移行では元のtokenizer実装/学習条件を照合してください。以前のmemmapを黙って流用しないため、tokenizer hash/encode契約のmanifestがないキャッシュは拒否します。既存キャッシュを退避し、新しい出力先で作り直してください。自動削除はしません。

パスはリポジトリを基準に解決します。学習は `python src2/main.py`、HyenaのGPU学習は `python src2/mainh.py`、IMEは `python src2/ime.py`。`config.py` のCORPUS等は自分のデータへ設定してください。

### 最小回帰テスト

依存：`requirements.txt`、`requirements-dev.txt` と、公式配布のCPU版PyTorch。Python 3.12.14 / PyTorch 2.14.1+cpu / tokenizers 0.23.2で確認しました。

```sh
python -m pytest -q
```

モデルは幅8・1層のCPU fixtureです。`model(x)` は従来どおり `[B,T,V]`、`model(x,last_only=True)` は `[B,V]`。greedy/確率生成、空/長いseed、plain/LoRAの保存往復、旧pickle拒否、tokenizer照合、保存失敗保全、プレビュー失敗と短時間学習の最終保存を検証します。外部モデル・実コーパス・GPU学習・Windows IME操作は未検証です。
