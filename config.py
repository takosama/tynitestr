# ====== 基本設定 ======
from datetime import datetime
from pathlib import Path


CORPUS = Path(r"E:\test2\data.csv")   # ← 自分のデータに合わせて
TOKENIZER_JSON = Path("../tokenizer.json")
FORCE_RETRAIN_TOKENIZER = False

# モデルサイズ切替: "small" or "large"
MODEL_SIZE = "small"

# ハイパ5
VOCAB_SIZE = 30000
WINDOW = 1024
CSV_TEXT_COL, CSV_SEP = "text", ","
BATCH_SIZE, EPOCHS, LR = 512, 8, 1e-4
BETAS, WEIGHT_DECAY = (0.9, 0.99), 1e-2
ACCUM_STEPS = 16
# DataLoader workers（Windowsは0が安定）
# DataLoader workers: Windows でも lazy-memmap で安定運用可。2 で I/O を並列化
NUM_WORKERS = 2
PIN_MEMORY = True

# Activation checkpointing: reduce peak memory by recomputing block activations in backward
GRAD_CHECKPOINT = True

# ====== Checkpoint settings ======
CKPT_DIR = Path("../checkpoints"); CKPT_DIR.mkdir(exist_ok=True)
RUN_ID = datetime.now().strftime("%Y%m%d_%H%M%S")
SAVE_EVERY = 5000          # 何stepごとに保存するか
KEEP_LAST = 3              # latestを何世代残すか
BEST_METRIC = "loss_ema"   # ベスト判定に使うメトリクス
# ====== memmap 構築 ======
TOK_BIN = Path("../corpus_tokens.u32")   # uint32 の連結トークン列
OFF_BIN = Path("../corpus_offsets.u64")  # 各ドキュメントの開始オフセット(累積長)




if MODEL_SIZE == "small":
    D_MODEL = 384; N_LAYER = 8; N_HEAD = 8
else:
    D_MODEL = 768; N_LAYER = 24; N_HEAD = 16

# ====== LoRA settings (最小追加) ======
USE_LORA = False           # ← LoRAを使わない時は False
LORA_R = 16              # ランク
LORA_ALPHA = 16           # スケーリング
LORA_DROPOUT = 0.05       # LoRA前段のdropout
LORA_TARGET_LM_HEAD = False  # lm_headにLoRAを当てる場合 True（通常はFalse推奨）

