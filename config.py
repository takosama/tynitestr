# ====== Basic Settings ======
import os
from datetime import datetime
from pathlib import Path

# ====== Torch/Runtime Env (set early) ======
# Ensure these defaults are applied before importing torch in training scripts.
os.environ.setdefault("TORCHDYNAMO_DISABLE", "0")
os.environ.setdefault("TORCH_COMPILE_DISABLE", "0")
os.environ.setdefault(
    "PYTORCH_CUDA_ALLOC_CONF", "max_split_size_mb:128,expandable_segments:True"
)

CORPUS = Path(r"E:\test2\data.csv")  # Adjust to your dataset
TOKENIZER_JSON = Path("../tokenizer.json")
FORCE_RETRAIN_TOKENIZER = False

# Model size preset: "small" or "large"
MODEL_SIZE = "large"

# Tokenization / data
VOCAB_SIZE = 30000

WINDOW = 64
CSV_TEXT_COL, CSV_SEP = "text", ","
BATCH_SIZE, EPOCHS, LR =128, 8, 1e-4
BETAS, WEIGHT_DECAY = (0.9, 0.99), 1e-2
ACCUM_STEPS = 1
# DataLoader workers: tune per machine
# DataLoader workers: Windows lazy-memmap OK; adjust for parallel I/O
NUM_WORKERS = 2
PIN_MEMORY = True

# Activation checkpointing: reduce peak memory by recomputing block activations in backward
GRAD_CHECKPOINT = True

# ====== Checkpoint settings ======
CKPT_DIR = Path("../checkpoints")
CKPT_DIR.mkdir(exist_ok=True)
RUN_ID = datetime.now().strftime("%Y%m%d_%H%M%S")
SAVE_EVERY = 5000  # Save every N optimizer steps
KEEP_LAST = 3  # How many latest snapshots to keep
BEST_METRIC = "loss_ema"  # Metric name for best snapshot
# ====== Memmap layout ======
TOK_BIN = Path("../corpus_tokens.u32")  # uint32 concatenated token stream
OFF_BIN = Path("../corpus_offsets.u64")  # u64 cumulative document offsets


# ====== Hyena trainer knobs (moved from mainh.py) ======
# Training dynamics
LABEL_SMOOTH = 0.03
WARMUP_STEPS = 100
EMA_BETA = 0.98
GRAD_CLIP_NORM = 1

# Checkpoint/preview cadence
HYENA_SAVE_EVERY = 50  # optimizer steps (Hyena-specific)
PREVIEW_EVERY = 10      # steps between preview generation

# torch.compile settings
HYENA_COMPILE_MODE = "max-autotune"


if MODEL_SIZE == "small":
    D_MODEL = 384
    N_LAYER = 8
    N_HEAD = 8
else:
    D_MODEL = 768
    N_LAYER = 24
    N_HEAD = 16

# ====== LoRA settings ======
USE_LORA = False  # Enable LoRA adapters or not
LORA_R = 16  # Low-rank dimension
LORA_ALPHA = 16  # Scaling
LORA_DROPOUT = 0.05  # Dropout before LoRA A
LORA_TARGET_LM_HEAD = False  # Whether to apply LoRA to lm_head (usually False)
