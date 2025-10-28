# ====== Basic Settings ======
from datetime import datetime
from pathlib import Path

CORPUS = Path(r"E:\test2\data.csv")  # Adjust to your dataset
TOKENIZER_JSON = Path("../tokenizer.json")
FORCE_RETRAIN_TOKENIZER = False

# Model size preset: "small" or "large"
MODEL_SIZE = "small"

# Tokenization / data
VOCAB_SIZE = 30000
WINDOW = 1024
CSV_TEXT_COL, CSV_SEP = "text", ","
BATCH_SIZE, EPOCHS, LR = 512, 8, 1e-4
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
