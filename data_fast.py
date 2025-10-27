from pathlib import Path
import numpy as np
import torch

from config import TOK_BIN, WINDOW


class FastMemmapNexTokDataset(torch.utils.data.Dataset):
    """
    Windows 向け最適化版 Dataset:
    - memmap はパスのみ保持し、各プロセスで遅延オープン
    - __getitem__ では numpy を返し、collate_fn 側で一括 Tensor 化
    """
    def __init__(self, tok_bin: Path = TOK_BIN, window: int = WINDOW):
        self.tok_path = str(tok_bin)
        self.tok = None  # 遅延オープン
        self.window = int(window)

    def _ensure_open(self):
        if self.tok is None:
            self.tok = np.memmap(self.tok_path, dtype=np.uint32, mode="r")
            if len(self.tok) <= self.window:
                raise RuntimeError("Token stream too short for the chosen WINDOW")

    def __len__(self):
        self._ensure_open()
        return max(1, (len(self.tok) - self.window - 1))

    def __getitem__(self, _):
        self._ensure_open()
        start = np.random.randint(0, len(self.tok) - self.window - 1)
        arr = np.asarray(self.tok[start:start + self.window + 1], dtype=np.int64)
        return arr[:-1], arr[1:]


def fast_collate_long(batch):
    """高速バッチ化: numpy -> torch.long をまとめて行う。"""
    xs, ys = zip(*batch)
    x = torch.from_numpy(np.stack(xs, axis=0)).long()
    y = torch.from_numpy(np.stack(ys, axis=0)).long()
    return x, y

