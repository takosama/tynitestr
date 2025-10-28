import re
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from config import CSV_SEP, CSV_TEXT_COL, OFF_BIN, TOK_BIN, WINDOW
from tokenizer import ByteBPETokenizer


def preprocess_corpus(corpus_path: Path, out_suffix: str = ".clean.csv") -> Path:
    """
    入力CSV/TSVの対象カラムに置換を施して、新しいCSVを作る。
    既存があれば再処理しない。
    """
    ext = corpus_path.suffix.lower()
    if ext not in [".csv", ".tsv"]:
        return corpus_path

    out_path = corpus_path.with_suffix(out_suffix)
    if out_path.exists():
        print(f"✅ cleaned corpus already exists: {out_path}")
        return out_path

    sep = "\t" if ext == ".tsv" else CSV_SEP
    print(f"🚧 cleaning corpus → {out_path.name} (sep='{sep}')")
    # カラム検出
    try:
        target_col = detect_text_col(corpus_path)
    except Exception:
        target_col = CSV_TEXT_COL

    first = True
    for chunk in pd.read_csv(
        corpus_path, sep=sep, dtype=str, on_bad_lines="skip", chunksize=100_000
    ):
        if target_col not in chunk.columns:
            obj_cols = [c for c in chunk.columns if chunk[c].dtype == object]
            if obj_cols:
                target_col = max(
                    obj_cols, key=lambda c: chunk[c].astype(str).str.len().mean()
                )
            else:
                target_col = chunk.columns[0]
        chunk[target_col] = chunk[target_col].astype(str).map(_clean_text_for_ime)
        chunk.to_csv(
            out_path,
            sep=sep,
            index=False,
            mode=("w" if first else "a"),
            header=first,
            encoding="utf-8",
        )
        first = False

    print(f"✅ cleaned corpus saved: {out_path}")
    return out_path


def _clean_text_for_ime(s: str) -> str:
    if not isinstance(s, str):
        return ""
    s = unicodedata.normalize("NFKC", s)
    for key, pat in _PATTERNS.items():
        s = re.sub(pat, _REPLS[key], s, flags=re.IGNORECASE)
    return s.strip()


_PATTERNS = {
    "URL": r"(https?|ftp)://[^\s　]+",
    "EMAIL": r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b",
    "MENTION": r"@[A-Za-z0-9_]{1,15}",
    "HASHTAG": r"#[A-Za-z0-9_\u3040-\u30FF\u4E00-\u9FFF]+",
    "RT": r"(?i)(?:\bRT\b[\s:：]*|ＲＴ[\s:：]*)",
    "USER": r"<USER>",
}
_REPLS = {k: "" for k in _PATTERNS}


def build_memmap_tokens(
    corpus_path: Path,
    tokenizer: ByteBPETokenizer,
    tok_bin=TOK_BIN,
    off_bin=OFF_BIN,
    chunksize=100_000,
):
    if tok_bin.exists() and off_bin.exists():
        print("✅ memmap already exists — skipping build")
        return
    print("🚧 Building memmap tokens (streaming)…")

    col = detect_text_col(corpus_path)
    ext = corpus_path.suffix.lower()
    total = 0
    offsets = [0]

    with open(tok_bin, "wb") as f_out:
        if ext in [".csv", ".tsv"]:
            for chunk in pd.read_csv(
                corpus_path,
                sep="\t" if ext == ".tsv" else CSV_SEP,
                dtype=str,
                on_bad_lines="skip",
                usecols=[col],
                chunksize=chunksize,
            ):
                for s in chunk[col].dropna():
                    s = unicodedata.normalize("NFKC", str(s))
                    ids = tokenizer.encode(s) + [tokenizer.special["<eos>"]]
                    arr = np.asarray(ids, dtype=np.uint32)
                    f_out.write(arr.tobytes())
                    total += arr.size
                    offsets.append(total)
        else:
            text = unicodedata.normalize(
                "NFKC", corpus_path.read_text(encoding="utf-8")
            )
            ids = tokenizer.encode(text) + [tokenizer.special["<eos>"]]
            arr = np.asarray(ids, dtype=np.uint32)
            f_out.write(arr.tobytes())
            total += arr.size
            offsets.append(total)

    np.asarray(offsets, dtype=np.uint64).tofile(off_bin)
    print(f"✅ memmap built: tokens={total:,} docs={len(offsets)-1}")


# ====== Dataset ======
class MemmapNexTokDataset(Dataset):
    """
    ディスク常駐の連結トークン列から、ランダム開始で (x=WINDOW tokens, y=次トークン列[WINDOW]) を返す。
    ドキュメント境界は未考慮（必要なら OFF_BIN を使って避ける）。
    """

    def __init__(self, tok_bin=TOK_BIN, window=WINDOW):
        self.tok = np.memmap(tok_bin, dtype=np.uint32, mode="r")
        self.window = int(window)
        if len(self.tok) <= self.window:
            raise RuntimeError("Token stream too short for the chosen WINDOW")

    def __len__(self):
        return max(1, (len(self.tok) - self.window - 1))

    def __getitem__(self, _):
        start = np.random.randint(0, len(self.tok) - self.window - 1)
        span = torch.from_numpy(
            self.tok[start : start + self.window + 1].astype(np.int64)
        )
        x = span[:-1]  # 長さ=WINDOW
        y = span[1:]  # 長さ=WINDOW（xを右へ1シフトした教師）
        return x, y


def detect_text_col(path: Path) -> str:
    ext = path.suffix.lower()
    if ext in [".csv", ".tsv"]:
        df_head = pd.read_csv(
            path,
            sep="\t" if ext == ".tsv" else CSV_SEP,
            dtype=str,
            on_bad_lines="skip",
            nrows=2000,
        )
        if CSV_TEXT_COL in df_head.columns:
            return CSV_TEXT_COL
        cand = [c for c in df_head.columns if df_head[c].dtype == object]
        return max(cand, key=lambda c: df_head[c].astype(str).str.len().mean())
    raise ValueError("Unsupported file")
