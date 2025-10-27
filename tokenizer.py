# tokenizer.py
# Byte-BPE の超軽量実装 + 便利ユーティリティ
from collections import Counter, deque
from pathlib import Path
import json
import unicodedata

import pandas as pd

from config import CSV_SEP, CSV_TEXT_COL, TOKENIZER_JSON

class ByteBPETokenizer:
    def __init__(self, tokfile="tokenizer.json"):
        obj = json.loads(Path(tokfile).read_text("utf-8"))
        self.vocab = {k.encode("latin1"): int(v) for k, v in obj["vocab"].items()}
        self.id2bytes = {int(v): k.encode("latin1") for k, v in obj["vocab"].items()}
        self.pair2id = {(a, b): nid for a, b, nid in obj.get("merges", [])}
        self.special = {k: int(v) for k, v in obj.get("special_tokens", {}).items()}
        self.unk_id = self.special.get("<unk>", 0)

    def _iter_bytes(self, text):
        for ch in text:
            for b in ch.encode("utf-8"):
                yield b

    def encode(self, text, add_bos=False, add_eos=False):
        seq = [self.vocab.get(bytes([b]), self.unk_id) for b in self._iter_bytes(text)]
        dq = deque(seq); changed = True
        while changed:
            changed = False; tmp = deque()
            while dq:
                a = dq.popleft()
                if dq and (nid := self.pair2id.get((a, dq[0]))):
                    tmp.append(nid); dq.popleft(); changed = True
                else:
                    tmp.append(a)
            dq = tmp
        seq = list(dq)
        if add_bos: seq = [self.special.get("<bos>", 0)] + seq
        if add_eos: seq += [self.special.get("<eos>", 1)]
        return seq

    def decode(self, ids):
        return b"".join([self.id2bytes.get(i, b"") for i in ids]).decode("utf-8", "replace")

def train_bpe_from_text(text: str, vocab_size=500, tokfile=Path("tokenizer.json")):
    data = text.encode("utf-8")
    vocab, merges, next_id = {bytes([i]): i for i in range(256)}, [], 256
    for (a, b), _ in Counter(zip(data, data[1:])).most_common(max(0, vocab_size-256)):
        nb = bytes([a]) + bytes([b])
        if nb not in vocab:
            vocab[nb] = next_id
            merges.append((a, b, next_id))
            next_id += 1
    tokfile.write_text(json.dumps({
        "type": "byte_bpe",
        "vocab": {k.decode("latin1"): v for k, v in vocab.items()},
        "merges": merges,
        "special_tokens": {"<bos>": next_id, "<eos>": next_id+1, "<unk>": next_id+2}
    }, ensure_ascii=False), encoding="utf-8")
    return next_id + 3  # vocab_size 実績を返す

def load_corpus_text(path: Path, text_col="text", sep=",") -> str:
    """トークナイザ学習など“一括テキストが必要な時”のみ使用"""
    import pandas as pd
    ext = path.suffix.lower()
    if ext in [".csv", ".tsv"]:
        df = pd.read_csv(path, sep="\t" if ext == ".tsv" else sep,
                         dtype=str, on_bad_lines="skip")
        col = text_col if text_col in df.columns else max(
            [c for c in df.columns if df[c].dtype == object],
            key=lambda c: df[c].astype(str).str.len().mean()
        )
        s = df[col].dropna().drop_duplicates()
        return "\n".join([unicodedata.normalize("NFKC", str(x)) for x in s])
    else:
        return unicodedata.normalize("NFKC", path.read_text(encoding="utf-8"))
def train_bpe_from_text(text, vocab_size=500, tokfile=TOKENIZER_JSON):
    data = text.encode("utf-8")
    vocab, merges, next_id = {bytes([i]): i for i in range(256)}, [], 256
    for (a, b), _ in Counter(zip(data, data[1:])).most_common(max(0, vocab_size-256)):
        nb = bytes([a]) + bytes([b])
        if nb not in vocab:
            vocab[nb] = next_id
            merges.append((a, b, next_id))
            next_id += 1
    tokfile.write_text(json.dumps({
        "type": "byte_bpe",
        "vocab": {k.decode("latin1"): v for k, v in vocab.items()},
        "merges": merges,
        "special_tokens": {"<bos>": next_id, "<eos>": next_id+1, "<unk>": next_id+2}
    }, ensure_ascii=False), encoding="utf-8")
    print("✅ tokenizer.json created")
 
# ====== CSVテキスト全読み（トークナイザ学習用） ======
def load_corpus_text(path: Path) -> str:
    """トークナイザ再学習など“一括テキストが必要な時”のみ使用"""
    ext = path.suffix.lower()
    if ext in [".csv", ".tsv"]:
        df = pd.read_csv(path, sep="\t" if ext == ".tsv" else CSV_SEP,
                         dtype=str, on_bad_lines="skip")
        col = CSV_TEXT_COL if CSV_TEXT_COL in df.columns else max(
            [c for c in df.columns if df[c].dtype == object],
            key=lambda c: df[c].astype(str).str.len().mean()
        )
        s = df[col].dropna().drop_duplicates()
        return "\n".join([unicodedata.normalize("NFKC", str(x)) for x in s])
    else:
        text = path.read_text(encoding="utf-8")
        return unicodedata.normalize("NFKC", text)
