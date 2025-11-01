
"""
Streaming tokenizer training using Hugging Face `tokenizers` (BPE only).

- Trains **BPE** tokenizer from large corpora without loading all text into memory.
- Supports plain text or CSV/TSV (select a text column).
- Adds sampling, per-line truncation, and ByteLevel initial alphabet to avoid slow merges.
- Saves a HuggingFace-compatible `tokenizer.json` at repo root by default.

Usage examples:

  # From a large plain text file
  python coupus/tokenizer.py --input coupus/coupus_chunks_n_stride_m.txt \
      --vocab-size 16000 --min-frequency 50 --output tokenizer.json \
      --sample-rate 0.05 --max-lines 1000000 --max-chars-per-line 1024 --lowercase

  # From CSV (selecting the text column)
  python coupus/tokenizer.py --input E:\\data\\big.csv \
      --csv-col text --csv-sep , --vocab-size 16000 --min-frequency 50 \
      --sample-rate 0.05 --max-lines 1000000 --max-chars-per-line 1024
"""

from __future__ import annotations

import argparse
import csv
import random
import sys
import time
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Optional

# Enlarge CSV field size limit to handle giant cells (Windows-safe fallback)
try:
    csv.field_size_limit(sys.maxsize)
except (OverflowError, AttributeError, ValueError):
    csv.field_size_limit(2**31 - 1)  # ~2GB


def _norm(s: str, lowercase: bool) -> str:
    s = unicodedata.normalize("NFKC", s)
    return s.lower() if lowercase else s


@dataclass
class CorpusIterator:
    path: Path
    csv_col: Optional[str] = None
    csv_sep: str = ","
    lowercase: bool = False
    encoding: str = "utf-8"
    errors: str = "ignore"
    # performance controls
    sample_rate: float = 1.0
    max_lines: Optional[int] = None
    max_chars_per_line: int = 1024
    seed: int = 42
    log_every: int = 100_000  # progress log to stderr

    def __iter__(self) -> Iterator[str]:
        rng = random.Random(self.seed)
        sent = 0
        last_log = time.time()

        def _push(text: str) -> Optional[str]:
            nonlocal sent, last_log
            # normalize newlines and trim
            text = text.replace("\n", " ").replace("\r", " ").strip()
            if not text:
                return None
            # sampling
            if self.sample_rate < 1.0 and rng.random() > self.sample_rate:
                return None
            # truncate very long lines to prevent pair explosion
            if self.max_chars_per_line and len(text) > self.max_chars_per_line:
                text = text[: self.max_chars_per_line]
            sent += 1
            # progress log
            if self.log_every and (sent % self.log_every == 0 or (time.time() - last_log) > 10):
                print(f"[iter] sampled={sent}", file=sys.stderr)
                last_log = time.time()
            return _norm(text, self.lowercase)

        p = self.path
        suf = p.suffix.lower()
        if suf in {".csv", ".tsv"}:
            sep = "\t" if suf == ".tsv" else (self.csv_sep or ",")
            with p.open("r", encoding=self.encoding, errors=self.errors, newline="") as f:
                reader = csv.DictReader(f, delimiter=sep)
                col = self.csv_col or (reader.fieldnames[0] if reader.fieldnames else None)
                if not col:
                    return
                for row in reader:
                    val = row.get(col)
                    if not val:
                        continue
                    out = _push(str(val))
                    if out is not None:
                        yield out
                        if self.max_lines and sent >= self.max_lines:
                            return
        else:
            with p.open("r", encoding=self.encoding, errors=self.errors) as f:
                for line in f:
                    out = _push(line)
                    if out is not None:
                        yield out
                        if self.max_lines and sent >= self.max_lines:
                            return


def train_tokenizer(
    input_path: Path,
    output_path: Path,
    vocab_size: int = 16000,
    min_frequency: int = 50,
    limit_alphabet: Optional[int] = None,
    lowercase: bool = False,
    csv_col: Optional[str] = None,
    csv_sep: str = ",",
    sample_rate: float = 0.05,
    max_lines: Optional[int] = 1_000_000,
    max_chars_per_line: int = 1024,
    seed: int = 42,
) -> None:
    try:
        from tokenizers import Tokenizer
        from tokenizers.models import BPE
        from tokenizers.normalizers import NFKC, Lowercase, Sequence
        from tokenizers.pre_tokenizers import ByteLevel
        from tokenizers.trainers import BpeTrainer
        from tokenizers.processors import TemplateProcessing
    except Exception:
        print(
            "[error] Missing dependency: tokenizers. Install with: pip install tokenizers",
            file=sys.stderr,
        )
        raise

    it = CorpusIterator(
        path=input_path,
        csv_col=csv_col,
        csv_sep=csv_sep,
        lowercase=lowercase,
        sample_rate=sample_rate,
        max_lines=max_lines,
        max_chars_per_line=max_chars_per_line,
        seed=seed,
    )

    special_tokens = ["<pad>", "<unk>", "<bos>", "<eos>", "<mask>"]

    tok = Tokenizer(BPE(unk_token="<unk>"))
    tok.normalizer = Sequence([NFKC(), Lowercase()]) if lowercase else NFKC()
    tok.pre_tokenizer = ByteLevel(add_prefix_space=False)

    # Use ByteLevel initial alphabet to stabilize early merges and speed up training
    initial_ab = ByteLevel.alphabet()

    bpe_kwargs = dict(
        vocab_size=int(vocab_size),
        min_frequency=int(min_frequency),
        special_tokens=special_tokens,
        initial_alphabet=initial_ab,
        show_progress=True,
    )
    if limit_alphabet is not None:
        bpe_kwargs["limit_alphabet"] = int(limit_alphabet)

    trainer = BpeTrainer(**bpe_kwargs)
    tok.train_from_iterator(it, trainer=trainer)

    # Resolve special token IDs dynamically (safer than hard-coding)
    bos_id = tok.token_to_id("<bos>")
    eos_id = tok.token_to_id("<eos>")
    if bos_id is None or eos_id is None:
        raise RuntimeError("Special tokens <bos>/<eos> not found in vocabulary.")

    tok.post_processor = TemplateProcessing(
        single="<bos> $A <eos>",
        pair="<bos> $A <eos> <bos> $B <eos>",
        special_tokens=[("<bos>", bos_id), ("<eos>", eos_id)],
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    tok.save(str(output_path))
    print(f"[ok] Saved tokenizer to: {output_path}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Train a streaming BPE tokenizer (fast & memory-safe)")
    parser.add_argument("--input", default="coupus/coupus_clean.csv", type=Path, help="Input corpus file (txt/csv/tsv)")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "tokenizer.json",
        help="Where to save tokenizer.json (default: repo root)",
    )
    parser.add_argument("--vocab-size", type=int, default=30000)
    parser.add_argument("--min-frequency", type=int, default=50)
    parser.add_argument("--limit-alphabet", type=int, default=None)
    parser.add_argument("--lowercase", action="store_true")
    parser.add_argument("--csv-col", type=str, default=None, help="CSV/TSV text column name")
    parser.add_argument("--csv-sep", type=str, default=",", help="CSV separator (default ,)")

    # performance controls
    parser.add_argument("--sample-rate", type=float, default=0.1, help="Sampling rate for lines (0<r<=1)")
    parser.add_argument("--max-lines", type=int, default=1_000_000, help="Max sampled lines to feed trainer")
    parser.add_argument("--max-chars-per-line", type=int, default=1024, help="Truncate each line to this many chars")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for sampling")

    args = parser.parse_args(argv)

    if args.vocab_size is None or args.vocab_size <= 0:
        raise ValueError("--vocab-size must be a positive integer")
    if args.min_frequency is None or args.min_frequency <= 0:
        raise ValueError("--min-frequency must be a positive integer")
    if args.limit_alphabet is not None and args.limit_alphabet <= 0:
        raise ValueError("--limit-alphabet must be a positive integer when provided")
    if not (0 < args.sample_rate <= 1.0):
        raise ValueError("--sample-rate must be in (0, 1]")

    train_tokenizer(
        input_path=args.input,
        output_path=args.output,
        vocab_size=args.vocab_size,
        min_frequency=args.min_frequency,
        limit_alphabet=args.limit_alphabet,
        lowercase=args.lowercase,
        csv_col=args.csv_col,
        csv_sep=args.csv_sep,
        sample_rate=args.sample_rate,
        max_lines=args.max_lines,
        max_chars_per_line=args.max_chars_per_line,
        seed=args.seed,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

