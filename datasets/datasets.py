"""
Dataset utilities for handling pre-chunked corpus files.

Features:
  - Load `coupus/csv_chunks_n_offset.txt`.
  - Shuffle the entire dataset and write to `datasets/shuffled.txt`.
  - Tokenize the data and save an `int32` tensor to `datasets/cache.pt`.
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import Iterable, List, Sequence

import torch
from tokenizers import Tokenizer
from tqdm.auto import tqdm

CHUNKS_PATH = Path("coupus/csv_chunks_n_offset.txt")
DATASETS_DIR = Path("datasets")
SHUFFLED_PATH = DATASETS_DIR / "shuffled.txt"
CACHE_PATH = DATASETS_DIR / "cache.pt"


def load_chunks(path: Path = CHUNKS_PATH) -> List[str]:
    """Load chunked text lines from file."""
    if not path.exists():
        raise FileNotFoundError(f"Chunk file not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        return [line.rstrip("\n") for line in f]


def shuffle_data(lines: Iterable[str], output: Path = SHUFFLED_PATH) -> List[str]:
    """Shuffle lines and persist to output."""
    data = list(lines)
    random.shuffle(data)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="\n") as f:
        for line in tqdm(data, desc="Writing shuffled data", unit="line"):
            f.write(line)
            f.write("\n")
    return data


def make_cache(
    lines: Iterable[str],
    tokenizer_path: Path = Path("tokenizer.json"),
    output: Path = CACHE_PATH,
) -> torch.Tensor:
    """
    Tokenize lines and store as an int32 tensor on disk.
    - Use tok.encode(line).ids to get token IDs.
    - Pad with the <pad> token ID when available, otherwise 0.
    """
    tok = Tokenizer.from_file(str(tokenizer_path))

    pad_id = tok.token_to_id("<pad>")
    if pad_id is None:
        pad_id = 0

    lines_list: Sequence[str] = list(lines)

    encoded: List[List[int]] = []
    for line in tqdm(lines_list, desc="Encoding", unit="line"):
        if not line:
            continue
        encoded.append(tok.encode(line).ids)

    if not encoded:
        raise ValueError("No data to cache after tokenization.")

    max_len = max(len(seq) for seq in encoded)
    tensor = torch.full(
        (len(encoded), max_len),
        fill_value=int(pad_id),
        dtype=torch.int32,
    )
    for i, seq in enumerate(tqdm(encoded, desc="Padding", unit="seq", leave=False)):
        if not seq:
            continue
        tensor[i, : len(seq)] = torch.tensor(seq, dtype=torch.int32)

    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(tensor, output)
    return tensor


def build_cache(
    chunk_path: Path = CHUNKS_PATH,
    tokenizer_path: Path = Path("tokenizer.json"),
    shuffle: bool = True,
    seed: int | None = None,
    shuffle_output: Path = SHUFFLED_PATH,
    cache_output: Path = CACHE_PATH,
) -> torch.Tensor:
    """
    Convenience wrapper that loads, shuffles (optional), and caches token IDs.
    """
    if seed is not None:
        random.seed(seed)

    lines = load_chunks(chunk_path)
    if shuffle:
        lines = shuffle_data(lines, output=shuffle_output)
    else:
        lines = list(lines)

    return make_cache(lines, tokenizer_path=tokenizer_path, output=cache_output)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Load corpus chunks, optionally shuffle, and build an int32 cache."
    )
    parser.add_argument(
        "--chunk-path",
        type=Path,
        default=CHUNKS_PATH,
        help="Path to chunk descriptor file.",
    )
    parser.add_argument(
        "--tokenizer",
        type=Path,
        default=Path("tokenizer.json"),
        help="Path to tokenizer JSON.",
    )
    parser.add_argument(
        "--no-shuffle",
        action="store_true",
        help="Disable shuffling before caching.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional RNG seed for deterministic shuffling.",
    )
    parser.add_argument(
        "--shuffle-output",
        type=Path,
        default=SHUFFLED_PATH,
        help="Where to save the shuffled corpus.",
    )
    parser.add_argument(
        "--cache-output",
        type=Path,
        default=CACHE_PATH,
        help="Where to save the cached tensor.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    tensor = build_cache(
        chunk_path=args.chunk_path,
        tokenizer_path=args.tokenizer,
        shuffle=not args.no_shuffle,
        seed=args.seed,
        shuffle_output=args.shuffle_output,
        cache_output=args.cache_output,
    )
    print(f"Cached tensor with shape {tuple(tensor.shape)} at {args.cache_output}")


__all__ = [
    "load_chunks",
    "shuffle_data",
    "make_cache",
    "build_cache",
    "CHUNKS_PATH",
    "SHUFFLED_PATH",
    "CACHE_PATH",
]


if __name__ == "__main__":
    main()
