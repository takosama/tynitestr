"""
CSV to text converter with chunking utilities.

Workflow:
  1) Replace every newline in the CSV text with a single space (-> one long line)
  2) Split into fixed-size chunks of length n
  3) Split again with offset m (0 < m < n), otherwise skip

Outputs:
  - Single-line joined text file
  - Base chunks (offset 0)
  - Offset chunks (only when 0 < m < n)
"""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path
from typing import Iterable, List, Optional, Union

# ---- Defaults
DEFAULT_INPUT = Path("coupus/coupus_clean.csv")
DEFAULT_JOINED = Path("coupus/csv_single_line.txt")
DEFAULT_CHUNKS = Path("coupus/csv_chunks_n.txt")
DEFAULT_OFFSET_CHUNKS = Path("coupus/csv_chunks_n_offset.txt")

# Enlarge CSV field size limit to handle giant text cells safely
try:
    import sys
    csv.field_size_limit(sys.maxsize)
except Exception:
    try:
        csv.field_size_limit(2**31 - 1)
    except Exception:
        pass


def clean_newlines(s: str) -> str:
    """Replace CR/LF sequences with spaces and collapse redundant whitespace."""
    s = s.replace("\r\n", " ").replace("\n", " ").replace("\r", " ")
    return re.sub(r"\s+", " ", s).strip()


def chunk_fixed(s: str, n: int, keep_tail: bool = True) -> List[str]:
    out: List[str] = []
    for i in range(0, len(s), n):
        piece = s[i : i + n]
        if len(piece) == n or (keep_tail and piece):
            out.append(piece)
    return out


def chunk_with_offset(s: str, n: int, offset: int, keep_tail: bool = True) -> List[str]:
    out: List[str] = []
    if offset <= 0:
        return out
    pos = offset
    while pos < len(s):
        piece = s[pos : pos + n]
        if len(piece) == n or (keep_tail and piece):
            out.append(piece)
        if len(piece) < n:
            break
        pos += n
    return out


def write_lines(path: Path, lines: Iterable[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for line in lines:
            f.write(line)
            f.write("\n")


def _parse_col_selector(text_col: Optional[str]) -> Union[None, int, str]:
    """Allow either a column name or a 0-based integer index as a string."""
    if text_col is None:
        return None
    # if it's an integer like "0" or "2"
    if text_col.isdigit():
        return int(text_col)
    return text_col


def read_texts(
    path: Path,
    sep: str = ",",
    text_col: Optional[str] = None,
    has_header: bool = True,
    encoding: str = "utf-8",
    errors: str = "ignore",
) -> List[str]:
    """
    Read texts from CSV/TSV (selecting a column) or from plain .txt.
    - text_col: column name or "0"/"1"/... (index). If None, first column is used.
    - has_header=False: use csv.reader and take index (name not available).
    """
    suf = path.suffix.lower()
    if suf in {".txt"}:
        # Read entire file as text lines
        with path.open("r", encoding=encoding, errors=errors) as f:
            return [line.rstrip("\n") for line in f if line.strip()]
    # CSV / TSV
    delimiter = "\t" if suf == ".tsv" else (sep or ",")
    col_sel = _parse_col_selector(text_col)

    texts: List[str] = []
    with path.open("r", encoding=encoding, errors=errors, newline="") as f:
        if has_header:
            reader = csv.DictReader(f, delimiter=delimiter)
            # choose first field if text_col None
            if reader.fieldnames is None or len(reader.fieldnames) == 0:
                return texts
            if isinstance(col_sel, int):
                # map index to name
                if 0 <= col_sel < len(reader.fieldnames):
                    target = reader.fieldnames[col_sel]
                else:
                    raise IndexError(f"--text-col index out of range (0..{len(reader.fieldnames)-1})")
            else:
                target = col_sel or reader.fieldnames[0]
                if target not in reader.fieldnames:
                    raise KeyError(f"--text-col '{target}' not found; available: {reader.fieldnames}")
            for row in reader:
                val = row.get(target, "")
                if val:
                    texts.append(str(val))
        else:
            reader = csv.reader(f, delimiter=delimiter)
            for row in reader:
                if not row:
                    continue
                if isinstance(col_sel, int):
                    idx = col_sel
                else:
                    idx = 0  # default to first column if no header and name not usable
                if idx >= len(row):
                    continue
                texts.append(str(row[idx]))
    return texts


def main() -> None:
    ap = argparse.ArgumentParser(description="Convert CSV text column to chunks")
    ap.add_argument("--input", type=Path, default=DEFAULT_INPUT, help="Input CSV/TXT path")
    ap.add_argument("--out-single", type=Path, default=DEFAULT_JOINED, help="Output path for single-line text")
    ap.add_argument("--out-chunks", type=Path, default=DEFAULT_CHUNKS, help="Output path for base chunks")
    ap.add_argument("--out-offset", type=Path, default=DEFAULT_OFFSET_CHUNKS, help="Output path for offset chunks")
    ap.add_argument("--text-col", type=str, default=None, help="CSV column (name or zero-based index as string)")
    ap.add_argument("--sep", type=str, default=",", help="CSV separator (default ',')")
    ap.add_argument("--no-header", action="store_true", help="Set if CSV has no header row")
    ap.add_argument("--n", type=int, default=1024, help="Chunk length")
    ap.add_argument("--m", type=int, default=128, help="Offset length for second pass")
    ap.add_argument("--keep-tail", action="store_true", help="Keep trailing chunks shorter than n characters")

    args = ap.parse_args()

    if args.n <= 0:
        raise ValueError("--n must be a positive integer")
    if args.m < 0:
        raise ValueError("--m must be non-negative")

    texts = read_texts(
        args.input,
        sep=args.sep,
        text_col=args.text_col,
        has_header=not args.no_header,
    )
    if not texts:
        print("No texts found; nothing to do.")
        return

    normalized = [clean_newlines(t) for t in texts if t]
    joined = " ".join([t for t in normalized if t]).strip()
    if not joined:
        print("Normalized text is empty; nothing to do.")
        return

    args.out_single.parent.mkdir(parents=True, exist_ok=True)
    args.out_single.write_text(joined, encoding="utf-8")

    base_chunks = chunk_fixed(joined, args.n, keep_tail=args.keep_tail)
    write_lines(args.out_chunks, base_chunks)

    offset_chunks: List[str] = []
    if 0 < args.m < args.n:
        offset_chunks = chunk_with_offset(joined, args.n, args.m, keep_tail=args.keep_tail)
        write_lines(args.out_offset, offset_chunks)
    else:
        try:
            args.out_offset.unlink()
        except FileNotFoundError:
            pass

    print(f"Single-line length: {len(joined):,}")
    print(f"Chunks (n={args.n}, keep_tail={args.keep_tail}): {len(base_chunks):,}")
    if offset_chunks:
        print(f"Offset chunks (m={args.m}): {len(offset_chunks):,}")
    else:
        print(f"Offset chunking skipped (m={args.m} >= n or m == 0).")


if __name__ == "__main__":
    main()
