"""
Corpus cleaner for newline-separated CSV.

Replaces URL, mention, email, phone, and retweet markers with placeholders:
  <url> <mention> <email> <phone> <retweet>

Additionally, each cleaned text is wrapped as:
  <bos>TEXT<eos>

Usage:
  python -m coupus.coupus [--input coupus/coupus.csv] [--output coupus/coupus_clean.csv] [--text-col text]
"""

from __future__ import annotations

import argparse
import re
import unicodedata
from pathlib import Path
from typing import Dict

import pandas as pd

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover - optional dependency
    def tqdm(iterable, **_: object):
        return iterable


DEFAULT_INPUT = Path("./coupus.csv")
DEFAULT_OUTPUT = Path("coupus/coupus_clean.csv")


# Patterns (kept readable and conservative)
RE_URL = re.compile(r"(https?|ftp)://[^\s　]+", re.IGNORECASE)
RE_EMAIL = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")
RE_MENTION = re.compile(r"@[A-Za-z0-9_]{1,15}")

# Phone numbers (international-ish, conservative to avoid over-matching plain numbers)
# - +country
# - (0xx) or 0x blocks
# - separated by spaces or hyphens
RE_PHONE = re.compile(
    r"(?<!\w)(?:\+?\d{1,3}[- ]?)?(?:\(\d{2,4}\)|\d{2,4})[- ]?\d{2,4}[- ]?\d{3,4}(?!\w)"
)

# Retweet markers like: RT @user: ... or standalone RT:
RE_RETWEET = re.compile(r"(?im)^(?:\s*)RT\s*:?")


BOS_TOKEN = "<bos>"
EOS_TOKEN = "<eos>"

PLACEHOLDERS: Dict[str, str] = {
    "url": "<url>",
    "mention": "<mention>",
    "email": "<email>",
    "phone": "<phone>",
    "retweet": "<retweet>",
}


def normalize_text(s: str) -> str:
    if not isinstance(s, str):
        return ""
    return unicodedata.normalize("NFKC", s)


def replace_tokens(s: str) -> tuple[str, Dict[str, int]]:
    """Replace URL/mention/email/phone/retweet with placeholders, returning counts."""
    s = normalize_text(s)
    counts = {k: 0 for k in PLACEHOLDERS}

    def sub_with_count(pattern: re.Pattern[str], repl: str, text: str, key: str) -> str:
        nonlocal counts
        def _repl(_: re.Match[str]) -> str:  # count matches
            counts[key] += 1
            return repl
        return pattern.sub(_repl, text)

    # Replace more specific items first
    s = sub_with_count(RE_URL, PLACEHOLDERS["url"], s, "url")
    s = sub_with_count(RE_EMAIL, PLACEHOLDERS["email"], s, "email")
    s = sub_with_count(RE_PHONE, PLACEHOLDERS["phone"], s, "phone")
    s = sub_with_count(RE_MENTION, PLACEHOLDERS["mention"], s, "mention")

    # Retweet markers at line starts; keep mentions already replaced
    s = sub_with_count(RE_RETWEET, PLACEHOLDERS["retweet"], s, "retweet")

    # Collapse excessive whitespace around placeholders
    s = re.sub(r"\s+", " ", s).strip()
    # Wrap with BOS/EOS tokens
    s = f"{BOS_TOKEN}{s}{EOS_TOKEN}"
    return s, counts


def detect_text_column(df: pd.DataFrame, preferred: str | None) -> str:
    if preferred and preferred in df.columns:
        return preferred
    obj_cols = [c for c in df.columns if df[c].dtype == object]
    if not obj_cols:
        # Fallback: if single column, use it
        if len(df.columns) == 1:
            return df.columns[0]
        raise ValueError("No suitable text column found.")
    return max(obj_cols, key=lambda c: df[c].astype(str).str.len().mean())


def main() -> None:
    ap = argparse.ArgumentParser(description="Clean corpus CSV with placeholders")
    ap.add_argument("--input", type=Path, default=DEFAULT_INPUT, help="Input CSV path")
    ap.add_argument(
        "--output", type=Path, default=DEFAULT_OUTPUT, help="Output CSV path"
    )
    ap.add_argument(
        "--text-col",
        type=str,
        default=None,
        help="Target text column (auto-detect if omitted)",
    )
    ap.add_argument(
        "--sep",
        type=str,
        default=",",
        help="CSV separator (default ',')",
    )
    args = ap.parse_args()

    # Try reading with header; if file has no header and a single column, treat as text column
    try:
        df = pd.read_csv(args.input, sep=args.sep, dtype=str, on_bad_lines="skip")
    except Exception:
        # Fallback to no header
        df = pd.read_csv(
            args.input, sep=args.sep, dtype=str, on_bad_lines="skip", header=None
        )

    # If no columns or empty DataFrame
    if df.empty:
        print("Input is empty; nothing to do.")
        return

    if args.text_col is None and len(df.columns) == 1:
        # Name the single column for clarity
        df.columns = ["text"]
        target_col = "text"
    else:
        target_col = detect_text_column(df, args.text_col)

    total_counts = {k: 0 for k in PLACEHOLDERS}
    cleaned = []
    for s in tqdm(
        df[target_col].astype(str).fillna(""),
        desc="Cleaning corpus",
        unit="row",
    ):
        cs, cnts = replace_tokens(s)
        cleaned.append(cs)
        for k in total_counts:
            total_counts[k] += cnts[k]

    df[target_col] = cleaned
    args.output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output, sep=args.sep, index=False, encoding="utf-8")

    print(f"Saved cleaned CSV to: {args.output}")
    print("Replacement counts:")
    for k, v in total_counts.items():
        print(f"  {k:8s}: {v}")


if __name__ == "__main__":
    main()
