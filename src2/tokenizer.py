"""Byte-level BPE adapter backed by the committed Hugging Face tokenizer JSON."""

import csv
from pathlib import Path

from tokenizers import (
    Tokenizer,
    decoders,
    models,
    normalizers,
    pre_tokenizers,
    trainers,
)

SPECIAL_TOKENS = ["<pad>", "<unk>", "<bos>", "<eos>", "<mask>"]


class ByteBPETokenizer:
    def __init__(self, path):
        import hashlib

        self.fingerprint = hashlib.sha256(Path(path).read_bytes()).hexdigest()
        self.encoding_version = "hf-bytelevel-no-added-specials-v1"
        self.tokenizer = Tokenizer.from_file(str(path))
        # Older committed JSON has ByteLevel encoding but no decoder.
        if self.tokenizer.decoder is None:
            import json

            spec = json.loads(Path(path).read_text(encoding="utf-8"))
            if spec.get("pre_tokenizer", {}).get("type") != "ByteLevel":
                raise ValueError("A tokenizer decoder is required")
            self.tokenizer.decoder = decoders.ByteLevel()
        self.vocab = self.tokenizer.get_vocab()
        self.special = {
            token: self.vocab[token] for token in SPECIAL_TOKENS if token in self.vocab
        }
        if "<eos>" not in self.special:
            raise ValueError("Tokenizer requires <eos>")
        self.bos_id = self.special.get("<bos>", self.special["<eos>"])

    def encode(self, text: str) -> list[int]:
        # Dataset code adds EOS; generation must not insert EOS after its prompt.
        return self.tokenizer.encode(text, add_special_tokens=False).ids

    def decode(self, ids: list[int]) -> str:
        return self.tokenizer.decode([int(i) for i in ids], skip_special_tokens=True)


def load_corpus_text(path):
    path = Path(path)
    if path.suffix.lower() in {".csv", ".tsv"}:
        from config import CSV_SEP, CSV_TEXT_COL

        with path.open(encoding="utf-8-sig", newline="") as stream:
            reader = csv.DictReader(
                stream, delimiter="\t" if path.suffix.lower() == ".tsv" else CSV_SEP
            )
            if not reader.fieldnames or CSV_TEXT_COL not in reader.fieldnames:
                raise ValueError(f"Missing text column: {CSV_TEXT_COL}")
            return "\n".join(row[CSV_TEXT_COL] or "" for row in reader)
    return path.read_text(encoding="utf-8")


def train_bpe_from_text(text: str, vocab_size: int, output_path=None):
    if output_path is None:
        from config import TOKENIZER_JSON

        output_path = TOKENIZER_JSON
    tokenizer = Tokenizer(models.BPE(unk_token="<unk>"))
    tokenizer.normalizer = normalizers.NFKC()
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=SPECIAL_TOKENS,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        show_progress=False,
    )
    tokenizer.train_from_iterator([text], trainer=trainer)
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tokenizer.save(str(path))
    return path


if __name__ == "__main__":
    from config import CORPUS, VOCAB_SIZE

    print(train_bpe_from_text(load_corpus_text(CORPUS), VOCAB_SIZE))
