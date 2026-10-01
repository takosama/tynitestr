# -*- coding: utf-8 -*-
"""
ime_tui.py — TinyGPT2 単語IME（確率付き・BPE/UNK/絵文字対応・TUI・5〜10文字補完）
- composing: 入力中の語
- Space（半角/全角）: 変換候補表示／巡回
- Enter: 候補確定（選択中）／composingをcommittedへ確定（非選択時）
- Backspace: 1文字戻る（選択中は候補解除）
- 数字1-9: 候補ダイレクト選択
- ESC: 終了

特長:
- BPEの部分トークンや <unk> を含む間は「未完成」とみなし候補に出さない
- 絵文字など多トークン文字は1グラフェムが完成するまで展開を継続
- 5〜10文字程度のまとまり（短文・語句）を補完候補として生成
"""

import math
import os
import re
import sys
import time
import traceback
import unicodedata
from dataclasses import dataclass

import torch
from tqdm.auto import tqdm


# ===== Windows UTF-8 強制 =====
def _force_utf8_console():
    try:
        import ctypes

        ctypes.windll.kernel32.SetConsoleOutputCP(65001)
        ctypes.windll.kernel32.SetConsoleCP(65001)
    except Exception:
        pass
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass


_force_utf8_console()

# ===== 設定 =====
DEBUG = False
CKPT = None  # Set to a versioned state-dict bundle, or select latest checkpoint.
from config import TOKENIZER_JSON

BEAM_SIZE = 16
FIRST_STEP_TOPK = 120
STEP_TOPK = 64
MAX_STEPS = 12
TEMPERATURE = 0.8
TARGET_MIN_CHARS = 10
TARGET_MAX_CHARS = 15

BAR_LEN = 8
TOP_EMOJIS = []

try:
    import msvcrt

    IS_WINDOWS = True
except Exception:
    IS_WINDOWS = False

# ===== 文字クラス =====
_HIRA, _KATA, _KANJI, _LATIN, _DIGIT = (
    re.compile(r"[\u3040-\u309F]"),
    re.compile(r"[\u30A0-\u30FF\uFF66-\uFF9D]"),
    re.compile(r"[\u4E00-\u9FFF]"),
    re.compile(r"[A-Za-z]"),
    re.compile(r"[0-9]"),
)
_MARK = re.compile(
    r'[、。．，・…！？!?\.\,\(\)\[\]{}「」『』《》〈〉“”"\'／/\\:;〜～—\s]'
)
SEP_CHARS = set(
    " 　、。．，・…！？!?。、/\\#@：「」『』《》〈〉“”\"'()（）[]{}：；;〜～—\n\r\t"
)


def _char_class(ch: str) -> str:
    if _MARK.match(ch):
        return "M"
    if _HIRA.match(ch):
        return "H"
    if _KATA.match(ch):
        return "K"
    if _KANJI.match(ch):
        return "C"
    if _DIGIT.match(ch):
        return "D"
    if _LATIN.match(ch):
        return "L"
    return "M"


def _good_chunk(w: str) -> bool:
    if not w:
        return False
    if all(ch in SEP_CHARS for ch in w):
        return False
    if any(ord(ch) < 0x20 for ch in w):
        return False
    return True


def _longest_common_prefix_len(a: str, b: str) -> int:
    i = 0
    L = min(len(a), len(b))
    while i < L and a[i] == b[i]:
        i += 1
    return i


def _contains_unk(tokenizer, ids: list[int], decoded: str) -> bool:
    try:
        sp = getattr(tokenizer, "special", None) or {}
        unk_id = sp.get("<unk>", None)
    except Exception:
        unk_id = None
    if unk_id is not None and any(t == unk_id for t in ids):
        return True
    if "<unk>" in decoded or "\ufffd" in decoded:
        return True
    return False


def _incomplete_grapheme(s: str) -> bool:
    if not s:
        return True
    c = s[-1]
    if unicodedata.combining(c):
        return True
    if c in ("\ufe0f", "\u200d"):
        return True
    RI_START, RI_END = 0x1F1E6, 0x1F1FF
    ri_tail = 0
    for ch in reversed(s):
        cp = ord(ch)
        if RI_START <= cp <= RI_END:
            ri_tail += 1
        else:
            break
    return ri_tail % 2 == 1


def _clamp_idx(idx, cands):
    return 0 if not cands else max(0, min(idx, len(cands) - 1))


def _prob_bar(p: float) -> str:
    return "█" * int(round(p * BAR_LEN)) + "░" * (BAR_LEN - int(round(p * BAR_LEN)))


# ===== モデルロード =====
def _load_model_and_tokenizer(checkpoint_path=None, tokenizer_path=None, device=None):
    from pathlib import Path
    from config import CKPT_DIR
    from model_io import load_model_bundle
    from tokenizer import ByteBPETokenizer

    path = checkpoint_path or CKPT
    if path is None:
        candidates = sorted(CKPT_DIR.glob("*_latest_*.pt"))
        if not candidates:
            raise FileNotFoundError("No versioned checkpoint found; export a tensor checkpoint first")
        path = candidates[-1]
    tokenizer_path = Path(tokenizer_path or TOKENIZER_JSON)
    tok = ByteBPETokenizer(tokenizer_path)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model_bundle(path, device=device, tokenizer_path=tokenizer_path)
    if max(tok.vocab.values()) >= model.wte.num_embeddings:
        raise ValueError("Tokenizer ids exceed model vocabulary")
    return model, tok


@dataclass
class Beam:
    ids: list
    logp: float
    text: str
    finished: bool


# ===== 5〜10文字補完 =====
@torch.no_grad()
def prob_span_candidates(
    seed_text: str, model, tokenizer, n=10, prefix_constraint: str = ""
):
    device = next(model.parameters()).device

    def _enc(txt: str):
        ids = tokenizer.encode(txt or "")
        return ids if ids else (tokenizer.encode(" ") or [tokenizer.bos_id])

    seed_ids = _enc(seed_text)
    seed = torch.tensor(seed_ids, dtype=torch.long, device=device)[None, :]
    prefix_ids = _enc(prefix_constraint) if prefix_constraint else []
    prefix_text = tokenizer.decode(prefix_ids) if prefix_ids else ""
    beams = [Beam(ids=list(prefix_ids), logp=0.0, text="", finished=False)]
    results: dict[str, float] = {}
    max_steps = MAX_STEPS + 8
    pbar = tqdm(total=max_steps, desc="🔮", leave=False, disable=(not DEBUG))
    for step in range(max_steps):
        new_beams = []
        expanded = False
        for b in beams:
            if b.finished:
                new_beams.append(b)
                continue
            x = torch.cat(
                [seed, torch.tensor([b.ids], dtype=torch.long, device=device)], dim=1
            )
            x = x[:, -model.wpe.num_embeddings:]
            logits = model(x, last_only=True) / TEMPERATURE
            probs = torch.softmax(logits, dim=-1)
            k = min(FIRST_STEP_TOPK if step == 0 else STEP_TOPK, probs.size(-1))
            pv, pi = torch.topk(probs, k=k, dim=-1)
            for tid, p in zip(pi[0].tolist(), pv[0].tolist()):
                ids2 = b.ids + [tid]
                logp2 = b.logp + math.log(max(p, 1e-12))
                cont = tokenizer.decode(ids2)
                cut = (
                    _longest_common_prefix_len(cont, prefix_text) if prefix_text else 0
                )
                suffix = cont[cut:]
                if _contains_unk(tokenizer, ids2, cont) or _incomplete_grapheme(suffix):
                    new_beams.append(Beam(ids2, logp2, b.text, False))
                    expanded = True
                    continue
                cand = suffix
                g_len = len(cand)
                finished = False
                out_text = cand
                if g_len < TARGET_MIN_CHARS:
                    finished = False
                elif g_len > TARGET_MAX_CHARS:
                    out_text = cand[:TARGET_MAX_CHARS]
                    finished = True
                else:
                    finished = True
                if not _good_chunk(out_text):
                    continue
                new_beams.append(Beam(ids2, logp2, out_text, finished))
                expanded = True
        new_beams.sort(key=lambda b: b.logp, reverse=True)
        beams = new_beams[:BEAM_SIZE]
        for b in beams:
            if b.finished and _good_chunk(b.text):
                results[b.text] = max(results.get(b.text, -1e9), b.logp)
        pbar.update(1)
        if (len(results) >= n and step >= 3) or not expanded:
            break
    pbar.close()
    if not results:
        return []
    words, lps = zip(*results.items())
    m = max(lps)
    exps = [math.exp(lp - m) for lp in lps]
    Z = sum(exps) or 1
    pairs = [(w, e / Z) for w, e in zip(words, exps) if _good_chunk(w)]
    pairs.sort(
        key=lambda t: (
            -t[1],
            abs(len(t[0]) - (TARGET_MIN_CHARS + TARGET_MAX_CHARS) / 2),
        )
    )
    return pairs[:n]


# ===== UI =====
def _render(committed, composing, cand_probs, idx, selecting):
    sys.stdout.write("\r" + " " * 200 + "\r")
    sys.stdout.write(f"入力: {committed}|{composing}\n")
    if selecting and cand_probs:
        lines = []
        for i, (w, p) in enumerate(cand_probs, 1):
            mark = "▶" if i - 1 == idx else "  "
            lines.append(f"{mark} {i:>2}. {w} {p*100:5.1f}% {_prob_bar(p)}")
        sys.stdout.write("\n".join(lines))
        sys.stdout.write("\n（Space 次 / Enter 確定 / 1-9 選択 / BS 戻る / ESC 終了）")
    else:
        sys.stdout.write("候補: （Spaceで変換。Enterで確定。全角/半角Spaceどちらも可）")
    sys.stdout.flush()


# ===== ループ =====
def _windows_loop(model, tokenizer):
    os.system("")
    try:
        os.system("chcp 65001 >NUL")
    except Exception:
        pass
    committed, composing = "", ""
    selecting = False
    cand_probs = []
    idx = 0
    last_space = 0
    print("TinyGPT2 IME — Space変換 / Enter確定 / ESC終了")
    _render(committed, composing, cand_probs, idx, selecting)
    while True:
        ch = msvcrt.getwch()
        if ch in ("\x00", "\xe0"):
            _ = msvcrt.getwch()
            continue
        if ch == "\x1b":
            print("\n⏹ 終了")
            break
        if ch == "\x08":
            if selecting:
                selecting = False
                cand_probs = []
            else:
                if composing:
                    composing = composing[:-1]
                elif committed:
                    committed = committed[:-1]
            _render(committed, composing, cand_probs, idx, selecting)
            continue
        if ch == "\r":
            if selecting and cand_probs:
                composing += cand_probs[idx][0]
                selecting = False
                cand_probs = []
            else:
                committed += composing
                composing = ""
            _render(committed, composing, cand_probs, idx, selecting)
            continue
        if ch in (" ", "\u3000"):
            now = time.time()
            if now - last_space < 0.06:
                continue
            last_space = now
            if selecting:
                idx = (idx + 1) % len(cand_probs) if cand_probs else 0
                _render(committed, composing, cand_probs, idx, selecting)
                continue
            if not composing:
                tail = re.findall(
                    r'([^\s、。！？…・「」『』《》〈〉“”"\'/\\:;〜～—-]+)$', committed
                )
                if tail:
                    composing = tail[0]
                    committed = committed[: -len(composing)]
                else:
                    committed += ch
                    _render(committed, composing, cand_probs, idx, selecting)
                    continue
            selecting = True
            try:
                pairs = prob_span_candidates(
                    committed, model, tokenizer, n=30, prefix_constraint=composing
                )
                cand_probs = pairs[:10] if pairs else []
            except Exception as e:
                if DEBUG:
                    print(e)
                    traceback.print_exc()
                cand_probs = []
            if not cand_probs:
                selecting = False
                _render(committed, composing, cand_probs, idx, selecting)
                continue
            idx = 0
            _render(committed, composing, cand_probs, idx, selecting)
            continue
        if "1" <= ch <= "9" and selecting and cand_probs:
            i = ord(ch) - ord("1")
            if i < len(cand_probs):
                idx = i
            _render(committed, composing, cand_probs, idx, selecting)
            continue
        composing += ch
        if selecting:
            selecting = False
            cand_probs = []
        _render(committed, composing, cand_probs, idx, selecting)


def _fallback_loop(model, tokenizer):
    print("TinyGPT2 IME（簡易） — :space 変換 / :enter 確定 / :quit 終了")
    committed, composing = "", ""
    while True:
        print(f"\n入力: {committed}|{composing}")
        cmd = input("[文字入力] / [:space] / [:enter] / [:quit] > ").strip()
        if cmd == ":quit":
            print("bye")
            break
        if cmd == ":enter":
            committed += composing
            composing = ""
            continue
        if cmd == ":space":
            pairs = prob_span_candidates(
                committed, model, tokenizer, n=30, prefix_constraint=composing
            )
            pairs = pairs[:10] if pairs else []
            for i, (w, p) in enumerate(pairs, 1):
                print(f"{i:>2}. {w} {p*100:5.1f}% {_prob_bar(p)}")
            sel = input("番号 / Enterで1番確定 > ").strip()
            if sel.isdigit():
                i = int(sel) - 1
                if 0 <= i < len(pairs):
                    composing += pairs[i][0]
            else:
                if pairs:
                    composing += pairs[0][0]
            continue
        composing += cmd


def main():
    try:
        model, tokenizer = _load_model_and_tokenizer()
    except Exception as e:
        print("[load error]", e)
        traceback.print_exc()
        return
    if IS_WINDOWS:
        _windows_loop(model, tokenizer)
    else:
        _fallback_loop(model, tokenizer)


if __name__ == "__main__":
    main()
