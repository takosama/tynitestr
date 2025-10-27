# chengemodeleasy.py
# checkpoints/*.pt（dict形式）→ torch.load 一発で動く “モデル本体” に変換
# 出力: checkpoints/*.model.pt

import torch
from model import TinyGPT2
from lora import _apply_lora_to_model, _remap_old_linear_keys_to_lora
from config import LORA_R, LORA_ALPHA, LORA_DROPOUT, LORA_TARGET_LM_HEAD, USE_LORA, VOCAB_SIZE

def _strip_module_prefix(sd: dict):
    if any(k.startswith("module.") for k in sd.keys()):
        return {k.replace("module.", "", 1): v for k, v in sd.items()}
    return sd

def _infer_ckpt_vocab(sd: dict) -> int:
    if "wte.weight" in sd:
        return sd["wte.weight"].shape[0]
    if "lm_head.weight" in sd:
        return sd["lm_head.weight"].shape[0]
    raise ValueError("state_dict に wte.weight / lm_head.weight が見つからないため、vocab_sizeを推定できません。")

def convert_checkpoint_to_executable_model(pt_path: str, out_path: str = None, force_vocab_size: int | None = None):
    """
    旧 checkpoint(.pt) を torch.load で実行できる “モデル本体” に変換
      - DataParallelの 'module.' プレフィクス除去
      - LoRA旧→新のキー移植
      - vocab不一致: デフォルトは ckpt側に合わせて再構築
        （force_vocab_size を指定したら部分コピーで拡張）
    """
    ckpt = torch.load(pt_path, map_location="cpu")
    if "model" not in ckpt:
        raise ValueError(f"{pt_path} はbundle形式ではありません（'model'キーが無い）。weights-onlyは変換不要です。")

    sd = _strip_module_prefix(ckpt["model"])

    # ---- LoRAの有無に合わせてキー整形（旧→新） ----
    has_lora_in_ckpt = any((".lora_A." in k) or (".lora_B." in k) or (".base." in k) for k in sd.keys())
    # モデル骨格は後で作るけど、_remap_old_linear_keys_to_lora が model 参照する想定なので
    # いったん一時モデル（最小サイズ）でターゲットを把握しやすくする場合はここで TinyGPT2 を作ってもよい。
    # 今回は後段で本体を作ってから、必要時もう一度 remap を呼ぶ運用にする。

    # ---- まず ckpt 側の vocab を推定 ----
    ckpt_vocab = _infer_ckpt_vocab(sd)

    # ---- 使う vocab を決定 ----
    # 既定: ckpt側に合わせる（最も安全）
    # 指定があれば force_vocab_size に拡張（>= ckpt_vocab を想定）
    target_vocab = ckpt_vocab if (force_vocab_size is None) else force_vocab_size

    # ---- モデルを構築（target_vocabで） ----
    model = TinyGPT2(vocab_size=target_vocab)
    if USE_LORA:
        _apply_lora_to_model(
            model,
            r=LORA_R,
            alpha=LORA_ALPHA,
            dropout=LORA_DROPOUT,
            target_lm_head=LORA_TARGET_LM_HEAD,
        )

    # （LoRA旧→新のキー移植が必要ならこの時点で実行）
    if USE_LORA and not has_lora_in_ckpt:
        sd = _remap_old_linear_keys_to_lora(sd, model)

    # ---- vocab 不一致への対処 ----
    if target_vocab == ckpt_vocab:
        # サイズ一致：そのまま読み込み
        ret = model.load_state_dict(sd, strict=False)
    else:
        # サイズ不一致：埋め込みとlm_headだけ手動で部分コピー、他はload_state_dictで流し込む
        if target_vocab < ckpt_vocab:
            raise ValueError(f"force_vocab_size={target_vocab} < ckpt_vocab={ckpt_vocab} はサポート外（縮小は不可）。")

        # 1) まず埋め込み/ヘッド以外をロード
        exclude = {"wte.weight", "lm_head.weight"}
        other_sd = {k: v for k, v in sd.items() if k not in exclude}
        ret = model.load_state_dict(other_sd, strict=False)

        # 2) 埋め込みとlm_headを手動コピー（共通部分のみ）
        with torch.no_grad():
            n = ckpt_vocab  # コピー可能な語彙数
            if "wte.weight" in sd:
                model.wte.weight[:n].copy_(sd["wte.weight"][:n])
            if "lm_head.weight" in sd:
                model.lm_head.weight[:n].copy_(sd["lm_head.weight"][:n])
            # 末尾(target_vocab - ckpt_vocab) は既定初期化のまま（必要ならここで好きな初期化をする）

    print(f"[convert] missing={len(ret.missing_keys)}, unexpected={len(ret.unexpected_keys)}")
    if ret.missing_keys[:5]:
        print("  missing(head):", ret.missing_keys[:5])
    if ret.unexpected_keys[:5]:
        print("  unexpected(head):", ret.unexpected_keys[:5])

    model.eval()

    # ---- CPUで保存（可搬性↑） ----
    if out_path is None:
        out_path = pt_path.replace(".pt", ".model.pt")
    torch.save(model.cpu(), out_path)
    print(f"[convert] Saved executable model → {out_path}")

    return out_path


if __name__ == "__main__":
    PATH = r"checkpoints\20251021_175651_latest_000002000.pt"
    # そのままなら ckpt の 26965 語彙で保存
    convert_checkpoint_to_executable_model(PATH)
    # もし 30000 語彙に“拡張して”保存したければ↓
    # convert_checkpoint_to_executable_model(PATH, force_vocab_size=30000)
