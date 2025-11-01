import math

import torch
import torch.nn as nn

from config import LORA_ALPHA, LORA_DROPOUT, LORA_R, LORA_TARGET_LM_HEAD

##########
# lora.py – 旧ckpt(非LoRA) → 新モデル(LoRA) 互換リマップの決定版
##########
# lora.py




def _remap_old_linear_keys_to_lora(sd_old: dict, model: nn.Module) -> dict:
    """
    旧: *.weight（非LoRA）
    新: *.base.weight（LoRAの土台） or そのまま（Embedding/LayerNorm など LoRA非対象）
    1) まず「同名キーがモデルに存在するか」を試す
    2) なければ「.base.*」への写像を試す
    LoRAの A/B は ckpt には無い前提なので触らない
    """
    # DP由来の "module." を剥がす
    if any(k.startswith("module.") for k in sd_old):
        sd_old = {k.replace("module.", "", 1): v for k, v in sd_old.items()}

    tgt = model.module if hasattr(model, "module") else model
    model_keys = set(tgt.state_dict().keys())
    sd_new = {}

    def to_base(k: str) -> str:
        if k.endswith(".weight"):
            return k[: -len(".weight")] + ".base.weight"
        if k.endswith(".bias"):
            return k[: -len(".bias")] + ".base.bias"
        return k

    for k_old, tensor in sd_old.items():
        # LoRA関連は旧ckptに無いはずだが、あってもスキップ
        if ".lora_A." in k_old or ".lora_B." in k_old:
            continue

        # 1) まず「そのまま」入れてみる（Embedding, LayerNorm などはこちらで一致）
        if k_old in model_keys:
            sd_new[k_old] = tensor
            continue

        # 2) ダメなら ".base.*" へ
        k_base = to_base(k_old)
        if k_base in model_keys:
            sd_new[k_base] = tensor
            continue

        # 3) lm_head の救済（旧: lm_head.weight → 新: lm_head.base.weight）
        if (
            k_old.startswith("lm_head.")
            and ("lm_head.base" + k_old[len("lm_head") :]) in model_keys
        ):
            sd_new["lm_head.base" + k_old[len("lm_head") :]] = tensor
            continue

        # 4) それ以外は見送り（unexpected を出さないため無視）

    # モデルにないキーは落としておく（安全）
    sd_new = {k: v for k, v in sd_new.items() if k in model_keys}
    return sd_new


##########


class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, r: int, alpha: int, dropout: float):
        super().__init__()
        self.base = base
        self.r = int(r)
        self.scaling = alpha / float(r) if r > 0 else 1.0
        self.enabled = self.r > 0
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        if self.enabled:
            # A/B を base と同じ device & dtype で作成（型不一致クラッシュ防止）
            dev = base.weight.device
            dt = base.weight.dtype
            self.lora_A = nn.Linear(base.in_features, r, bias=False).to(
                device=dev, dtype=dt
            )
            self.lora_B = nn.Linear(r, base.out_features, bias=False).to(
                device=dev, dtype=dt
            )

            nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B.weight)
        else:
            # no-train params (state_dict占有を避けるなら register_buffer でもOK)
            self.register_parameter(
                "lora_A_dummy", nn.Parameter(torch.zeros(0), requires_grad=False)
            )
            self.register_parameter(
                "lora_B_dummy", nn.Parameter(torch.zeros(0), requires_grad=False)
            )

    def forward(self, x):
        out = self.base(x)
        if self.enabled:
            # Dropout -> A -> B を加算
            out = out + self.lora_B(self.lora_A(self.dropout(x))) * self.scaling
        return out

    @staticmethod
    def from_linear(m: nn.Linear, r: int, alpha: int, dropout: float):
        # 既にLoRALinearならそのまま返す（再ラップ防止）
        if isinstance(m, LoRALinear):
            return m
        return LoRALinear(m, r=r, alpha=alpha, dropout=dropout)


def _apply_lora_to_model(
    model: nn.Module,
    r=LORA_R,
    alpha=LORA_ALPHA,
    dropout=LORA_DROPOUT,
    target_lm_head=LORA_TARGET_LM_HEAD,
):
    """
    モデル内の nn.Linear を LoRALinear に置換。
    - 既に LoRALinear の .base は再ラップしない
    - 2回目の適用でも安全
    """
    # スナップショットしてから書き換え
    for name, module in list(model.named_modules()):
        # 既にLoRALinearそのものはスキップ
        if isinstance(module, LoRALinear):
            continue

        # lm_head を外す設定なら外す
        if (not target_lm_head) and (name.endswith("lm_head") or ".lm_head." in name):
            continue

        if isinstance(module, nn.Linear):
            # 親モジュールを辿る
            parent_name = ".".join(name.split(".")[:-1])
            child_name = name.split(".")[-1]
            parent = model
            if parent_name:
                for p in parent_name.split("."):
                    parent = getattr(parent, p)

            # 親が LoRALinear かつ child が "base" の場合は再ラップ禁止
            if isinstance(parent, LoRALinear) and child_name == "base":
                continue

            setattr(
                parent, child_name, LoRALinear.from_linear(module, r, alpha, dropout)
            )


def _mark_only_lora_trainable(model: nn.Module):
    for p in model.parameters():
        p.requires_grad = False
    for n, p in model.named_parameters():
        if ("lora_A" in n) or ("lora_B" in n):
            p.requires_grad = True


def _collect_lora_params(model: nn.Module):
    return [
        p
        for n, p in model.named_parameters()
        if p.requires_grad and (("lora_A" in n) or ("lora_B" in n))
    ]


@torch.no_grad()
def merge_lora_into_base(model: nn.Module):
    """LoRA をベースにマージして単体重みに戻す"""
    for m in model.modules():
        if isinstance(m, LoRALinear) and m.enabled:
            # (out,r) @ (r,in) = (out,in)
            delta = (m.lora_B.weight @ m.lora_A.weight) * m.scaling
            m.base.weight.add_(delta)
            m.enabled = False
            # LoRA側は更新されないよう requires_grad を落としておく
            if hasattr(m, "lora_A"):
                for p in m.lora_A.parameters():
                    p.requires_grad = False
            if hasattr(m, "lora_B"):
                for p in m.lora_B.parameters():
                    p.requires_grad = False
