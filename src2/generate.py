"""Text generation utilities."""

import torch
import torch.nn.functional as F


def _fallback_bos(tokenizer):
    # If tokenizer exposes a BOS id, use it
    for k in ("bos_id", "BOS", "bos"):
        t = getattr(tokenizer, k, None)
        if isinstance(t, int):
            return t
    # Otherwise, fallback to the id of a space or 0
    enc_space = tokenizer.encode(" ")
    return enc_space[0] if enc_space else 0


def _top_k_top_p_filtering(logits, top_k=None, top_p=None):
    # top-k filtering
    if top_k is not None and top_k > 0:
        top_k = min(top_k, logits.size(-1))
        vals, idx = torch.topk(logits, top_k)
        mask = torch.full_like(logits, float("-inf"))
        mask.scatter_(1, idx, vals)
        logits = mask
    # top-p (nucleus) filtering
    if top_p is not None and 0.0 < top_p < 1.0:
        sorted_logits, sorted_idx = torch.sort(logits, descending=True, dim=-1)
        probs = F.softmax(sorted_logits, dim=-1)
        cum = torch.cumsum(probs, dim=-1)
        cutoff = cum > top_p
        # Keep the very first element
        cutoff[..., 0] = False
        sorted_logits[cutoff] = float("-inf")
        logits = torch.full_like(logits, float("-inf"))
        logits.scatter_(1, sorted_idx, sorted_logits)
    return logits


@torch.no_grad()
def generate_text(
    model,
    tokenizer,
    seed_text=None,
    seed_ids=None,
    max_new_tokens=64,
    temperature=1.0,
    top_k=0,
    top_p=0.0,
):
    device = next(model.parameters()).device
    # --- Build seed ---
    if seed_ids is None:
        ids = tokenizer.encode(seed_text or "")
    else:
        ids = list(seed_ids)
    # --- Safety: ensure context does not exceed position embedding length ---
    # DataParallel stores the model under .module
    model_for_meta = getattr(model, "module", model)
    max_pos = getattr(getattr(model_for_meta, "wpe", None), "num_embeddings", None)
    if isinstance(max_pos, int) and max_pos > 0 and len(ids) > max_pos:
        # If seed is too long, keep only the last max_pos tokens
        ids = ids[-max_pos:]
        print(f"[generate_text] seed too long: truncated to last {max_pos} tokens")

    if len(ids) == 0:
        # If empty, start with a single token
        ids = [_fallback_bos(tokenizer)]

    x = torch.tensor(ids, dtype=torch.long, device=device)[None, :]  # [1, T]

    for _ in range(max_new_tokens):
        # Ensure context does not exceed position embedding length
        if isinstance(max_pos, int) and max_pos > 0 and x.size(1) > max_pos:
            x_in = x[:, -max_pos:]
        else:
            x_in = x
        logits = model(x_in, last_only=True)  # [1, T, V]
        if logits.size(1) == 0:
            # Double-safety; typically this never happens
            x = torch.cat(
                [x, torch.tensor([[_fallback_bos(tokenizer)]], device=device)], dim=1
            )
            continue
        # Sample next token
        if temperature <= 0:
            next_id = torch.argmax(logits, dim=-1)
        else:
            logits = logits / temperature
            logits = _top_k_top_p_filtering(logits, top_k=top_k, top_p=top_p)
            probs = F.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1).squeeze(-1)  # [1]
        x = torch.cat([x, next_id[:, None]], dim=1)

    return tokenizer.decode(x[0].tolist())
