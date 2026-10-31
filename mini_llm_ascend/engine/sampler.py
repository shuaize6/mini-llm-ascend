"""首版只做 greedy；固定长度实验在 argmax 前屏蔽 EOS。"""

import torch


def normalize_eos(eos_token_id, vocab_size):
    ids = [] if eos_token_id is None else ([eos_token_id] if isinstance(eos_token_id, int) else list(eos_token_id))
    if any(type(i) is not int or not 0 <= i < vocab_size for i in ids):
        raise ValueError("EOS ID 超出词表或不是整数。")
    return set(ids)


def greedy_select(logits, eos_ids=(), suppress_eos=False):
    if logits.ndim != 3 or tuple(logits.shape[:2]) != (1, 1):
        raise ValueError("采样只接受 [1,1,vocab] 的最后位置 logits。")
    if not bool(torch.isfinite(logits).all().item()):
        raise ValueError("logits 存在 NaN/Inf。")
    scores = logits[0, 0].clone()
    eos_ids = normalize_eos(eos_ids, scores.numel())
    if suppress_eos and eos_ids:
        if len(eos_ids) == scores.numel():
            raise ValueError("屏蔽 EOS 后没有可选 token。")
        scores[list(sorted(eos_ids))] = -float("inf")
    return int(scores.argmax().item())
