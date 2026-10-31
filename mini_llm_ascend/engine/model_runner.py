"""请求级 prefill/decode 循环，结束和异常均回收逻辑缓存状态。"""

import torch

from .sampler import greedy_select, normalize_eos


class ModelRunner:
    def __init__(self, model, capacity=512, *, cache=None):
        self.model = model.eval()
        # 注入 M2 请求缓存时，以 cache.capacity 为准；默认路径保持 M1 行为。
        self.cache = model.new_cache(capacity) if cache is None else cache

    @torch.inference_mode()
    def generate(self, input_ids, max_new_tokens, *, eos_token_id=None, fixed_length=False, on_token=None):
        if type(max_new_tokens) is not int or max_new_tokens <= 0:
            raise ValueError("输出预算必须为正整数。")
        if input_ids.ndim != 2 or input_ids.shape[0] != 1 or input_ids.shape[1] == 0:
            raise ValueError("prompt 必须为非空 [1,S]。")
        if input_ids.shape[1] + max_new_tokens > self.cache.capacity:
            raise ValueError("prompt + 输出预算超过缓存容量。")
        eos_ids = normalize_eos(eos_token_id, self.model.config.vocab_size)
        output_ids = []
        self.cache.reset()
        try:
            logits = self.model(input_ids, self.cache)
            for index in range(max_new_tokens):
                token = greedy_select(logits, eos_ids, suppress_eos=fixed_length)
                output_ids.append(token)
                if on_token is not None:
                    on_token(token)  # token 已同步到 host；文本解码可在生成结束后进行。
                if (not fixed_length and token in eos_ids) or index + 1 == max_new_tokens:
                    break
                next_input = torch.tensor([[token]], dtype=torch.long, device=input_ids.device)
                logits = self.model(next_input, self.cache)
            return output_ids
        finally:
            self.cache.reset()
