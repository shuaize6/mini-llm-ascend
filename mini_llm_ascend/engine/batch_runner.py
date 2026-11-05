"""变长 token 打包：线性层合批，注意力按请求隔离，无 padding。"""

import torch
from torch.nn import functional as F

from .model import apply_rope, causal_mask, rotary_factors


@torch.inference_mode()
def forward_batch(model, inputs, caches):
    """返回 [请求数,1,vocab]；调用者负责失败后的 close/release。"""
    if not inputs or len(inputs) != len(caches) or len({id(c) for c in caches}) != len(caches):
        raise ValueError("输入与独立缓存必须一一对应，且批次不能为空。")
    config = model.config
    weight = model.model.embed_tokens.weight
    lengths, factors, masks = [], [], []
    # 先检查整个批次，再开启缓存事务，避免无效输入修改前面的请求。
    for ids, cache in zip(inputs, caches):
        if ids.ndim != 2 or ids.shape[0] != 1 or ids.shape[1] == 0:
            raise ValueError("每个输入必须是非空 [1,Q]。")
        if ids.dtype != torch.long or ids.device != weight.device:
            raise ValueError("输入必须是模型设备上的 long tensor。")
        if bool(((ids < 0) | (ids >= config.vocab_size)).any().item()):
            raise ValueError("token ID 超出词表。")
        size = ids.shape[1]
        if cache.length + size > min(cache.capacity, config.max_position_embeddings):
            raise ValueError("请求超过位置/缓存上限。")
        if len(cache.keys) != config.num_hidden_layers:
            raise ValueError("缓存层数错误。")
        positions = torch.arange(cache.length, cache.length + size, device=weight.device)
        factors.append(rotary_factors(positions, config.head_dim, config.rope_theta, weight.dtype))
        masks.append(causal_mask(size, cache.length, device=weight.device, dtype=weight.dtype))
        lengths.append(size)
    begun = []
    try:
        for size, cache in zip(lengths, caches):
            cache.begin_step(size)
            begun.append(cache)
        cos = torch.cat([pair[0] for pair in factors], dim=2)
        sin = torch.cat([pair[1] for pair in factors], dim=2)
        x = model.model.embed_tokens(torch.cat(inputs, dim=1))
        total, d = sum(lengths), config.head_dim
        for layer in model.model.layers:
            attn = layer.self_attn
            normalized = layer.input_layernorm(x)
            q = attn.q_proj(normalized).reshape(1, total, config.num_attention_heads, d)
            k = attn.k_proj(normalized).reshape(1, total, config.num_key_value_heads, d)
            v = attn.v_proj(normalized).reshape(1, total, config.num_key_value_heads, d).transpose(1, 2)
            q = apply_rope(attn.q_norm(q).transpose(1, 2), cos, sin)
            k = apply_rope(attn.k_norm(k).transpose(1, 2), cos, sin)
            contexts, offset = [], 0
            for size, cache, mask in zip(lengths, caches, masks):
                end = offset + size
                cache.write_layer(attn.layer_idx, k[:, :, offset:end], v[:, :, offset:end])
                keys, values = cache.view_layer(attn.layer_idx)
                groups = config.num_attention_heads // config.num_key_value_heads
                keys = keys.repeat_interleave(groups, dim=1)
                values = values.repeat_interleave(groups, dim=1)
                scores = torch.matmul(q[:, :, offset:end], keys.transpose(-2, -1)) * (d ** -0.5)
                probs = F.softmax(scores + mask, dim=-1, dtype=torch.float32).to(q.dtype)
                contexts.append(torch.matmul(probs, values).transpose(1, 2).contiguous())
                offset = end
            merged = torch.cat(contexts, dim=1).reshape(1, total, config.num_attention_heads * d)
            x = x + attn.o_proj(merged)
            x = x + layer.mlp(layer.post_attention_layernorm(x))
        # 每个请求仅取末位置。保留 token 顺序，避免读到邻居的 logits。
        indices, offset = [], 0
        for size in lengths:
            offset += size
            indices.append(offset - 1)
        index = torch.tensor(indices, device=weight.device, dtype=torch.long)
        last = model.model.norm(x).index_select(1, index).transpose(0, 1)
        logits = model.lm_head(last)
        for cache in caches:
            cache.commit_step()
        return logits
    except BaseException:
        for cache in begun:
            cache.abort_step()
        raise
