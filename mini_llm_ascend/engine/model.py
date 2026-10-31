"""可逐行阅读的 Qwen3 eager 前向，不调用 HF 模型计算。

数学顺序参照 Transformers v4.51.3 Qwen3 (Apache-2.0)：
https://github.com/huggingface/transformers/blob/v4.51.3/src/transformers/models/qwen3/modeling_qwen3.py
这里只支持单请求、无 padding、默认 RoPE、无滑动窗口的推理。
"""

from dataclasses import dataclass, fields
import math

import torch
from torch import nn
from torch.nn import functional as F

from mini_llm_ascend.cache.contiguous_kv import ContiguousKVCache


@dataclass(frozen=True)
class ModelConfig:
    vocab_size: int
    hidden_size: int
    intermediate_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    max_position_embeddings: int
    rms_norm_eps: float = 1e-6
    rope_theta: float = 1000000.0
    tie_word_embeddings: bool = True

    def __post_init__(self):
        for name in ("vocab_size", "hidden_size", "intermediate_size", "num_hidden_layers",
                     "num_attention_heads", "num_key_value_heads", "head_dim", "max_position_embeddings"):
            if type(getattr(self, name)) is not int or getattr(self, name) <= 0:
                raise ValueError(name + " 必须为正整数。")
        if self.num_attention_heads % self.num_key_value_heads or self.head_dim % 2:
            raise ValueError("Q heads 必须整除 KV heads，head_dim 必须为偶数。")
        if not all(math.isfinite(v) and v > 0 for v in (self.rms_norm_eps, self.rope_theta)):
            raise ValueError("eps 和 rope_theta 必须为有限正数。")

    @classmethod
    def from_dict(cls, data):
        if data.get("model_type") != "qwen3":
            raise ValueError("只支持 Qwen3。")
        unsupported = (data.get("attention_bias", False), data.get("use_sliding_window", False),
                       data.get("sliding_window") is not None, data.get("rope_scaling") is not None,
                       data.get("hidden_act", "silu") != "silu")
        if any(unsupported):
            raise ValueError("M1 不支持 attention bias、滑动窗口、RoPE scaling 或非 silu 激活。")
        return cls(**{f.name: data[f.name] for f in fields(cls) if f.name in data})


class RMSNorm(nn.Module):
    def __init__(self, width, eps):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(width))
        self.eps = eps

    def forward(self, x):
        # 统计量用 FP32；乘权重的顺序与参考实现一致。
        value = x.float()
        value = value * torch.rsqrt(value.square().mean(-1, keepdim=True) + self.eps)
        return self.weight * value.to(x.dtype)


def rotary_factors(positions, head_dim, theta, dtype):
    """位置 [Q] → cos/sin [1,1,Q,D]；前半与后半维配对。"""
    # HF 在 CPU 初始化频率，再在目标设备做 FP32 batched matmul。
    # 在 NPU 上，等价的逐元素乘法与矩阵乘法可能选择不同精度路径；
    # 对拍版保留相同的计算顺序，而不是只保持代数公式等价。
    inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2, device="cpu").float() / head_dim))
    inv_freq = inv_freq.to(positions.device)
    angles = (inv_freq[None, :, None] @ positions[None, None, :].float()).transpose(1, 2)
    angles = torch.cat((angles, angles), dim=-1)
    return angles.cos().to(dtype)[:, None], angles.sin().to(dtype)[:, None]


def apply_rope(x, cos, sin):
    half = x.shape[-1] // 2
    rotated = torch.cat((-x[..., half:], x[..., :half]), dim=-1)
    return x * cos + rotated * sin


def causal_mask(q_len, past_len, *, device, dtype):
    """有历史前缀时，query i 可访问 key j <= past_len+i。"""
    query = torch.arange(past_len, past_len + q_len, device=device)[:, None]
    key = torch.arange(past_len + q_len, device=device)[None, :]
    mask = torch.zeros((q_len, past_len + q_len), device=device, dtype=dtype)
    return mask.masked_fill(key > query, torch.finfo(dtype).min)[None, None]


class Attention(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        h, d = config.hidden_size, config.head_dim
        self.q_proj = nn.Linear(h, config.num_attention_heads * d, bias=False)
        self.k_proj = nn.Linear(h, config.num_key_value_heads * d, bias=False)
        self.v_proj = nn.Linear(h, config.num_key_value_heads * d, bias=False)
        self.o_proj = nn.Linear(config.num_attention_heads * d, h, bias=False)
        self.q_norm = RMSNorm(d, config.rms_norm_eps)
        self.k_norm = RMSNorm(d, config.rms_norm_eps)

    def forward(self, x, cos, sin, mask, cache):
        b, q_len, _ = x.shape
        c = self.config
        q = self.q_proj(x).reshape(b, q_len, c.num_attention_heads, c.head_dim)
        k = self.k_proj(x).reshape(b, q_len, c.num_key_value_heads, c.head_dim)
        v = self.v_proj(x).reshape(b, q_len, c.num_key_value_heads, c.head_dim).transpose(1, 2)
        q = apply_rope(self.q_norm(q).transpose(1, 2), cos, sin)
        k = apply_rope(self.k_norm(k).transpose(1, 2), cos, sin)
        if cache is not None:
            # 当前 token 的 K/V 也必须参与本层注意力。
            cache.write_layer(self.layer_idx, k, v)
            k, v = cache.view_layer(self.layer_idx)
        groups = c.num_attention_heads // c.num_key_value_heads
        # 只扩展计算用的临时张量，永久缓存仍保存 Hkv 个头。
        k = k.repeat_interleave(groups, dim=1)
        v = v.repeat_interleave(groups, dim=1)
        scores = torch.matmul(q, k.transpose(-2, -1)) * (c.head_dim ** -0.5)
        probabilities = F.softmax(scores + mask, dim=-1, dtype=torch.float32).to(q.dtype)
        context = torch.matmul(probabilities, v).transpose(1, 2).contiguous()
        return self.o_proj(context.reshape(b, q_len, c.num_attention_heads * c.head_dim))


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        h, i = config.hidden_size, config.intermediate_size
        self.gate_proj = nn.Linear(h, i, bias=False)
        self.up_proj = nn.Linear(h, i, bias=False)
        self.down_proj = nn.Linear(i, h, bias=False)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class DecoderLayer(nn.Module):
    def __init__(self, config, index):
        super().__init__()
        self.input_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.self_attn = Attention(config, index)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.mlp = MLP(config)

    def forward(self, x, cos, sin, mask, cache):
        x = x + self.self_attn(self.input_layernorm(x), cos, sin, mask, cache)
        return x + self.mlp(self.post_attention_layernorm(x))


class ModelBody(nn.Module):
    """保持 checkpoint 的 model.* 命名，方便严格匹配权重。"""
    def __init__(self, config):
        super().__init__()
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([DecoderLayer(config, i) for i in range(config.num_hidden_layers)])
        self.norm = RMSNorm(config.hidden_size, config.rms_norm_eps)


class Qwen3(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.model = ModelBody(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        if config.tie_word_embeddings:
            self.lm_head.weight = self.model.embed_tokens.weight

    def new_cache(self, capacity):
        c = self.config
        if capacity > c.max_position_embeddings:
            raise ValueError("缓存容量超过模型位置上限。")
        weight = self.model.embed_tokens.weight
        return ContiguousKVCache(c.num_hidden_layers, c.num_key_value_heads, capacity,
                                 c.head_dim, device=weight.device, dtype=weight.dtype)

    @torch.inference_mode()
    def forward(self, input_ids, cache=None, *, last_only=True, trace=None):
        """trace(name,tensor) 仅供对拍；生产计时不启用中间张量拷贝。"""
        c = self.config
        if input_ids.ndim != 2 or input_ids.shape[0] != 1 or input_ids.shape[1] == 0:
            raise ValueError("只接受非空、无 padding 的 [1,Q] 输入。")
        if input_ids.dtype != torch.long or input_ids.device != self.model.embed_tokens.weight.device:
            raise ValueError("IDs 必须为同设备的 torch.long。")
        if bool(((input_ids < 0) | (input_ids >= c.vocab_size)).any().item()):
            raise ValueError("token ID 超出词表。")
        q_len = input_ids.shape[1]
        past_len = cache.length if cache is not None else 0
        if past_len + q_len > c.max_position_embeddings:
            raise ValueError("输入超过模型位置上限。")
        if cache is not None:
            if len(cache.keys) != c.num_hidden_layers:
                raise ValueError("缓存层数与模型不一致。")
            cache.begin_step(q_len)
        try:
            x = self.model.embed_tokens(input_ids)
            if trace is not None:
                trace("embedding", x)
            positions = torch.arange(past_len, past_len + q_len, device=x.device)
            cos, sin = rotary_factors(positions, c.head_dim, c.rope_theta, x.dtype)
            mask = causal_mask(q_len, past_len, device=x.device, dtype=x.dtype)
            for index, layer in enumerate(self.model.layers):
                x = layer(x, cos, sin, mask, cache)
                if trace is not None:
                    trace("layer.{}".format(index), x)
            x = self.model.norm(x)
            if trace is not None:
                trace("norm", x)
            logits = self.lm_head(x[:, -1:, :] if last_only else x)
            if cache is not None:
                cache.commit_step()
            return logits
        except BaseException:
            if cache is not None:
                cache.abort_step()
            raise
