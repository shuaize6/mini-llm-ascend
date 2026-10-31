"""严格加载本地单文件 safetensors；未知/遗漏键不静默跳过。"""

import json
from pathlib import Path

import torch
from safetensors import safe_open

from .model import ModelConfig, Qwen3


def load_model(path, *, device="cpu", dtype=torch.float32):
    path = Path(path)
    config = ModelConfig.from_dict(json.loads((path / "config.json").read_text(encoding="utf-8")))
    if (path / "model.safetensors.index.json").exists():
        raise ValueError("M1 当前只支持单文件 model.safetensors，不支持分片权重。")
    # meta 上只建立结构，避免先生成一份无用的大模型随机权重。
    with torch.device("meta"):
        model = Qwen3(config)
    model = model.to(dtype=dtype).to_empty(device="cpu")
    # to_empty 可能打破共享参数，装载前显式恢复绑定。
    if config.tie_word_embeddings:
        model.lm_head.weight = model.model.embed_tokens.weight
    targets = model.state_dict()
    with safe_open(str(path / "model.safetensors"), framework="pt", device="cpu") as source:
        keys = set(source.keys())
        optional = {"lm_head.weight"} if config.tie_word_embeddings else set()
        missing, extra = set(targets) - keys - optional, keys - set(targets)
        if missing or extra:
            raise ValueError("权重键不匹配：missing={}, extra={}".format(sorted(missing), sorted(extra)))
        for name in sorted(keys):
            if tuple(source.get_slice(name).get_shape()) != tuple(targets[name].shape):
                raise ValueError("权重 shape 不匹配：" + name)
        if config.tie_word_embeddings and "lm_head.weight" in keys:
            if not torch.equal(source.get_tensor("lm_head.weight"), source.get_tensor("model.embed_tokens.weight")):
                raise ValueError("声明共享的 embedding 与 lm_head 权重不一致。")
        with torch.no_grad():
            for name in sorted(keys):
                if config.tie_word_embeddings and name == "lm_head.weight":
                    continue
                value = source.get_tensor(name).to(dtype=dtype)
                if not bool(torch.isfinite(value).all().item()):
                    raise ValueError("转换后权重存在 NaN/Inf：" + name)
                targets[name].copy_(value)
    model = model.to(device).eval()
    if config.tie_word_embeddings:
        model.lm_head.weight = model.model.embed_tokens.weight
    return model
