"""M1 命令共用：本地模型身份、输入准备、严格误差统计。"""

import hashlib
import json
from pathlib import Path


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def provenance(model_path):
    root = Path(__file__).resolve().parents[1]
    paths = sorted((root / "mini_llm_ascend").rglob("*.py")) + sorted((root / "bench").glob("*.py"))
    model_path = Path(model_path)
    files = [model_path / "model.safetensors"] + sorted(model_path.glob("*.json"))
    return {"source_sha256": {p.relative_to(root).as_posix(): file_sha256(p) for p in paths},
            "model_files": {p.name: {"size_bytes": p.stat().st_size, "sha256": file_sha256(p)} for p in files},
            "upstream_revision": "unknown-local-copy"}


def prepare_input(path, prompt, device):
    from transformers import AutoTokenizer
    path = Path(path)
    tokenizer = AutoTokenizer.from_pretrained(str(path), local_files_only=True, trust_remote_code=False)
    rendered = tokenizer.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False,
                                             add_generation_prompt=True, enable_thinking=False)
    ids = tokenizer(rendered, add_special_tokens=False, return_tensors="pt")["input_ids"].to(device)
    config = json.loads((path / "config.json").read_text(encoding="utf-8"))
    generation_path = path / "generation_config.json"
    generation = json.loads(generation_path.read_text(encoding="utf-8")) if generation_path.is_file() else {}
    eos = generation.get("eos_token_id", config.get("eos_token_id"))
    return tokenizer, ids, rendered, eos


def compare_tensors(actual, reference, *, atol, rtol, logits=False):
    """先搬到 CPU FP32 再统计，不让设备归约误差影响验收统计。"""
    import torch
    a, r = actual.detach().float().cpu(), reference.detach().float().cpu()
    if a.shape != r.shape:
        return {"passed": False, "reason": "shape_mismatch", "actual_shape": list(a.shape), "reference_shape": list(r.shape)}
    finite = bool(torch.isfinite(a).all() and torch.isfinite(r).all())
    result = {"shape": list(a.shape), "finite": finite, "passed": False}
    if not finite:
        result["reason"] = "nonfinite"
        return result
    error = (a - r).abs()
    relative = error / r.abs().clamp_min(1e-6)
    allowed = atol + rtol * r.abs()
    worst_flat = int(error.reshape(-1).argmax())
    result.update(passed=bool((error <= allowed).all()), max_abs=float(error.max()),
                  mean_abs=float(error.mean()), max_relative=float(relative.max()),
                  mean_relative=float(relative.mean()), worst_flat_index=worst_flat,
                  worst_actual=float(a.reshape(-1)[worst_flat]), worst_reference=float(r.reshape(-1)[worst_flat]),
                  failing_elements=int((error > allowed).sum()))
    if logits:
        same = a.argmax(-1) == r.argmax(-1)
        top = r.topk(min(2, r.shape[-1]), dim=-1)
        result["top1_agreement"] = float(same.float().mean())
        if not bool(same.all()):
            position = int((~same).reshape(-1).nonzero()[0])
            row_a, row_r = a.reshape(-1, a.shape[-1])[position], r.reshape(-1, r.shape[-1])[position]
            result["first_top1_mismatch"] = {"position_flat": position,
                "actual_id": int(row_a.argmax()), "reference_id": int(row_r.argmax()),
                "reference_top2_ids": top.indices.reshape(-1, top.indices.shape[-1])[position].tolist(),
                "reference_top2_gap": float(row_r.topk(2).values.diff().abs()[0]) if row_r.numel() >= 2 else None}
    return result
