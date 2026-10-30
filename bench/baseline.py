"""Qwen3 单请求 HF 基线：先测清楚，再实现自己的模型前向。

用 generate 的 streamer 观察 token，而不把打印/文本解码放入计时区间。
默认离线加载、非思考模式、greedy、eager attention、batch=1。
"""

import argparse
import hashlib
import json
import statistics
import time
from pathlib import Path

from bench.common import environment_info, open_device, synchronize, write_report


class TokenTimer:
    """HF 第一次 put 是整个 prompt，之后每次 put 才是一个新 token。

generate 在回调前将 token 搬到 CPU，因此时间戳表示 host 观察到 token
的时刻，包含设备到主机传输；它不是纯 NPU kernel 时间。
"""

    def __init__(self):
        self.skip_prompt = True
        self.timestamps = []

    def put(self, value):
        if self.skip_prompt:
            self.skip_prompt = False
            return
        if value.numel() != 1:
            raise ValueError("基线只允许 batch=1、每次输出一个 token。")
        self.timestamps.append(time.perf_counter())

    def end(self):
        pass


def timing_metrics(start, end, timestamps):
    """只计算真实观测的 token；单 token 的 TPOT 为 None。"""
    if not timestamps:
        raise ValueError("未观察到输出 token。")
    points = [start] + list(timestamps) + [end]
    if any(right < left for left, right in zip(points, points[1:])) or end <= start:
        raise ValueError("时间戳必须单调且总耗时大于零。")
    intervals = [(b - a) * 1000 for a, b in zip(timestamps, timestamps[1:])]
    duration = end - start
    return {
        "output_tokens": len(timestamps), "generate_s": duration,
        "ttft_ms": (timestamps[0] - start) * 1000,
        "tpot_ms": statistics.mean(intervals) if intervals else None,
        "itl_ms": intervals,
        "output_tokens_per_s": len(timestamps) / duration,
        "finalization_ms": (end - timestamps[-1]) * 1000,
    }


def memory_snapshot(torch, device):
    if device.type != "npu":
        return {"note": "CPU 模式不报告 NPU 内存。"}
    result = {}
    for name in ("memory_allocated", "memory_reserved", "max_memory_allocated"):
        try:
            result[name + "_bytes"] = getattr(torch.npu, name)()
        except Exception as exc:
            result[name + "_error"] = str(exc)
    return result


def model_manifest(path):
    """记录配置 hash 和权重文件大小；不将文件大小冒充权重校验和。"""
    files = {}
    for file in sorted(path.iterdir()):
        if file.is_file() and (file.suffix in (".json", ".safetensors", ".txt")):
            entry = {"size_bytes": file.stat().st_size}
            if file.suffix == ".json":
                entry["sha256"] = hashlib.sha256(file.read_bytes()).hexdigest()
            files[file.name] = entry
    return {"files": files, "weight_sha256": "not computed; use --model-revision to record provenance"}


def run_once(model, inputs, generation_config, torch, device):
    timer = TokenTimer()
    synchronize(torch, device)
    start = time.perf_counter()
    with torch.inference_mode():
        output = model.generate(
            **inputs, generation_config=generation_config, streamer=timer,
            logits_to_keep=1, use_model_defaults=False,
        )
    synchronize(torch, device)
    end = time.perf_counter()
    # output: [1, prompt_len + generated_len]；只保存新生成部分。
    generated = output[0, inputs["input_ids"].shape[1]:].detach().cpu().tolist()
    if len(generated) != len(timer.timestamps):
        raise RuntimeError("streamer 观测数量与实际输出长度不一致，不能报告性能。")
    result = timing_metrics(start, end, timer.timestamps)
    result["output_ids"] = generated
    return result


def benchmark(args, report):
    model_path = Path(args.model).resolve()
    if not (model_path / "config.json").is_file():
        raise FileNotFoundError("--model 必须是包含 config.json 的本地模型目录。")
    config = json.loads((model_path / "config.json").read_text(encoding="utf-8"))
    if config.get("model_type") != "qwen3":
        raise ValueError("本基线仅支持 Qwen3。")

    torch, device, dtype = open_device(args.device, args.dtype)
    from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

    torch.manual_seed(args.seed)
    report["actual_device"] = str(device)
    report["dtype"] = str(dtype)
    report["model_manifest"] = model_manifest(model_path)
    report["model_config"] = config

    # 不用 device_map='auto'，避免自动卸载或设备切换污染基线。
    print("开始离线加载 tokenizer 和 Qwen3；首次搬运可能较慢。", flush=True)
    load_start = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(
        str(model_path), local_files_only=True, trust_remote_code=False
    )
    model = AutoModelForCausalLM.from_pretrained(
        str(model_path), local_files_only=True, trust_remote_code=False,
        use_safetensors=True, torch_dtype=dtype, attn_implementation="eager",
    ).to(device).eval()
    synchronize(torch, device)
    report["load_s"] = time.perf_counter() - load_start

    text = tokenizer.apply_chat_template(
        [{"role": "user", "content": args.prompt}], tokenize=False,
        add_generation_prompt=True, enable_thinking=False,
    )
    # chat template 已包含特殊 token，不再额外插入。
    inputs = tokenizer(text, add_special_tokens=False, return_tensors="pt").to(device)
    prompt_tokens = inputs["input_ids"].shape[1]
    if prompt_tokens + args.max_new_tokens > config["max_position_embeddings"]:
        raise ValueError("prompt + 输出预算超过模型上下文上限。")
    report["input_ids"] = inputs["input_ids"][0].cpu().tolist()
    report["prompt_tokens"] = prompt_tokens
    report["rendered_prompt"] = text

    # 独立配置去掉模型默认采样参数；greedy 是可复现实验设置，不是质量推荐。
    generation_config = GenerationConfig(
        do_sample=False, num_beams=1, use_cache=True,
        max_new_tokens=args.max_new_tokens,
        min_new_tokens=args.max_new_tokens if args.fixed_length else 0,
        eos_token_id=model.generation_config.eos_token_id,
        bos_token_id=model.generation_config.bos_token_id,
        pad_token_id=tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id,
    )
    report["generation_config"] = generation_config.to_dict()
    report["use_model_defaults"] = False
    print("模型已加载：{} 个输入 token；开始预热。".format(prompt_tokens), flush=True)
    for _ in range(args.warmup):
        run_once(model, inputs, generation_config, torch, device)
    if device.type == "npu":
        try:
            torch.npu.reset_peak_memory_stats()
        except Exception as exc:
            report["memory_reset_error"] = str(exc)
    report["runs"] = []
    for index in range(args.runs):
        result = run_once(model, inputs, generation_config, torch, device)
        if args.fixed_length and result["output_tokens"] != args.max_new_tokens:
            raise RuntimeError("固定长度实验未生成指定数量的 token。")
        result["text"] = tokenizer.decode(result["output_ids"], skip_special_tokens=True)
        result["memory"] = memory_snapshot(torch, device)
        report["runs"].append(result)
        # 每轮结束后写文件，后续失败也保留已完成样本。
        write_report(args.output, report)
        print("样本 {}：{} tokens，TTFT {:.1f} ms，{:.2f} token/s".format(
            index + 1, result["output_tokens"], result["ttft_ms"], result["output_tokens_per_s"]), flush=True)
    report["summary"] = {}
    for key in ("ttft_ms", "tpot_ms", "generate_s", "output_tokens_per_s"):
        values = [run[key] for run in report["runs"] if run[key] is not None]
        report["summary"][key] = {
            "mean": statistics.mean(values) if values else None,
            "median": statistics.median(values) if values else None,
        }
    report["summary"]["sample_count"] = len(report["runs"])
    report["summary"]["identical_output_ids"] = all(
        run["output_ids"] == report["runs"][0]["output_ids"] for run in report["runs"]
    )
    print("生成文本：\n" + report["runs"][0]["text"])


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="本地 Qwen3 模型目录")
    parser.add_argument("--model-revision", default="unknown-local-copy", help="人工记录来源 revision，不自动下载")
    parser.add_argument("--device", choices=("cpu", "npu"), default="npu")
    parser.add_argument("--dtype", choices=("float16", "float32", "bfloat16"))
    parser.add_argument("--prompt", default="请用一句话解释什么是KV缓存。")
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--fixed-length", action="store_true", help="抑制提前 EOS，固定输出长度，仅用于性能实验")
    parser.add_argument("--output", default="results/baseline.json")
    args = parser.parse_args()
    if args.max_new_tokens < 1 or args.runs < 1 or args.warmup < 0:
        parser.error("max-new-tokens/runs 必须为正数，warmup 不能为负数。")
    return args


def main():
    args = parse_args()
    report = {"schema_version": 1, "status": "running", "args": vars(args),
              "environment": environment_info(), "batch_size": 1,
              "timing_scope": "HF generate + host token observation; excludes load, tokenize, input transfer, text decode",
              "warnings": ["Greedy is a reproducibility setting, not a Qwen quality recommendation.",
                           "Small samples do not establish P99 latency; CPU is not an NPU performance baseline."]}
    try:
        benchmark(args, report)
        report["status"] = "ok"
    except Exception as exc:
        report["status"] = "error"
        report["error"] = "{}: {}".format(type(exc).__name__, exc)
        print(report["error"])
    write_report(args.output, report)
    return 0 if report["status"] == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
