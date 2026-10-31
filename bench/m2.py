"""M2 必要验收：一次加载，非连续块短对拍，再测 M1/M2 同时段性能。"""

import argparse
import gc
import math
import statistics

from bench.baseline import memory_snapshot
from bench.common import environment_info, open_device, synchronize, write_report
from bench.m1 import run_once
from bench.m1_common import compare_tensors, prepare_input, provenance


def validate(model, ids, args, eos, torch):
    """两步足以覆盖跨块 Prefill、gather 与 Decode；CPU 测试补全状态边界。"""
    from mini_llm_ascend.cache.paged_kv import PagedKVPool
    from mini_llm_ascend.engine.sampler import greedy_select, normalize_eos
    if ids.shape[1] <= args.block_size:
        raise ValueError("本对拍要求 prompt token 数大于 block_size，以覆盖跨块；请调整 prompt 或块大小。")
    needed = (ids.shape[1] + 1 + args.block_size - 1) // args.block_size
    pool = PagedKVPool.for_model(model, needed + 1, args.block_size)
    left, guard, right = [pool.new_cache(args.capacity) for _ in range(3)]
    paged = pool.new_cache(args.capacity)
    contiguous = model.new_cache(args.capacity)
    result = {"cases": [], "note": "validation-only small pool, distinct from benchmark pool"}
    try:
        # 仅预留块，不读取其中未写入的内容；保留块 1，令目标请求用 [0,2,...]。
        for cache in (left, guard, right):
            cache.begin_step(1)
        right.close()
        left.close()
        inputs = ids
        for name in ("prefill", "decode"):
            expected = model(inputs, contiguous)
            actual = model(inputs, paged)
            comparison = compare_tensors(actual, expected, atol=args.atol, rtol=args.rtol, logits=True)
            result["cases"].append({"name": name, "input_ids": inputs[0].cpu().tolist(),
                                    "block_table": list(paged.block_table), "length": paged.length,
                                    "comparison": comparison})
            token = greedy_select(expected, normalize_eos(eos, model.config.vocab_size), suppress_eos=True)
            inputs = torch.tensor([[token]], dtype=torch.long, device=ids.device)
        table = result["cases"][0]["block_table"]
        result["noncontiguous_layout"] = len(table) >= 2 and table[1] != table[0] + 1
        result["passed"] = result["noncontiguous_layout"] and all(c["comparison"]["passed"] for c in result["cases"])
    finally:
        for cache in (left, guard, right, paged):
            cache.close()
        contiguous.reset()
        result["pool_after_close"] = pool.snapshot()
    if result["pool_after_close"]["allocated_blocks"] != 0:
        raise RuntimeError("数值验证结束后仍有块未归还。")
    return result


def measure_backend(name, model, ids, args, eos, torch, device, result, save, expected_ids=None):
    from mini_llm_ascend.cache.paged_kv import PagedKVPool
    from mini_llm_ascend.engine.model_runner import ModelRunner
    pool = None
    if name == "paged":
        pool = PagedKVPool.for_model(model, args.num_blocks, args.block_size)
        runner = ModelRunner(model, cache=pool.new_cache(args.capacity))
        result["pool_before"] = pool.snapshot()
        result["kv_reserved_bytes"] = pool.nbytes
    else:
        runner = ModelRunner(model, args.capacity)
        result["kv_reserved_bytes"] = runner.cache.nbytes
    result.update(warmup_runs=[], runs=[])

    def sample():
        nonlocal expected_ids
        run = run_once(runner, ids, args, eos, torch, device)
        if expected_ids is None:
            expected_ids = run["output_ids"]
        if run["output_ids"] != expected_ids:
            raise RuntimeError("{} 输出 IDs 与控制组不一致。".format(name))
        if pool is not None:
            run["request_kv"] = runner.cache.last_request_stats
            run["pool_after_request"] = pool.snapshot()
            if run["pool_after_request"]["allocated_blocks"] != 0:
                raise RuntimeError("请求结束后分页池仍有未归还块。")
        return run

    try:
        for i in range(args.warmup):
            save("{}_warmup_{}".format(name, i + 1))
            result["warmup_runs"].append(sample())
            save("{}_warmup_{}_done".format(name, i + 1))
        if device.type == "npu":
            torch.npu.reset_peak_memory_stats()
        for i in range(args.runs):
            save("{}_run_{}".format(name, i + 1))
            run = sample()
            run["memory"] = memory_snapshot(torch, device)
            result["runs"].append(run)
            save("{}_run_{}_done".format(name, i + 1))
            print("{} 样本 {}：{:.3f} token/s".format(name, i + 1, run["metrics"]["output_tokens_per_s"]), flush=True)
        result["summary"] = {"sample_count": len(result["runs"]), "identical_output_ids": True}
        for key in ("ttft_ms", "tpot_ms", "generate_s", "output_tokens_per_s"):
            values = [r["metrics"][key] for r in result["runs"] if r["metrics"][key] is not None]
            result["summary"][key] = {"mean": statistics.mean(values) if values else None,
                                      "min": min(values) if values else None, "max": max(values) if values else None}
        result["output_ids"] = expected_ids
    finally:
        if pool is not None:
            runner.cache.close()
            result["pool_after_close"] = pool.snapshot()
        else:
            runner.cache.reset()
    return expected_ids


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--device", choices=("cpu", "npu"), default="npu")
    parser.add_argument("--dtype", choices=("float16", "float32", "bfloat16"))
    parser.add_argument("--prompt", default="请用一句话解释什么是KV缓存。")
    parser.add_argument("--capacity", type=int, default=512)
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--num-blocks", type=int, default=32)
    parser.add_argument("--max-new-tokens", type=int, default=8)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--atol", type=float)
    parser.add_argument("--rtol", type=float)
    parser.add_argument("--output", default="results/m2.json")
    parser.set_defaults(fixed_length=True)
    args = parser.parse_args()
    if min(args.capacity, args.block_size, args.num_blocks, args.max_new_tokens, args.runs) <= 0 or args.warmup < 0:
        parser.error("容量、块参数、输出长度和 runs 必须为正，warmup 不能为负。")
    fp32 = (args.dtype or ("float16" if args.device == "npu" else "float32")) == "float32"
    args.atol = args.atol if args.atol is not None else (1e-5 if fp32 else 0.01)
    args.rtol = args.rtol if args.rtol is not None else (1e-4 if fp32 else 0.01)
    if any(not math.isfinite(v) or v < 0 for v in (args.atol, args.rtol)):
        parser.error("容差必须为非负有限数。")
    return args


def main():
    args = parse_args()
    report = {"schema_version": 1, "status": "running", "stage": "setup", "args": vars(args),
              "environment": environment_info(), "passed": False,
              "timing_scope": "generate + host observation; excludes load, tokenize, input transfer, initial KV pool allocation and statistics; includes request block allocation/release and gather",
              "backend_order": ["contiguous", "paged"],
              "warnings": ["Small serial samples do not establish concurrency or P99 performance.",
                           "Paged gather is a reference path, not a native PagedAttention kernel."]}

    def save(stage):
        report["stage"] = stage
        write_report(args.output, report)
        print(stage, flush=True)

    save("setup")
    try:
        torch, device, dtype = open_device(args.device, args.dtype)
        from mini_llm_ascend.engine.weights import load_model
        save("hashing_model")
        report["provenance"] = provenance(args.model)
        tokenizer, ids, rendered, eos = prepare_input(args.model, args.prompt, device)
        report.update(input_ids=ids[0].cpu().tolist(), rendered_prompt=rendered,
                      device=str(device), dtype=str(dtype), eos_token_id=eos)
        if ids.shape[1] + args.max_new_tokens > args.capacity:
            raise ValueError("prompt + 输出预算超过请求 capacity。")
        save("loading_model")
        model = load_model(args.model, device=device, dtype=dtype)
        save("validating_noncontiguous_blocks")
        report["validation"] = validate(model, ids, args, eos, torch)
        save("validation_complete")
        if not report["validation"]["passed"]:
            raise RuntimeError("M1/M2 数值对拍未通过，停止性能测量。")
        report["backends"] = {}
        expected = None
        for name in report["backend_order"]:
            # 验证池和前一个后端不常驻，避免额外 KV 容量污染峰值统计。
            gc.collect()
            synchronize(torch, device)
            if device.type == "npu":
                torch.npu.empty_cache()
            report["backends"][name] = {}
            expected = measure_backend(name, model, ids, args, eos, torch, device,
                                       report["backends"][name], save, expected)
        report.update(status="ok", stage="complete", passed=True,
                      text=tokenizer.decode(expected, skip_special_tokens=True))
        print(report["text"], flush=True)
        code = 0
    except KeyboardInterrupt:
        report.update(status="interrupted", error="KeyboardInterrupt")
        code = 130
    except Exception as exc:
        report.update(status="error", error="{}: {}".format(type(exc).__name__, exc))
        code = 1
    write_report(args.output, report)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
