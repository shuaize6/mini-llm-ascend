"""运行自研 M1 单请求生成，不调用 HF 模型 forward/generate。"""

import argparse
import statistics
import time

from bench.baseline import memory_snapshot, timing_metrics
from bench.common import environment_info, open_device, synchronize, write_report
from bench.m1_common import prepare_input, provenance


def run_once(runner, ids, args, eos, torch, device):
    """只测一次生成；模型、输入和连续 KV 容量已在计时前准备好。"""
    timestamps = []
    synchronize(torch, device)
    start = time.perf_counter()
    output = runner.generate(ids, args.max_new_tokens, eos_token_id=eos, fixed_length=args.fixed_length,
                             on_token=lambda _: timestamps.append(time.perf_counter()))
    synchronize(torch, device)
    end = time.perf_counter()
    if len(output) != len(timestamps) or (args.fixed_length and len(output) != args.max_new_tokens):
        raise RuntimeError("实际输出数量、观测数量或固定长度不一致。")
    return {"output_ids": output, "metrics": timing_metrics(start, end, timestamps)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--device", choices=("cpu", "npu"), default="npu")
    parser.add_argument("--dtype", choices=("float16", "float32", "bfloat16"))
    parser.add_argument("--prompt", default="请用一句话解释什么是KV缓存。")
    parser.add_argument("--max-new-tokens", type=int, default=8)
    parser.add_argument("--capacity", type=int, default=512)
    parser.add_argument("--fixed-length", action="store_true")
    parser.add_argument("--warmup", type=int, default=0, help="完整请求预热次数，不进入 summary")
    parser.add_argument("--runs", type=int, default=1, help="预热后的测量次数")
    parser.add_argument("--output", default="results/m1-generation.json")
    args = parser.parse_args()
    if args.max_new_tokens <= 0 or args.capacity <= 0 or args.runs <= 0 or args.warmup < 0:
        parser.error("输出预算、容量和 runs 必须为正，warmup 不能为负。")
    report = {"schema_version": 1, "status": "running", "stage": "setup", "args": vars(args),
              "environment": environment_info(),
              "timing_scope": "generate + host token observation; excludes model load, tokenize, input transfer, KV allocation, text decode",
              "warmup_policy": "full independent requests; warmup samples excluded from summary",
              "warnings": ["Small samples do not establish P99 or long-context performance.",
                           "Preallocated KV is reused as storage; each request resets logical cache state."]}
    write_report(args.output, report)
    try:
        torch, device, dtype = open_device(args.device, args.dtype)
        from mini_llm_ascend.engine.weights import load_model
        from mini_llm_ascend.engine.model_runner import ModelRunner
        report["provenance"] = provenance(args.model)
        tokenizer, ids, rendered, eos = prepare_input(args.model, args.prompt, device)
        report.update(input_ids=ids[0].cpu().tolist(), rendered_prompt=rendered, eos_token_id=eos,
                      device=str(device), dtype=str(dtype), stage="loading_model")
        write_report(args.output, report)
        runner = ModelRunner(load_model(args.model, device=device, dtype=dtype), args.capacity)
        report.update(stage="warmup", kv_capacity_bytes=runner.cache.nbytes, warmup_runs=[], runs=[])
        write_report(args.output, report)
        for index in range(args.warmup):
            report["stage"] = "warmup_{}".format(index + 1)
            write_report(args.output, report)
            report["warmup_runs"].append(run_once(runner, ids, args, eos, torch, device))
            write_report(args.output, report)
        if device.type == "npu":
            try:
                torch.npu.reset_peak_memory_stats()
            except Exception as exc:
                report["memory_reset_error"] = str(exc)
        for index in range(args.runs):
            report["stage"] = "measuring_run_{}".format(index + 1)
            write_report(args.output, report)
            result = run_once(runner, ids, args, eos, torch, device)
            result.update(text=tokenizer.decode(result["output_ids"], skip_special_tokens=True),
                          memory=memory_snapshot(torch, device))
            report["runs"].append(result)
            write_report(args.output, report)
            print("样本 {}：TTFT {:.2f} ms，TPOT {} ms，{:.3f} token/s".format(
                index + 1, result["metrics"]["ttft_ms"], result["metrics"]["tpot_ms"],
                result["metrics"]["output_tokens_per_s"]), flush=True)
        report["summary"] = {"sample_count": len(report["runs"]), "identical_output_ids": all(
            run["output_ids"] == report["runs"][0]["output_ids"] for run in report["runs"])}
        for key in ("ttft_ms", "tpot_ms", "generate_s", "output_tokens_per_s"):
            values = [run["metrics"][key] for run in report["runs"] if run["metrics"][key] is not None]
            report["summary"][key] = {"mean": statistics.mean(values) if values else None,
                                      "median": statistics.median(values) if values else None,
                                      "min": min(values) if values else None, "max": max(values) if values else None}
        # 单次启动保留原有顶层字段；多次测量请读取 runs/summary。
        if args.runs == 1:
            report.update(report["runs"][0])
        report.update(status="ok", stage="complete")
        print(report["runs"][0]["text"], flush=True)
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
