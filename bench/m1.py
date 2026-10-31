"""运行自研 M1 单请求生成，不调用 HF 模型 forward/generate。"""

import argparse
import time

from bench.baseline import memory_snapshot, timing_metrics
from bench.common import environment_info, open_device, synchronize, write_report
from bench.m1_common import prepare_input, provenance


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--device", choices=("cpu", "npu"), default="npu")
    parser.add_argument("--dtype", choices=("float16", "float32", "bfloat16"))
    parser.add_argument("--prompt", default="请用一句话解释什么是KV缓存。")
    parser.add_argument("--max-new-tokens", type=int, default=8)
    parser.add_argument("--capacity", type=int, default=512)
    parser.add_argument("--fixed-length", action="store_true")
    parser.add_argument("--output", default="results/m1-generation.json")
    args = parser.parse_args()
    if args.max_new_tokens <= 0 or args.capacity <= 0:
        parser.error("输出预算和容量必须为正。")
    report = {"schema_version": 1, "status": "running", "stage": "setup", "args": vars(args),
              "environment": environment_info(), "timing_scope": "unwarmed single request + host token observation; not an optimization comparison"}
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
        report.update(stage="generating", kv_capacity_bytes=runner.cache.nbytes)
        write_report(args.output, report)
        timestamps = []
        synchronize(torch, device)
        start = time.perf_counter()
        output = runner.generate(ids, args.max_new_tokens, eos_token_id=eos, fixed_length=args.fixed_length,
                                 on_token=lambda _: timestamps.append(time.perf_counter()))
        synchronize(torch, device)
        end = time.perf_counter()
        report.update(status="ok", stage="complete", output_ids=output,
                      text=tokenizer.decode(output, skip_special_tokens=True),
                      metrics=timing_metrics(start, end, timestamps), memory=memory_snapshot(torch, device))
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
