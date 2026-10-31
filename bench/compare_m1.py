"""同设备、同 dtype 的 HF / M1 逐层与逐 token 对拍；不用于性能计时。"""

import argparse
import gc
import math
from pathlib import Path

from bench.common import environment_info, open_device, synchronize, write_report
from bench.m1_common import compare_tensors, prepare_input, provenance


def capture_hf(model, input_ids, continuation, torch, atol, rtol):
    """HF 独立运行，保存 CPU 参考；下一阶段释放大模型再加载 M1。"""
    snapshots, controls, handles = [], [], []
    current = {}
    capturing = True
    def hook(name):
        def save(module, inputs, output):
            if not capturing:
                return
            value = output[0] if isinstance(output, tuple) else output
            current[name] = value.detach().float().cpu().clone()
        return save
    handles.append(model.model.embed_tokens.register_forward_hook(hook("embedding")))
    for index, layer in enumerate(model.model.layers):
        handles.append(layer.register_forward_hook(hook("layer.{}".format(index))))
    handles.append(model.model.norm.register_forward_hook(hook("norm")))
    cache = None
    prefix = input_ids
    try:
        for step in range(len(continuation)):
            current = {}
            ids = input_ids if step == 0 else torch.tensor([[continuation[step - 1]]], device=input_ids.device)
            if step:
                prefix = torch.cat((prefix, ids), dim=1)
            with torch.inference_mode():
                output = model(ids, past_key_values=cache, use_cache=True)
            cache = output.past_key_values
            current["logits"] = output.logits.detach().float().cpu().clone()
            snapshots.append(current)
            # 控制组：HF 自己的 cache 与重算也可能因 GEMM shape 而有舍入差。
            # 记录它，避免把参考后端固有的精度差异误判为缓存状态错误。
            capturing = False
            with torch.inference_mode():
                full = model(prefix, use_cache=False).logits[:, -ids.shape[1]:]
            controls.append(compare_tensors(current["logits"], full, atol=atol, rtol=rtol, logits=True))
            del full
            capturing = True
            print("HF teacher forcing {}/{}".format(step + 1, len(continuation)), flush=True)
    finally:
        for handle in handles:
            handle.remove()
    return snapshots, controls


def compare(args, report):
    torch, device, dtype = open_device(args.device, args.dtype)
    torch.set_num_threads(args.cpu_threads)
    from transformers import AutoModelForCausalLM, GenerationConfig
    from mini_llm_ascend.engine.model_runner import ModelRunner
    from mini_llm_ascend.engine.weights import load_model

    def stage(name):
        report["stage"] = name
        write_report(args.output, report)
        print(name, flush=True)

    stage("hashing_model")
    report["provenance"] = provenance(args.model)
    report["device"], report["dtype"] = str(device), str(dtype)
    tokenizer, ids, rendered, eos = prepare_input(args.model, args.prompt, device)
    report.update(input_ids=ids[0].cpu().tolist(), rendered_prompt=rendered, eos_token_id=eos)
    if ids.shape[1] + args.max_new_tokens > args.capacity:
        raise ValueError("prompt + 输出预算超过缓存容量。")
    stage("loading_hf")
    hf = AutoModelForCausalLM.from_pretrained(str(Path(args.model)), local_files_only=True,
        trust_remote_code=False, use_safetensors=True, torch_dtype=dtype, attn_implementation="eager").to(device).eval()
    if args.capacity > hf.config.max_position_embeddings:
        raise ValueError("缓存容量超过模型位置上限。")
    generation = GenerationConfig(do_sample=False, num_beams=1, use_cache=True,
        max_new_tokens=args.max_new_tokens, min_new_tokens=args.max_new_tokens,
        eos_token_id=eos, pad_token_id=tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id)
    report["generation_config"] = generation.to_dict()
    stage("hf_greedy")
    with torch.inference_mode():
        output = hf.generate(ids, attention_mask=torch.ones_like(ids), generation_config=generation, use_model_defaults=False)
    continuation = output[0, ids.shape[1]:].cpu().tolist()
    if len(continuation) != args.max_new_tokens:
        raise RuntimeError("HF 固定长度生成数量不符。")
    report["hf_output_ids"] = continuation
    stage("hf_teacher_forcing")
    reference, controls = capture_hf(hf, ids, continuation, torch, args.atol, args.rtol)
    report["hf_cache_vs_recompute"] = controls
    del hf, output
    gc.collect()
    if device.type == "npu":
        synchronize(torch, device)
        torch.npu.empty_cache()

    stage("loading_m1")
    model = load_model(args.model, device=device, dtype=dtype)
    cache = model.new_cache(args.capacity)
    report["kv_capacity_bytes"] = cache.nbytes
    report["cases"] = []
    prefix = ids
    for step, expected in enumerate(reference):
        stage("m1_teacher_forcing_{}".format(step))
        current = {}
        def trace(name, value):
            current[name] = value.detach().float().cpu().clone()
        inputs = ids if step == 0 else torch.tensor([[continuation[step - 1]]], device=device)
        if step:
            prefix = torch.cat((prefix, inputs), dim=1)
        logits = model(inputs, cache, last_only=False, trace=trace)
        current["logits"] = logits.detach().float().cpu()
        comparisons = {name: compare_tensors(current[name], tensor, atol=args.atol, rtol=args.rtol,
                                             logits=name == "logits") for name, tensor in expected.items()}
        # 重算完整前缀：独立验证自研 cache 的结果，不复用 HF Cache。
        full_logits = model(prefix, last_only=False)[:, -inputs.shape[1]:]
        comparisons["cache_vs_recompute"] = compare_tensors(logits, full_logits, atol=args.atol, rtol=args.rtol, logits=True)
        case = {"step": step, "input_ids": inputs[0].cpu().tolist(), "prefix_ids": prefix[0].cpu().tolist(),
                "cache_length": cache.length, "comparisons": comparisons,
                "passed": all(item["passed"] for item in comparisons.values())}
        report["cases"].append(case)
        write_report(args.output, report)
    cache.reset()
    del cache
    stage("m1_free_generation")
    generated = ModelRunner(model, args.capacity).generate(ids, args.max_new_tokens, eos_token_id=eos, fixed_length=True)
    report["m1_output_ids"] = generated
    report["m1_text"] = tokenizer.decode(generated, skip_special_tokens=True)
    report["identical_output_ids"] = generated == continuation
    report["first_generation_mismatch"] = next((i for i, pair in enumerate(zip(generated, continuation)) if pair[0] != pair[1]), None)
    report["passed"] = (report["identical_output_ids"] and all(case["passed"] for case in report["cases"])
                        and all(item["passed"] for item in controls))
    report["first_failed_comparison"] = next(({"step": case["step"], "name": name} for case in report["cases"]
        for name, item in case["comparisons"].items() if not item["passed"]), None)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--device", choices=("cpu", "npu"), default="npu")
    parser.add_argument("--dtype", choices=("float16", "float32", "bfloat16"))
    parser.add_argument("--prompt", default="请用一句话解释什么是KV缓存。")
    parser.add_argument("--max-new-tokens", type=int, default=4)
    parser.add_argument("--capacity", type=int, default=512)
    parser.add_argument("--cpu-threads", type=int, default=4)
    parser.add_argument("--atol", type=float)
    parser.add_argument("--rtol", type=float)
    parser.add_argument("--output", default="results/m1-compare.json")
    args = parser.parse_args()
    if min(args.max_new_tokens, args.capacity, args.cpu_threads) < 1:
        parser.error("输出预算、容量和线程数必须为正。")
    fp32 = (args.dtype or ("float16" if args.device == "npu" else "float32")) == "float32"
    args.atol = args.atol if args.atol is not None else (1e-5 if fp32 else 1e-2)
    args.rtol = args.rtol if args.rtol is not None else (1e-4 if fp32 else 1e-2)
    if any(not math.isfinite(v) or v < 0 for v in (args.atol, args.rtol)):
        parser.error("误差阈值必须为非负有限数。")
    return args


def main():
    args = parse_args()
    report = {"schema_version": 1, "status": "running", "args": vars(args), "environment": environment_info(),
              "threshold_policy": "initial explicit tolerances; not a calibrated universal accuracy guarantee",
              "scope": "same-device same-dtype HF comparison; not cross-device accuracy or performance",
              "passed": False, "cases": [], "warnings": ["保留终端 stderr；JSON 不自动捕获后端告警。"]}
    write_report(args.output, report)
    try:
        compare(args, report)
        report["status"] = "ok" if report["passed"] else "failed"
        report["stage"] = "complete"
        code = 0 if report["passed"] else 1
    except KeyboardInterrupt:
        report.update(status="interrupted", error="KeyboardInterrupt")
        code = 130
    except Exception as exc:
        report.update(status="error", error="{}: {}".format(type(exc).__name__, exc))
        code = 1
    write_report(args.output, report)
    print("M1 对拍状态：{}".format(report["status"]), flush=True)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
