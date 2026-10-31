"""用小张量检查真实 head_dim=128 的 RoPE；无需加载大模型权重。

同时保留旧的代数等价写法作为诊断对照，不用于模型前向。
"""

import argparse

from bench.common import environment_info, open_device, write_report
from bench.m1_common import compare_tensors


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "npu"), default="npu")
    parser.add_argument("--output", default="results/m1-rope.json")
    args = parser.parse_args()
    report = {"status": "running", "environment": environment_info(), "cases": []}
    try:
        torch, device, dtype = open_device(args.device, "float16")
        from transformers import Qwen3Config
        from transformers.models.qwen3.modeling_qwen3 import Qwen3RotaryEmbedding
        from mini_llm_ascend.engine.model import rotary_factors
        config = Qwen3Config(head_dim=128, rope_theta=1000000.0, sliding_window=None)
        reference = Qwen3RotaryEmbedding(config).to(device)
        x = torch.zeros(1, 1, 1024, device=device, dtype=dtype)
        for positions in (list(range(21)), [21], [22], [127]):
            ids = torch.tensor(positions, device=device)
            with torch.inference_mode():
                expected = reference(x, ids[None])
                current = rotary_factors(ids, 128, config.rope_theta, dtype)
                legacy_inv = 1.0 / (config.rope_theta ** (torch.arange(0, 128, 2, device=device).float() / 128))
                alternatives = {"legacy_device_frequency_outer": legacy_inv,
                                "cpu_frequency_outer": reference.inv_freq}
                comparisons = {"current_" + name: compare_tensors(a[:, 0], b, atol=0, rtol=0)
                               for name, a, b in zip(("cos", "sin"), current, expected)}
                for label, inv in alternatives.items():
                    angles = ids.float()[:, None] * inv[None, :]
                    angles = torch.cat((angles, angles), dim=-1)[None]
                    for name, actual, target in zip(("cos", "sin"), (angles.cos().to(dtype), angles.sin().to(dtype)), expected):
                        comparisons[label + "_" + name] = compare_tensors(actual, target, atol=0, rtol=0)
            report["cases"].append({"positions": positions, "comparisons": comparisons})
            write_report(args.output, report)
        report["passed"] = all(c["comparisons"][name]["passed"] for c in report["cases"] for name in ("current_cos", "current_sin"))
        report["status"] = "ok" if report["passed"] else "failed"
        code = 0 if report["passed"] else 1
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
