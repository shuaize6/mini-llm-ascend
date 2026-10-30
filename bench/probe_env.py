"""环境探测：python -m bench.probe_env --device npu。

只读环境信息并运行小张量，不安装软件，不改变 CANN 或驱动。
"""

import argparse
import time
from pathlib import Path

from bench.common import command_result, environment_info, open_device, synchronize, write_report


def smoke_tests(torch, device, dtype):
    """与 CPU FP32 参考对比：能调用算子不等于算对了。"""
    torch.manual_seed(0)
    x_cpu = torch.randn(16, 32)
    w_cpu = torch.randn(32, 24)
    # 参考输入先转目标 dtype 再升回 FP32，排除输入舍入造成的误差。
    x_cpu = x_cpu.to(dtype).float()
    w_cpu = w_cpu.to(dtype).float()
    x, w = x_cpu.to(device, dtype), w_cpu.to(device, dtype)
    indices = torch.tensor([0, 3, 7, 15], device=device)
    cases = {
        "matmul": (lambda: x @ w, x_cpu @ w_cpu),
        "index_select": (lambda: x.index_select(0, indices), x_cpu[[0, 3, 7, 15]]),
        "softmax": (lambda: torch.softmax(x, dim=-1), torch.softmax(x_cpu, dim=-1)),
        "argmax": (lambda: x.argmax(dim=-1), x_cpu.argmax(dim=-1)),
    }
    results = {}
    for name, (operation, reference) in cases.items():
        print("正在检查 {}（首次调用可能编译算子）……".format(name), flush=True)
        try:
            for _ in range(2):
                operation()
            synchronize(torch, device)
            start = time.perf_counter()
            for _ in range(5):
                actual = operation()
            synchronize(torch, device)
            elapsed_ms = (time.perf_counter() - start) * 1000 / 5
            actual = actual.detach().cpu().float()
            reference = reference.float()
            tolerance = 0.05 if dtype == torch.bfloat16 else 0.01
            finite = bool(torch.isfinite(actual).all())
            ok = torch.allclose(actual, reference, atol=tolerance, rtol=tolerance)
            results[name] = {
                "ok": bool(ok), "shape": list(actual.shape),
                "finite": finite,
                "max_abs_error": (actual - reference).abs().max().item() if finite else None,
                "mean_ms": elapsed_ms, "atol": tolerance, "rtol": tolerance,
            }
        except Exception as exc:
            results[name] = {"ok": False, "error": "{}: {}".format(type(exc).__name__, exc)}
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "npu"), default="npu")
    parser.add_argument("--dtype", choices=("float16", "float32", "bfloat16"))
    parser.add_argument("--output", default="results/environment.json")
    args = parser.parse_args()
    report = {"schema_version": 1, "environment": environment_info(),
              "requested_device": args.device, "ok": False}
    report["npu_smi"] = command_result(["npu-smi", "info"])
    report["system_memory"] = command_result(["free", "-m"])
    cann = Path("/usr/local/Ascend/ascend-toolkit/latest/aarch64-linux/ascend_toolkit_install.info")
    report["cann_install_info"] = cann.read_text(errors="replace") if cann.is_file() else None
    try:
        torch, device, dtype = open_device(args.device, args.dtype)
        report["actual_device"] = str(device)
        report["dtype"] = str(dtype)
        if device.type == "npu":
            report["device_name"] = torch.npu.get_device_name(0)
        report["smoke_tests"] = smoke_tests(torch, device, dtype)
        report["ok"] = all(item["ok"] for item in report["smoke_tests"].values())
    except Exception as exc:
        report["error"] = "{}: {}".format(type(exc).__name__, exc)
    write_report(args.output, report)
    print("探测通过" if report["ok"] else "探测失败，请查看报告中的 error 或失败算子。")
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
