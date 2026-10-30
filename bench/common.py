"""基准共用的小工具：设备选择、记录环境和输出 JSON。

模块顶层只依赖标准库；即使没有安装 torch，也能生成诊断报告。
"""

import importlib.metadata
import hashlib
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


def command_result(command, timeout=15):
    """外部诊断命令失败也保留原因，不将“没采集到”写成零。"""
    try:
        result = subprocess.run(
            command, capture_output=True, text=True, errors="replace", timeout=timeout
        )
        return {"returncode": result.returncode, "stdout": result.stdout.strip(),
                "stderr": result.stderr.strip()}
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"error": str(exc)}


def environment_info():
    versions = {}
    for name in ("torch", "torch-npu", "transformers", "safetensors", "numpy",
                 "tokenizers", "huggingface-hub", "einops"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    root = Path(__file__).resolve().parents[1]
    deployed = root / ".deployed_commit"
    return {
        "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
        "python": sys.version, "executable": sys.executable,
        "platform": platform.platform(), "machine": platform.machine(),
        "packages": versions,
        "git_commit": command_result(["git", "-C", str(root), "rev-parse", "HEAD"]),
        "git_status": command_result(["git", "-C", str(root), "status", "--porcelain"]),
        "deployed_commit": deployed.read_text().strip() if deployed.is_file() else None,
        # 未提交代码也能追溯，避免仅记录 HEAD 而误认运行的是该提交。
        "benchmark_source_sha256": {
            str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted((root / "bench").glob("*.py"))
        },
    }


def open_device(name, dtype_name):
    """显式选择 CPU 或 NPU，NPU 失败时不悄悄回退。"""
    import torch

    if name == "npu":
        import torch_npu  # noqa: F401，导入后注册 torch.npu

        if not torch.npu.is_available():
            raise RuntimeError("NPU 不可用，请先检查 CANN 环境变量与驱动。")
        torch.npu.set_device(0)
        device = torch.device("npu:0")
    else:
        device = torch.device("cpu")
    dtype_name = dtype_name or ("float16" if name == "npu" else "float32")
    return torch, device, getattr(torch, dtype_name)


def synchronize(torch, device):
    if device.type == "npu":
        torch.npu.synchronize()


def write_report(path, report):
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False),
                      encoding="utf-8")
    print("报告已保存：{}".format(target))
