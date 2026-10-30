# NanoServe-Ascend

面向昇腾 310B4（香橙派 AIpro 8T）的手写轻量 LLM 推理引擎。

**技术路线：混合栈** —— torch_npu 主引擎（调度/缓存/API）+ AscendC 原生 kernel（热点算子）。
**目标模型：Qwen3-0.6B**（28 层 / hidden 1024 / GQA 16:8 / head_dim 128 / vocab 151936）

> 这不是 vLLM 的移植，而是一次受限平台上的系统设计练习：
> 这块板子上**没有 PagedAttention 算子**（CANN 8.1 的 211 个 310B 算子中不存在，torch_npu / MindSpore 也没有），
> 所以分页注意力必须自己写——先写 torch 版（gather + 掩码 dense），再写 AscendC kernel 版。

---

## 状态

| 阶段 | 内容 | 状态 |
| --- | --- | --- |
| 设计 | 技术设计文档 v0.2 | ✅ 完成 |
| M0 | 环境标定 + 模型部署 + HF 基线 | ⬜ 未开始 |
| M1 | 手写 forward（含 q/k-norm）+ KV cache | ⬜ |
| M2 | PagedAttention（torch 版） | ⬜ |
| M3 | Continuous Batching + Chunked Prefill | ⬜ |
| M4 | OpenAI 兼容 API + 流式 + 指标 | ⬜ |
| M5 | 框架层算子融合 + 消融 | ⬜ |
| M6 | Prefix Caching | ⬜ |
| M7 | 投机解码（prompt-lookup） | ⬜ |
| M8 | W8A16 量化 | ⬜ |
| M9 | 报告与复现 | ⬜ |
| M10 | AscendC 原生 kernel（冲刺） | ⬜ |

## 文档

- [技术设计文档](docs/技术设计文档.md) —— 架构、关键设计、内存/性能模型、里程碑、风险登记册

## 目录职责

```text
nanoserve/                ← 仓库根
├── nanoserve/            Python 包
│   ├── engine/           前向计算、调度器、采样
│   ├── cache/            块管理、分页 KV、前缀哈希
│   ├── api/              FastAPI / OpenAI 兼容 / SSE
│   ├── ops/              自定义算子绑定（pybind11 → aclnn）
│   └── utils/            配置、计时、日志
├── ascendc/              AscendC 算子工程（op_host tiling + op_kernel）
├── bench/                基准与消融脚本
├── tests/                单测与对拍
├── configs/              模型/引擎配置
├── scripts/              部署与板端初始化
└── docs/                 设计文档
```

## 环境（板上）

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh    # CANN 8.1.RC1
/usr/local/miniconda3/bin/python                      # Python 3.9.2 + torch 2.1.0 + torch_npu 2.1.0.post2
```

| 项 | 值 |
| --- | --- |
| 芯片 | Ascend 310B4，16 GB 内存（与 NPU 共享） |
| 算力实测 | ≈1.55 TFLOPS fp16（1024³ matmul，722 次/秒） |
| 模型权重 | Qwen3-0.6B，fp16 唯一参数 ≈1.11 GiB |
| KV 开销 | 112 KiB/token（block=16 时 1.75 MiB/块） |

## 开发与部署

**唯一真源是本仓库（笔记本）**，板子是部署目标：

```powershell
# 笔记本：提交并推送到板上（首次需先跑 board_init.sh）
git push board main
```

板上 `/root/nanoserve` 会通过 `post-receive` 钩子自动 checkout 成工作副本，可版本化回滚。

模型权重不进 git，放在板上 `/root/models/Qwen3-0.6B/`。

## 约束（重要）

- Python 3.9 与 torch 2.1.0 / torch_npu 2.1.0.post2 **不可升级**（310B 只有这一个可用轮子）。
- 板子无法访问 HuggingFace / ModelScope / GitHub / 华为 OBS；依赖与权重需在笔记本代理下载后传输。
- `npu_prompt_flash_attention` 在 310B 上不可用；`npu_rms_norm` / `npu_rotary_mul` / `npu_fusion_attention` 可用。