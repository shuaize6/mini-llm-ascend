# mini-llm-ascend

面向昇腾 310B 的教学型轻量 LLM 推理引擎，围绕 **KV Cache、连续批处理与性能分析**逐步实现。

An educational LLM inference engine for Ascend NPUs, exploring paged KV cache, continuous batching, and profiling.

> **当前状态：M0 环境探测与 HF 单请求基线已实现。** 包含中文学习说明、环境报告、小模型测试和离线基准脚本；自研推理引擎尚未实现。完整模型数值正确性将在 M1 对拍验证。

## 项目目标

- 从 Qwen3-0.6B 的单请求前向开始，理解 Prefill、Decode 和 KV Cache。
- 实现分页 KV 管理、连续批处理、流式服务与请求取消回收。
- 用同条件实验分析吞吐、延迟和内存，记录优化收益与代价。
- 在 profiling 证据支持下，选择热点尝试 AscendC 自定义算子。

项目借鉴 vLLM 等推理系统的公开设计，与 vLLM 官方无隶属关系。项目名统一为 `mini-llm-ascend`，Python 包名使用合法的下划线形式 `mini_llm_ascend`。

## 学习方式

英文标识符配中文解释，明确张量 shape、模块职责和状态变化。先实现易读的参考版本，再增加优化路径；每阶段配最小示例、必要测试和实验说明。

阅读与实现顺序：

`模型前向 → 连续 KV → 块管理 → 分页注意力参考实现 → 调度器 → 服务 → 性能分析 → 算子优化`

## 路线与进度

| 阶段 | 内容 | 状态 |
| --- | --- | --- |
| 方案 | 可行性评审、首版范围、学习路线 | 已整理 |
| A | 环境探测、模型加载、HF 基线 | 已完成板端短请求初测，7 项测试通过 |
| B | 手写前向、连续 KV、正确性对拍 | 待实现 |
| C | 分页 KV、连续批处理、流式服务 | 待实现 |
| D | Chunked Prefill、Prefix Caching、热点优化 | 后续阶段 |
| E | 性能报告与复现指南 | 待实验 |

投机解码、W8A16 和完整 PagedAttention kernel 为扩展方向，不作为首版交付前提。gather + dense 注意力用于验证分页 KV 管理，其性能需要实测。

## 文档

- [M0 环境探测与单请求基线](docs/M0环境与基线.md)：安装、运行、指标口径与代码阅读顺序。
- [M0 代码设计与原理](docs/M0代码设计与原理.md)：逐函数说明、Prefill/Decode/KV Cache、张量形状、计时公式与 7 项测试解读。
- [M1 代码设计与原理](docs/M1代码设计与原理.md)：待实现设计，讲解手写 Qwen3 前向、连续 KV 状态、生成循环与 HF 对拍方案。
- [目录与部署](docs/目录与部署.md)：统一的板端目录、环境激活及 Git 推送方式。
- [310B4 初测与原始报告](docs/experiments/m0-310b4/README.md)：21-token 输入、8-token 输出，预热后 3 轮平均 TTFT 154.45 ms、吞吐 6.53 token/s；含 CPU 回退与数值警告，完整精度待对拍。
- [可行性评估与学习路线](docs/可行性评估与学习路线.md)：优先阅读，包含设计勘误、验收关卡与学习约定。
- [原始技术设计](docs/技术设计文档.md)：详细架构与扩展设想，结合评审说明阅读；预估数字不是已实现成果。

## 目录

```text
mini-llm-ascend/
├── .git/             Git 历史和配置
├── .venv/            板端 Python 环境（不入库）
├── models/           模型权重（不入库）
├── results/          本地实验输出（不入库）
├── mini_llm_ascend/   引擎、缓存、服务及算子绑定的预留目录
├── ascendc/           自定义算子预留目录
├── bench/            环境探测与 HF 单请求基线
├── tests/            计时边界、固定长度与提前 EOS 测试
├── configs/          配置草案，暂非可运行配置
├── scripts/          板端部署辅助脚本
└── docs/             设计、评审与学习路线
```

## 目标环境

目标为香橙派 AIpro 8T / Ascend 310B4。环境记录为 CANN 8.1.RC1、Python 3.9.2、PyTorch 2.1.0 和板载 torch_npu 2.1.0.post2。M0 使用隔离环境安装 `requirements-m0.txt` 中的模型加载依赖，探测脚本保存实际版本。

```bash
# 在仓库根目录，用已准备好的 Python 环境运行
python -m bench.probe_env --device npu
python -m bench.baseline --model ./models/Qwen3-0.6B --device npu \
  --max-new-tokens 1 --warmup 0 --runs 1
```

暂不直接升级板载软件栈。目标模型是 Qwen3-0.6B，模型权重不进入 Git。按模型配置计算，FP16 KV 为 112 KiB/token，block_size=16 时为 1.75 MiB/块；容量规划还需考虑激活、临时张量和共享系统内存。

## 板端部署辅助

脚本只用于同步源码，不会安装依赖或启动推理服务。首次使用前阅读脚本并检查目标目录。

板端统一使用 `/root/mini-llm-ascend/`，其中的 `.git` 是普通 Git 仓库，`.venv` 是项目环境。Git 接收推送时会更新工作副本；已跟踪文件有未提交改动时会拒绝更新，需要先处理改动。

```bash
# 开发板：首次初始化项目目录和环境
bash scripts/board_init.sh
```

```powershell
# 本机：将占位地址替换为自己的开发板地址，使用本机 SSH 配置认证
git remote add board root@YOUR_BOARD_HOST:/root/mini-llm-ascend
git push board main

# 后续也可使用辅助脚本
.\scripts\deploy.ps1
```

GitHub 远端 `origin` 与部署远端 `board` 各自独立。凭据、模型与设备私有配置不应提交到仓库。

## 参考资料

- [vLLM / PagedAttention 论文](https://arxiv.org/abs/2309.06180)
- [Qwen3-0.6B 官方模型](https://huggingface.co/Qwen/Qwen3-0.6B)
- [vLLM-Ascend](https://github.com/vllm-project/vllm-ascend)
- [昇腾文档中心](https://www.hiascend.com/document)
