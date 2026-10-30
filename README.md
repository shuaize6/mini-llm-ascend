# mini-llm-ascend

面向昇腾 310B 的教学型轻量 LLM 推理引擎，围绕 **KV Cache、连续批处理与性能分析**逐步实现。

An educational LLM inference engine for Ascend NPUs, exploring paged KV cache, continuous batching, and profiling.

> **当前状态：设计与开发准备阶段。** 仓库包含设计文档、配置草案和部署辅助脚本，尚无可运行的推理引擎。板端兼容性与端到端性能需要后续验证。

## 项目目标

- 从 Qwen3-0.6B 的单请求前向开始，理解 Prefill、Decode 和 KV Cache。
- 实现分页 KV 管理、连续批处理、流式服务与请求取消回收。
- 用同条件实验分析吞吐、延迟和内存，记录优化收益与代价。
- 在 profiling 证据支持下，选择热点尝试 AscendC 自定义算子。

项目借鉴 vLLM 等推理系统的公开设计，与 vLLM 官方无隶属关系。早期文档使用 `vLLM-ShengTeng` 名称；本地目录和预留 Python 包名暂保留 `vllm_shengteng`。

## 学习方式

英文标识符配中文解释，明确张量 shape、模块职责和状态变化。先实现易读的参考版本，再增加优化路径；每阶段配最小示例、必要测试和实验说明。

阅读与实现顺序：

`模型前向 → 连续 KV → 块管理 → 分页注意力参考实现 → 调度器 → 服务 → 性能分析 → 算子优化`

## 路线与进度

| 阶段 | 内容 | 状态 |
| --- | --- | --- |
| 方案 | 可行性评审、首版范围、学习路线 | 已整理 |
| A | 环境探测、模型加载、HF 基线 | 待实现与验证 |
| B | 手写前向、连续 KV、正确性对拍 | 待实现 |
| C | 分页 KV、连续批处理、流式服务 | 待实现 |
| D | Chunked Prefill、Prefix Caching、热点优化 | 后续阶段 |
| E | 性能报告与复现指南 | 待实验 |

投机解码、W8A16 和完整 PagedAttention kernel 为扩展方向，不作为首版交付前提。gather + dense 注意力用于验证分页 KV 管理，其性能需要实测。

## 文档

- [可行性评估与学习路线](docs/可行性评估与学习路线.md)：优先阅读，包含设计勘误、验收关卡与学习约定。
- [原始技术设计](docs/技术设计文档.md)：详细架构与扩展设想，结合评审说明阅读；预估数字不是已实现成果。

## 目录

```text
mini-llm-ascend/
├── vllm_shengteng/    引擎、缓存、服务及算子绑定的预留目录
├── ascendc/           自定义算子预留目录
├── bench/            性能实验预留目录
├── tests/            正确性测试预留目录
├── configs/          配置草案，暂非可运行配置
├── scripts/          板端部署辅助脚本
└── docs/             设计、评审与学习路线
```

## 目标环境

目标为香橙派 AIpro 8T / Ascend 310B4。既有环境记录为 CANN 8.1.RC1、Python 3.9.2、PyTorch 2.1.0 和板载 torch_npu 2.1.0.post2；后续将通过探测脚本生成可复现的环境记录与依赖锁定文件。

暂不直接升级板载软件栈。目标模型是 Qwen3-0.6B，模型权重不进入 Git。按模型配置计算，FP16 KV 为 112 KiB/token，block_size=16 时为 1.75 MiB/块；容量规划还需考虑激活、临时张量和共享系统内存。

## 板端部署辅助

脚本只用于同步源码，不会安装依赖或启动推理服务。首次使用前阅读脚本并检查目标目录。

当前 `board_init.sh` 面向开发板的 `/root/vllm_shengteng` 工作目录，会安装自动 checkout 的接收钩子；请勿在该目录存放需要保留的未提交代码。

```bash
# 开发板：首次初始化裸仓库与接收钩子
bash scripts/board_init.sh
```

```powershell
# 本机：将占位地址替换为自己的开发板地址，使用本机 SSH 配置认证
 git remote add board root@YOUR_BOARD_HOST:/root/vllm_shengteng.git
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
