# M0：Ascend 310B4 上的 Qwen3-0.6B 单请求初测

整理日期：2026-10-31。基线原始报告时间为 2026-10-30 14:33 UTC（北京时间 22:33）。

**结论：环境探测、HF 模型加载与短请求生成链路已跑通。以下是短负载的初步性能锚点；完整模型数值正确性尚未完成对拍。**

## 环境与负载

- Ascend 310B4，CANN 8.1.RC1，Python 3.9.2。
- torch 2.1.0，torch_npu 2.1.0.post2+git64bdab5。
- Transformers 4.51.3，huggingface-hub 0.30.2，tokenizers 0.21.4，safetensors 0.7.0，einops 0.8.1。
- Qwen3-0.6B，FP16，eager attention，batch=1，KV Cache 开启。
- prompt：“请用一句话解释什么是KV缓存。”；经过 chat template 后为 **21 个输入 token**。
- 非思考模式、greedy；`use_model_defaults=False`；固定 **8 个输出 token**。
- 同一进程预热 1 轮，再连续测量 3 轮。未测试多请求、长上下文或持续负载。
- 运行源码对应部署提交 `8e9009c54a0ea9064a2a786594dba028ac51472c`，基线报告同时保存了各脚本 SHA-256。

板端使用裸仓库部署的工作副本，没有 `.git` 目录，因此报告内 Git 查询失败是预期情况；请用 `deployed_commit` 和 `benchmark_source_sha256` 追溯运行版本。

## 预热后结果

| 样本 | TTFT（ms） | TPOT（ms/token） | 输出吞吐（token/s） |
| --- | ---: | ---: | ---: |
| 1 | 144.97 | 155.43 | 6.48 |
| 2 | 165.28 | 155.66 | 6.37 |
| 3 | 153.09 | 147.67 | 6.73 |
| 算术平均 | **154.45** | **152.92** | **6.53** |

三轮输出 token IDs 完全一致。解码文本为：“KV缓存是一种用于存储和检索”。输出长度受实验预算限制，不能据此评估完整回答质量。

模型加载及搬到设备耗时约 5.03 秒，未计入上表。NPU 框架分配器记录的峰值 allocated 为 1,200,909,312 字节（约 1.12 GiB），不包含整个系统和所有驱动内存。

TTFT 从已分词、已传输输入后的 `generate` 开始计时，到 host 收到首 token；TPOT 是后续 host token 间隔平均值；吞吐为实际输出 token 数除以整个生成调用耗时。结果包含 HF 框架、host 观测和相关数据搬运成本，不包含网络与服务端排队。

原始数据：[qwen3-npu.json](qwen3-npu.json)。

## 首次运行的成本

前一轮首次成功短生成没有进程内预热，TTFT 约 **315.57 秒**，8-token 生成调用耗时约 **1,026.95 秒**。其运行中持续出现编译活动，不能用这个吞吐代表预热后的性能。

这不是清空全部编译缓存后的严格冷启动实验：此前已经做过算子探测和模型诊断；该轮也没有启用固定长度 EOS 抑制。这里只保留它作为启动成本的诊断证据，不与上表计算“优化加速倍数”。原始数据：[qwen3-smoke.json](qwen3-smoke.json)。

## 正确性与限制

- matmul、index_select、softmax、argmax 的小张量 NPU 对拍通过；阈值和误差见 [environment-npu.json](environment-npu.json)。
- 7 项测试通过：计时边界、单 token、无 token/非单调时钟、跳过 prompt、中断保存，以及随机小 Qwen3 的提前 EOS、固定长度与 HF 直接生成对照。小模型集成测试运行在开发板 CPU 上。
- 日志包含 `aten::isin` 回退 CPU、64 位 add 性能提示和 `storage_offset ... result is untrustworthy` 警告。原始运行日志见 [runtime-output.txt](runtime-output.txt)。因此只能称为选择 NPU 设备的混合执行链路，不能宣称全部算子在 NPU 上执行或整模型精度已经验证。
- `npu-smi` 的 Health 字段为 Alarm。此次没有排除其原因，也没有完成温度/频率的持续采样，结果仅描述这次设备状态。
- 三个短请求不能推出稳定 P99、长上下文吞吐或生产服务能力。下一阶段应做完整 logits/逐层对拍，并扩大输入输出长度与样本量。

## 权重与复现

本机与板端权重重新传输后，SHA-256 一致：

```text
f47f71177f32bcd101b7573ec9171e6a57f4f4d31148d38e382306f42996874b  model.safetensors
```

这是本次本地权重副本的身份记录，不是官方模型 revision 认证。首次传输校验不一致的副本未用于上表结果。

安装与探测方法见 [M0 指南](../../M0环境与基线.md)。复现上表使用：

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
/root/mini-llm-m0-venv/bin/python -m bench.baseline \
  --model /root/models/Qwen3-0.6B --device npu \
  --max-new-tokens 8 --fixed-length --warmup 1 --runs 3 \
  --output results/qwen3-npu.json
```

后续更改 dtype、attention 实现、输入长度或输出长度时，可能触发新的编译。32-token 实验尚未执行，不能直接沿用这组数字。
