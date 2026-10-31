# M1 正确性验证与 RoPE 排错记录

日期：2026-10-31。目标模型为本地 Qwen3-0.6B，目标设备为 Ascend 310B4。这里记录正确性验证，不报告 M1 相对 HF 的速度提升。

## 最终 NPU 短输入结果

[完整对拍报告](m1-npu.json) 和 [终端日志](m1-npu-output.txt) 保存了本次结果。环境为板载 torch 2.1.0 / torch_npu 2.1.0.post2、CANN 8.1.RC1，计算 dtype 为 FP16。

输入 21 个 token，固定输出 4 个 token，容量 512，`atol=0.01, rtol=0.01`。结果为 `status=ok, passed=true`：

| 比较项目 | 本次结果 |
| --- | --- |
| 4 步 teacher forcing 中每层输出对 HF | 最大绝对误差均为 0 |
| Prefill 全位置和 Decode logits 对 HF | 最大绝对误差均为 0，top-1 一致率 100% |
| 自研缓存与完整前缀重算 | 4 步最大绝对误差均为 0 |
| HF 自身缓存与完整前缀重算控制组 | 4 步最大绝对误差均为 0 |
| 两套实现独立 greedy 生成 | IDs 均为 `[82707, 99982, 24360, 101158]` |

这里的零误差只描述当前短案例、相同设备和 dtype，不是跨设备或任意输入的保证。

## 测试集与实际启动命令

[开发板测试日志](unit-tests.txt) 记录全部 27 项测试通过，包括原有 M0 的 7 项和 M1 新增的 20 项。日志中故障注入测试会主动生成 `error/interrupted` 报告，最终套件结果仍为 `OK`，不要把这些预期测试输出误认成运行失败。

[自研生成报告](m1-generation.json) 与 [启动日志](m1-generation-output.txt) 验证了实际入口：

```bash
python -m bench.m1 --model ./models/Qwen3-0.6B --device npu \
  --max-new-tokens 8 --fixed-length --output results/m1-generation.json
```

实际输出 8 个 token，文本为“KV缓存是一种用于存储和检索”，IDs 与 [M0 已归档的 HF 8-token 基线](../m0-310b4/qwen3-npu.json) 相同。这是独立启动验证，不是重新运行的一组性能对照实验。命令未单独预热，编译缓存又受到此前运行影响，因此其计时既不代表稳定性能，也不是严格干净的冷启动测量。

## 实验口径

- 对拍使用相同权重、tokenizer、无 thinking 聊天模板、输入 IDs、计算 dtype 和 greedy/EOS 约束。
- HF 使用 Transformers 4.51.3 的 eager attention。先保存 CPU 参考张量，再释放 HF 模型、装载自研模型，避免两份完整模型同时驻留 NPU。
- 相同后续 token 的 teacher forcing 覆盖 embedding、28 层输出、final norm 和全位置 logits；另外检查自研与 HF 各自的缓存/完整前缀重算。
- 独立 greedy 生成还要比较完整输出 IDs。数值比较通过和 token 相同分别记录。
- 原始报告保存源码哈希、权重哈希、阈值和环境。NPU 实验使用 `MAX_COMPILE_CORE_NUMBER=2`、`TE_PARALLEL_COMPILER=2`，并串行运行。

权重 SHA256 为 `f47f71177f32bcd101b7573ec9171e6a57f4f4d31148d38e382306f42996874b`。上游 revision 仍未知，不用本地文件哈希冒充上游版本号。

## 一次真实的 NPU 数值排错

初版在 CPU 小模型上通过，但真实 NPU 第一轮 decode 超过了固定阈值 `atol=0.01, rtol=0.01`：Prefill 的 HF/M1 logits 最大绝对误差为 0；第一轮 decode 为 `0.0390625`，923 个 logits 元素超标。逐层误差从第 19 层（从 0 编号）起超过阈值，而自研缓存与自研完整前缀重算一致。

这段证据保存在 [修复前的部分报告](npu-before-rope-fix.partial.json)。该文件是在运行中复制出的快照，`status=running`，只有当时完成的案例，**不是完整成功报告**。后续发生 SSH 连接超时和开发板重启，临时目录中的原进程日志没有完整保留；现有证据不足以断言重启原因。

随后用 [RoPE 小探针](rope-npu.json) 隔离问题。head_dim=128、位置 21 时，旧路径在 NPU 上初始化频率，cos 与 HF 相差 `3.814697265625e-06`；同一位置改为 CPU 初始化频率后，逐元素组合与 HF 参考一致。最终代码同时保持 HF 的 CPU 频率初始化和 batched matmul 顺序，探针的 Prefill、21、22、127 位置 cos/sin 均完全一致。

探针的原始终端输出见 [RoPE 日志](rope-npu-output.txt)。

这个例子说明：公式等价并不保证后端浮点路径一致，小张量差异也可能在深层网络中放大。修复没有放宽 NPU 验收阈值。探针只证明选定位置的行为，不能据此外推所有位置和所有 CANN 版本。

## CPU 完整权重控制组

CPU 实验在 Windows 的隔离环境运行，Python 3.11、torch 2.11.0+cu126，实际计算设备为 CPU、dtype 为 FP32；这与板载 torch 2.1.0 环境分开记录，不做性能排名。

| 实验 | 输入 / 输出 token | 容差 | 结果 |
| --- | --- | --- | --- |
| [严格起点与控制组](cpu-calibration-strict.json) | 21 / 4 | atol=1e-5，rtol=1e-4 | 整体判失败；同形状 HF/M1 logits 完全一致，但两者各自缓存/重算都产生最大约 5.0545e-5 的误差 |
| [另一 prompt 的验证](cpu-heldout.json) | 22 / 4 | 显式 atol=1e-4，rtol=1e-4 | 通过；HF/M1 logits 最大误差为 0，缓存/重算最大误差约 3.2902e-5，输出 IDs 相同 |

严格报告没有被覆盖成“通过”。先观察 HF 自身控制组在不同矩阵形状下的舍入差，再在另一 prompt 上验证明确记录的容差。该容差仅是本轮短输入实验的依据，不是通用精度保证；CPU 小模型单测仍保留更严格的阈值。

## 来源与限制

开发时的 NPU 验证先在临时源码副本运行，之后改到项目内 `results/m1-validation/`。历史报告中的执行路径保持原样；这些运行发生在提交前，不能只凭 `git_commit` 判断实际源码，必须同时检查 `provenance.source_sha256`。日常运行请使用项目根目录及 [M1 运行指南](../../M1运行与学习指南.md)。

最终 NPU 对拍、CPU 控制组/另一 prompt 验证及 8-token 生成报告的源码指纹，已逐文件核对，与实现提交 `83ddd67` 的 Git 文件内容一致。修复前部分报告对应旧实现，刻意保留其旧指纹，不能用它代表最终代码。

本阶段范围是单请求、无 padding、连续 KV 和 greedy。短输入对拍不能证明长上下文、并发、分页缓存、随机采样或不同设备间完全一致。NPU 的 CPU fallback、storage_offset 等后端告警需要随日志保留；同设备 HF 一致性也不能替代与高精度跨设备参考的完整精度验证。
