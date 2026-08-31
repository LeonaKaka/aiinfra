# AI Infra from Zero

一个从基础知识逐层学习到 **Megatron Core** 与 **vLLM KV Connector** 的学习型网站。

目标不是做名词百科，而是用同一条学习路径把：

`Tensor / Transformer → GPU → Distributed → Megatron → LLM Inference → vLLM → KV Connector`

真正串起来，并且让每个概念都能落到可运行实验与真实源码。

## Curriculum

1. Foundations
2. Transformer Fundamentals
3. GPU Systems
4. Distributed Systems for AI
5. Megatron Core
6. LLM Inference
7. vLLM
8. KV Connector

8 个核心模块正文已经搭建完成。

## Hands-on Labs

当前有 14 个可运行实验，分成两条完整主线：

**Training · A1–A7**

- A1 Mini Megatron TP
- A2 Mini TP Backward
- A3 Data Parallel + Gradient Accumulation
- A4 Mini Distributed Optimizer
- A5 Bucketed Async Reduce-Scatter
- A6 TP × DP 2D Topology
- A7 Profiler-ready Overlap

**Inference · B1–B7**

- B1 Mini KV Handoff
- B2 Mini Block KV Handoff
- B3 Mini Async KV Transfer
- B4 Layer-wise KV Streaming
- B5 KV Handshake + Lifetime
- B6 Registered Region Descriptor
- B7 KV Lease / Expiry

默认支持 CPU/Gloo；检测到足够 CUDA GPU 时自动切到 NCCL。实验首先验证 correctness 与依赖结构，不把小 tensor / CPU timing 当成真实 GPU、NIXL 或 RDMA benchmark。

## Source-reading workflow

**Source Map** 不再按仓库目录罗列文件，而是按 `Control → State → Data → Sync` 的阅读镜头组织，并把每个关键源码入口直接映射到对应课程或 Lab。

Megatron 主线：

`TP layers/mappings → parallel groups → PP schedule/P2P → distributed optimizer → param/grad buckets + overlap`

vLLM / NIXL 主线：

`Engine/Scheduler → KV block state → GPU model runner → KVConnector contract → NIXL scheduler lifecycle → metadata → memory registration → pull transfer/completion`

这样读源码时先知道“这一层正在解决什么问题”，再进入 class、buffer、process group 和 transfer handle。

## Source audit snapshot

课程中的版本敏感表述会优先按当前上游源码复查，而不是把旧教程中的类名、默认值或实现细节当成永久定义。

最近一次系统复查（2026-08-31）对照：

- vLLM `main`: `44fe2a392b71d52a8d72faf2f8278834379482c9`
- NVIDIA/Megatron-LM `main`: `713ea6ad3f373235bab32faa60ab6d470ab698c3`

网页中的源码链接仍指向上游 `main`，方便继续阅读最新代码；上面的 SHA 只是说明这轮课程语义核对时使用的具体快照。上游继续演化后，应重新复查版本敏感内容。

## Reading tools

- **Source Map**：按问题、数据流与生命周期阅读 Megatron / vLLM，并直接跳到对应 Lab。
- **Glossary**：只收录课程真正用到的概念，并解释它在训练/推理系统里为什么出现。
- **Labs**：先用小规模 reference 跑通机制，再回源码确认真实工程约束。

## Quality checks

每次 push 会运行站点检查，验证：HTML 阅读元数据、重复 ID、本地页面/fragment/asset 链接、CSS imports/`url(...)`、`app.js` 动态 lesson/Lab routes、已知语义回归字符串、过期 placeholder，以及所有 Lab Python 源码的语法编译。

这个轻量 CI 不会伪装成 GPU/NCCL/NIXL 性能测试；需要真实硬件才能下结论的内容，在课程和 Lab 中会显式标出。

## Site

GitHub Pages: https://leonakaka.github.io/aiinfra/

- Labs: https://leonakaka.github.io/aiinfra/labs/
- Source Map: https://leonakaka.github.io/aiinfra/source-map/
- Glossary: https://leonakaka.github.io/aiinfra/glossary/
