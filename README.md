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

当前有 20 个可运行实验，分成两条完整主线：

**Training · A1–A12**

- A1 Mini Megatron TP
- A2 Mini TP Backward
- A3 Data Parallel + Gradient Accumulation
- A4 Mini Distributed Optimizer
- A5 Bucketed Async Reduce-Scatter
- A6 TP × DP 2D Topology
- A7 Profiler-ready Overlap
- A8 Mini Sequence Parallel
- A9 Mini Pipeline Parallel
- A10 Mini Context Parallel
- A11 Mini Expert Parallel
- A12 Mini Training System

**Inference · B1–B8**

- B1 Mini KV Handoff
- B2 Mini Block KV Handoff
- B3 Mini Async KV Transfer
- B4 Layer-wise KV Streaming
- B5 KV Handshake + Lifetime
- B6 Registered Region Descriptor
- B7 KV Lease / Expiry
- B8 Mini Inference Engine

默认支持 CPU/Gloo；检测到足够 CUDA GPU 时自动切到 NCCL。实验首先验证 correctness 与依赖结构，不把小 tensor / CPU timing 当成真实 GPU、NIXL 或 RDMA benchmark。

Megatron 05 模块与 KV Connector 08 模块的核心课程现在会在 checkpoint 之后直接给出对应的 **HANDS-ON · 对应实验**：A1–A12 与 B1–B8 都能从概念课直接落到最小可运行实验，再回到 Source Map / 上游源码，而不需要读者先离开课程去 Labs 首页猜下一步。

## Source-reading workflow

**Source Map** 不再按仓库目录罗列文件，而是按 `Control → State → Data → Sync` 的阅读镜头组织，并把每个关键源码入口直接映射到对应课程或 Lab。

Megatron 主线：

`TP layers/mappings → parallel groups → PP schedule/P2P → CP attention / EP dispatcher → DDP gradient sync → distributed optimizer → param/grad buckets + overlap`

vLLM / NIXL 主线：

`Engine/Scheduler → KV block state → GPU ModelRunner → KVConnector contract → NIXL scheduler lifecycle → metadata → memory registration → pull / push transfer + completion/failure`

这样读源码时先知道“这一层正在解决什么问题”，再进入 class、buffer、process group 和 transfer handle。

## Source audit snapshot

课程中的版本敏感表述会优先按当前上游源码复查，而不是把旧教程中的类名、默认值或实现细节当成永久定义。

最近一次系统复查（2026-09-13）对照：

- vLLM `main`: `7ee8a6dd013819838da8012ca549d724bee7c6c6`
- NVIDIA/Megatron-LM `main`: `4fe0daffe4fc45efc231efc7b5bfc3018d81d516`

本轮按课程真正依赖的版本敏感边界做了定点核验，而不是机械追逐每个上游提交。vLLM 侧重新检查了 Scheduler 的 token budget 与 Prefill/Decode 调度语义、`gpu_worker.py` 的 V1/V2 ModelRunner selector、KVCacheManager / Prefix Cache、KV load failure policy、`KVConnectorBase_V1.get_transfer_results()` 的完成/失败闭环，以及 NixlConnector 的 UCX 默认 backend、KV layout、pull/push 路径、handshake、registration geometry、lease/heartbeat 与 heterogeneous P/D 支持边界。Megatron 侧重新核对了训练执行主干、TP mappings/layers、PP schedules/P2P、Distributed Optimizer、`param_and_grad_buffer.py` 的 reduce-scatter/all-reduce 与 overlap 生命周期、CP attention 路径，以及 MoE router / dispatcher 的当前配置与实验性 flex backend。课程继续把这些具体默认值和支持矩阵标记为“当前实现”，把 TP / PP / SP / CP / EP、request lifecycle、KV ownership 与 communication dependency 作为更稳定的主心智模型。网页中的源码链接仍指向上游 `main`；上述 SHA 只表示这轮语义复查使用的具体快照，上游继续演化后应重新核验版本敏感内容。

## Reading tools

- **Source Map**：按问题、数据流与生命周期阅读 Megatron / vLLM，并直接跳到对应 Lab。
- **Glossary**：精选跨课程反复出现、容易混淆或影响源码阅读的核心概念；完整课内术语仍放在每课末尾。
- **Labs**：先用小规模 reference 跑通机制，再回源码确认真实工程约束。

## Quality checks

每次 push 会运行站点检查，验证：HTML 阅读元数据、重复 ID、本地页面/fragment/asset 链接、CSS imports/`url(...)`、`app.js` 动态 lesson/Lab routes、已发布课程的旧 `locked` / `muted-next` 导航能否解析到真实目标、已知语义回归字符串、过期 placeholder，以及所有 Lab Python 源码的语法编译。

术语与文风现在有多层保护：`scripts/lesson_terms.py` 维护 canonical term registry 与课尾术语表；`scripts/audit_lesson_acronyms.py` 扫描新的高置信未注册缩写；`scripts/audit_first_use_terms.py` 按 01.1→08.5→Labs 的教学顺序检查技术英文首次出现是否带“中文术语 + 简短解释”，并清理后续重复解释；`scripts/audit_term_density.py` 与 `scripts/audit_language_mix.py` 继续检查术语密度、中文优先和异常中英混写；`scripts/check_glossary_terms.py` 只约束总站精选核心词，避免把 Glossary 膨胀成全部课内术语的复制品。

已人工验收为 lesson-native 的核心 SVG 另外由 `scripts/check_diagrams.py` 做宽度回归保护，防止重新退回“超宽小字号画布再整体缩小”的旧模式。

Lab 代码或 smoke harness 变化还会触发独立的 **Lab Correctness Smoke**：在干净的 GitHub runner 上安装 CPU PyTorch，用 `torchrun` / Gloo 实际执行 A1–A12 与 B1–B8 共 20 个实验。每个实验都必须正常退出并输出自身的独立 `PASS`，否则 workflow 失败。这个检查验证 reference 数值、collective/state-machine correctness 与脚本可运行性；CPU timing 仍不被解释为 GPU、NCCL、NIXL、RDMA 或 GPUDirect 性能 benchmark。

这个轻量 CI 不会伪装成 GPU/NCCL/NIXL 性能测试；需要真实硬件才能下结论的内容，在课程和 Lab 中会显式标出。

## Site

GitHub Pages: https://leonakaka.github.io/aiinfra/

- Labs: https://leonakaka.github.io/aiinfra/labs/
- Source Map: https://leonakaka.github.io/aiinfra/source-map/
- Glossary: https://leonakaka.github.io/aiinfra/glossary/
