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

当前有 18 个可运行实验，分成两条完整主线：

**Training · A1–A11**

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

最近一次系统复查（2026-09-01）对照：

- vLLM `main`: `f9c7c6e0909eadc23f1aa2510a233f91692ed437`
- NVIDIA/Megatron-LM `main`: `6c440e61fd5f8e6ad526d3225c02ec85be8e8fa8`

这轮继续复核了 vLLM GPU runner selector / V1-V2 runner 分流、Scheduler、KV Connector / NIXL lifecycle，以及 Megatron Expert Parallel dispatcher、Distributed Optimizer、Context Parallel 与 communication-overlap 生命周期。相对上一快照，vLLM 又前进了 5 个提交：大部分集中在测试与 ROCm MLA，但其中 KV cache coordinator / prefix-cache tests 已明确覆盖 hybrid full-attention + Mamba `align` 的 fine-grained partial prefix-cache hits，因此 07.5 已补上“完整 hash block 是基础心智模型、不是所有当前路径的全局不变量”这一例外；这些提交未改变本课程的 continuous-batching / KV budget 主契约。Megatron 当前 `main` 仍是本次复核使用的快照，最新提交只重构 experimental FSDP gradient-readiness countdown，未触及本课程使用的 DDP bucket、CP、MoE dispatcher 或 overlap contract。网页中的源码链接仍指向上游 `main`，方便继续阅读最新代码；上面的 SHA 只是说明这轮课程语义核对时使用的具体快照。上游继续演化后，应重新复查版本敏感内容。

## Reading tools

- **Source Map**：按问题、数据流与生命周期阅读 Megatron / vLLM，并直接跳到对应 Lab。
- **Glossary**：精选跨课程反复出现、容易混淆或影响源码阅读的核心概念；完整课内术语仍放在每课末尾。
- **Labs**：先用小规模 reference 跑通机制，再回源码确认真实工程约束。

## Quality checks

每次 push 会运行站点检查，验证：HTML 阅读元数据、重复 ID、本地页面/fragment/asset 链接、CSS imports/`url(...)`、`app.js` 动态 lesson/Lab routes、已发布课程的旧 `locked` / `muted-next` 导航能否解析到真实目标、已知语义回归字符串、过期 placeholder，以及所有 Lab Python 源码的语法编译。

术语另有三层保护：`scripts/lesson_terms.py` 维护 canonical term registry，并验证每课首次展开与课尾术语表没有 drift；`scripts/audit_lesson_acronyms.py` 反向扫描 learner-facing `<p>/<li>` 正文，新的高置信未注册缩写会直接让 CI 失败；`scripts/check_glossary_terms.py` 则只约束总站精选核心词，确保对应卡片仍存在并包含 canonical English name，而不会把总站 Glossary 膨胀成全部课内术语的复制品。

已人工验收为 lesson-native 的核心 SVG 另外由 `scripts/check_diagrams.py` 做宽度回归保护，防止重新退回“超宽小字号画布再整体缩小”的旧模式。

Lab 代码或 smoke harness 变化还会触发独立的 **Lab Correctness Smoke**：在干净的 GitHub runner 上安装 CPU PyTorch，用 `torchrun` / Gloo 实际执行 A1–A11 与 B1–B7 共 18 个实验。每个实验都必须正常退出并输出自身的独立 `PASS`，否则 workflow 失败。这个检查验证 reference 数值、collective/state-machine correctness 与脚本可运行性；CPU timing 仍不被解释为 GPU、NCCL、NIXL、RDMA 或 GPUDirect 性能 benchmark。

这个轻量 CI 不会伪装成 GPU/NCCL/NIXL 性能测试；需要真实硬件才能下结论的内容，在课程和 Lab 中会显式标出。

## Site

GitHub Pages: https://leonakaka.github.io/aiinfra/

- Labs: https://leonakaka.github.io/aiinfra/labs/
- Source Map: https://leonakaka.github.io/aiinfra/source-map/
- Glossary: https://leonakaka.github.io/aiinfra/glossary/
