# AI Infra from Zero

一个从基础知识逐层学习到 **Megatron Core** 与 **vLLM KV Connector** 的学习型网站。

当前已经完成 8 个核心学习模块，并进入可运行的 Hands-on Labs、Source Map 与 Glossary 阶段。

## Curriculum

1. Foundations
2. Transformer Fundamentals
3. GPU Systems
4. Distributed Systems for AI
5. Megatron Core
6. LLM Inference
7. vLLM
8. KV Connector

## Hands-on Labs

Training Infra：
- **A1 Mini Megatron TP**：Column Parallel + Row Parallel forward。
- **A2 Mini TP Backward**：手算 TP backward，并与 dense autograd 核对 dX/dW。

Inference Infra：
- **B1 Mini KV Handoff**：Prefill rank 生成 K/V，Decode rank 接管并与 recompute reference 核对。
- **B2 Mini Block KV Handoff**：加入 request metadata、token blocks、physical KV pool 与 block table。

实验均支持 CPU/Gloo fallback；至少两张 CUDA GPU 时自动使用 NCCL。当前 correctness 路径已在 CPU/Gloo 上验证。

## Reference

- **Source Map**：按问题而不是按目录阅读 Megatron Core / vLLM 源码。
- **Glossary**：按学习路径整理术语，并强调“它解决什么问题”。

## Site

GitHub Pages: https://leonakaka.github.io/aiinfra/

- Labs: https://leonakaka.github.io/aiinfra/labs/
- Source Map: https://leonakaka.github.io/aiinfra/source-map/
- Glossary: https://leonakaka.github.io/aiinfra/glossary/
