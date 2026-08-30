# AI Infra from Zero

一个从基础知识逐层学习到 **Megatron Core** 与 **vLLM KV Connector** 的学习型网站。

当前已经完成 8 个核心学习模块，并开始进入可运行的 Hands-on Labs。

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

- **Mini Megatron TP**：两进程 Column Parallel + Row Parallel MLP，并与 dense reference 核对。
- **Mini KV Handoff**：Prefill rank 生成 K/V，Decode rank 接收后直接继续 attention，并与 recompute reference 核对。

两个实验都支持 CPU/Gloo fallback；存在至少两张 CUDA GPU 时自动使用 NCCL。

## Site

GitHub Pages: https://leonakaka.github.io/aiinfra/

Labs: https://leonakaka.github.io/aiinfra/labs/
