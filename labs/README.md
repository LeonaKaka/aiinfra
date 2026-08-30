# Hands-on Labs

这部分不是复刻 Megatron / vLLM，而是把课程里最关键的系统契约压缩成可以直接运行、可以和 reference 核对的小实验。

## Training Infra

### Lab A1 — Mini Megatron TP

```bash
torchrun --standalone --nproc-per-node=2 labs/code/mini_megatron_tp.py
```

验证 Column Parallel + Row Parallel 的 sharded MLP 与 dense reference 数值一致。

### Lab A2 — Mini TP Backward

```bash
torchrun --standalone --nproc-per-node=2 labs/code/mini_tp_backward.py
```

手算 TP MLP 的反向传播，分别检查 forward、dX、dW1、dW2，并与 PyTorch dense autograd reference 对齐。重点观察：W1/W2 gradients 天然保持 sharded，而 replicated dX 需要跨 TP ranks 求和。

## Inference Infra

### Lab B1 — Mini KV Handoff

```bash
torchrun --standalone --nproc-per-node=2 labs/code/mini_kv_handoff.py
```

Rank 0 产生 prompt K/V，Rank 1 接收后直接 Decode，并与 Decode 端重新计算 prompt K/V 的 reference 对齐。

### Lab B2 — Mini Block KV Handoff

```bash
torchrun --standalone --nproc-per-node=2 labs/code/mini_block_kv_handoff.py
```

在 B1 基础上加入 request metadata、固定 token blocks、physical KV pool 和 logical→physical block table。默认例子故意把逻辑 blocks `[0,1,2]` 放到物理 slots `[7,6,5]`，验证 attention 结果与物理排布无关。

## Environment

需要 Python 3.10+ 与带 `torch.distributed` 的 PyTorch。

- CPU：自动使用 Gloo。
- 至少 2 张 CUDA GPU：自动使用 NCCL，每个 rank 一张 GPU。
- 本仓库不固定 CUDA wheel；PyTorch GPU 安装方式应与本机 CUDA / driver 匹配。

## What the timings mean

默认 tensor 很小。脚本中的 timing 主要用来观察执行顺序，不应当作为 GPU、NCCL、NVLink、InfiniBand 或 RDMA 的性能 benchmark。当前阶段的第一验收标准是 correctness；后续再放大 workload、做异步传输与 profiler timeline。
