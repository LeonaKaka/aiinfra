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

手算 TP MLP 的反向传播，分别检查 forward、dX、dW1、dW2，并与 PyTorch dense autograd reference 对齐。

### Lab A3 — Data Parallel + Gradient Accumulation

```bash
torchrun --standalone --nproc-per-node=2 labs/code/mini_dp_grad_accum.py
```

两个 rank 持有相同参数、处理不同样本；每个 rank 先累积多个 microbatch 的梯度，再只做一次 all-reduce。脚本与完整 global-batch reference 对齐，重点区分“本地 gradient accumulation”和“跨 DP ranks gradient synchronization”。

### Lab A4 — Mini Distributed Optimizer

```bash
torchrun --standalone --nproc-per-node=2 labs/code/mini_distributed_optimizer.py
```

用 `reduce_scatter_tensor` 让 global gradient 直接落成 local shard；每个 rank 只保留自己的 momentum / parameter shard，更新后再 `all_gather_into_tensor` 重新 materialize 完整参数。这个实验只实现核心数据流，不代表完整 Megatron Distributed Optimizer / ZeRO stage 语义。

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

加入 request metadata、固定 token blocks、physical KV pool 和 logical→physical block table，验证 attention 结果与物理排布无关。

### Lab B3 — Mini Async KV Transfer

```bash
torchrun --standalone --nproc-per-node=2 labs/code/mini_async_kv_transfer.py
```

用 `isend` / `irecv` + `Work.wait()` 把 KV block transfer 拆成 launch、independent work 和 dependency 三个阶段。CPU/Gloo timing 只用于观察执行顺序，不是网络性能 benchmark。

## Environment

需要 Python 3.10+ 与带 `torch.distributed` 的 PyTorch。

- CPU：自动使用 Gloo。
- 至少 2 张 CUDA GPU：自动使用 NCCL，每个 rank 一张 GPU。
- 本仓库不固定 CUDA wheel；PyTorch GPU 安装方式应与本机 CUDA / driver 匹配。

## Validation rule

1. **先 correctness**：每个实验都与 dense autograd / recompute reference 对齐。
2. **再 system semantics**：明确每个 rank 持有什么、哪个 collective/transfer 必须发生。
3. **最后 performance**：只有在 workload、backend、GPU topology 与 profiler 都明确时才解释 timing。

下一阶段会加入 bucketed async reduce-scatter、TP × DP process groups、GPU profiler timeline，以及 layer-by-layer KV transfer / NIXL memory-registration 源码实验。
