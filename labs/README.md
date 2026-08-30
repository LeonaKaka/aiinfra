# Hands-on Labs

这部分不是复刻 Megatron / vLLM，而是把课程里最关键的系统契约压缩成可以直接运行的小实验。

## Lab A — Mini Megatron TP

```bash
torchrun --standalone --nproc-per-node=2 labs/code/mini_megatron_tp.py
```

- CPU：自动使用 Gloo。
- 至少 2 张 CUDA GPU：自动使用 NCCL，每个 rank 一张 GPU。
- 验证：Column Parallel + Row Parallel 的 sharded MLP 与 dense reference 数值一致。

## Lab B — Mini KV Handoff

```bash
torchrun --standalone --nproc-per-node=2 labs/code/mini_kv_handoff.py
```

- Rank 0：Prefill producer，生成 prompt K/V。
- Rank 1：Decode consumer，接收 K/V 后直接做第一步 attention。
- 验证：handoff 路径与 Decode 端重新计算 prompt K/V 的 reference 数值一致。

## Environment

需要 Python 3.10+ 与带 `torch.distributed` 的 PyTorch。PyTorch 的 CUDA 安装方式与 CUDA/驱动版本有关，因此本仓库不固定一个通用 GPU wheel。

这两个脚本的默认 tensor 很小，内置 timing 主要用于理解执行顺序，不应当当作 GPU、NCCL、NVLink 或 RDMA 的性能 benchmark。
