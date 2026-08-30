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
两个 rank 持有相同参数、处理不同样本；每个 rank 先累积多个 microbatch 的梯度，再只做一次 all-reduce。

### Lab A4 — Mini Distributed Optimizer
```bash
torchrun --standalone --nproc-per-node=2 labs/code/mini_distributed_optimizer.py
```
用 `reduce_scatter_tensor` 让 global gradient 直接落成 local shard；每个 rank 只保留自己的 momentum / parameter shard，更新后再 `all_gather_into_tensor` materialize 完整参数。

### Lab A5 — Bucketed Async Reduce-Scatter
```bash
torchrun --standalone --nproc-per-node=2 labs/code/mini_bucketed_reduce_scatter.py
```
手动构造 W2 → W1 的 gradient-ready 顺序。W2 bucket ready 后立即启动 `async_op=True` 的 reduce-scatter，再继续计算 dH / ReLU backward / dW1。CPU/Gloo timing 只展示 launch 与 wait 的依赖顺序，不代表真实硬件 overlap。

### Lab A6 — TP × DP 2D Topology
```bash
torchrun --standalone --nproc-per-node=4 labs/code/mini_tp_dp_2d.py
```
4 ranks 组成 TP=2 × DP=2。脚本分别创建 TP groups `[[0,1],[2,3]]` 与 DP groups `[[0,2],[1,3]]`，验证 TP layer communication 与 DP gradient sync 只能发生在各自的 process group。

### Lab A7 — Profiler-ready Overlap
```bash
torchrun --standalone --nproc-per-node=2 labs/code/mini_profiler_overlap.py --trace-dir profiler_traces
```
用 `torch.profiler` 标记 forward、W2 gradient ready、async reduce-scatter、继续 backward 与最终 wait，并导出每个 rank 的 Chrome trace。CPU/Gloo 已验证 correctness 与 range 顺序；只有 GPU/NCCL trace 出现 NCCL kernel 与 compute kernel 时间重叠时，才可以宣称真实 overlap。

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
用 `isend` / `irecv` + `Work.wait()` 把 KV block transfer 拆成 launch、independent work 和 dependency 三个阶段。

### Lab B4 — Layer-wise KV Streaming
```bash
torchrun --standalone --nproc-per-node=2 labs/code/mini_layerwise_kv_stream.py
```
Prefill 每算完一层 KV 就立即发送；Decode 只在真正进入 layer L 前等待 KV[L] ready。该实验直接对应 `KVConnectorBase_V1` 的 `start_load_kv()` / `wait_for_layer_load(layer_name)` 这类 per-layer dependency。

### Lab B5 — KV Handshake + Lifetime
```bash
torchrun --standalone --nproc-per-node=2 labs/code/mini_kv_handshake_lifetime.py
```
用 toy control-plane descriptor 表达 protocol/request/shape/bytes，Consumer 先验证再接收 K/V，最后发送 completion，Producer 才允许 region generation 前进。该脚本明确不模拟真实 RDMA registration/rkey，只教学 handshake 与 lifetime contract。

### Lab B6 — Registered Region Descriptor
```bash
torchrun --standalone --nproc-per-node=2 labs/code/mini_registered_region_descriptor.py
```
用一个底层 storage allocation 承载多个 layer KV views，分别计算 region offset、block length、block stride，再交换几何 metadata 并验证 block transfer。对应当前 vLLM NIXL `register_kv_caches()` 里“allocation registration 与 logical transfer regions 分开”的设计，但不模拟真实 `register_memory`、rkey、UCX 或 RDMA。

### Lab B7 — KV Lease / Expiry
```bash
torchrun --standalone --nproc-per-node=2 labs/code/mini_kv_lease_expiry.py
```
用 deterministic virtual clock 跑两条资源生命周期：一个 request 通过 heartbeat 延长 lease 后正常 completion；另一个 request 完全失联，最终靠 expiry 回收。另用 toy generation ID 演示 stale heartbeat 不能续约已复用资源。该状态机教学 lifetime invariant，不宣称复刻当前 vLLM 的全部 lease 实现。

## Environment

需要 Python 3.10+ 与带 `torch.distributed` 的 PyTorch。

- CPU：自动使用 Gloo。
- 至少 2 张 CUDA GPU：自动使用 NCCL，每个 rank 一张 GPU。
- 本仓库不固定 CUDA wheel；PyTorch GPU 安装方式应与本机 CUDA / driver 匹配。

## Validation rule

1. **先 correctness**：实验与 dense autograd / recompute reference 对齐。
2. **再 system semantics**：明确每个 rank 持有什么、哪个 collective/transfer 必须发生、什么时候必须等待。
3. **最后 performance**：只有在 workload、backend、GPU topology 与 profiler 都明确时才解释 timing。

软件实验主线到 B7 已闭合。下一阶段只做有证据的硬件验证：真实 GPU/NCCL profiler overlap，以及具备对应 NIC/driver/software stack 时的 NIXL/UCX/RDMA/GPUDirect 数据面。
