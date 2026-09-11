#!/usr/bin/env python3
"""Shared first-use policy for source-facing AI Infra terminology.

Teaching prose keeps the source spelling, then explains it once:
    shape（形状：描述 tensor 每一维的长度）
Later uses keep the source spelling without repeating the explanation.

Exact class/function/field/path identifiers are never rewritten internally.
"""

# term: (Chinese name, short explanation)
SOURCE_TERMS = {
    "tensor": ("张量", "多维数组，是模型中数据流动和计算的基本载体"),
    "shape": ("形状", "描述 tensor 每一维的长度"),
    "dtype": ("数据类型", "规定每个元素用什么数值格式存储"),
    "device": ("设备", "表示 tensor 当前位于 CPU、哪张 GPU 或其它设备"),
    "storage": ("底层存储", "真正承载 tensor 数据的内存区域"),
    "layout": ("内存布局", "描述逻辑维度怎样映射到底层存储"),
    "stride": ("步长", "描述某一维索引增加 1 时底层地址跨多少元素"),
    "contiguous": ("连续存储", "表示数据是否按当前逻辑维度顺序连续排列"),
    "bias": ("偏置", "Linear 中加在矩阵乘法结果上的可学习参数"),
    "forward": ("前向计算", "用当前参数从输入计算模型输出"),
    "backward": ("反向传播", "沿计算图反向计算梯度"),
    "gradient": ("梯度", "损失对参数或中间量的导数"),
    "parameter": ("参数", "训练过程中由 optimizer 更新的可学习 tensor"),
    "optimizer state": ("优化器状态", "optimizer 为更新参数额外保存的状态"),
    "activation": ("激活值", "forward 中产生并可能供 backward 使用的中间 tensor"),
    "kernel": ("GPU 内核", "一次提交到 GPU 上执行的计算程序"),
    "stream": ("CUDA 流", "GPU 上用于组织异步工作与依赖关系的执行队列"),
    "host memory": ("主机内存", "CPU 侧系统内存"),
    "device memory": ("设备显存", "GPU 等设备本地可直接访问的内存"),
    "compute-bound": ("计算受限", "性能主要受算力而不是数据搬运限制"),
    "memory-bound": ("带宽受限", "性能主要受显存带宽或数据搬运限制"),
    "launch overhead": ("启动开销", "CPU 提交 kernel 等工作本身产生的固定成本"),
    "profiler": ("性能分析工具", "记录 kernel、通信和时间线以定位性能瓶颈"),
    "process": ("进程", "一个独立运行的程序实例"),
    "rank": ("进程编号", "分布式作业中标识一个 process 的编号"),
    "world size": ("总进程数", "同一 distributed world 中参与的 process 数量"),
    "local rank": ("本地进程编号", "process 在当前节点上的编号"),
    "process group": ("进程组", "参与同一组 collective 的 ranks 集合"),
    "backend": ("通信后端", "实际执行 collective 或 P2P 的通信实现"),
    "collective": ("集合通信", "一组 ranks 共同参与的通信操作"),
    "shard": ("分片", "完整 tensor 或 state 被切分后的一部分"),
    "replica": ("副本", "同一逻辑对象在不同 ranks 上保存的复制版本"),
    "stage": ("流水线阶段", "Pipeline Parallel 中负责一段模型层的执行单元"),
    "microbatch": ("微批次", "为填充 pipeline 而从 batch 进一步切出的较小批次"),
    "bucket": ("通信桶", "把多块参数或梯度聚合后统一通信的分组"),
    "overlap": ("通信重叠", "让通信与独立计算并发以减少暴露等待"),
    "router": ("路由器", "MoE 中决定 token 应送到哪些 experts 的组件"),
    "dispatcher": ("分发器", "按 routing 结果搬运 token 并恢复顺序的组件"),
    "assignment": ("分配关系", "一次 token 到某个 expert 的路由记录"),
    "request": ("请求", "在线推理系统跟踪的一次用户生成任务"),
    "scheduler": ("调度器", "决定本轮哪些 requests 推进多少 token 的组件"),
    "iteration": ("执行轮次", "scheduler 规划并执行一次模型工作的循环"),
    "buffer": ("缓冲区", "暂存 tensor 或传输数据的一块内存"),
    "metadata": ("元数据", "描述数据位置、映射或状态而不是数据本体的信息"),
    "block": ("块", "KV cache 等系统中固定粒度的内存管理单位"),
    "block table": ("块表", "把逻辑 KV block 映射到物理 block ID 的表"),
    "slot mapping": ("槽位映射", "把当前 token 映射到 KV cache 精确写入位置"),
    "prefix cache": ("前缀缓存", "复用已计算 prompt 前缀 KV 状态的缓存机制"),
    "preemption": ("抢占", "资源不足时暂停 request 并回收其资源"),
    "handoff": ("交接", "把已有 KV 状态及其映射信息交给另一个实例继续使用"),
    "descriptor": ("描述符", "描述一段可传输内存位置和属性的结构"),
    "region": ("内存区域", "传输层注册或操作的一段内存范围"),
    "ownership": ("归属关系", "说明某个资源当前由哪个 request、rank 或组件持有"),
    "lease": ("租约", "在有效期内保证远端资源继续保留的生命周期机制"),
    "heartbeat": ("心跳", "周期性证明消费者仍存活并续租资源的控制消息"),
    "producer": ("生产端", "产生并提供 KV 状态的一侧"),
    "consumer": ("消费端", "接收并继续使用 KV 状态的一侧"),
    "throughput": ("吞吐量", "单位时间内系统完成的 token 或 request 数"),
    "latency": ("延迟", "一次 request 或系统阶段从开始到完成所需时间"),
    "workload": ("工作负载", "系统需要处理的一组请求形态和计算特征"),
}

IDENTIFIER_ALLOWLIST = {
    "ColumnParallelLinear", "RowParallelLinear", "SchedulerOutput",
    "ModelRunnerOutput", "KVCacheManager", "BlockPool", "EngineCore",
    "ModelRunner", "GPUModelRunner",
}
