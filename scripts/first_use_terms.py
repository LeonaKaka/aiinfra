#!/usr/bin/env python3
"""Shared first-use policy for source-facing AI Infra terminology.

Teaching prose keeps source/code spelling and explains it once when the course
first teaches the concept:

    shape（形状）。它表示 tensor 各维度的长度。
    scheduler（调度器）。它决定每轮哪些 requests 推进多少 token。

The parenthesis contains only the Chinese term. A short beginner explanation stays
in normal prose immediately around the first occurrence. Later uses keep the
English/source spelling so the prose continues to match code, profiler output,
docs and repository identifiers.

Exact class/function/field/path identifiers are never translated internally.
"""

# term: (Chinese name, short beginner explanation)
SOURCE_TERMS = {
    # 01 · Foundations
    "PyTorch": ("深度学习框架", "本课程用它演示 tensor、自动求导和训练代码"),
    "tensor": ("张量", "多维数组，也是模型计算中最基本的数据对象"),
    "memory": ("内存", "保存数据、参数和中间状态的存储空间"),
    "shape": ("形状", "表示 tensor 各维度的长度"),
    "dtype": ("数据类型", "表示 tensor 元素用什么数值格式存储"),
    "device": ("设备位置", "表示数据位于 CPU 还是哪张 GPU"),
    "storage": ("底层存储", "真正承载 tensor 数据的内存区域"),
    "layout": ("内存布局", "描述逻辑元素怎样映射到底层 storage"),
    "stride": ("步长", "表示某一维索引增加 1 时底层地址跨多少元素"),
    "contiguous": ("连续存储", "表示数据是否按当前逻辑维度顺序紧密排列"),
    "view": ("视图", "在不复制数据时用另一种逻辑形状解释同一块 storage"),
    "reshape": ("重塑形状", "改变 tensor 的逻辑 shape，必要时可能发生数据重排"),
    "transpose": ("转置维度", "交换 tensor 的维度顺序而通常不立即复制数据"),
    "permute": ("维度重排", "改变 tensor 各维度的顺序"),
    "copy": ("复制", "真正生成或写入一份新的数据副本"),
    "Linear": ("线性层", "对输入做矩阵乘法，并可再加 bias"),
    "bias": ("偏置", "Linear 中加在矩阵乘法结果上的可学习参数"),
    "loss": ("损失", "衡量模型预测与目标之间的误差"),
    "forward": ("前向计算", "用当前参数从输入计算模型输出和中间状态"),
    "backward": ("反向传播", "沿计算图反向计算参数 gradient"),
    "gradient": ("梯度", "表示 loss 对参数变化的敏感方向"),
    "parameter": ("参数", "训练中会被 optimizer 更新的可学习数值"),
    "optimizer": ("优化器", "根据 gradient 更新模型 parameter"),
    "optimizer state": ("优化器状态", "optimizer 为更新参数额外保存的历史量"),
    "activation": ("激活值", "forward 产生并可能在 backward 中复用的中间结果"),
    "autograd": ("自动求导", "PyTorch 自动记录计算图并计算 gradient 的机制"),
    "Adam": ("Adam 优化器", "利用一阶和二阶动量统计更新参数的常用 optimizer"),

    # 02 · Transformer
    "Transformer": ("Transformer 模型", "现代大语言模型常用的基础网络结构"),
    "Attention": ("注意力机制", "让当前 token 按相关性读取上下文信息"),
    "token": ("词元", "模型处理文本时使用的离散序列单位"),
    "sequence": ("序列", "按顺序排列的一串 token"),
    "hidden dimension": ("隐藏维度", "每个 token 隐藏表示的向量宽度"),
    "projection": ("投影", "用线性变换把表示映射到新的特征空间"),
    "attention score": ("注意力分数", "衡量 Query 与 Key 相关性的数值"),
    "causal mask": ("因果掩码", "阻止当前位置读取未来 token"),
    "softmax": ("归一化函数", "把一组 scores 变成和为 1 的权重"),
    "attention head": ("注意力头", "在一个子空间中独立计算 Attention 的一路表示"),
    "output projection": ("输出投影", "把多头 Attention 结果重新映射回隐藏维度"),
    "LayerNorm": ("层归一化", "按特征维对单个样本的表示做归一化"),
    "RMSNorm": ("均方根归一化", "用均方根尺度归一化表示的常见方法"),
    "residual connection": ("残差连接", "把子层输入直接加回输出以稳定深层网络"),

    # 03 · GPU Systems
    "Tensor Core": ("张量核心", "NVIDIA GPU 中专门加速矩阵乘法的硬件单元"),
    "host memory": ("主机内存", "CPU 侧的系统内存"),
    "device memory": ("设备显存", "GPU 直接访问的本地设备内存"),
    "kernel": ("GPU 内核", "提交到 GPU 上执行的一段并行计算"),
    "stream": ("CUDA 流", "按顺序组织 GPU 异步工作的执行队列"),
    "compute-bound": ("计算受限", "性能主要受计算吞吐限制"),
    "memory-bound": ("内存带宽受限", "性能主要受数据读写带宽限制"),
    "communication-bound": ("通信受限", "性能主要受跨设备通信限制"),
    "launch overhead": ("启动开销", "提交 kernel 等工作本身带来的固定成本"),
    "profiler": ("性能分析器", "记录算子、时间线和资源使用情况的工具"),
    "buffer": ("缓冲区", "临时或持久保存待处理数据的一块内存"),

    # 04 · Distributed
    "process": ("进程", "独立运行程序并拥有自己状态的执行实例"),
    "rank": ("进程编号", "分布式作业中标识一个 process 的编号"),
    "global rank": ("全局进程编号", "在整个分布式作业中唯一标识一个 process"),
    "local rank": ("本地进程编号", "标识 process 在当前机器上的本地序号"),
    "world size": ("总进程数", "表示当前 distributed world 中有多少个 processes"),
    "process group": ("进程组", "定义一组共同参与通信的 ranks"),
    "backend": ("后端", "提供具体通信或执行实现的一层接口"),
    "collective": ("集合通信", "一组 ranks 共同参与的数据通信操作"),
    "all-reduce": ("全归约", "归约各 rank 数据并把结果返回给所有参与者"),
    "all-gather": ("全收集", "收集各 rank 分片并让所有参与者得到完整结果"),
    "reduce-scatter": ("归约分散", "先归约再让每个 rank 只保留一份分片"),
    "broadcast": ("广播", "由一个 rank 把同一份数据发送给其它参与者"),
    "rendezvous": ("会合初始化", "让分布式 processes 发现彼此并建立通信所需信息"),
    "communicator": ("通信器", "通信库内部管理参与者与连接状态的对象"),
    "topology": ("拓扑", "描述 GPU、CPU、NIC 与网络之间的连接关系"),

    # 05 · Megatron
    "Megatron": ("大模型训练框架", "用于学习多维并行和分布式训练的源码实现"),
    "shard": ("分片", "把一份 tensor 或 state 拆成多份分到不同 ranks"),
    "replica": ("副本", "在不同 ranks 上保存的同一逻辑模型或 state"),
    "pipeline stage": ("流水线阶段", "负责模型连续一部分 layers 的执行阶段"),
    "microbatch": ("微批次", "把一个 batch 再切小以形成 Pipeline Parallel"),
    "bucket": ("通信桶", "把多块小数据合并后统一通信的分组单位"),
    "overlap": ("重叠执行", "让计算与通信并行以隐藏部分等待时间"),
    "expert": ("专家子网络", "MoE 中由 router 选择执行的稀疏前馈子网络"),
    "router": ("路由器", "根据规则决定 token 或 request 应被送往哪里"),
    "dispatcher": ("分发器", "按 routing 结果把 token 送到对应 expert 或目标"),
    "assignment": ("分配关系", "一次 token 到某个 expert 的 routing 记录"),

    # 06 · Inference
    "Prefill": ("预填充阶段", "一次处理 prompt 并建立历史 KV state 的阶段"),
    "Decode": ("解码阶段", "利用已有 KV state 一步步生成新 token 的阶段"),
    "prompt": ("输入提示", "送给模型作为生成上下文的输入 token 序列"),
    "request": ("请求", "一次进入推理服务并持续推进的生成任务"),
    "scheduler": ("调度器", "决定每轮哪些 requests 推进以及推进多少 token"),
    "iteration": ("执行轮次", "scheduler 规划并执行一次模型工作的循环"),
    "batch": ("批次", "一次共同送入模型执行的一组数据或 token 工作"),
    "continuous batching": ("连续批处理", "每个调度轮次都可以重新组成执行 batch"),
    "engine": ("推理引擎", "把 request 管理、scheduler 和模型执行组织成服务循环"),
    "workload": ("工作负载", "系统实际接收的一组 request 规模与分布特征"),
    "prefix cache": ("前缀缓存", "复用相同输入前缀已经计算好的 KV state"),
    "preemption": ("抢占", "资源不足时暂时移出正在运行的 request 并稍后恢复"),
    "throughput": ("吞吐量", "单位时间内系统完成的 request 或 token 数量"),
    "latency": ("延迟", "一次 request 或一步操作从开始到完成所花的时间"),
    "benchmark": ("基准测试", "在固定条件下测量和比较系统性能"),

    # 07 · vLLM
    "worker": ("工作进程", "实际持有设备并执行模型或数据操作的 process"),
    "executor": ("执行器", "把 scheduler 的计划分发到一个或多个 workers 执行"),
    "metadata": ("元数据", "描述数据身份、布局和执行方式的控制信息"),
    "block": ("块", "KV Cache 等系统中固定粒度的内存管理单位"),
    "block table": ("块表", "把逻辑 KV blocks 映射到物理 block ID"),
    "slot mapping": ("槽位映射", "把本轮 token 映射到具体 KV 写入位置"),
    "eviction": ("淘汰", "回收可丢弃 cache 以腾出资源"),
    "ownership": ("归属关系", "说明某个资源当前由哪个 request、rank 或组件持有"),

    # 08 · KV Connector / NIXL
    "producer": ("生产端", "产生并提供 KV state 的一侧"),
    "consumer": ("消费端", "接收 KV 并继续使用它的一侧"),
    "control plane": ("控制面", "负责 scheduler、状态和传输计划等控制信息"),
    "data plane": ("数据面", "负责真正的大块 KV 数据移动"),
    "transfer": ("传输", "把数据从一个位置移动到另一个位置"),
    "transport": ("传输后端", "实现网络或设备数据移动的底层机制"),
    "handoff": ("交接", "把已有 KV state 转移给另一个实例继续使用"),
    "descriptor": ("描述符", "描述一段可传输内存位置和属性的结构"),
    "region": ("内存区域", "传输层注册或操作的一段内存范围"),
    "lease": ("租约", "在有效期内保证远端资源继续保留的生命周期机制"),
    "heartbeat": ("心跳", "周期性证明 consumer 仍存活并维持资源有效期的消息"),
    "completion": ("完成状态", "表示异步操作已经安全结束的确认"),
    "peer": ("对端", "通信或 transfer 中的另一侧 process 或实例"),
}

# Exact identifiers are preserved verbatim; the surrounding prose explains them.
IDENTIFIER_ALLOWLIST = {
    "ColumnParallelLinear", "RowParallelLinear", "SchedulerOutput",
    "ModelRunnerOutput", "KVCacheManager", "BlockPool", "KVCacheBlock",
    "EngineCore", "RequestStatus", "InputBatch", "BlockTable", "BlockTables",
    "ModelRunner", "GPUModelRunner",
}
