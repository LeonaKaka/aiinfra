#!/usr/bin/env python3
"""One-shot fixes for high-signal term-density/placement warnings."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def replace_once(path: Path, old: str, new: str) -> None:
    text = path.read_text(encoding="utf-8")
    if new in text:
        return
    if old not in text:
        raise RuntimeError(f"anchor not found in {path}: {old[:140]!r}")
    path.write_text(text.replace(old, new, 1), encoding="utf-8")


def fix_gpu_bottlenecks() -> None:
    path = ROOT / "learn" / "03-gpu-systems" / "gpu-bottlenecks.html"
    replace_once(
        path,
        '<div class="latency-bandwidth"><div><small>SMALL MESSAGE</small><b>Latency matters</b><span>消息很小时，几十微秒级的固定开销都可能占很大比例。</span></div><div><small>LARGE MESSAGE</small><b>Bandwidth matters</b><span>数据量大时，NVLink / Peripheral Component Interconnect Express (PCIe，高速外设组件互连) / InfiniBand / RDMA over Converged Ethernet (RoCE，基于融合以太网的 RDMA) 的有效吞吐决定搬完要多久。</span></div></div>',
        '<p>这里先认两个后面会反复出现的链路名：Peripheral Component Interconnect Express (PCIe，高速外设组件互连) 是常见主机互连；RDMA over Converged Ethernet (RoCE，基于融合以太网的 RDMA) 则是在以太网上承载 RDMA 语义的网络技术。它们和 NVLink、InfiniBand 分别处在不同硬件/网络路径上。</p><div class="latency-bandwidth"><div><small>SMALL MESSAGE</small><b>Latency matters</b><span>消息很小时，几十微秒级的固定开销都可能占很大比例。</span></div><div><small>LARGE MESSAGE</small><b>Bandwidth matters</b><span>数据量大时，NVLink / PCIe / InfiniBand / RoCE 的有效吞吐决定搬完要多久。</span></div></div>',
    )
    replace_once(
        path,
        '<h2>不要靠猜，用不同 profiler 回答不同层级的问题。</h2>\n        <div class="profile-grid"><div class="profile-card"><small>TIMELINE</small><b>Nsight Systems</b><span>看 CPU、CUDA kernels、NVIDIA Collective Communications Library (NCCL，NVIDIA 集体通信库)、streams、空档和 overlap：系统“什么时候在等”。</span></div>',
        '<h2>不要靠猜，用不同 profiler 回答不同层级的问题。</h2>\n        <p>时间线里你会频繁看到 NVIDIA Collective Communications Library (NCCL，NVIDIA 集体通信库) 发起的 collective / P2P；它是通信软件层，不是 NVLink、InfiniBand 或以太网链路本身。</p>\n        <div class="profile-grid"><div class="profile-card"><small>TIMELINE</small><b>Nsight Systems</b><span>看 CPU、CUDA kernels、NCCL、streams、空档和 overlap：系统“什么时候在等”。</span></div>',
    )
    replace_once(
        path,
        '<h2>GPU Systems 到这里结束，下一模块开始研究“通信到底怎么发生”。</h2>\n        <div class="bridge-grid"><div><small>TRAINING · MEGATRON</small>',
        '<h2>GPU Systems 到这里结束，下一模块开始研究“通信到底怎么发生”。</h2>\n        <p>推理侧后面还会遇到 Prefill/Decode Disaggregation (P/D，Prefill/Decode 分离)：把两阶段放到不同资源池后，除了调度，还要为 KV handoff 支付新的通信成本。</p>\n        <div class="bridge-grid"><div><small>TRAINING · MEGATRON</small>',
    )
    replace_once(
        path,
        '<span>continuous batching 提高利用率；Paged KV 管 HBM；Prefill/Decode Disaggregation (P/D，Prefill/Decode 分离) disaggregation 与 KV Connector 再引入跨设备/节点数据传输。</span>',
        '<span>continuous batching 提高利用率；Paged KV 管 HBM；P/D 与 KV Connector 再引入跨设备/节点数据传输。</span>',
    )


def fix_process_rank() -> None:
    path = ROOT / "learn" / "04-distributed" / "process-rank.html"
    replace_once(
        path,
        '<h2>rank / group 是训练与推理系统共同的底层语言。</h2>\n        <div class="distributed-truths">',
        '<h2>rank / group 是训练与推理系统共同的底层语言。</h2>\n        <p>推理侧后面会遇到 Prefill/Decode Disaggregation (P/D，Prefill/Decode 分离)：即使 Prefill 与 Decode 被拆到不同实例，worker 身份、device mapping 和通信域仍然必须先定义清楚。</p>\n        <div class="distributed-truths">',
    )
    replace_once(
        path,
        '<span>多 GPU TP workers、跨进程执行器、Prefill/Decode Disaggregation (P/D，Prefill/Decode 分离) 节点同样需要明确 worker 身份、device 映射和通信域。</span>',
        '<span>多 GPU TP workers、跨进程执行器、P/D 节点同样需要明确 worker 身份、device 映射和通信域。</span>',
    )


def fix_overlap() -> None:
    path = ROOT / "learn" / "05-megatron" / "communication-overlap.html"
    replace_once(
        path,
        '<p class="dek">前面每加一种并行都会出现通信：Data Parallel (DP，数据并行) 有 gradient reduce / parameter gather，Tensor Parallel (TP，张量并行) 有 activation All-Gather / Reduce-Scatter，Pipeline Parallel (PP，流水线并行) 有 stage Peer-to-Peer (P2P，点对点通信)，Expert Parallel (EP，专家并行) 有 token exchange。Overlap 的目标不是把这些 bytes 变没，而是让通信尽早开始、让独立计算继续推进，把真正的 wait 推迟到数据第一次必须被消费的位置。</p>',
        '<p class="dek">前面每加一种并行都会出现通信：数据并行要同步梯度或参数，张量并行要交换 activation，流水线并行要在 stages 之间传数据，专家并行还要搬运 routed tokens。Overlap 的目标不是把这些 bytes 变没，而是让通信尽早开始、让独立计算继续推进，把真正的 wait 推迟到数据第一次必须被消费的位置。</p>',
    )
    replace_once(
        path,
        '<h2>NCCL 与 GEMM 同时出现在 timeline 上，也可能互相拖慢。</h2>\n        <div class="stream-grid">',
        '<h2>NCCL 与 GEMM 同时出现在 timeline 上，也可能互相拖慢。</h2>\n        <p>跨设备路径里还会反复看到 Peripheral Component Interconnect Express (PCIe，高速外设组件互连)：它是 CPU、GPU、NIC 等设备之间常见的主机互连之一，不能和 NVLink 或网络 fabric 当成同一层。</p>\n        <div class="stream-grid">',
    )
    replace_once(
        path,
        '<span>依赖 NVLink/Peripheral Component Interconnect Express (PCIe，高速外设组件互连)/Network Interface Card (NIC，网络接口卡) 等互连，同时也可能使用 SM、HBM bandwidth 与 Graphics Processing Unit (GPU，图形处理器)-side communication resources；算法和平台不同，资源画像不同。</span>',
        '<span>依赖 NVLink/PCIe/Network Interface Card (NIC，网络接口卡) 等互连，同时也可能使用 SM、HBM bandwidth 与 Graphics Processing Unit (GPU，图形处理器)-side communication resources；算法和平台不同，资源画像不同。</span>',
    )


def fix_nixl_rdmа() -> None:
    path = ROOT / "learn" / "08-kv-connector" / "nixl-rdma.html"
    replace_once(
        path,
        '<p class="dek">前面三课都在讲“应该传什么、什么时候传、谁来决定”。这一课终于下到数据面。先避免缩写堆砌：NVIDIA Inference Xfer Library (NIXL，NVIDIA 推理传输库) 不是 Key-Value (KV，键-值) Cache 算法，Unified Communication X (UCX，统一通信 X) 不是 vLLM Scheduler，Remote Direct Memory Access (RDMA，远程直接内存访问) 也不是“网络更快”的同义词。我们从最朴素的 Graphics Processing Unit (GPU，图形处理器)→Central Processing Unit (CPU，中央处理器)→网络→CPU→GPU 路径开始，看为什么 memory registration、direct access、transport backend、handshake compatibility、KV layout 和异步 handle 会决定 KV transfer 的真实成本与正确性。</p>',
        '<p class="dek">前面三课都在讲“应该传什么、什么时候传、谁来决定”。这一课终于下到数据面。先记住两件事：NVIDIA Inference Xfer Library (NIXL，NVIDIA 推理传输库) 负责的是传输基础设施，而 Key-Value (KV，键-值) Cache 仍是模型推理状态。下面从最朴素的“设备→主机→网络→主机→设备”路径开始，再逐层拆 transport backend、远程内存访问、memory registration、handshake compatibility、KV layout 和异步 handle。</p>',
    )


def fix_why_move_kv() -> None:
    path = ROOT / "learn" / "08-kv-connector" / "why-move-kv.html"
    replace_once(
        path,
        '<section class="article-section" id="three-problems"><div class="section-no">04 · THREE DIFFERENT PROBLEMS</div><h2>“把 Prefill 交给 Decode”其实至少包含三条链。</h2><div class="meta-grid">',
        '<section class="article-section" id="three-problems"><div class="section-no">04 · THREE DIFFERENT PROBLEMS</div><h2>“把 Prefill 交给 Decode”其实至少包含三条链。</h2><p>后面会把数据面拆成三个名字：NVIDIA Inference Xfer Library (NIXL，NVIDIA 推理传输库) 提供传输基础设施，Unified Communication X (UCX，统一通信 X) 可以作为底层通信框架，而 Remote Direct Memory Access (RDMA，远程直接内存访问) 描述的是远程内存访问机制。先把层级分开，再看它们怎样组合。</p><div class="meta-grid">',
    )
    replace_once(
        path,
        '<span>GPU→GPU、GPU→host→network→GPU，还是通过 NVIDIA Inference Xfer Library (NIXL，NVIDIA 推理传输库)/Unified Communication X (UCX，统一通信 X)/Remote Direct Memory Access (RDMA，远程直接内存访问) 等路径。</span>',
        '<span>GPU→GPU、GPU→host→network→GPU，还是通过 NIXL / UCX / RDMA 等数据路径。</span>',
    )


def main() -> None:
    fix_gpu_bottlenecks()
    fix_process_rank()
    fix_overlap()
    fix_nixl_rdmа()
    fix_why_move_kv()
    print("Applied high-signal term readability fixes.")


if __name__ == "__main__":
    main()
