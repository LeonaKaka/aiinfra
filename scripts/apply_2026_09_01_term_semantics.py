#!/usr/bin/env python3
"""One-shot maintenance for the 2026-09-01 terminology/source audit.

This script is intentionally narrow and idempotent. It updates the canonical
lesson term registry, adds a few non-glossary spellings to the reverse acronym
audit, refreshes the vLLM source snapshot, and tightens the 07.5 prefix-cache
semantics after upstream hybrid/Mamba partial-hit support landed.

After the generated content is committed, this helper should be removed.
"""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def replace_once(path: Path, old: str, new: str) -> None:
    text = path.read_text(encoding="utf-8")
    if new in text:
        return
    if old not in text:
        raise RuntimeError(f"anchor not found in {path}: {old[:100]!r}")
    path.write_text(text.replace(old, new, 1), encoding="utf-8")


def insert_before(path: Path, anchor: str, block: str) -> None:
    text = path.read_text(encoding="utf-8")
    if block.strip() in text:
        return
    if anchor not in text:
        raise RuntimeError(f"anchor not found in {path}: {anchor!r}")
    path.write_text(text.replace(anchor, block + anchor, 1), encoding="utf-8")


def update_term_registry() -> None:
    path = ROOT / "scripts" / "lesson_terms.py"

    insert_before(
        path,
        '    "RAM": ("Random Access Memory",',
        '    "DRAM": ("Dynamic Random-Access Memory", "动态随机存取存储器", "主机内存和部分设备内存常用的易失性存储技术；容量层级与带宽特征要和缓存、设备显存区分。", True),\n',
    )
    insert_before(
        path,
        '    "GEMM": ("General Matrix Multiply",',
        '    "CLI": ("Command-Line Interface", "命令行界面", "通过终端参数和命令配置、启动或调试程序的接口。", True),\n',
    )
    insert_before(
        path,
        '    "MHA": ("Multi-Head Attention",',
        '    "GPT": ("Generative Pre-trained Transformer", "生成式预训练 Transformer", "以自回归 Transformer 为基础的一类生成式语言模型命名。", True),\n'
        '    "LM": ("Language Model", "语言模型", "对 token 序列概率或下一 token 分布进行建模的模型。", True),\n'
        '    "LoRA": ("Low-Rank Adaptation", "低秩适配", "用低秩增量参数高效适配预训练模型的方法；也可能进入 prefix-cache identity。", True),\n',
    )
    insert_before(
        path,
        '    "FP8": ("8-bit Floating Point",',
        '    "INT8": ("8-bit Integer", "8 位整数", "常见低精度整数表示，量化推理中用于降低存储与计算成本。", True),\n',
    )
    insert_before(
        path,
        '    "PCIe": ("Peripheral Component Interconnect Express",',
        '    "PC": ("Program Counter", "程序计数器", "执行单元用于跟踪下一条指令位置的控制状态；GPU SIMT 控制流讨论中会遇到。", True),\n'
        '    "IOMMU": ("Input-Output Memory Management Unit", "输入输出内存管理单元", "为设备 DMA 提供地址转换和访问隔离的硬件单元，影响设备内存映射与直通路径。", True),\n',
    )
    insert_before(
        path,
        '    "1F1B": ("One Forward One Backward",',
        '    "VPP": ("Virtual Pipeline Parallelism", "虚拟流水线并行", "在一个物理 pipeline rank 上放多个模型分段，以更细粒度的 interleaving 降低流水线 bubble。", True),\n'
        '    "GTP": ("Generalized Tensor Parallelism", "广义张量并行", "Megatron 当前的新型张量并行路径；源码中的 GTP remat/replica 语义不能和经典 TP 简单混为一谈。", True),\n',
    )
    insert_before(
        path,
        '    "OOM": ("Out Of Memory",',
        '    "RNG": ("Random Number Generator", "随机数生成器", "dropout、初始化和采样等操作依赖的随机状态；并行训练中需要正确管理不同 rank 的 RNG stream。", True),\n'
        '    "ZeRO": ("Zero Redundancy Optimizer", "零冗余优化器", "通过数据并行 ranks 分片优化器状态、梯度或参数来减少训练状态冗余的一类方法。", True),\n',
    )
    replace_once(
        path,
        '    "APC": ("Automatic Prefix Caching", "自动前缀缓存", "vLLM 中复用完整缓存 block 的前缀缓存机制。", True),',
        '    "APC": ("Automatic Prefix Caching", "自动前缀缓存", "vLLM 通过 prefix hash identity 复用已计算 KV 的机制；普通 full-attention 路径以完整 hash block 为基础心智模型，当前 hybrid/Mamba 路径存在细粒度 partial-hit 例外。", True),',
    )
    insert_before(
        path,
        '    "RPC": ("Remote Procedure Call",',
        '    "MPI": ("Message Passing Interface", "消息传递接口", "HPC 中常见的进程间消息传递标准；理解 NCCL 与传统分布式通信生态时会遇到。", True),\n'
        '    "MTU": ("Maximum Transmission Unit", "最大传输单元", "网络链路一次可承载的数据包大小上限，配置不当会影响 RoCE/RDMA 网络效率与连通性。", True),\n',
    )
    insert_before(
        path,
        '    "LBHNC": ("LBHNC KV layout",',
        '    "GB": ("Gigabyte", "GB 十进制容量单位", "十进制容量单位，1 GB = 10^9 bytes；与 GiB 不同。", False),\n'
        '    "GiB": ("Gibibyte", "GiB 二进制容量单位", "二进制容量单位，1 GiB = 2^30 bytes。", False),\n'
        '    "MiB": ("Mebibyte", "MiB 二进制容量单位", "二进制容量单位，1 MiB = 2^20 bytes。", False),\n',
    )


def update_reverse_audit() -> None:
    path = ROOT / "scripts" / "audit_lesson_acronyms.py"
    insert_before(
        path,
        'ALLOW.update(TERMS)\n',
        '# Operation/class/enum spellings that learners may see but that are not\n'
        '# independent glossary abbreviations. Keep this list narrow.\n'
        'NON_GLOSSARY_TOKENS = {\n'
        '    "AllGather", "AllReduce", "AllToAll", "FC1", "INFO", "MAX",\n'
        '    "NixlConnector", "READ", "README", "SUM", "SWAP", "WRITE",\n'
        '}\n'
        'ALLOW.update(NON_GLOSSARY_TOKENS)\n',
    )


def update_prefix_cache_lesson() -> None:
    path = ROOT / "learn" / "07-vllm" / "prefix-cache-preemption.html"
    replace_once(
        path,
        '<li>理解标准 Automatic Prefix Caching (APC，自动前缀缓存) 为什么按可缓存 hash block 识别 prefix，以及 shared blocks 为什么需要 ref count。</li>',
        '<li>理解 Automatic Prefix Caching (APC，自动前缀缓存) 如何用 hash identity 识别可复用 prefix，并知道普通 full-attention 的完整 hash-block 心智模型在当前 hybrid/Mamba 路径存在细粒度 partial-hit 例外。</li>',
    )
    replace_once(
        path,
        '<p>标准 Automatic Prefix Caching 的教学规则仍然是“缓存完整 hash blocks”：block hash 会包含 parent hash、当前 block tokens，以及 LoRA / multimodal / cache salt 等必要额外信息。当前 <code>KVCacheBlock</code> 内部还记录 group-aware hash 和 <code>_block_hash_num_tokens</code>，说明现代 V1 的 hash granularity、physical allocation block、hybrid cache group 已经不应被当成永远一一等同。</p>',
        '<p>对普通 full-attention APC，先用“命中完整 hash blocks”建立基础心智模型仍然最稳：block hash 会包含 parent hash、当前 block tokens，以及 LoRA / multimodal / cache salt 等必要额外信息。但这已经不能写成全局实现不变量。当前 V1 的 <code>HybridKVCacheCoordinator</code> 在 full-attention + Mamba <code>align</code> 等满足条件的 hybrid cache 中，可以启用 <code>enable_partial_hash_hits</code>，让 cache-hit alignment 下放到更细的 <code>hash_block_size</code>；scheduler 还会为 Mamba partial tail 增加对齐 stop。也就是说，<strong>“必须命中完整 physical allocation block”现在是过度简化</strong>。当前 <code>KVCacheBlock</code> 记录 group-aware hash 和 <code>_block_hash_num_tokens</code>，进一步说明 hash granularity、physical allocation block、scheduler block 与 hybrid cache group 不应被当成永远一一等同。</p>',
    )
    replace_once(
        path,
        '<div class="concept-note"><p><strong>先学稳定语义，不要绑死内部尺寸。</strong>“相同 prefix → 可验证 hash identity → 找到可复用 KV”是核心；physical KV block size、hash block size、attention kernel block size 在当前 V1 可能是不同层级的概念。</p></div>',
        '<div class="concept-note"><p><strong>先学稳定语义，不要绑死内部尺寸。</strong>“相同 prefix → 可验证 hash identity → 找到可复用 KV”是核心；普通 full-attention APC 常以完整 hash block 理解，而当前 hybrid/Mamba 的 fine-grained partial hit 是明确例外。physical KV block size、hash block size、scheduler block size 与 attention kernel block size 在当前 V1 可能属于不同层级。</p></div>',
    )
    replace_once(
        path,
        '<div class="source-note">设计说明：<a href="https://github.com/vllm-project/vllm/blob/main/docs/design/prefix_caching.md">Automatic Prefix Caching</a>；当前 metadata 定义见 <a href="https://github.com/vllm-project/vllm/blob/main/vllm/v1/core/kv_cache_utils.py">kv_cache_utils.py</a>。</div>',
        '<div class="source-note">设计说明：<a href="https://github.com/vllm-project/vllm/blob/main/docs/design/prefix_caching.md">Automatic Prefix Caching</a>；当前 block metadata 见 <a href="https://github.com/vllm-project/vllm/blob/main/vllm/v1/core/kv_cache_utils.py">kv_cache_utils.py</a>；hybrid/Mamba fine-grained hit 的当前协调逻辑见 <a href="https://github.com/vllm-project/vllm/blob/main/vllm/v1/core/kv_cache_coordinator.py">kv_cache_coordinator.py</a> 与 <a href="https://github.com/vllm-project/vllm/blob/main/tests/v1/core/prefix_cache/test_partial_prefix_cache_hits.py">partial prefix-cache tests</a>。</div>',
    )


def refresh_readme_snapshot() -> None:
    path = ROOT / "README.md"
    replace_once(
        path,
        '- vLLM `main`: `39e276eaeb9daed06a180f6a8d187bbb8790e97b`',
        '- vLLM `main`: `f9c7c6e0909eadc23f1aa2510a233f91692ed437`',
    )
    replace_once(
        path,
        '这轮继续复核了 vLLM GPU runner selector / V1-V2 runner 分流、Scheduler、KV Connector / NIXL lifecycle，以及 Megatron Expert Parallel dispatcher、Distributed Optimizer、Context Parallel 与 communication-overlap 生命周期。相对上一快照，vLLM 新增的 scheduler 变化集中在 structured-output / `min_tokens` stop-ordering，未改变本课程的 continuous-batching / KV budget contract；Megatron 新提交只重构 experimental FSDP gradient-readiness countdown，未触及本课程使用的 DDP bucket、CP、MoE dispatcher 或 overlap contract。',
        '这轮继续复核了 vLLM GPU runner selector / V1-V2 runner 分流、Scheduler、KV Connector / NIXL lifecycle，以及 Megatron Expert Parallel dispatcher、Distributed Optimizer、Context Parallel 与 communication-overlap 生命周期。相对上一快照，vLLM 又前进了 5 个提交：大部分集中在测试与 ROCm MLA，但其中 KV cache coordinator / prefix-cache tests 已明确覆盖 hybrid full-attention + Mamba `align` 的 fine-grained partial prefix-cache hits，因此 07.5 已补上“完整 hash block 是基础心智模型、不是所有当前路径的全局不变量”这一例外；这些提交未改变本课程的 continuous-batching / KV budget 主契约。Megatron 当前 `main` 仍是本次复核使用的快照，最新提交只重构 experimental FSDP gradient-readiness countdown，未触及本课程使用的 DDP bucket、CP、MoE dispatcher 或 overlap contract。',
    )


def main() -> None:
    update_term_registry()
    update_reverse_audit()
    update_prefix_cache_lesson()
    refresh_readme_snapshot()
    print("Applied 2026-09-01 terminology/source maintenance.")


if __name__ == "__main__":
    main()
