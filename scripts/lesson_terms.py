#!/usr/bin/env python3
"""Keep every lesson's acronym terminology self-contained.

Policy:
- every lesson gets a final "本课术语表" before the next-lesson link;
- every glossary-eligible abbreviation that appears in lesson prose is listed;
- first substantive prose use is normalized to:
    English Full Name (ABBR，中文名)
- product names with no official expansion are never given a fake expansion.

Run:
  python scripts/lesson_terms.py --write   # normalize lesson HTML
  python scripts/lesson_terms.py           # check idempotence / coverage
"""
from __future__ import annotations

import argparse
import html as html_lib
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LESSONS = ROOT / "learn"

# abbr: (English full name / official name, Chinese name, one-line course meaning, expand_first_use)
TERMS = {
    "AI": ("Artificial Intelligence", "人工智能", "本课程讨论的大模型训练、推理与系统基础设施所服务的计算领域。", True),
    "API": ("Application Programming Interface", "应用程序编程接口", "软件组件之间约定的调用接口；源码阅读时先看 contract，再看实现。", True),
    "CPU": ("Central Processing Unit", "中央处理器", "主要承担 Python 控制流、调度、数据准备与部分通信控制。", True),
    "GPU": ("Graphics Processing Unit", "图形处理器", "大模型训练与推理中执行 GEMM、Attention 等高并行计算的主要设备。", True),
    "RAM": ("Random Access Memory", "随机存取存储器", "这里通常指主机侧系统内存。", True),
    "VRAM": ("Video Random Access Memory", "显存", "工程语境里常泛指 GPU device memory；不等于特指 HBM。", True),
    "HBM": ("High Bandwidth Memory", "高带宽内存", "数据中心 GPU/加速器常见的高带宽设备内存技术。", True),
    "GDDR": ("Graphics Double Data Rate", "图形双倍数据速率存储器", "很多消费级/工作站 GPU 使用的显存技术。", True),
    "SM": ("Streaming Multiprocessor", "流式多处理器", "NVIDIA GPU 上承载线程/warp 与计算资源的核心执行单元。", True),
    "ALU": ("Arithmetic Logic Unit", "算术逻辑单元", "执行通用算术与逻辑运算的硬件单元。", True),
    "CUDA": ("CUDA", "NVIDIA GPU 并行计算平台", "官方产品名，没有应当强行展开的现代英文全称；本课按 CUDA 平台/编程栈理解。", False),
    "GEMM": ("General Matrix Multiply", "通用矩阵乘法", "Linear、MLP 等大量计算最终会落到的矩阵乘核心操作。", True),
    "MLP": ("Multi-Layer Perceptron", "多层感知机", "Transformer block 中常见的前馈子层。", True),
    "LLM": ("Large Language Model", "大语言模型", "本课程训练与推理系统的主要模型对象。", True),
    "MHA": ("Multi-Head Attention", "多头注意力", "让多个 attention head 在不同子空间并行计算。", True),
    "GQA": ("Grouped-Query Attention", "分组查询注意力", "多个 Query heads 共享较少的 KV heads，以减少 KV 成本。", True),
    "MQA": ("Multi-Query Attention", "多查询注意力", "所有 Query heads 共享单组 K/V 的 Attention 变体。", True),
    "MLA": ("Multi-head Latent Attention", "多头潜在注意力", "通过潜在表示压缩注意力状态的模型结构；会影响 KV 表示与兼容性。", True),
    "QKV": ("Query-Key-Value", "查询-键-值", "Attention 中 Q、K、V 三类投影/状态的合称。", True),
    "KV": ("Key-Value", "键-值", "自回归 Attention 需要复用的历史 Key/Value 状态。", True),
    "RoPE": ("Rotary Position Embedding", "旋转位置编码", "把位置信息注入 Query/Key 的常见位置编码方法。", True),
    "GELU": ("Gaussian Error Linear Unit", "高斯误差线性单元", "Transformer/MLP 中常见的非线性激活函数。", True),
    "FP32": ("32-bit Floating Point", "32 位浮点", "常见高精度浮点格式，元素通常占 4 bytes。", True),
    "FP16": ("16-bit Floating Point", "16 位浮点", "半精度浮点格式，常用于降低显存与提升矩阵计算吞吐。", True),
    "BF16": ("Brain Floating Point 16", "BF16 16 位浮点", "与 FP16 同为 16 位，但指数范围更接近 FP32，训练中很常见。", True),
    "FP8": ("8-bit Floating Point", "8 位浮点", "更低精度的浮点表示，需要硬件与数值策略共同支持。", True),
    "TF32": ("TensorFloat-32", "TensorFloat-32 浮点格式", "NVIDIA Tensor Core 上用于部分 FP32 工作负载的计算格式。", True),
    "PCIe": ("Peripheral Component Interconnect Express", "高速外设组件互连", "CPU、GPU、NIC 等设备之间常见的主机互连总线。", True),
    "DMA": ("Direct Memory Access", "直接内存访问", "设备在较少 CPU 搬运参与下直接读写内存的数据移动机制。", True),
    "NUMA": ("Non-Uniform Memory Access", "非一致内存访问", "多插槽系统里不同 CPU/内存位置具有不同访问代价的拓扑特征。", True),
    "NVMe": ("Non-Volatile Memory Express", "非易失性存储器高速接口", "面向高速 SSD 的协议/接口，在分层存储与 offload 场景中常见。", True),
    "DP": ("Data Parallel", "数据并行", "不同 replica 处理不同数据，再同步训练状态；它主要切数据而不是切单层模型。", True),
    "DDP": ("Distributed Data Parallel", "分布式数据并行", "PyTorch 常用的数据并行训练方式，每个 rank 通常持有模型副本并同步梯度。", True),
    "TP": ("Tensor Parallel", "张量并行", "把单层内部的大矩阵/计算沿张量维度分到多个 ranks。", True),
    "PP": ("Pipeline Parallel", "流水线并行", "沿模型深度切成多个 pipeline stages，并用 microbatch 流水。", True),
    "SP": ("Sequence Parallel", "序列并行", "在 TP 域内沿 sequence 维分片部分 activation，减少重复保存。", True),
    "CP": ("Context Parallel", "上下文并行", "让 Attention 沿 context/sequence 维跨 ranks 工作并交换必要上下文。", True),
    "EP": ("Expert Parallel", "专家并行", "把 MoE experts 分布到不同 ranks，并路由 token 到对应 expert。", True),
    "FSDP": ("Fully Sharded Data Parallel", "全分片数据并行", "在数据并行域进一步分片参数、梯度或优化器状态的训练方式。", True),
    "MoE": ("Mixture of Experts", "混合专家模型", "由 router 为 token 选择少数 experts 的稀疏模型结构。", True),
    "P2P": ("Peer-to-Peer", "点对点通信", "两个 peer/rank 之间直接发送和接收数据的通信模式。", True),
    "A2A": ("All-to-All", "全对全通信", "每个 rank 给不同 peers 发送不同数据；MoE dispatch 中很典型。", True),
    "AG": ("All-Gather", "全收集", "把各 rank 的 shard 收集成完整结果并让参与者获得。", True),
    "RS": ("Reduce-Scatter", "归约分散", "先做 reduction，再让每个 rank 只保留一个 reduced shard。", True),
    "NCCL": ("NVIDIA Collective Communications Library", "NVIDIA 集体通信库", "GPU 集群中常用的 collective / P2P 通信软件库。", True),
    "1F1B": ("One Forward One Backward", "一前向一反向", "Pipeline steady state 中交替执行 forward/backward microbatch 的调度方式。", True),
    "OOM": ("Out Of Memory", "内存不足/显存不足", "所需内存超过可用容量时的失败状态。", True),
    "TTFT": ("Time To First Token", "首 Token 延迟", "从请求进入系统到第一个输出 token 可见的端到端时间。", True),
    "ITL": ("Inter-Token Latency", "Token 间延迟", "流式生成时相邻可见输出 token 之间的时间间隔。", True),
    "TPOT": ("Time Per Output Token", "每输出 Token 时间", "生成阶段按输出 token 聚合后的平均耗时指标。", True),
    "TPS": ("Tokens Per Second", "每秒 Token 数", "用 token 数衡量的吞吐率。", True),
    "QPS": ("Queries Per Second", "每秒查询数", "系统每秒处理请求/查询数量的吞吐指标。", True),
    "SLO": ("Service Level Objective", "服务级目标", "系统希望达到的延迟、可用性或吞吐目标。", True),
    "SLA": ("Service Level Agreement", "服务级协议", "对服务质量指标的正式约定。", True),
    "APC": ("Automatic Prefix Caching", "自动前缀缓存", "vLLM 中复用完整缓存 block 的前缀缓存机制。", True),
    "LRU": ("Least Recently Used", "最近最少使用", "按最近使用时间决定淘汰优先级的经典策略。", True),
    "TTL": ("Time To Live", "生存时间/有效期", "资源或元数据在过期前允许继续有效的时间窗口。", True),
    "RPC": ("Remote Procedure Call", "远程过程调用", "跨进程/节点调用远端服务接口的通信抽象。", True),
    "IPC": ("Inter-Process Communication", "进程间通信", "同机或跨边界进程交换控制/数据的通用机制。", True),
    "NIC": ("Network Interface Card", "网络接口卡", "主机连接网络 fabric 的硬件接口。", True),
    "IB": ("InfiniBand", "InfiniBand 高速互连", "HPC/AI 集群常见的低延迟高带宽网络技术。", True),
    "RDMA": ("Remote Direct Memory Access", "远程直接内存访问", "允许远端内存传输减少 CPU 数据搬运参与的网络访问机制。", True),
    "RoCE": ("RDMA over Converged Ethernet", "基于融合以太网的 RDMA", "在以太网上承载 RDMA 语义的网络技术。", True),
    "UCX": ("Unified Communication X", "统一通信 X", "为 HPC/AI 提供多种传输后端抽象的通信框架。", True),
    "NIXL": ("NVIDIA Inference Xfer Library", "NVIDIA 推理传输库", "为 AI 推理框架提供跨多类 memory/storage 的点对点数据传输抽象。", True),
    "P/D": ("Prefill/Decode Disaggregation", "Prefill/Decode 分离", "把 Prefill 与 Decode 放到不同资源池/实例并协调 KV handoff。", True),
    "DCP": ("Decode Context Parallelism", "Decode 上下文并行", "在 Decode 阶段沿上下文维并行 Attention 的执行策略。", True),
    "LBHNC": ("LBHNC KV layout", "LBHNC KV 布局", "当前 NIXL/vLLM 文档中的 KV cache 维度布局记号，本身不是需要强行展开的英文缩写。", False),
    "LBNHC": ("LBNHC KV layout", "LBNHC KV 布局", "另一种 KV cache 维度布局记号，本身是维度顺序代码。", False),
}

# Avoid expanding inside code/diagrams/titles; "first use" means first substantive prose use.
SKIP_TAGS = {"code", "pre", "script", "style", "svg", "table", "h1", "h2", "h3", "nav", "aside"}
SKIP_CLASSES = {"lesson-kicker", "section-no", "breadcrumb", "mobile-course-bar", "lesson-terms", "toc"}
TERM_SECTION_RE = re.compile(r"\s*<section class=\"lesson-terms\".*?</section>\s*", re.S)
MAIN_RE = re.compile(r"(?P<open><main\b[^>]*class=\"[^\"]*\barticle\b[^\"]*\"[^>]*>)(?P<body>.*?)(?P<close></main>)", re.S | re.I)
TAG_RE = re.compile(r"(<[^>]+>)")
OPEN_TAG_RE = re.compile(r"<\s*([A-Za-z0-9]+)\b([^>]*)>")
CLOSE_TAG_RE = re.compile(r"</\s*([A-Za-z0-9]+)\s*>")
CLASS_RE = re.compile(r'class\s*=\s*[\"\']([^\"\']*)[\"\']', re.I)
NEXT_RE = re.compile(r"(?P<indent>\n\s*)(?=<a\b[^>]*class=\"[^\"]*\bnext-lesson\b|<div\b[^>]*class=\"[^\"]*\bnext-lesson\b)", re.I)


def term_pattern(abbr: str) -> re.Pattern[str]:
    if abbr == "P/D":
        return re.compile(r"(?<![A-Za-z0-9])P/D(?![A-Za-z0-9])")
    return re.compile(rf"(?<![A-Za-z0-9]){re.escape(abbr)}(?![A-Za-z0-9])")


def tokenize_body(body: str):
    """Yield (kind, text, blocked) preserving exact HTML."""
    stack: list[tuple[str, bool]] = []
    for part in TAG_RE.split(body):
        if not part:
            continue
        if not part.startswith("<"):
            yield "text", part, any(blocked for _, blocked in stack)
            continue
        close = CLOSE_TAG_RE.match(part)
        if close:
            tag = close.group(1).lower()
            # Pop back to the matching tag; tolerant of compact hand-written HTML.
            for i in range(len(stack) - 1, -1, -1):
                if stack[i][0] == tag:
                    del stack[i:]
                    break
            yield "tag", part, any(blocked for _, blocked in stack)
            continue
        op = OPEN_TAG_RE.match(part)
        if op and not part.rstrip().endswith("/>") and not part.startswith("<!--"):
            tag = op.group(1).lower()
            attrs = op.group(2)
            classes = set()
            m = CLASS_RE.search(attrs)
            if m:
                classes = set(m.group(1).split())
            blocked = tag in SKIP_TAGS or bool(classes & SKIP_CLASSES)
            stack.append((tag, blocked))
        yield "tag", part, any(blocked for _, blocked in stack)


def visible_text(body: str) -> str:
    chunks = []
    for kind, text, blocked in tokenize_body(body):
        if kind == "text" and not blocked:
            chunks.append(html_lib.unescape(text))
    return " ".join(chunks)


def found_terms(body: str) -> list[str]:
    text = visible_text(body)
    positions = []
    for abbr in TERMS:
        m = term_pattern(abbr).search(text)
        if m:
            positions.append((m.start(), abbr))
    return [abbr for _, abbr in sorted(positions)]


def expand_first_uses(body: str, abbreviations: list[str]) -> str:
    pending = {abbr for abbr in abbreviations if TERMS[abbr][3]}
    if not pending:
        return body
    parts = []
    for kind, text, blocked in tokenize_body(body):
        if kind != "text" or blocked or not pending:
            parts.append(text)
            continue
        for abbr in list(pending):
            english, chinese, _, _ = TERMS[abbr]
            pat = term_pattern(abbr)
            m = pat.search(text)
            if not m:
                continue
            # Already normalized in this prose fragment: do not duplicate it.
            window = html_lib.unescape(text[max(0, m.start() - 120): m.end() + 120])
            if english in window and chinese in window:
                pending.remove(abbr)
                continue
            replacement = f"{english} ({abbr}，{chinese})"
            text = text[:m.start()] + replacement + text[m.end():]
            pending.remove(abbr)
        parts.append(text)
    return "".join(parts)


def term_section(abbreviations: list[str]) -> str:
    rows = []
    for abbr in abbreviations:
        english, chinese, meaning, _ = TERMS[abbr]
        rows.append(
            "        <tr>"
            f"<td><code>{html_lib.escape(abbr)}</code></td>"
            f"<td>{html_lib.escape(english)}</td>"
            f"<td>{html_lib.escape(chinese)}</td>"
            f"<td>{html_lib.escape(meaning)}</td>"
            "</tr>"
        )
    if not rows:
        rows.append("        <tr><td>—</td><td>—</td><td>—</td><td>本课没有需要额外展开的技术缩写。</td></tr>")
    return (
        "\n      <section class=\"lesson-terms\" id=\"lesson-terms\">\n"
        "        <div class=\"section-no\">TERMS · 本课术语表</div>\n"
        "        <h2>本课出现的缩写与术语</h2>\n"
        "        <p class=\"lesson-terms-intro\">第一次在正文使用时采用 <strong>English Full Name (ABBR，中文名)</strong>；这里集中复习，避免学习时来回跳总站 Glossary。</p>\n"
        "        <div class=\"lesson-terms-scroll\"><table class=\"lesson-terms-table\">\n"
        "          <thead><tr><th>缩写</th><th>English full name</th><th>中文名</th><th>本课怎么理解</th></tr></thead>\n"
        "          <tbody>\n" + "\n".join(rows) + "\n          </tbody>\n"
        "        </table></div>\n"
        "      </section>\n"
    )


def normalize_html(source: str) -> str:
    m = MAIN_RE.search(source)
    if not m:
        raise ValueError("missing <main class=\"article\">")
    body = TERM_SECTION_RE.sub("\n", m.group("body"))
    abbreviations = found_terms(body)
    body = expand_first_uses(body, abbreviations)
    # Re-scan after expansion (same semantic set, but keeps ordering stable).
    abbreviations = found_terms(body)
    section = term_section(abbreviations)
    next_match = NEXT_RE.search(body)
    if next_match:
        body = body[:next_match.start()] + section + body[next_match.start():]
    else:
        body = body.rstrip() + section + "\n"
    return source[:m.start()] + m.group("open") + body + m.group("close") + source[m.end():]


def lesson_files() -> list[Path]:
    return sorted(p for p in LESSONS.glob("**/*.html") if p.is_file())


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()

    changed = []
    total_terms = 0
    for path in lesson_files():
        old = path.read_text(encoding="utf-8")
        try:
            new = normalize_html(old)
        except ValueError as exc:
            raise SystemExit(f"{path.relative_to(ROOT)}: {exc}")
        total_terms += len(found_terms(MAIN_RE.search(new).group("body")))
        if new != old:
            changed.append(path)
            if args.write:
                path.write_text(new, encoding="utf-8")

    if args.write:
        print(f"Terminology normalized: {len(lesson_files())} lessons; {len(changed)} file(s) changed; {total_terms} lesson-term occurrences indexed.")
        return 0
    if changed:
        print("Terminology normalization required:")
        for path in changed:
            print(f"  - {path.relative_to(ROOT)}")
        print("Run: python scripts/lesson_terms.py --write")
        return 1
    print(f"Lesson terminology checked: {len(lesson_files())} lessons; {total_terms} lesson-term occurrences indexed; 0 drift.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
