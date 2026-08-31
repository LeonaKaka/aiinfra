from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def replace_once(text: str, old: str, new: str, label: str) -> str:
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"{label}: expected one anchor, found {count}")
    return text.replace(old, new, 1)


def patch_readme() -> None:
    path = ROOT / "README.md"
    text = path.read_text(encoding="utf-8")
    text = replace_once(text, "当前有 14 个可运行实验，分成两条完整主线：", "当前有 18 个可运行实验，分成两条完整主线：", "README lab count")
    text = replace_once(text, "**Training · A1–A7**", "**Training · A1–A11**", "README training range")
    text = replace_once(
        text,
        "- A7 Profiler-ready Overlap\n\n**Inference · B1–B7**",
        "- A7 Profiler-ready Overlap\n- A8 Mini Sequence Parallel\n- A9 Mini Pipeline Parallel\n- A10 Mini Context Parallel\n- A11 Mini Expert Parallel\n\n**Inference · B1–B7**",
        "README new labs",
    )
    text = replace_once(
        text,
        "用 `torchrun` / Gloo 实际执行 A1–A7 与 B1–B7 共 14 个实验。",
        "用 `torchrun` / Gloo 实际执行 A1–A11 与 B1–B7 共 18 个实验。",
        "README smoke count",
    )
    path.write_text(text, encoding="utf-8")


def patch_labs_readme() -> None:
    path = ROOT / "labs/README.md"
    text = path.read_text(encoding="utf-8")
    anchor = "用 `torch.profiler` 标记 forward、W2 gradient ready、async `reduce_scatter_single`、继续 backward 与最终 wait，并导出每个 rank 的 Chrome trace。CPU/Gloo 已验证 correctness 与 range 顺序；只有 GPU/NCCL trace 出现 NCCL kernel 与 compute kernel 时间重叠时，才可以宣称真实 overlap。\n\n## Inference Infra"
    insert = """用 `torch.profiler` 标记 forward、W2 gradient ready、async `reduce_scatter_single`、继续 backward 与最终 wait，并导出每个 rank 的 Chrome trace。CPU/Gloo 已验证 correctness 与 range 顺序；只有 GPU/NCCL trace 出现 NCCL kernel 与 compute kernel 时间重叠时，才可以宣称真实 overlap。

### Lab A8 — Mini Sequence Parallel
```bash
torchrun --standalone --nproc-per-node=2 labs/code/mini_sequence_parallel.py
```
模拟 TP row-parallel partial output 直接通过 `reduce_scatter_single` 落成 sequence-sharded activation；局部算子后再 `all_gather_single`，与 dense `tanh(x @ W)` reference 对齐。重点是 TP layout → SP layout 的转换，而不是把 SP 简化成普通 sequence slicing。

### Lab A9 — Mini Pipeline Parallel
```bash
torchrun --standalone --nproc-per-node=2 labs/code/mini_pipeline_parallel.py
```
两个 ranks 分别拥有两段模型，按 microbatch 传 activation，backward 再传 activation gradient。实验采用最简单的 GPipe-style `F0 F1 F2 → B2 B1 B0` 教学 schedule，并明确不冒充 Megatron production 1F1B。

### Lab A10 — Mini Context Parallel
```bash
torchrun --standalone --nproc-per-node=2 labs/code/mini_context_parallel.py
```
每个 rank 的 local Q 保持本地，KV chunks 按 ring 轮转；causal mask 始终用 global token positions，最后每个 local output 与 dense causal attention 对应切片对齐。ring order 只是教学实现，不是当前 Megatron CP 的唯一算法。

### Lab A11 — Mini Expert Parallel
```bash
torchrun --standalone --nproc-per-node=2 labs/code/mini_expert_parallel.py
```
确定性 balanced router 把 token 通过 `all_to_all_single` 发给 expert owners，本地 expert compute 后 reverse all-to-all，再恢复原 token 顺序并与 reference 对齐。该实验只教学 ownership / dispatch / combine 生命周期，不模拟 capacity、dropping 或 load balancing。

## Inference Infra"""
    text = replace_once(text, anchor, insert, "labs README A8-A11")
    text = replace_once(
        text,
        "软件实验主线到 B7 已闭合。下一阶段只做有证据的硬件验证",
        "软件实验现在覆盖 Training A1–A11 与 Inference B1–B7。下一阶段优先做跨机制 capstone 与有证据的硬件验证",
        "labs README closing",
    )
    path.write_text(text, encoding="utf-8")


def patch_labs_index() -> None:
    path = ROOT / "labs/index.html"
    text = path.read_text(encoding="utf-8")
    text = replace_once(text, "<span>14 experiments</span>", "<span>18 experiments</span>", "labs index count")
    text = replace_once(
        text,
        "Training 线从 TP forward/backward 走到 DP、Distributed Optimizer、bucketed overlap、TP×DP 与 profiler 证据层；",
        "Training 线从 TP forward/backward 走到 DP、Distributed Optimizer、bucketed overlap、TP×DP、SP、PP、CP、EP 与 profiler 证据层；",
        "labs index hero",
    )
    a7 = '<a class="lab-card train" href="./mini-profiler-overlap.html"><small>LAB A7 · PROFILER EVIDENCE</small><h2>Profiler-ready Overlap</h2><p>用 torch.profiler 标出 gradient-ready、async reduce-scatter、继续 backward 与 wait；CPU 验顺序，GPU/NCCL 才能证明真实 overlap。</p><footer><span>核心问题</span><strong>怎样证明 async 真的隐藏了通信？</strong></footer></a>'
    new_cards = a7 + '''
      <a class="lab-card train" href="./mini-sequence-parallel.html"><small>LAB A8 · SEQUENCE PARALLEL</small><h2>Mini Sequence Parallel</h2><p>TP partial output 不先复制成 full activation，而是 reduce-scatter 直接落成 sequence shard，再与 dense reference 对齐。</p><footer><span>核心问题</span><strong>SP 为什么通常和 TP 的 layout transition 绑在一起？</strong></footer></a>
      <a class="lab-card train" href="./mini-pipeline-parallel.html"><small>LAB A9 · PIPELINE PARALLEL</small><h2>Mini Pipeline Parallel</h2><p>两个 stages 分别拥有模型的一半；microbatch forward 传 activation，backward 把 activation gradient 送回来。</p><footer><span>核心问题</span><strong>PP 的 stage boundary 到底传什么？</strong></footer></a>
      <a class="lab-card train" href="./mini-context-parallel.html"><small>LAB A10 · CONTEXT PARALLEL</small><h2>Mini Context Parallel</h2><p>Q shard 保持本地，KV chunks 轮转；用 global positions 保持 causal semantics，并对齐 dense attention。</p><footer><span>核心问题</span><strong>sequence 切开后，Attention 怎样仍看到全局 context？</strong></footer></a>
      <a class="lab-card train" href="./mini-expert-parallel.html"><small>LAB A11 · EXPERT PARALLEL</small><h2>Mini Expert Parallel</h2><p>router → all-to-all dispatch → local experts → reverse combine；同时验证数值和 token identity。</p><footer><span>核心问题</span><strong>token 为什么必须去 expert owner，再回到原顺序？</strong></footer></a>'''
    text = replace_once(text, a7, new_cards, "labs index cards")
    text = text.replace("14 个脚本目前的第一验收标准", "18 个脚本目前的第一验收标准")
    path.write_text(text, encoding="utf-8")


def patch_source_map() -> None:
    path = ROOT / "source-map/index.html"
    text = path.read_text(encoding="utf-8")
    text = replace_once(
        text,
        '<a href="../labs/mini-tp-backward.html">A2 →</a></div>',
        '<a href="../labs/mini-tp-backward.html">A2 →</a><a href="../labs/mini-sequence-parallel.html">A8 SP →</a></div>',
        "source map SP link",
    )
    text = replace_once(
        text,
        '<a href="../learn/05-megatron/pipeline-parallel.html">05.4 →</a></div>',
        '<a href="../learn/05-megatron/pipeline-parallel.html">05.4 →</a><a href="../labs/mini-pipeline-parallel.html">A9 →</a></div>',
        "source map PP schedule link",
    )
    text = replace_once(
        text,
        '<a href="../learn/05-megatron/pipeline-parallel.html">05.4 →</a></div>\n        </div>\n        <div class="source-row">\n          <span class="order">06</span>',
        '<a href="../learn/05-megatron/pipeline-parallel.html">05.4 →</a><a href="../labs/mini-pipeline-parallel.html">A9 →</a></div>\n        </div>\n        <div class="source-row">\n          <span class="order">CP</span>\n          <div><em class="phase-label">DATA · CONTEXT</em><b>Context Parallel attention</b><code>megatron/core/transformer/attention.py · parallel_state.py</code></div>\n          <p>带着“local Q 留在哪里、remote context/KV 为什么要来、global position 如何保持”去读。先掌握 05.6 的语义，再看当前具体 partition / communication strategy。</p>\n          <div class="source-actions"><a href="https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/core/transformer/attention.py">attention ↗</a><a href="../learn/05-megatron/context-parallel.html">05.6 →</a><a href="../labs/mini-context-parallel.html">A10 →</a></div>\n        </div>\n        <div class="source-row">\n          <span class="order">EP</span>\n          <div><em class="phase-label">DATA · MOE</em><b>MoE token dispatcher</b><code>megatron/core/transformer/moe/token_dispatcher.py</code></div>\n          <p>按 route → dispatch → expert compute → combine 的生命周期读；先看 token ownership / permutation，再区分 all-gather、all-to-all 或其他 dispatcher backend。</p>\n          <div class="source-actions"><a href="https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/core/transformer/moe/token_dispatcher.py">源码 ↗</a><a href="../learn/05-megatron/expert-parallel.html">05.7 →</a><a href="../labs/mini-expert-parallel.html">A11 →</a></div>\n        </div>\n        <div class="source-row">\n          <span class="order">06</span>',
        "source map CP EP rows",
    )
    path.write_text(text, encoding="utf-8")


def main() -> None:
    patch_readme()
    patch_labs_readme()
    patch_labs_index()
    patch_source_map()
    print("Synced A8-A11 into README, Labs index, Labs README, and Source Map")


if __name__ == "__main__":
    main()
