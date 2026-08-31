from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def replace_once(text: str, old: str, new: str, label: str) -> str:
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"{label}: expected exactly one anchor, found {count}")
    return text.replace(old, new, 1)


def patch_tensor_foundation() -> None:
    path = ROOT / "learn/01-foundations/tensor.html"
    text = path.read_text(encoding="utf-8")
    if 'id="storage-layout"' in text:
        return

    text = replace_once(
        text,
        '<div class="lesson-kicker"><span>LESSON 01.1 · FOUNDATION</span><span>约 25 min</span></div>',
        '<div class="lesson-kicker"><span>LESSON 01.1 · FOUNDATION</span><span>约 35 min</span></div>',
        "01.1 reading time",
    )
    text = replace_once(
        text,
        '<li>用一个实际句子解释 batch、sequence length、hidden size。</li>\n          <li>理解为什么以后学 Megatron 时，“沿哪个维度切 tensor”会成为核心问题。</li>',
        '<li>用一个实际句子解释 batch、sequence length、hidden size。</li>\n          <li>区分 tensor 的逻辑 shape 与底层 storage/layout，理解 view、stride、contiguous 为什么会影响 kernel 与数据搬运。</li>\n          <li>理解为什么以后学 Megatron 时，“沿哪个维度切 tensor”会成为核心问题。</li>',
        "01.1 learning goal",
    )
    text = replace_once(
        text,
        '<div class="why-card"><b>shape</b><p>有多少个元素、每个维度多长。它决定了后面矩阵乘法怎么做，也决定能不能沿某一维切分。</p></div>',
        '<div class="why-card"><b>shape</b><p>给出各个维度的长度；总元素数由这些维度长度的乘积决定。它决定后面矩阵乘法怎么做，也决定能不能沿某一维切分。</p></div>',
        "01.1 shape precision",
    )

    storage_section = '''      <section class="article-section" id="storage-layout">
        <div class="section-no">05 · STORAGE / VIEW / STRIDE</div>
        <h2>同一个 shape，不一定以同一种方式躺在内存里。</h2>
        <p>到目前为止，我们一直把 tensor 当成“带 shape 的数字块”。这对读模型计算已经够用，但读 AI Infra 还差一层：<strong>shape 描述的是逻辑索引空间，storage/layout 描述这些元素怎样映射到底层内存。</strong>两块 tensor 可以拥有相同 shape，却有不同的 stride，甚至共享同一块 storage。</p>
        <div class="code-block"><div class="code-head"><span>python</span><span>shape vs layout</span></div><pre><code>x = torch.arange(12).reshape(3, 4)
y = x.transpose(0, 1)

print(x.shape, x.stride())  <span class="comment"># [3,4], (4,1)</span>
print(y.shape, y.stride())  <span class="comment"># [4,3], (1,4)</span>
print(y.is_contiguous())    <span class="comment"># False</span>

z = y.contiguous()
print(z.stride())           <span class="comment"># materialize a contiguous layout</span></code></pre></div>
        <div class="why-grid">
          <div class="why-card"><b>storage</b><p>底层真正拥有内存的 allocation。多个 tensor/view 可以引用同一块 storage，而不是每次都复制数据。</p></div>
          <div class="why-card"><b>stride</b><p>某一维索引增加 1 时，在底层 storage 里要跨过多少个元素。它把“逻辑坐标”映射到“物理位置”。</p></div>
          <div class="why-card"><b>contiguous</b><p>元素是否按当前逻辑维度顺序紧密排列。某些 kernel 或 transfer path 可以处理任意 stride，另一些路径会要求或更偏好 contiguous buffer。</p></div>
        </div>
        <p><code>transpose</code> 常常只改变 view 的 shape/stride，而不立即复制 storage；<code>contiguous()</code> 则可能真的分配新内存并重排数据。以后看到 reshape、permute、view、copy 时，要开始问：<strong>这是只改了“怎么看”，还是实际搬了数据？</strong></p>
        <div class="concept-note"><p><strong>为什么这对后面重要：</strong>07.4 的 block table 和 08.4 的 registered memory / transfer region 都会把“逻辑 tensor”继续落到更具体的地址、offset、stride 与 lifetime。现在先建立一个底线：<strong>logical shape ≠ physical layout</strong>。</p></div>
      </section>

'''
    text = replace_once(
        text,
        '      <section class="article-section" id="megatron">\n        <div class="section-no">05 · WHERE THIS LEADS</div>',
        storage_section + '      <section class="article-section" id="megatron">\n        <div class="section-no">06 · WHERE THIS LEADS</div>',
        "01.1 storage section",
    )
    text = replace_once(
        text,
        '<div class="quiz-a">因为跨 GPU 使用数据往往意味着通信或拷贝。AI Infra 的大量性能问题，就是在研究哪些数据必须移动、移动多少、什么时候移动。</div>\n        </div>\n      </section>',
        '<div class="quiz-a">因为跨 GPU 使用数据往往意味着通信或拷贝。AI Infra 的大量性能问题，就是在研究哪些数据必须移动、移动多少、什么时候移动。</div>\n        </div>\n        <div class="quiz-item">\n          <div class="quiz-q">4. 两个 tensor 的 shape 一样，是否意味着它们的底层内存布局也一定一样？</div>\n          <button class="quiz-toggle" type="button">看答案</button>\n          <div class="quiz-a">不一定。shape 只说明各维长度；transpose/view 等操作可以让 tensor 拥有不同 stride，甚至共享同一 storage。是否 contiguous、是否需要实际 copy，是另一层问题。</div>\n        </div>\n      </section>',
        "01.1 checkpoint",
    )
    text = replace_once(
        text,
        '<a href="#memory">Infra 视角</a>\n      <a href="#megatron">连接 Megatron</a>',
        '<a href="#memory">Infra 视角</a>\n      <a href="#storage-layout">Storage / Stride</a>\n      <a href="#megatron">连接 Megatron</a>',
        "01.1 toc",
    )
    path.write_text(text, encoding="utf-8")


def patch_transformer_position_bridge() -> None:
    path = ROOT / "learn/02-transformer/transformer-block.html"
    text = path.read_text(encoding="utf-8")
    if 'id="position-semantics"' in text:
        return

    text = replace_once(
        text,
        '<div class="lesson-kicker"><span>LESSON 02.3 · TRANSFORMER</span><span>约 45 min</span></div>',
        '<div class="lesson-kicker"><span>LESSON 02.3 · TRANSFORMER</span><span>约 55 min</span></div>',
        "02.3 reading time",
    )
    text = replace_once(
        text,
        '<li>区分“一个 block 的 forward”和外层 training loop / optimizer。</li><li>指出 Tensor Parallel、Pipeline Parallel、Key-Value (KV，键-值) Cache 分别会落在 block 的什么位置。</li>',
        '<li>区分“一个 block 的 forward”和外层 training loop / optimizer。</li><li>解释 token position 如何进入 Attention，并理解可复用 KV 还依赖一致的位置与模型语义。</li><li>指出 Tensor Parallel、Pipeline Parallel、Key-Value (KV，键-值) Cache 分别会落在 block 的什么位置。</li>',
        "02.3 learning goal",
    )
    text = replace_once(
        text,
        '<p>这里用 PyTorch 的高级 <code>MultiheadAttention</code> 是因为上一课已经手写过 Attention shape；这一课重点是观察一个 block 如何组合子模块。真实 LLM 的 attention、mask、Rotary Position Embedding (RoPE，旋转位置编码)、RMSNorm、gated MLP 等会更具体。</p>',
        '<p>这里用 PyTorch 的高级 <code>MultiheadAttention</code> 是因为上一课已经手写过 Attention shape；这一课重点是观察一个 block 如何组合子模块。真实 LLM 的 attention、mask、RoPE、RMSNorm、gated MLP 等会更具体。</p>',
        "02.3 RoPE duplicate expansion",
    )

    position_section = '''      <section class="article-section" id="position-semantics">
        <div class="section-no">06 · POSITION IS PART OF ATTENTION STATE</div>
        <h2>同一个 token 出现在第 10 位和第 100 位，Attention 不能把它们当成同一件事。</h2>
        <p>decoder 不只需要知道“token 是什么”，还需要知道“token 在序列的什么位置”。很多现代 LLM 使用 Rotary Position Embedding (RoPE，旋转位置编码)：在常见实现里，Query 和 Key 会根据 <code>positions</code> 做位置相关变换，再进入 Attention。Value 通常不做同样的旋转。</p>
        <div class="code-block"><div class="code-head"><span>mental model</span><span>position-conditioned attention</span></div><pre><code>hidden states
    │
    ├─→ Q ─┐
    ├─→ K ─┼─→ RoPE(positions, Q, K) ─→ Attention
    └─→ V ────────────────────────────→ Attention

<span class="comment"># common decoder idea, not a promise about kernel fusion order</span></code></pre></div>
        <p>这会直接影响后面的 KV Cache：缓存不是“随便保存一份 K/V 数字以后就能到处用”。要安全复用本地或远端 KV，至少要保证它仍对应<strong>同一模型语义、同一层/头布局、同一 token 前缀与一致的位置编码约定</strong>。如果 position 或 RoPE 配置不兼容，即使字节数和 shape 都对，语义也可能已经错了。</p>
        <div class="why-grid"><div class="why-card"><b>token identity</b><p>缓存必须对应你以为的那段 prefix，而不是另一段“长度刚好一样”的序列。</p></div><div class="why-card"><b>position semantics</b><p>位置编号、RoPE 参数/缩放等必须与生成这些 K/V 时的约定兼容。</p></div><div class="why-card"><b>layout / format</b><p>layer、head、dtype、cache layout 也必须与消费端契约一致；这会在 06–08 模块继续具体化。</p></div></div>
        <div class="concept-note"><p><strong>实现和语义要分开：</strong>真实 runtime 可以把 QK norm、RoPE、KV cache write 等步骤融合进 kernel；你不需要背某个函数调用顺序。需要牢牢记住的是：<strong>KV reuse 复用的是带上下文语义的模型状态，不只是搬一段显存。</strong></p></div>
        <div class="source-note">当前 vLLM 的 Qwen3 路径就是一个直观例子：<code>positions</code> 传入 attention，Q/K 先经过 <code>rotary_emb(positions, q, k)</code>，随后进入 attention。源码入口：<a href="https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/models/qwen3.py">vLLM · qwen3.py</a>。具体模型与 fused kernel 可能采用不同实现组织，因此这里把它作为语义示例，不把函数排列当永久 API。</div>
      </section>

'''
    text = replace_once(
        text,
        '      <section class="article-section" id="stack">\n        <div class="section-no">06 · FROM ONE BLOCK TO A MODEL</div>',
        position_section + '      <section class="article-section" id="stack">\n        <div class="section-no">07 · FROM ONE BLOCK TO A MODEL</div>',
        "02.3 position section",
    )
    text = replace_once(text, '<div class="section-no">07 · BLOCK VS TRAINING LOOP</div>', '<div class="section-no">08 · BLOCK VS TRAINING LOOP</div>', "02.3 section 08")
    text = replace_once(text, '<div class="section-no">08 · INFRA MAP OF A BLOCK</div>', '<div class="section-no">09 · INFRA MAP OF A BLOCK</div>', "02.3 section 09")
    text = replace_once(text, '<div class="section-no">09 · THE ROAD TO INFERENCE</div>', '<div class="section-no">10 · THE ROAD TO INFERENCE</div>', "02.3 section 10")
    text = replace_once(
        text,
        '后面会正式推导通信。</div></div></section>',
        '后面会正式推导通信。</div></div><div class="quiz-item"><div class="quiz-q">4. 为什么远端拿到 shape 完全正确的 KV，仍然不代表一定可以安全复用？</div><button class="quiz-toggle" type="button">看答案</button><div class="quiz-a">因为 KV 还依赖生成它时的模型、layer/head layout、token prefix、position/RoPE 语义和 cache format。字节数正确只是必要条件，不是语义兼容的充分条件。</div></div></section>',
        "02.3 checkpoint",
    )
    text = replace_once(
        text,
        '<a href="#code">Tiny Block</a><a href="#stack">堆成模型</a>',
        '<a href="#code">Tiny Block</a><a href="#position-semantics">Position / RoPE</a><a href="#stack">堆成模型</a>',
        "02.3 toc",
    )
    path.write_text(text, encoding="utf-8")


def patch_model_parallel_wording() -> None:
    path = ROOT / "learn/05-megatron/why-model-parallel.html"
    text = path.read_text(encoding="utf-8")
    old = '<li>区分 Tensor Parallel (TP，张量并行)、Pipeline Parallel (PP，流水线并行)、Data Parallel (DP，数据并行) 分别在切模型的哪个维度。</li>'
    new = '<li>区分 Data Parallel (DP，数据并行) 如何切数据/副本，以及 Tensor Parallel (TP，张量并行)、Pipeline Parallel (PP，流水线并行) 如何分别切 layer 内 tensor 与模型深度。</li>'
    if new not in text:
        text = replace_once(text, old, new, "05.1 DP wording")
    path.write_text(text, encoding="utf-8")


def main() -> None:
    patch_tensor_foundation()
    patch_transformer_position_bridge()
    patch_model_parallel_wording()
    print("Content foundation review applied to 01.1, 02.3, 05.1")


if __name__ == "__main__":
    main()
