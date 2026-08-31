# Diagram Quality Standard

This project uses architecture-first teaching diagrams. The accepted baseline is the reviewed KV Cache visual integrated in `learn/06-llm-inference/kv-cache.html`.

## 1. Match the site before drawing the diagram

- Reuse the site palette: `--bg`, `--paper`, `--ink`, `--line`, `--line-dark`, `--train`, `--train-soft`, `--infer`, `--infer-soft`.
- Reuse the site typography: system sans for explanation, monospace for IDs / shapes / states.
- Reuse the site visual grammar: thin neutral borders, restrained fills, 8–16 px radii, off-white page, white diagram canvas.
- Do not introduce a separate infographic visual identity inside a lesson.

## 2. Architecture first, cards second

A core diagram should first expose one coherent system or mechanism on a single canvas:

`input/state → transformation/control → persistent state → execution/output`

Use boxes only when they represent real conceptual boundaries. Do not turn every sentence into a card.

## 3. Arrow routing rules

- Every arrow must have a clear source and destination.
- Terminate arrows at box edges / ports, never in the middle of labels.
- Route independent flows through separate lanes when possible.
- Do not run a line through another node, label, or matrix.
- Do not place text directly on a line unless there is enough dedicated whitespace.
- Prefer straight orthogonal routes over decorative curves.
- If arrows become spaghetti, replace them with an explicit mapping table rather than forcing more lines.

## 4. No overlap / no accidental tangency

Before accepting a diagram, check:

- text does not touch borders;
- arrowheads do not sit on text;
- labels do not overlap each other;
- boxes have visible gaps;
- callouts do not cover data cells;
- lines do not graze unrelated boxes;
- the final row/column is not clipped by the wrapper.

## 5. Information density

- One visual should answer one main question.
- Use 3–4 semantic layers maximum on one canvas.
- Keep the most important path visually dominant.
- Use color for persistent semantic categories, not decoration.
- Small labels are allowed only for secondary metadata; the main mechanism must remain readable at lesson-column width.

## 6. Responsive behavior

Desktop target: lesson column around 760 px inside the normal three-column site layout.

Mobile target: 390 px viewport.

For dense architecture diagrams, preserve the desktop composition and allow local horizontal scrolling inside the diagram wrapper. Do not make the entire page overflow horizontally.

## 7. Required review loop

For every new core diagram:

1. Draft in the real site visual language.
2. Render inside a lesson-like page, not as an isolated poster.
3. Inspect the complete desktop composition.
4. Inspect a close crop of the diagram.
5. Inspect mobile behavior / overflow.
6. Fix arrow routing, spacing, clipping, and text collisions.
7. Repeat until there are no visible overlaps or misplaced elements.
8. Only then integrate into the lesson.
9. Run Site Quality Check and GitHub Pages deployment.

Do not mark a diagram complete after code generation alone.

## 8. Current reference implementations

- `learn/05-megatron/tensor-parallel.html` — replicated input → two TP rank lanes → Column shards → local hidden shards → Row shards → partial outputs → SUM reduction; the no-gather boundary must not cross either rank's data path.
- `learn/05-megatron/sequence-parallel.html` — sequence-sharded state → central all-gather → separate per-rank TP compute windows → Row-parallel partials → central reduce-scatter → local sequence-sharded state; collective boundaries stay distinct from rank-local compute.
- `learn/05-megatron/pipeline-parallel.html` — stage × time schedule with forward/backward microbatch cells, dependency-created idle holes, and warmup/steady/cooldown bands; never imply all stages synchronously switch F/B together.
- `learn/05-megatron/distributed-optimizer.html` — four DP local-gradient lanes → central reduce-scatter → per-rank reduced-gradient/optimizer ownership → central parameter all-gather → synchronized model-parameter replicas; distinguish long-lived sharded optimizer state from forward-visible replicated compute parameters.
- `learn/05-megatron/context-parallel.html` — local context ownership → P2P KV ring → per-rank attention accumulation while remote KV rotates → context output stays local; distinguish the simple contiguous teaching slice from current zigzag CP partitioning.
- `learn/05-megatron/expert-parallel.html` — source-token ownership → destination-dependent token–expert assignments → A2A dispatch → expert-owner local compute → reverse A2A combine → restored source-token order; distinguish token ownership, expert-parameter ownership, and temporary compute ownership.
- `learn/05-megatron/communication-overlap.html` — bucket READY event → asynchronous collective launch → independent compute window → first required WAIT, with separate gradient-reduce and parameter-prefetch lanes; distinguish communication duration from exposed critical-path tail and verify overlap by wait points rather than visual concurrency alone.
- `learn/06-llm-inference/kv-cache.html` — autoregressive loop + one attention layer + persistent KV + cache growth.
- `learn/07-vllm/kv-cache-manager.html` — logical blocks + block table + BlockPool + scattered physical KV + append rule.
- `learn/07-vllm/model-runner-paged-attention.html` — separate KV write lane and paged-attention read lane; block-table selection stays visually distinct from data movement.
- `learn/08-kv-connector/why-move-kv.html` — token IDs vs computed KV state; explicitly separates the recompute-only token path from the valid state-handoff path.
- `learn/08-kv-connector/connector-architecture.html` — scheduler control plane + worker data plane + ConnectorMetadata + remote KV transport.
- `learn/08-kv-connector/transfer-lifecycle.html` — Scheduler / Worker / Memory swimlanes with request-level completion, layer-level readiness, and source-block lifetime on one timeline.
- `learn/08-kv-connector/nixl-rdma.html` — side-channel handshake is visually separated from registered-memory bulk KV transfer through NIXL / UCX / RDMA layers.
- `learn/08-kv-connector/production-pd.html` — Router + P/D queues + Prefill + pinned source KV + handoff + Decode paged KV + heartbeat + TTFT/ITL on one end-to-end production canvas.

The infrastructure references are regression baselines for multi-lane diagrams: control traffic, bulk data, memory lifetime, invalid/recompute paths, collective boundaries, schedule dependencies, ownership transitions, critical-path wait points, and user-visible SLOs must remain visually distinct.
