# Teaching Style

Lesson prose follows these rules. The reference implementation is `learn/01-foundations/tensor.html`.

## 1. Plain language, no inline glossary

- Write in natural Chinese. Keep English only for names that appear in code, docs or logs (`shape`, `all-reduce`, `optimizer.step()`).
- Do not attach a parenthetical definition to every term (`tensor（张量：……）`). Explain a term inside the sentence the first time it matters, in plain words. The per-lesson term table at the end keeps the full English names.
- One idea per sentence. Lead with the conclusion, then the details.
- Avoid defensive hedging (“通常”“常见”“在某些实现中”) unless the exception actually matters to the reader right now. Put real exceptions after the main explanation.

## 2. Do not cut information to make it simpler

Simplifying the wording must not drop technical content. When rewriting, list what the old version covered and check every item still has a place (main text, a later section, or a clearly marked “进阶” section).

## 3. Concrete, not metaphorical

- Do not use everyday analogies that are unrelated to the subject (moving companies, Excel, ID cards, books).
- Instead, show the real thing: actual numbers, actual memory addresses, actual shapes, a runnable snippet and its real output, a worked calculation.
- Compute real-world magnitudes (“≈ 17 GB”, “≈ 1.1 TFLOP”) instead of saying “可能非常大”.
- Every number in a lesson must be verified (run it, or compute it in a script) before publishing.

## 4. Options must come with “when to use which”

Whenever a lesson lists alternatives (dtypes, optimizers, parallelism modes, collectives, transports…), it must say:

1. what the default is;
2. when you switch to another one;
3. what concretely goes wrong if you pick the wrong one — ideally a reproducible symptom (`inf`, the update disappears, the loss diverges).

Listing properties alone (“范围大、精度低”) is not enough.

## 5. Visuals where text is weakest

- Add a visual only where it answers a question the text struggles with (layout in memory, which cells take part in a computation, how a tensor is split, how state changes over steps).
- Make it interactive when the reader learns by trying (indexing, stepping a training loop, moving a slider).
- Follow `design/DIAGRAM_QUALITY_STANDARD.md`: site palette, no overlaps, check desktop and 390 px mobile before publishing.
- Shared visual components for the foundations module live in `learn/01-foundations/foundations-visuals.css`.
