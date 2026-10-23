/* Homepage design experiment only — no dependencies, no impact on lessons. */
(() => {
  const map = document.querySelector('.award-home .hero-map');
  if (!map) return;

  const terminal = map.querySelector('.map-terminal pre');
  const title = map.querySelector('.terminal-head code');
  const buttons = [...map.querySelectorAll('.route-switch button')];
  const status = map.querySelector('.map-status');
  const views = {
    training: {
      title: 'training.trace',
      status: 'TRAINING PATH ACTIVE',
      lines: '<span class="muted">$</span> trace Megatron Core\n<span class="ok">✓</span> tensor → parallel groups\n<span class="ok">✓</span> forward → backward → sync\n<span class="ok">✓</span> TP / DP / PP / SP / CP / EP\n\n<span class="muted">$</span> run experiments A1–A12\n<span class="ok">✓</span> from diagrams to runnable code'
    },
    inference: {
      title: 'inference.trace',
      status: 'INFERENCE PATH ACTIVE',
      lines: '<span class="muted">$</span> trace vLLM request\n<span class="ok">✓</span> scheduler → KV Cache\n<span class="ok">✓</span> prefill → transfer → decode\n<span class="ok">✓</span> connector → NIXL / RDMA\n\n<span class="muted">$</span> run experiments B1–B8\n<span class="ok">✓</span> follow KV ownership end-to-end'
    }
  };

  function focusRoute(name) {
    const view = views[name];
    if (!view) return;
    map.dataset.focus = name;
    buttons.forEach((button) => {
      button.setAttribute('aria-pressed', String(button.dataset.focus === name));
    });
    if (terminal) terminal.innerHTML = view.lines;
    if (title) title.textContent = view.title;
    if (status) status.textContent = view.status;
  }

  buttons.forEach((button) => {
    button.addEventListener('click', () => focusRoute(button.dataset.focus));
  });
  focusRoute('training');

  // Motion is progressively enhanced; all content remains visible without JS.
  const prefersLessMotion = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
  if (prefersLessMotion || !('IntersectionObserver' in window)) return;

  const targets = [...document.querySelectorAll(
    '.award-home .section-heading, .award-home .destination-card, .award-home .roadmap-item, .award-home .method-grid article, .award-home .quick-links a'
  )];
  targets.forEach((target) => target.classList.add('reveal'));
  document.body.classList.add('motion-ready');

  const observer = new IntersectionObserver((entries) => {
    for (const entry of entries) {
      if (!entry.isIntersecting) continue;
      entry.target.classList.add('is-visible');
      observer.unobserve(entry.target);
    }
  }, { threshold: 0.08, rootMargin: '0px 0px 50px 0px' });
  targets.forEach((target) => observer.observe(target));
})();
