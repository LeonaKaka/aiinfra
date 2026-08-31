const progress = document.getElementById('readingProgress');
const navToggle = document.getElementById('navToggle');
const siteNav = document.getElementById('siteNav');
const siteRoot = new URL('./', document.currentScript?.src || window.location.href);

function siteUrl(path) {
  return new URL(path, siteRoot).href;
}

function updateReadingProgress() {
  const scrollable = document.documentElement.scrollHeight - window.innerHeight;
  const ratio = scrollable > 0 ? window.scrollY / scrollable : 0;
  if (progress) {
    progress.style.width = `${Math.max(0, Math.min(1, ratio)) * 100}%`;
  }
}

function addNavLink(nav, label, path) {
  const exists = [...nav.querySelectorAll('a')].some(
    (link) => link.textContent.trim().toLowerCase() === label.toLowerCase()
  );
  if (exists) return;
  const link = document.createElement('a');
  link.href = siteUrl(path);
  link.textContent = label;
  const cta = nav.querySelector('.nav-cta');
  nav.insertBefore(link, cta || null);
}

window.addEventListener('scroll', updateReadingProgress, { passive: true });
window.addEventListener('resize', updateReadingProgress);
updateReadingProgress();

// The homepage exposes the complete reference path. Lesson headers stay compact
// and only receive the Labs shortcut so mobile navigation does not overflow.
if (siteNav) {
  addNavLink(siteNav, 'Source Map', 'source-map/index.html');
  addNavLink(siteNav, 'Glossary', 'glossary/index.html');
  addNavLink(siteNav, 'Labs', 'labs/index.html');
} else {
  document.querySelectorAll('.site-nav').forEach((nav) => {
    addNavLink(nav, 'Labs', 'labs/index.html');
  });
}

navToggle?.addEventListener('click', () => {
  const open = siteNav.classList.toggle('open');
  navToggle.setAttribute('aria-expanded', String(open));
});

siteNav?.querySelectorAll('a').forEach((link) => {
  link.addEventListener('click', () => {
    siteNav.classList.remove('open');
    navToggle?.setAttribute('aria-expanded', 'false');
  });
});

const lessonPreviewButton = document.querySelector('.lesson-preview button');
if (lessonPreviewButton) {
  lessonPreviewButton.disabled = false;
  lessonPreviewButton.textContent = '进入第一课 →';
  lessonPreviewButton.addEventListener('click', () => {
    window.location.href = siteUrl('learn/01-foundations/tensor.html');
  });
}

const lessonRoutes = {
  '01.1': 'learn/01-foundations/tensor.html',
  '01.2': 'learn/01-foundations/linear.html',
  '01.3': 'learn/01-foundations/training-loop.html',
  '01.4': 'learn/01-foundations/autograd-optimizer.html',
  '02.1': 'learn/02-transformer/attention.html',
  '02.2': 'learn/02-transformer/mha-gqa.html',
  '02.3': 'learn/02-transformer/transformer-block.html',
  '03.1': 'learn/03-gpu-systems/gpu-mental-model.html',
  '03.2': 'learn/03-gpu-systems/gpu-memory.html',
  '03.3': 'learn/03-gpu-systems/gpu-bottlenecks.html',
  '04.1': 'learn/04-distributed/process-rank.html',
  '04.2': 'learn/04-distributed/collectives.html',
  '04.3': 'learn/04-distributed/nccl-topology.html',
  '05.1': 'learn/05-megatron/why-model-parallel.html',
  '05.2': 'learn/05-megatron/tensor-parallel.html',
  '05.3': 'learn/05-megatron/sequence-parallel.html',
  '05.4': 'learn/05-megatron/pipeline-parallel.html',
  '05.5': 'learn/05-megatron/distributed-optimizer.html',
  '05.6': 'learn/05-megatron/context-parallel.html',
  '05.7': 'learn/05-megatron/expert-parallel.html',
  '05.8': 'learn/05-megatron/communication-overlap.html',
  '06.1': 'learn/06-llm-inference/autoregressive-generation.html',
  '06.2': 'learn/06-llm-inference/prefill-decode.html',
  '06.3': 'learn/06-llm-inference/kv-cache.html',
  '07.1': 'learn/07-vllm/architecture.html',
  '07.2': 'learn/07-vllm/scheduler-continuous-batching.html',
  '07.3': 'learn/07-vllm/kv-cache-manager.html',
  '07.4': 'learn/07-vllm/model-runner-paged-attention.html',
  '07.5': 'learn/07-vllm/prefix-cache-preemption.html',
  '08.1': 'learn/08-kv-connector/why-move-kv.html',
  '08.2': 'learn/08-kv-connector/connector-architecture.html',
  '08.3': 'learn/08-kv-connector/transfer-lifecycle.html',
  '08.4': 'learn/08-kv-connector/nixl-rdma.html',
  '08.5': 'learn/08-kv-connector/production-pd.html',
};

const capstoneRoutes = [
  { pattern: /mini\s+megatron/i, route: 'labs/mini-megatron.html' },
  { pattern: /mini\s+kv\s+(connector|handoff)/i, route: 'labs/mini-kv-handoff.html' },
];

const firstLessonRouteByModule = Object.fromEntries(
  Object.entries(lessonRoutes)
    .filter(([key]) => key.endsWith('.1'))
    .map(([key, route]) => [key.slice(0, 2), route])
);

function resolveLockedLessonRoute(item) {
  const label = item.textContent.trim();
  const key = label.match(/^(\d{2}\.\d)/)?.[1];
  if (key && lessonRoutes[key]) return lessonRoutes[key];

  const capstoneRoute = capstoneRoutes.find(({ pattern }) => pattern.test(label))?.route;
  if (capstoneRoute) return capstoneRoute;

  // Older lesson sidebars sometimes used one generic locked label for an entire
  // module (for example “GPU、显存与 kernel” or “Scheduler · Paged KV”). The
  // curriculum is now complete, so route such legacy placeholders to that
  // module's first real lesson instead of leaving a dead disabled-looking item.
  const moduleKey = item
    .closest('.module-block')
    ?.querySelector('.module-name b')
    ?.textContent.trim()
    .match(/\d{2}/)?.[0];
  return moduleKey && firstLessonRouteByModule[moduleKey];
}

document.querySelectorAll('.lesson-link.locked').forEach((item) => {
  const route = resolveLockedLessonRoute(item);
  if (!route) return;

  const link = document.createElement('a');
  link.className = 'lesson-link';
  link.href = siteUrl(route);
  link.textContent = item.textContent;
  item.replaceWith(link);
});

const nextLesson = document.querySelector('.next-lesson.muted-next');
if (nextLesson) {
  const key = nextLesson.querySelector('small')?.textContent.match(/(\d{2}\.\d)/)?.[1];
  const route = key && lessonRoutes[key];
  if (route) {
    const link = document.createElement('a');
    link.className = 'next-lesson';
    link.href = siteUrl(route);
    link.innerHTML = nextLesson.innerHTML;
    nextLesson.replaceWith(link);
  }
}
