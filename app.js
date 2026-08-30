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

window.addEventListener('scroll', updateReadingProgress, { passive: true });
window.addEventListener('resize', updateReadingProgress);
updateReadingProgress();

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
};

document.querySelectorAll('.lesson-link.locked').forEach((item) => {
  const key = item.textContent.trim().match(/^(\d{2}\.\d)/)?.[1];
  const route = key && lessonRoutes[key];
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
