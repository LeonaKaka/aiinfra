const progress = document.getElementById('readingProgress');
const navToggle = document.getElementById('navToggle');
const siteNav = document.getElementById('siteNav');

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

// Homepage preview: progressively enhance the semantic fallback button.
const lessonPreviewButton = document.querySelector('.lesson-preview button');
if (lessonPreviewButton) {
  lessonPreviewButton.disabled = false;
  lessonPreviewButton.textContent = '进入第一课 →';
  lessonPreviewButton.addEventListener('click', () => {
    window.location.href = './learn/01-foundations/tensor.html';
  });
}

// Lessons are static HTML, so older pages may still contain a "locked" span
// when a later lesson gets published. Keep known routes in one small map and
// progressively upgrade those spans to links. If JavaScript is unavailable,
// the page still remains fully readable; only the shortcut stays locked.
const lessonRoutes = {
  '01.2': './linear.html',
  '01.3': './training-loop.html',
  '01.4': './autograd-optimizer.html',
};

document.querySelectorAll('.lesson-link.locked').forEach((item) => {
  const key = item.textContent.trim().match(/^(\d{2}\.\d)/)?.[1];
  const href = key && lessonRoutes[key];
  if (!href) return;

  const link = document.createElement('a');
  link.className = 'lesson-link';
  link.href = href;
  link.textContent = item.textContent;
  item.replaceWith(link);
});

// The same upgrade applies to a muted "next lesson" card once that lesson exists.
const nextLesson = document.querySelector('.next-lesson.muted-next');
if (nextLesson) {
  const key = nextLesson.querySelector('small')?.textContent.match(/(\d{2}\.\d)/)?.[1];
  const href = key && lessonRoutes[key];
  if (href) {
    const link = document.createElement('a');
    link.className = 'next-lesson';
    link.href = href;
    link.innerHTML = nextLesson.innerHTML;
    nextLesson.replaceWith(link);
  }
}
