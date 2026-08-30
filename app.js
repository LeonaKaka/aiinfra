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

// The homepage lesson preview starts as semantic fallback markup.
// Once JS loads, turn it into the real lesson entrance.
const lessonPreviewButton = document.querySelector('.lesson-preview button');
if (lessonPreviewButton) {
  lessonPreviewButton.disabled = false;
  lessonPreviewButton.textContent = '进入第一课 →';
  lessonPreviewButton.addEventListener('click', () => {
    window.location.href = './learn/01-foundations/tensor.html';
  });
}
