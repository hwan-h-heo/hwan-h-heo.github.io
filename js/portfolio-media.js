function initPortfolioBoxes(root = document) {
  const portfolioBoxes = root.querySelectorAll('.portfolio-project-cover-link:not([data-portfolio-bound])');
  const isTouchDevice = !matchMedia('(hover:hover) and (pointer:fine)').matches;
  const reduced = matchMedia('(prefers-reduced-motion:reduce)');

  portfolioBoxes.forEach(box => {
    box.dataset.portfolioBound = 'true';

    const video = box.querySelector('video');
    const image = box.querySelector('img[data-gif]');
    const spinner = box.querySelector('.loading-spinner');
    let staticSrc = image ? image.getAttribute('data-static') || image.src : null;

    if (video && spinner) {
      video.addEventListener('waiting', () => {
        spinner.style.display = 'block';
      });
      video.addEventListener('canplay', () => {
        spinner.style.display = 'none';
      });
    }

    const activateMedia = () => {
      if (reduced.matches || navigator.connection?.saveData || document.hidden) return;
      box.classList.add('is-active');

      if (video) {
        video.play().catch(error => {
          console.debug('Video preview was not started.', error);
        });
      }

      if (image && image.dataset.gif) {
        if (!staticSrc || staticSrc.includes('placeholder')) {
            staticSrc = image.src;
        }

        if (spinner) {
          spinner.style.display = 'block';
        }

        const gifLoader = new Image();
        gifLoader.onload = () => {
          if (!box.classList.contains('is-active') || document.hidden || reduced.matches || navigator.connection?.saveData) return;
          image.src = image.dataset.gif;
          if (spinner) {
            spinner.style.display = 'none';
          }
        };
        gifLoader.onerror = () => {
          if (spinner) {
            spinner.style.display = 'none';
          }
        };
        gifLoader.src = image.dataset.gif;
      }
    };

    const deactivateMedia = () => {
      box.classList.remove('is-active');

      if (video) {
        video.pause();
        video.currentTime = 0;
      }

      if (image && staticSrc) {
        image.src = staticSrc;
      }

      if (spinner) {
        spinner.style.display = 'none';
      }
    };

    document.addEventListener('portfolio:chapter-change', deactivateMedia);
    document.addEventListener('visibilitychange', () => { if (document.hidden) deactivateMedia(); });
    reduced.addEventListener('change', () => { if (reduced.matches) deactivateMedia(); });
    if (isTouchDevice) {
      return;
    }

    box.addEventListener('mouseenter', activateMedia);
    box.addEventListener('mouseleave', deactivateMedia);
    box.addEventListener('focus', activateMedia);
    box.addEventListener('blur', deactivateMedia);
  });
}

window.initPortfolioBoxes = initPortfolioBoxes;
document.addEventListener('DOMContentLoaded', () => initPortfolioBoxes());
