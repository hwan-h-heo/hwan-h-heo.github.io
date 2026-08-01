(function initPortfolioPreloader() {
  const preloader = document.getElementById('preloader');
  if (!preloader) return;

  const delay = (milliseconds) => new Promise(resolve => window.setTimeout(resolve, milliseconds));
  const coreReady = new Promise(resolve => {
    if (document.documentElement.dataset.portfolioCoreReady) {
      resolve();
      return;
    }
    document.addEventListener('portfolio:core-ready', resolve, { once: true });
  });
  const fontsReady = document.fonts?.ready?.catch(() => undefined) || Promise.resolve();
  let revealed = false;

  const reveal = () => {
    if (revealed) return;
    revealed = true;
    preloader.classList.add('is-hidden');
    preloader.setAttribute('aria-hidden', 'true');
    window.setTimeout(() => {
      preloader.dataset.revealComplete = 'true';
      document.dispatchEvent(new CustomEvent('portfolio:preloader-hidden'));
    }, 430);
  };

  Promise.all([
    delay(260),
    Promise.race([
      Promise.all([coreReady, fontsReady]),
      delay(1800)
    ])
  ]).then(reveal);

  window.addEventListener('pageshow', event => {
    if (event.persisted) reveal();
  }, { once: true });
  window.setTimeout(reveal, 2200);
})();

(function() {
  "use strict";
  /**
   * Lazy loading for project media
   */
  const lazyImages = document.querySelectorAll('.lazy-image');
  const lazyImageObserver = new IntersectionObserver((entries, observer) => {
    entries.forEach(entry => {
      if (entry.isIntersecting) {
        const lazyImage = entry.target;
        const lazySource = lazyImage.dataset.src;

        if (!lazySource) {
          lazyImage.classList.remove('lazy-image');
          observer.unobserve(lazyImage);
          return;
        }

        lazyImage.src = lazySource;
        lazyImage.classList.remove('lazy-image');
        observer.unobserve(lazyImage);
      }
    });
  });

  lazyImages.forEach(lazyImage => {
    lazyImageObserver.observe(lazyImage);
  });

  /**
   * Correct scrolling position upon page load for URLs containing hash links.
   */
  window.addEventListener('load', function(e) {
    if (window.location.hash) {
      if (document.querySelector(window.location.hash)) {
        setTimeout(() => {
          let section = document.querySelector(window.location.hash);
          let scrollMarginTop = getComputedStyle(section).scrollMarginTop;
          window.scrollTo({
            top: section.offsetTop - parseInt(scrollMarginTop),
            behavior: 'smooth'
          });
        }, 100);
      }
    }
  });

  /**
   * Navmenu Scrollspy
   */
  let navmenulinks = document.querySelectorAll('.navmenu a');

  function navmenuScrollspy() {
    const scrollPosition = window.scrollY + 200;
    const atPageEnd = window.innerHeight + window.scrollY >= document.documentElement.scrollHeight - 2;
    let activeLink = null;

    navmenulinks.forEach(navmenulink => {
      if (!navmenulink.hash) return;
      let section = document.querySelector(navmenulink.hash);
      if (!section) return;

      if (scrollPosition >= section.offsetTop && scrollPosition <= (section.offsetTop + section.offsetHeight)) {
        activeLink = navmenulink;
      }
    });

    if (atPageEnd) {
      activeLink = Array.from(navmenulinks).reverse().find((navmenulink) => {
        return navmenulink.hash && document.querySelector(navmenulink.hash);
      }) || activeLink;
    }

    document.querySelectorAll('.navmenu a.active').forEach(link => link.classList.remove('active'));
    activeLink?.classList.add('active');
  }
  window.addEventListener('load', navmenuScrollspy);
  document.addEventListener('scroll', navmenuScrollspy);

})();

document.addEventListener('DOMContentLoaded', () => {
  "use strict";

  /**
   * Scroll top button
   */
  const scrollTop = document.querySelector('#scroll-top');
  if (scrollTop) {
    const togglescrollTop = function() {
      window.scrollY > 100 ? scrollTop.classList.add('active') : scrollTop.classList.remove('active');
    }
    
    window.addEventListener('load', togglescrollTop);
    document.addEventListener('scroll', togglescrollTop);
    
    scrollTop.addEventListener('click', () => {
      window.scrollTo({
        top: 0,
        behavior: 'smooth'
      });
    });
  }

});

/* ===================================================================
 * Portfolio Hover Media Play (with Loading Spinner)
 * ------------------------------------------------------------------- */
function initPortfolioBoxes(root = document) {
  const portfolioBoxes = root.querySelectorAll('.portfolio-project-cover-link:not([data-portfolio-bound])');
  const isTouchDevice = ('ontouchstart' in window) || (navigator.maxTouchPoints > 0);

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
