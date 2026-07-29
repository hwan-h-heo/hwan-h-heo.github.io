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
   * Animation on scroll function and init
   */
  function aosInit() {
    if (!window.AOS || !document.querySelector('[data-aos]')) return;

    window.AOS.init({
      duration: 600,
      easing: 'ease-in-out',
      once: true,
      mirror: false
    });
  }
  window.addEventListener('load', aosInit);

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
    navmenulinks.forEach(navmenulink => {
      if (!navmenulink.hash) return;
      let section = document.querySelector(navmenulink.hash);
      if (!section) return;
      let position = window.scrollY + 200;
      if (position >= section.offsetTop && position <= (section.offsetTop + section.offsetHeight)) {
        document.querySelectorAll('.navmenu a.active').forEach(link => link.classList.remove('active'));
        navmenulink.classList.add('active');
      } else {
        navmenulink.classList.remove('active');
      }
    })
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
  const portfolioBoxes = root.querySelectorAll('.portfolio-box:not([data-portfolio-bound]), .portfolio-project-link:not([data-portfolio-bound])');
  const isTouchDevice = ('ontouchstart' in window) || (navigator.maxTouchPoints > 0);

  portfolioBoxes.forEach(box => {
    box.dataset.portfolioBound = 'true';

    const video = box.querySelector('video');
    const image = box.querySelector('img[data-gif]');
    const spinner = box.querySelector('.loading-spinner');
    const titleElement = box.querySelector('.polar_content h6, .portfolio-project-title');

    if (!titleElement) return;

    const originalTitleHtml = titleElement.innerHTML;
    const hoverTypingText = titleElement.getAttribute('data-hover-text');

    let staticSrc = image ? image.getAttribute('data-static') || image.src : null;
    let typeInterval;
    let isActive = false; // 터치 디바이스에서 활성 상태 추적

    if (hoverTypingText && !box.querySelector('.portfolio-card-summary, .portfolio-project-summary')) {
      const summaryElement = document.createElement('p');
      summaryElement.className = 'portfolio-card-summary';
      summaryElement.textContent = hoverTypingText;
      titleElement.insertAdjacentElement('afterend', summaryElement);
    }

    const activateMedia = () => {
      box.classList.add('is-active');

      if (video) {
        if (spinner) {
          video.addEventListener('waiting', () => {
            spinner.style.display = 'block';
          });
          video.addEventListener('canplay', () => {
            spinner.style.display = 'none';
          });
        }

        video.play().catch(error => {
          console.log("Video play was prevented.", error);
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
          console.log("Failed to load GIF.");
        };
        gifLoader.src = image.dataset.gif;
      }
    };

    const deactivateMedia = () => {
      clearInterval(typeInterval);
      box.classList.remove('is-active');

      titleElement.innerHTML = originalTitleHtml;
      titleElement.classList.remove('typing-done');

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

    // 터치 디바이스용 이벤트 처리
    if (isTouchDevice) {
      box.addEventListener('click', (e) => {
        // 링크로 이동하는 것을 막지 않되, 미디어 활성화 토글
        if (!isActive) {
          e.preventDefault();
          isActive = true;
          activateMedia();
        } else {
          // 두 번째 탭에서는 링크로 이동
          isActive = false;
        }
      });

      // 다른 곳을 터치하면 비활성화
      document.addEventListener('touchstart', (e) => {
        if (isActive && !box.contains(e.target)) {
          isActive = false;
          deactivateMedia();
        }
      });
    } else {
      // 데스크톱용 마우스 이벤트
      box.addEventListener('mouseenter', activateMedia);
      box.addEventListener('mouseleave', deactivateMedia);
    }
  });
}

window.initPortfolioBoxes = initPortfolioBoxes;
document.addEventListener('DOMContentLoaded', () => initPortfolioBoxes());
