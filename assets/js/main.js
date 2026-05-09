document.addEventListener('DOMContentLoaded', () => {
  const header = document.getElementById('header');
  const hero = document.getElementById('home');
  const headerToggleBtn = document.querySelector('.header-toggle');
  let observer;

  function headerToggle() {
    header.classList.toggle('hidden');
    headerToggleBtn.classList.toggle('bi-list');
    headerToggleBtn.classList.toggle('bi-x');
  }

  function setupDesktopObserver() {
    if (!hero) return;
    observer = new IntersectionObserver((entries) => {
      entries.forEach(entry => {
        if (entry.isIntersecting) {
          header.classList.add('hidden');
        } else {
          header.classList.remove('hidden');
        }
      });
    }, { threshold: 0.2 });
    observer.observe(hero);
  }

  function handleHeaderState() {
    if (window.innerWidth >= 1200) {
      headerToggleBtn.classList.remove('bi-x');
      headerToggleBtn.classList.add('bi-list');
      if (!observer) {
        setupDesktopObserver();
      }
    } else {
      if (observer) {
        observer.disconnect();
        observer = null; 
      }
      header.classList.add('hidden');
      headerToggleBtn.classList.remove('bi-x');
      headerToggleBtn.classList.add('bi-list');
    }
  }

  headerToggleBtn.addEventListener('click', headerToggle);

  document.querySelectorAll('#navmenu a').forEach(navmenu => {
    navmenu.addEventListener('click', () => {
      if (window.innerWidth < 1200 && !header.classList.contains('hidden')) {
        headerToggle();
      }
    });
  });

  handleHeaderState();
  window.addEventListener('resize', handleHeaderState);
});

(function() {
  "use strict";
  /**
   * Animation on scroll function and init
   */
  function aosInit() {
    AOS.init({
      duration: 600,
      easing: 'ease-in-out',
      once: true,
      mirror: false
    });
  }
  window.addEventListener('load', aosInit);

  /**
   * Init typed.js
   */
  const selectTyped = document.querySelector('.typed');
  if (selectTyped) {
    let typed_strings = selectTyped.getAttribute('data-typed-items');
    typed_strings = typed_strings.split(',');
    new Typed('.typed', {
      strings: typed_strings,
      loop: true,
      typeSpeed: 100,
      backSpeed: 50,
      backDelay: 2000
    });
  }

  /**
   * Initiate Pure Counter
   */
  new PureCounter();

  /**
   * Animate the skills items on reveal
   */
  let skillsAnimation = document.querySelectorAll('.skills-animation');
  skillsAnimation.forEach((item) => {
    new Waypoint({
      element: item,
      offset: '80%',
      handler: function(direction) {
        let progress = item.querySelectorAll('.progress .progress-bar');
        progress.forEach(el => {
          el.style.width = el.getAttribute('aria-valuenow') + '%';
        });
      }
    });
  });

  /**
   * Initiate glightbox
   */
  const glightbox = GLightbox({
    selector: '.glightbox'
  });

  /**
   * Init isotope layout and filters
   */
  const isotopeInstances = [];

  document.querySelectorAll('.isotope-layout').forEach(function(isotopeItem) {
    if (isotopeItem.dataset.dynamicSource) {
      return;
    }

    let layout = isotopeItem.getAttribute('data-layout') ?? 'masonry';
    let filter = isotopeItem.getAttribute('data-default-filter') ?? '*';
    let sort = isotopeItem.getAttribute('data-sort') ?? 'original-order';

    const isotopeContainer = isotopeItem.querySelector('.isotope-container');
    let initIsotope;

    imagesLoaded(isotopeContainer, function() {
      initIsotope = new Isotope(isotopeContainer, {
        itemSelector: '.isotope-item',
        layoutMode: layout,
        filter: filter,
        sortBy: sort,
        percentPosition: true
      });

      isotopeInstances.push(initIsotope);
    });

    isotopeItem.querySelectorAll('.isotope-filters li').forEach(function(filters) {
      filters.addEventListener('click', function() {
        isotopeItem.querySelector('.isotope-filters .filter-active').classList.remove('filter-active');
        this.classList.add('filter-active');
        initIsotope.arrange({
          filter: this.getAttribute('data-filter')
        });
        if (typeof aosInit === 'function') {
          aosInit();
        }
      }, false);
    });
  });

  // 3. Lazy Loading 
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

        lazyImage.addEventListener('load', () => {
          isotopeInstances.forEach(instance => {
            instance.layout();
          });
        }, { once: true });

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
   * Init swiper sliders
   */
  function initSwiper() {
    document.querySelectorAll(".init-swiper").forEach(function(swiperElement) {
      let config = JSON.parse(
        swiperElement.querySelector(".swiper-config").innerHTML.trim()
      );

      if (swiperElement.classList.contains("swiper-tab")) {
        initSwiperWithCustomPagination(swiperElement, config);
      } else {
        new Swiper(swiperElement, config);
      }
    });
  }

  window.addEventListener("load", initSwiper);

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
  const headerToggleBtn = document.querySelector('.header-toggle');

  if (headerToggleBtn) {
    const toggleheaderToggleBtn = function() {
      window.scrollY > 100 ? headerToggleBtn.classList.add('active') : headerToggleBtn.classList.remove('active');
    }
    window.addEventListener('load', toggleheaderToggleBtn);
    document.addEventListener('scroll', toggleheaderToggleBtn);
  }

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
  const portfolioBoxes = root.querySelectorAll('.portfolio-box:not([data-portfolio-bound])');
  const isTouchDevice = ('ontouchstart' in window) || (navigator.maxTouchPoints > 0);

  portfolioBoxes.forEach(box => {
    box.dataset.portfolioBound = 'true';

    const video = box.querySelector('video');
    const image = box.querySelector('img[data-gif]');
    const spinner = box.querySelector('.loading-spinner');
    const titleElement = box.querySelector('.polar_content h6');

    if (!titleElement) return;

    const originalTitleHtml = titleElement.innerHTML;
    const hoverTypingText = titleElement.getAttribute('data-hover-text');

    let staticSrc = image ? image.getAttribute('data-static') || image.src : null;
    let typeInterval;
    let isActive = false; // 터치 디바이스에서 활성 상태 추적

    if (hoverTypingText && !box.querySelector('.portfolio-card-summary')) {
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
