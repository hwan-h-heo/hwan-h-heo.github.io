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

        lazyImage.addEventListener('load', () => {
          isotopeInstances.forEach(instance => {
            instance.layout();
          });
        }, { once: true });

        lazyImage.src = lazyImage.dataset.src;
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
 * Portfolio Hover Media Play
 * ------------------------------------------------------------------- */
document.addEventListener('DOMContentLoaded', () => {
  "use strict";

  const portfolioBoxes = document.querySelectorAll('.portfolio-box');

  portfolioBoxes.forEach(box => {
    const video = box.querySelector('video');
    const image = box.querySelector('img[data-gif]');
    
    // HTML ex:
    // <img src="placeholder.jpg" 
    //      data-src="path/to/static.jpg" 
    //      data-static="path/to/static.jpg" 
    //      data-gif="path/to/animated.gif" ... >

    let staticSrc = image ? image.getAttribute('data-static') || image.src : null;

    // 마우스를 올렸을 때
    box.addEventListener('mouseenter', () => {
      // 비디오가 있으면 재생 시도
      if (video) {
        // play()는 프로미스를 반환하므로, 사용자가 페이지와 상호작용하기 전에 자동 재생을 시도할 때 발생하는 오류를 catch로 처리해주는 것이 좋습니다.
        video.play().catch(error => {
          console.log("Video play was prevented.", error);
        });
      }
      
      // GIF 이미지가 있으면 동적 GIF로 교체
      if (image && image.dataset.gif) {
        // 만약 staticSrc가 아직 설정되지 않았다면 (lazy loading 이전 등) 현재 src를 저장
        if (!staticSrc || staticSrc.includes('placeholder')) {
            staticSrc = image.src;
        }
        image.src = image.dataset.gif;
      }
    });

    // 마우스가 벗어났을 때
    box.addEventListener('mouseleave', () => {
      // 비디오가 있으면 일시정지하고 처음으로 되감기
      if (video) {
        video.pause();
        video.load();
      }
      
      // GIF 이미지가 있으면 저장해둔 정적 이미지로 복원
      if (image && staticSrc) {
        image.src = staticSrc;
      }
    });
  });
});