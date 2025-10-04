// scripts.js

/**
 * 상단 내비게이션 바의 스크롤 효과를 초기화합니다.
 */
function initializeNavbar() {
    const mainNav = document.getElementById('mainNav');
    if (!mainNav) return;

    let scrollPos = 0;
    const headerHeight = mainNav.clientHeight;

    window.addEventListener('scroll', function() {
        const currentTop = document.body.getBoundingClientRect().top * -1;
        if (currentTop < scrollPos) { // Scrolling Up
            if (currentTop > 0 && mainNav.classList.contains('is-fixed')) {
                mainNav.classList.add('is-visible');
            } else {
                mainNav.classList.remove('is-visible', 'is-fixed');
            }
        } else { // Scrolling Down
            mainNav.classList.remove('is-visible');
            if (currentTop > headerHeight && !mainNav.classList.contains('is-fixed')) {
                mainNav.classList.add('is-fixed');
            }
        }
        scrollPos = currentTop;
    });
}

/**
 * 목차(TOC) 관련 기능을 초기화합니다.
 * 스크롤에 따라 표시/숨김 및 현재 섹션 활성화 기능을 포함합니다.
 */
function initializeToc() {
    const toc = document.querySelector('.toc');
    if (!toc) return; // 페이지에 목차가 없으면 실행하지 않음

    let tocItems = [];

    // --- 스크롤에 따라 목차 표시/숨김 ---
    document.addEventListener("scroll", function() {
        const headerHeight = document.querySelector('.masthead')?.offsetHeight || 300;
        if (window.scrollY > headerHeight) {
            toc.style.display = 'block';
        } else {
            toc.style.display = 'none';
        }
    });
    // 초기 상태 설정
    const headerHeight = document.querySelector('.masthead')?.offsetHeight || 300;
    toc.style.display = window.scrollY > headerHeight ? 'block' : 'none';

    // --- 스크롤에 따라 현재 섹션 활성화 ---
    const TOP_MARGIN = 0.1;
    const BOTTOM_MARGIN = 0.2;

    function initTocItems() {
        tocItems = [].slice.call(toc.querySelectorAll('li'));
        tocItems = tocItems.map(function(item) {
            const anchor = item.querySelector('a');
            if (!anchor) return null;
            const href = anchor.getAttribute('href');
            if (!href || href === '#') return null;
            const target = document.getElementById(href.slice(1));
            return { listItem: item, anchor: anchor, target: target };
        }).filter(item => item && item.target); // 유효한 항목만 필터링
    }

    function syncToc() {
        const windowHeight = window.innerHeight;
        let currentSection = null;

        tocItems.forEach(function(item) {
            const targetBounds = item.target.getBoundingClientRect();
            if (targetBounds.top <= windowHeight * (1 - BOTTOM_MARGIN)) {
                currentSection = item;
            }
        });

        tocItems.forEach(function(item) {
            if (item === currentSection) {
                item.listItem.classList.add('active');
            } else {
                item.listItem.classList.remove('active');
            }
        });
    }

    initTocItems();
    syncToc(); // 초기 동기화
    window.addEventListener('scroll', syncToc, false);
}

/**
 * 언어 변경 기능을 초기화하고 localStorage에 선택을 저장합니다.
 */
function initializeLanguageToggle() {
    // setLanguage 함수를 전역에서 접근 가능하게 만듦
    window.setLanguage = function(lang) {
        localStorage.setItem('language', lang);
        document.querySelectorAll('.lang').forEach(el => {
            if (!el.classList.contains('fixed')) {
                // block 대신 ''을 사용하여 원래의 display 속성(inline 등)을 따르도록 함
                el.style.display = el.classList.contains(lang) ? '' : 'none';
            }
        });
    };

    const savedLang = localStorage.getItem('language') || 'eng';
    window.setLanguage(savedLang);
}

/**
 * 한글이 포함된 요소에 폰트 스타일을 적용합니다.
 */
function initializeKoreanFonts() {
    document.querySelectorAll('p, h1, h2, h3, h4, ul, ol, li, span').forEach(function(element) {
        if (/[가-힣]/.test(element.innerText)) {
            element.style.fontSize = '1.2rem';
            element.style.fontFamily = 'Noto Sans KR';
        }
    });
}

/**
 * '맨 위로 가기' 버튼 기능을 초기화합니다.
 */
function initializeScrollTopButton() {
    const scrollTop = document.querySelector('.scroll-top');
    if (!scrollTop) return;

    function toggleScrollTop() {
        window.scrollY > 100 ? scrollTop.classList.add('active') : scrollTop.classList.remove('active');
    }

    scrollTop.addEventListener('click', (e) => {
        e.preventDefault();
        window.scrollTo({ top: 0, behavior: 'smooth' });
    });

    window.addEventListener('load', toggleScrollTop);
    document.addEventListener('scroll', toggleScrollTop);
}

// --- 페이지 로드 시 항상 실행되는 기능들 ---
document.addEventListener('DOMContentLoaded', () => {
    initializeNavbar();
    initializeScrollTopButton(); // 스크롤 버튼은 콘텐츠와 무관하므로 바로 초기화
});