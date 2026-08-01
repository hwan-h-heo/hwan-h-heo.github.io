(function() {
    function setupScrollTop() {
        const scrollTop = document.querySelector('.scroll-top');

        if (!scrollTop || scrollTop.dataset.initialized === 'true') {
            return;
        }

        function toggleScrollTop() {
            if (window.scrollY > 100) {
                scrollTop.classList.add('active');
            } else {
                scrollTop.classList.remove('active');
            }
        }

        scrollTop.addEventListener('click', (event) => {
            event.preventDefault();
            window.scrollTo({ top: 0, behavior: 'smooth' });
        });

        window.addEventListener('load', toggleScrollTop);
        document.addEventListener('scroll', toggleScrollTop);
        toggleScrollTop();
        scrollTop.dataset.initialized = 'true';
    }

    function setupHomeTopbar() {
        const topbar = document.querySelector('.blog-home-topbar');

        if (!topbar || topbar.dataset.initialized === 'true') {
            return;
        }

        const documentRoot = document.documentElement;
        const hero = document.querySelector('.blog-home-hero');

        function updateTopbar() {
            const threshold = hero
                ? Math.max(96, hero.offsetHeight - topbar.offsetHeight)
                : 96;
            const scrolled = window.scrollY > threshold;

            topbar.classList.toggle('is-scrolled', scrolled);
            documentRoot.classList.toggle('blog-topbar-scrolled', scrolled);
        }

        window.addEventListener('load', updateTopbar);
        window.addEventListener('resize', updateTopbar);
        document.addEventListener('scroll', updateTopbar, { passive: true });
        updateTopbar();
        topbar.dataset.initialized = 'true';
    }

    function setupSearchForm(formSelector, inputSelector) {
        const searchForm = document.querySelector(formSelector);
        const searchInput = document.querySelector(inputSelector);

        if (!searchForm || !searchInput || searchForm.dataset.initialized === 'true') {
            return;
        }

        const searchButton = searchForm.querySelector('button[type="submit"]');
        const collapsible = searchForm.hasAttribute('data-collapsible-search');
        const compactQuery = window.matchMedia('(max-width: 767.98px)');

        function setExpanded(expanded) {
            searchForm.classList.toggle('is-expanded', expanded);
            searchButton?.setAttribute('aria-expanded', String(expanded));
        }

        function syncExpandableState() {
            if (!collapsible) {
                return;
            }
            if (!compactQuery.matches) {
                searchForm.classList.remove('is-expanded');
                searchButton?.removeAttribute('aria-expanded');
                return;
            }
            setExpanded(Boolean(searchInput.value.trim()) || document.activeElement === searchInput);
        }

        if (collapsible) {
            searchButton?.addEventListener('click', (event) => {
                if (!compactQuery.matches || searchForm.classList.contains('is-expanded')) {
                    return;
                }
                event.preventDefault();
                setExpanded(true);
                searchInput.focus();
            });

            searchInput.addEventListener('focus', () => {
                if (compactQuery.matches) {
                    setExpanded(true);
                }
            });

            searchInput.addEventListener('keydown', (event) => {
                if (event.key !== 'Escape' || !compactQuery.matches) {
                    return;
                }
                event.preventDefault();
                searchInput.value = '';
                searchInput.blur();
                setExpanded(false);
                searchButton?.focus();
            });

            document.addEventListener('pointerdown', (event) => {
                if (
                    compactQuery.matches
                    && !searchForm.contains(event.target)
                    && !searchInput.value.trim()
                ) {
                    setExpanded(false);
                }
            });

            compactQuery.addEventListener('change', syncExpandableState);
            syncExpandableState();
        }

        searchForm.addEventListener('submit', (event) => {
            event.preventDefault();

            if (collapsible && compactQuery.matches && !searchForm.classList.contains('is-expanded')) {
                setExpanded(true);
                searchInput.focus();
                return;
            }

            const searchTerm = searchInput.value.trim();

            if (searchTerm) {
                window.location.href = `/blogs/search/?q=${encodeURIComponent(searchTerm)}`;
            }
        });

        searchForm.dataset.initialized = 'true';
    }

    window.initBlogShell = function initBlogShell(options = {}) {
        const run = function() {
            setupScrollTop();
            setupHomeTopbar();

            if (options.formSelector && options.inputSelector) {
                setupSearchForm(options.formSelector, options.inputSelector);
            }
        };

        if (document.readyState === 'loading') {
            document.addEventListener('DOMContentLoaded', run, { once: true });
        } else {
            run();
        }
    };
})();
