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

    function setupSearchForm(formSelector, inputSelector) {
        const searchForm = document.querySelector(formSelector);
        const searchInput = document.querySelector(inputSelector);

        if (!searchForm || !searchInput || searchForm.dataset.initialized === 'true') {
            return;
        }

        searchForm.addEventListener('submit', (event) => {
            event.preventDefault();
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
