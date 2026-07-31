(function() {
    function getPostTitle(post, lang) {
        return post[`title_${lang}`] || post.title_eng;
    }

    function escapeHtml(value) {
        return String(value || '')
            .replace(/&/g, '&amp;')
            .replace(/</g, '&lt;')
            .replace(/>/g, '&gt;')
            .replace(/"/g, '&quot;')
            .replace(/'/g, '&#39;');
    }

    function formatSeriesDate(value) {
        const date = new Date(value);
        if (Number.isNaN(date.getTime())) {
            return '';
        }

        const year = date.getFullYear();
        const month = String(date.getMonth() + 1).padStart(2, '0');
        const day = String(date.getDate()).padStart(2, '0');
        return `${year}.${month}.${day}`;
    }

    function initializeMathRendering() {
        if (typeof renderMathInElement === 'undefined') {
            return;
        }

        renderMathInElement(document.body, {
            delimiters: [
                { left: '$$', right: '$$', display: true },
                { left: '$', right: '$', display: false },
                { left: '\\(', right: '\\)', display: false },
                { left: '\\[', right: '\\]', display: true }
            ],
            throwOnError: false
        });
    }

    function setupShareButton() {
        const copyButton = document.getElementById('copyButton');
        const shareModal = document.getElementById('myshare_modal');

        if (!copyButton || !shareModal) {
            return;
        }

        copyButton.setAttribute('type', 'button');
        copyButton.setAttribute('aria-label', 'Copy post link');
        copyButton.setAttribute('title', 'Copy link');
        copyButton.innerHTML = '<i class="bi bi-link-45deg" aria-hidden="true"></i>';

        shareModal.setAttribute('role', 'status');
        shareModal.setAttribute('aria-live', 'polite');
        shareModal.setAttribute('aria-hidden', 'true');

        const modalContent = shareModal.querySelector('.share_modal-content');
        const modalLabels = window.blogPostPageConfig?.lang === 'kor'
            ? {
                copied: '링크 복사됨',
                ready: '글 주소를 바로 공유할 수 있습니다.'
            }
            : {
                copied: 'Link copied',
                ready: 'Post URL is ready to share.'
            };

        if (modalContent) {
            modalContent.innerHTML = `
                <button class="share_modal_close" type="button" aria-label="Dismiss">
                    <i class="bi bi-x-lg" aria-hidden="true"></i>
                </button>
                <span class="share_modal-icon" aria-hidden="true">
                    <i class="bi bi-check2"></i>
                </span>
                <span class="share_modal-message">
                    <strong>${modalLabels.copied}</strong>
                    <small>${modalLabels.ready}</small>
                </span>
                <div class="copy_indicator-container">
                    <div class="copy_indicator" id="share_modalIndicator"></div>
                </div>
            `;
        }

        const closeModal = shareModal.querySelector('.share_modal_close');
        const indicator = document.getElementById('share_modalIndicator');

        if (!indicator) {
            return;
        }

        let animationId;

        function updateShareButtonVisibility() {
            const headerHeight = document.querySelector('.masthead')?.offsetHeight || 300;
            copyButton.style.display = window.innerWidth > 1280 && window.scrollY > headerHeight ? 'inline-flex' : 'none';
        }

        function animateIndicator() {
            let startTime = null;
            const duration = 1500;

            function step(timestamp) {
                if (!startTime) {
                    startTime = timestamp;
                }

                const progress = Math.min((timestamp - startTime) / duration, 1);
                indicator.style.width = `${(1 - progress) * 100}%`;

                if (progress < 1) {
                    animationId = requestAnimationFrame(step);
                } else {
                    shareModal.classList.remove('is-visible');
                    shareModal.setAttribute('aria-hidden', 'true');
                    shareModal.style.display = 'none';
                }
            }

            cancelAnimationFrame(animationId);
            animationId = requestAnimationFrame(step);
        }

        function closeShareModal() {
            cancelAnimationFrame(animationId);
            shareModal.classList.remove('is-visible');
            shareModal.setAttribute('aria-hidden', 'true');
            shareModal.style.display = 'none';
            indicator.style.width = '0%';
        }

        document.addEventListener('scroll', updateShareButtonVisibility);
        window.addEventListener('resize', updateShareButtonVisibility);

        copyButton.addEventListener('click', function() {
            const url = new URL(window.location.href);
            url.hash = '';

            navigator.clipboard.writeText(url.href).then(function() {
                shareModal.style.display = 'block';
                shareModal.setAttribute('aria-hidden', 'false');
                window.requestAnimationFrame(() => {
                    shareModal.classList.add('is-visible');
                });
                indicator.style.width = '100%';
                animateIndicator();
            }).catch(function(error) {
                console.error('Link copy failed:', error);
            });
        });

        if (closeModal) {
            closeModal.addEventListener('click', closeShareModal);
        }

        window.addEventListener('click', function(event) {
            if (event.target === shareModal) {
                closeShareModal();
            }
        });

        updateShareButtonVisibility();
    }

    function initializeTOC() {
        const toc = document.querySelector('.toc');
        if (!toc) {
            return;
        }

        let tocItems = [];

        function updateTocVisibility() {
            const headerHeight = document.querySelector('.masthead')?.offsetHeight || 300;
            const isVisible = window.innerWidth > 1280 && window.scrollY > headerHeight;
            toc.classList.toggle('is-visible', isVisible);
            toc.setAttribute('aria-hidden', isVisible ? 'false' : 'true');
        }

        function getDirectAnchor(item) {
            return Array.from(item.children).find((child) => child.tagName === 'A') || item.querySelector('a');
        }

        function initTocItems() {
            tocItems = Array.from(toc.querySelectorAll('li')).map((item) => {
                const anchor = getDirectAnchor(item);
                const href = anchor?.getAttribute('href');
                if (!href || href === '#') {
                    return null;
                }

                const parentItem = item.parentElement?.closest('li') || null;
                const level = item.dataset.level || (parentItem ? '3' : '2');
                item.dataset.level = level;
                item.classList.add('toc-item', `toc-item-level-${level}`);
                anchor.classList.add('toc-link');

                return {
                    listItem: item,
                    parentItem,
                    target: document.getElementById(href.slice(1))
                };
            }).filter((item) => item && item.target);
        }

        function updateSublistHeights() {
            toc.querySelectorAll('.toc-sublist, ul ul').forEach((list) => {
                list.style.setProperty('--toc-sublist-height', `${list.scrollHeight}px`);
            });
        }

        function syncToc() {
            const windowHeight = window.innerHeight;
            const activationOffset = Math.min(220, Math.max(120, windowHeight * 0.32));
            let currentSection = null;

            tocItems.forEach((item) => {
                const targetBounds = item.target.getBoundingClientRect();
                if (targetBounds.top <= activationOffset) {
                    currentSection = item;
                }
            });

            tocItems.forEach((item) => {
                item.listItem.classList.remove('active', 'is-current', 'is-parent', 'is-expanded');
            });

            if (!currentSection) {
                return;
            }

            currentSection.listItem.classList.add('active', 'is-current', 'is-expanded');

            let parentItem = currentSection.parentItem;
            while (parentItem) {
                parentItem.classList.add('is-parent', 'is-expanded');
                parentItem = parentItem.parentElement?.closest('li') || null;
            }
        }

        document.addEventListener('scroll', updateTocVisibility);
        window.addEventListener('scroll', syncToc, false);
        window.addEventListener('resize', () => {
            updateTocVisibility();
            updateSublistHeights();
        });
        updateTocVisibility();
        initTocItems();
        updateSublistHeights();
        if (document.fonts?.ready) {
            document.fonts.ready.then(updateSublistHeights).catch(() => {});
        }
        syncToc();
    }

    function renderSeriesNavigation() {
        const siteData = window.siteData;
        const config = window.blogPostPageConfig;
        const seriesContainer = document.getElementById('series-container');
        const navContainer = document.getElementById('post-navigation');

        if (!siteData || !config || !seriesContainer || !navContainer) {
            return;
        }

        const currentPost = siteData.posts.find((post) => post.id === config.postId);
        if (!currentPost || !currentPost.series) {
            return;
        }

        const postsInSeries = siteData.posts
            .filter((post) => post.series === currentPost.series)
            .sort((a, b) => new Date(b.date) - new Date(a.date));

        if (postsInSeries.length <= 1) {
            return;
        }

        const seriesTitle = siteData.series[currentPost.series]?.[config.lang]
            || siteData.series[currentPost.series]?.eng
            || 'Series';

        const labels = {
            kicker: 'Series',
            current: 'Current',
            older: 'Older Post',
            next: 'Next Post',
            explore: 'Explore Series'
        };

        const currentIndex = postsInSeries.findIndex((post) => post.id === config.postId);
        const countLabel = `${postsInSeries.length} posts`;
        const listItems = postsInSeries.map((post, index) => {
            const title = getPostTitle(post, config.lang);
            const slug = config.lang === 'eng' ? post.slug : `${post.slug}-kor`;
            const date = formatSeriesDate(post.date);
            const number = String(index + 1).padStart(2, '0');
            if (post.id === config.postId) {
                return `
                    <li class="series-post is-current" aria-current="page">
                        <span class="series-post-index">${number}</span>
                        <span class="series-post-copy">
                            <strong>${escapeHtml(title)}</strong>
                            ${date ? `<span>${date}</span>` : ''}
                        </span>
                        <span class="series-current-pill">${labels.current}</span>
                    </li>
                `;
            }
            return `
                <li class="series-post">
                    <a href="/blogs/posts/${slug}/">
                        <span class="series-post-index">${number}</span>
                        <span class="series-post-copy">
                            <strong>${escapeHtml(title)}</strong>
                            ${date ? `<span>${date}</span>` : ''}
                        </span>
                    </a>
                </li>
            `;
        }).join('');

        seriesContainer.innerHTML = `
            <details class="series-card">
                <summary class="series-summary">
                    <span class="series-icon" aria-hidden="true"><i class="bi bi-collection"></i></span>
                    <span class="series-summary-copy">
                        <span class="series-kicker">${labels.kicker}</span>
                        <strong>${escapeHtml(seriesTitle)}</strong>
                    </span>
                    <span class="series-meta">${currentIndex + 1} / ${postsInSeries.length}</span>
                    <span class="series-toggle">
                        <span>${countLabel}</span>
                        <i class="bi bi-chevron-down" aria-hidden="true"></i>
                    </span>
                </summary>
                <div class="series-body">
                    <ol class="series-list">${listItems}</ol>
                </div>
            </details>
        `;

        const olderPost = postsInSeries[currentIndex + 1];
        const nextPost = postsInSeries[currentIndex - 1];

        function truncatePostNavTitle(title) {
            return title.length > 58 ? `${title.slice(0, 55)}...` : title;
        }

        function renderPostNavCard({ href, type, label, title }) {
            return `
                <a class="post-nav-card is-${type}" href="${href}">
                    <span class="post-nav-kicker">
                        <span>${escapeHtml(label)}</span>
                    </span>
                    <strong>${escapeHtml(truncatePostNavTitle(title))}</strong>
                </a>
            `;
        }

        let navHtml = '<nav class="post-nav-grid" aria-label="Post navigation">';

        if (olderPost) {
            const olderTitle = getPostTitle(olderPost, config.lang);
            const olderSlug = config.lang === 'eng' ? olderPost.slug : `${olderPost.slug}-kor`;
            navHtml += renderPostNavCard({
                href: `/blogs/posts/${olderSlug}/`,
                type: 'older',
                label: labels.older,
                title: olderTitle
            });
        } else if (nextPost) {
            navHtml += '<span class="post-nav-spacer" aria-hidden="true"></span>';
        }

        if (nextPost) {
            const nextTitle = getPostTitle(nextPost, config.lang);
            const nextSlug = config.lang === 'eng' ? nextPost.slug : `${nextPost.slug}-kor`;
            navHtml += renderPostNavCard({
                href: `/blogs/posts/${nextSlug}/`,
                type: 'next',
                label: labels.next,
                title: nextTitle
            });
        } else {
            const otherSeriesIds = Object.keys(siteData.series).filter((seriesId) => seriesId !== currentPost.series);
            const randomSeriesId = otherSeriesIds[Math.floor(Math.random() * otherSeriesIds.length)];
            const recommendedPost = siteData.posts
                .filter((post) => post.series === randomSeriesId)
                .sort((a, b) => new Date(b.date) - new Date(a.date))[0];

            if (recommendedPost) {
                const recommendedSlug = config.lang === 'eng' ? recommendedPost.slug : `${recommendedPost.slug}-kor`;
                const recommendedSeriesTitle = siteData.series[randomSeriesId]?.[config.lang]
                    || siteData.series[randomSeriesId]?.eng
                    || 'Series';
                navHtml += renderPostNavCard({
                    href: `/blogs/posts/${recommendedSlug}/`,
                    type: 'explore',
                    label: labels.explore,
                    title: recommendedSeriesTitle
                });
            } else {
                navHtml += '<span class="post-nav-spacer" aria-hidden="true"></span>';
            }
        }

        navHtml += '</nav>';
        navContainer.innerHTML = navHtml;
    }

    function initializeSpecialViewers() {
        const postId = window.blogPostPageConfig?.postId;
        const runtimeFeatures = window.blogPostPageConfig?.runtimeFeatures || {};

        if (runtimeFeatures.gaussianSplats || postId === '240917_3djs') {
            import('/blogs/3DViewer/js/gaussian_viewer.js').then((module) => {
                module.initGaussianViewer();
            }).catch((error) => {
                console.error('Failed to load Gaussian viewer:', error);
            });
        }

        if (runtimeFeatures.simpleModelViewer || postId === '250310_model_viewer') {
            import('/js/simple-model-viewer.js').then((module) => {
                module.initGaussianViewer();
            }).catch((error) => {
                console.error('Failed to load model viewer:', error);
            });
        }
    }

    function syncLanguagePreference() {
        const config = window.blogPostPageConfig;
        if (!config?.lang) {
            return;
        }

        document.documentElement.lang = config.lang === 'kor' ? 'ko' : 'en';

        try {
            localStorage.setItem('language', config.lang);
        } catch (error) {
            console.warn('Failed to persist language preference:', error);
        }

        document.querySelectorAll('[data-language-target]').forEach((link) => {
            link.addEventListener('click', () => {
                const targetLang = link.dataset.languageTarget;
                if (!targetLang) {
                    return;
                }

                try {
                    localStorage.setItem('language', targetLang);
                } catch (error) {
                    console.warn('Failed to persist language preference:', error);
                }
            });
        });
    }

    function setupNavbarCollapse() {
        const nav = document.getElementById('mainNav');
        const toggle = nav?.querySelector('[data-nav-toggle]');
        const collapse = nav?.querySelector('.navbar-collapse');

        if (!toggle || !collapse) {
            return;
        }

        function setExpanded(expanded) {
            collapse.classList.toggle('show', expanded);
            toggle.setAttribute('aria-expanded', String(expanded));
            toggle.setAttribute(
                'aria-label',
                expanded ? toggle.dataset.closeLabel : toggle.dataset.openLabel
            );
        }

        toggle.addEventListener('click', () => {
            setExpanded(!collapse.classList.contains('show'));
        });

        collapse.querySelectorAll('a, button').forEach((item) => {
            item.addEventListener('click', () => {
                if (window.matchMedia('(max-width: 991.98px)').matches) {
                    setExpanded(false);
                }
            });
        });

        window.addEventListener('resize', () => {
            if (!window.matchMedia('(max-width: 991.98px)').matches) {
                setExpanded(false);
            }
        });
    }

    function setupAutoRevealNav() {
        const nav = document.getElementById('mainNav');
        if (!nav) {
            return;
        }

        let lastScrollY = window.scrollY;
        let ticking = false;

        function isMenuOpen() {
            return Boolean(nav.querySelector('.navbar-collapse.show'));
        }

        function updateNav() {
            const currentY = Math.max(window.scrollY, 0);
            const headerHeight = document.querySelector('.masthead')?.offsetHeight || 260;
            const pinStart = Math.max(96, Math.min(headerHeight * 0.5, headerHeight - 72));
            const delta = currentY - lastScrollY;

            if (currentY <= pinStart) {
                nav.classList.remove('is-fixed', 'is-visible');
            } else {
                nav.classList.add('is-fixed');

                if (isMenuOpen() || delta < -6) {
                    nav.classList.add('is-visible');
                } else if (delta > 6) {
                    nav.classList.remove('is-visible');
                }
            }

            lastScrollY = currentY;
            ticking = false;
        }

        function requestUpdate() {
            if (ticking) {
                return;
            }
            ticking = true;
            window.requestAnimationFrame(updateNav);
        }

        window.addEventListener('scroll', requestUpdate, { passive: true });
        window.addEventListener('resize', requestUpdate);
        nav.querySelector('.navbar-toggler')?.addEventListener('click', () => {
            window.setTimeout(requestUpdate, 0);
        });
        updateNav();
    }

    function initializePage() {
        syncLanguagePreference();
        initializeMathRendering();
        setupShareButton();
        setupNavbarCollapse();
        setupAutoRevealNav();
        initializeTOC();
        renderSeriesNavigation();
        initializeSpecialViewers();
    }

    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', initializePage, { once: true });
    } else {
        initializePage();
    }
})();
