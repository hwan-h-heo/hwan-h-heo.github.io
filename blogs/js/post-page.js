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
        const closeModal = shareModal ? shareModal.querySelector('.share_modal_close') : null;
        const indicator = document.getElementById('share_modalIndicator');

        if (!copyButton || !shareModal || !indicator) {
            return;
        }

        let animationId;

        function updateShareButtonVisibility() {
            const headerHeight = document.querySelector('.masthead')?.offsetHeight || 300;
            copyButton.style.display = window.innerWidth > 1280 && window.scrollY > headerHeight ? 'block' : 'none';
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
                    shareModal.style.display = 'none';
                }
            }

            cancelAnimationFrame(animationId);
            animationId = requestAnimationFrame(step);
        }

        function closeShareModal() {
            cancelAnimationFrame(animationId);
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

        const labels = config.lang === 'kor'
            ? {
                kicker: '시리즈',
                current: '현재 글'
            }
            : {
                kicker: 'Series',
                current: 'Current'
            };

        const currentIndex = postsInSeries.findIndex((post) => post.id === config.postId);
        const countLabel = config.lang === 'kor'
            ? `${postsInSeries.length}편`
            : `${postsInSeries.length} posts`;
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

        let navHtml = '<div class="d-flex justify-content-between mb-4">';

        if (olderPost) {
            const olderTitle = getPostTitle(olderPost, config.lang);
            const olderSlug = config.lang === 'eng' ? olderPost.slug : `${olderPost.slug}-kor`;
            navHtml += `<a class="btn btn-light text-uppercase" href="/blogs/posts/${olderSlug}/" style="width: 40%; text-align: center;">← Older Post<br><small style="font-size: 0.7rem; text-transform: none;">${olderTitle.length > 25 ? olderTitle.substring(0, 25) + '...' : olderTitle}</small></a>`;
        } else if (nextPost) {
            navHtml += '<div></div>';
        }

        if (nextPost) {
            const nextTitle = getPostTitle(nextPost, config.lang);
            const nextSlug = config.lang === 'eng' ? nextPost.slug : `${nextPost.slug}-kor`;
            navHtml += `<a class="btn btn-light text-uppercase" href="/blogs/posts/${nextSlug}/" style="width: 40%; text-align: center;">Next Post →<br><small style="font-size: 0.7rem; text-transform: none;">${nextTitle.length > 25 ? nextTitle.substring(0, 25) + '...' : nextTitle}</small></a>`;
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
                navHtml += `<a class="btn btn-outline-secondary text-uppercase" href="/blogs/posts/${recommendedSlug}/" style="width: 40%; text-align: center;">Explore Series<br><small style="font-size: 0.7rem; text-transform: none;">${recommendedSeriesTitle}</small></a>`;
            } else {
                navHtml += '<div></div>';
            }
        }

        navHtml += '</div>';
        navContainer.innerHTML = navHtml;
    }

    function initializeSpecialViewers() {
        const postId = window.blogPostPageConfig?.postId;

        if (postId === '240917_3djs') {
            import('/blogs/3DViewer/js/gaussian_viewer.js').then((module) => {
                module.initGaussianViewer();
            }).catch((error) => {
                console.error('Failed to load Gaussian viewer:', error);
            });
        }

        if (postId === '250310_model_viewer') {
            import('/js/simple-model-viewer.js').then((module) => {
                module.initGaussianViewer();
            }).catch((error) => {
                console.error('Failed to load model viewer:', error);
            });
        }
    }

    function initializePage() {
        initializeMathRendering();
        setupShareButton();
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
