(function() {
    function getPostTitle(post, lang) {
        return post[`title_${lang}`] || post.title_eng;
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
        const bottomMargin = 0.2;

        function updateTocVisibility() {
            const headerHeight = document.querySelector('.masthead')?.offsetHeight || 300;
            toc.style.display = window.scrollY > headerHeight ? 'block' : 'none';
        }

        function initTocItems() {
            tocItems = Array.from(toc.querySelectorAll('li')).map((item) => {
                const anchor = item.querySelector('a');
                const href = anchor?.getAttribute('href');
                if (!href || href === '#') {
                    return null;
                }

                return {
                    listItem: item,
                    target: document.getElementById(href.slice(1))
                };
            }).filter((item) => item && item.target);
        }

        function syncToc() {
            const windowHeight = window.innerHeight;
            let currentSection = null;

            tocItems.forEach((item) => {
                const targetBounds = item.target.getBoundingClientRect();
                if (targetBounds.top <= windowHeight * (1 - bottomMargin)) {
                    currentSection = item;
                }
            });

            tocItems.forEach((item) => {
                item.listItem.classList.toggle('active', item === currentSection);
            });
        }

        document.addEventListener('scroll', updateTocVisibility);
        window.addEventListener('scroll', syncToc, false);
        updateTocVisibility();
        initTocItems();
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

        const listItems = postsInSeries.map((post) => {
            const title = getPostTitle(post, config.lang);
            const slug = config.lang === 'eng' ? post.slug : `${post.slug}-kor`;
            if (post.id === config.postId) {
                return `<li><strong>${title}</strong></li>`;
            }
            return `<li><a href="/blogs/posts/${slug}/">${title}</a></li>`;
        }).join('');

        seriesContainer.innerHTML = `
            <div class="accordion mb-4" id="seriesAccordion">
                <div class="accordion-item">
                    <h2 class="accordion-header">
                        <button class="accordion-button collapsed" type="button" data-bs-toggle="collapse" data-bs-target="#seriesAccordionBody" aria-expanded="false" aria-controls="seriesAccordionBody">
                            <strong>${seriesTitle}</strong>
                        </button>
                    </h2>
                    <div id="seriesAccordionBody" class="accordion-collapse collapse" data-bs-parent="#seriesAccordion">
                        <div class="accordion-body">
                            <ol>${listItems}</ol>
                        </div>
                    </div>
                </div>
            </div>
        `;

        const currentIndex = postsInSeries.findIndex((post) => post.id === config.postId);
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

