document.addEventListener('DOMContentLoaded', async function() {
    const seriesContainer = document.querySelector('#series-tab');
    const featureContainer = document.querySelector('#blog-home-feature');
    const langToggleButton = document.getElementById('lang-toggle-main');
    const tabControls = Array.from(document.querySelectorAll('[role="tab"][data-tab-target]'));
    let featureFitFrame = 0;

    function fitFeaturedCopy() {
        const card = featureContainer?.querySelector('.blog-feature-card');
        const cover = card?.querySelector('.blog-feature-cover');
        const copyElement = card?.querySelector('.blog-feature-copy');
        const titleElement = copyElement?.querySelector('h2');
        const subtitleElement = copyElement?.querySelector('.blog-feature-subtitle');

        if (!cover || !copyElement || !titleElement || !subtitleElement) {
            return;
        }

        copyElement.classList.remove('is-height-fitted');
        copyElement.style.removeProperty('height');
        titleElement.style.removeProperty('font-size');
        subtitleElement.style.removeProperty('font-size');

        const coverRect = cover.getBoundingClientRect();
        const copyRect = copyElement.getBoundingClientRect();
        if (Math.abs(coverRect.top - copyRect.top) > 2) {
            return;
        }

        const titleBase = Number.parseFloat(getComputedStyle(titleElement).fontSize);
        const subtitleBase = Number.parseFloat(getComputedStyle(subtitleElement).fontSize);
        const titleFloor = titleBase * 0.9;
        const subtitleFloor = subtitleBase * 0.9;
        let titleSize = titleBase;
        let subtitleSize = subtitleBase;

        copyElement.style.height = `${coverRect.height}px`;
        copyElement.classList.add('is-height-fitted');

        while (copyElement.scrollHeight > copyElement.clientHeight + 0.5) {
            let resized = false;
            if (titleSize > titleFloor) {
                titleSize = Math.max(titleFloor, titleSize - 0.25);
                titleElement.style.fontSize = `${titleSize}px`;
                resized = true;
            }
            if (subtitleSize > subtitleFloor) {
                subtitleSize = Math.max(subtitleFloor, subtitleSize - 0.125);
                subtitleElement.style.fontSize = `${subtitleSize}px`;
                resized = true;
            }
            if (!resized) {
                break;
            }
        }
    }

    function scheduleFeaturedCopyFit() {
        cancelAnimationFrame(featureFitFrame);
        featureFitFrame = requestAnimationFrame(fitFeaturedCopy);
    }

    function activateTab(control, { focus = false } = {}) {
        if (!control) {
            return;
        }

        tabControls.forEach((tab) => {
            const isActive = tab === control;
            const panel = document.querySelector(tab.dataset.tabTarget);
            tab.classList.toggle('is-active', isActive);
            tab.setAttribute('aria-selected', String(isActive));
            tab.tabIndex = isActive ? 0 : -1;

            if (panel) {
                panel.classList.toggle('is-active', isActive);
                panel.hidden = !isActive;
            }
        });

        if (focus) {
            control.focus();
        }
    }

    tabControls.forEach((control, index) => {
        control.addEventListener('click', () => activateTab(control));
        control.addEventListener('keydown', (event) => {
            let targetIndex = index;

            if (event.key === 'ArrowLeft') {
                targetIndex = (index - 1 + tabControls.length) % tabControls.length;
            } else if (event.key === 'ArrowRight') {
                targetIndex = (index + 1) % tabControls.length;
            } else if (event.key === 'Home') {
                targetIndex = 0;
            } else if (event.key === 'End') {
                targetIndex = tabControls.length - 1;
            } else {
                return;
            }

            event.preventDefault();
            activateTab(tabControls[targetIndex], { focus: true });
        });
    });

    const { loadSiteData, getPostTitle, getPostSubtitle, getPostUrl } = window.siteDataClient;
    const coverMedia = window.blogCoverMedia;
    const siteData = await loadSiteData();
    const sortedPosts = [...siteData.posts].sort((a, b) => new Date(b.date) - new Date(a.date));
    const postById = new Map(sortedPosts.map((post) => [post.id, post]));

    const labels = {
        eng: {
            searchPlaceholder: 'Search...',
            archiveTitle: 'Articles',
            oldArchiveTitle: 'Old Archive',
            tabPosts: 'Posts',
            tabNotes: 'Notes',
            tabSeries: 'Series',
            featuredLabel: 'Featured',
            readPost: 'Read post',
            items: 'items',
            latest: 'Latest'
        },
        kor: {
            searchPlaceholder: 'Search...',
            archiveTitle: 'Articles',
            oldArchiveTitle: 'Old Archive',
            tabPosts: 'Posts',
            tabNotes: 'Notes',
            tabSeries: 'Series',
            featuredLabel: 'Featured',
            readPost: '글 읽기',
            items: 'items',
            latest: 'Latest'
        }
    };

    function copy(lang, key) {
        return labels[lang]?.[key] || labels.eng[key] || '';
    }

    function formatDate(dateString, lang) {
        const locale = lang === 'kor' ? 'ko-KR' : 'en-US';
        return new Date(dateString).toLocaleDateString(locale, {
            year: 'numeric',
            month: 'long',
            day: 'numeric'
        });
    }

    function formatShortDate(dateString, lang) {
        const locale = lang === 'kor' ? 'ko-KR' : 'en-US';
        return new Date(dateString).toLocaleDateString(locale, {
            year: 'numeric',
            month: 'short'
        });
    }

    function getSeriesTitle(post, lang) {
        return siteData.series[post.series]?.[lang] || siteData.series[post.series]?.eng || 'Series';
    }

    function getTextLang(value, lang) {
        return lang === 'kor' && /[\u1100-\u11ff\u3130-\u318f\uac00-\ud7af]/.test(String(value || '')) ? 'ko' : 'en';
    }

    function setText(selector, value) {
        const element = document.querySelector(selector);
        if (element) {
            element.textContent = value;
        }
    }

    function updateStaticCopy(lang) {
        document.documentElement.lang = lang === 'kor' ? 'ko' : 'en';

        document.querySelectorAll('[data-i18n]').forEach((element) => {
            element.textContent = copy(lang, element.dataset.i18n);
        });

        document.querySelectorAll('[data-placeholder-i18n]').forEach((element) => {
            element.setAttribute('placeholder', copy(lang, element.dataset.placeholderI18n));
        });

        const postCount = sortedPosts.filter((post) => post.category === 'post').length;
        const noteCount = sortedPosts.filter((post) => post.category === 'note').length;
        const seriesCount = new Set(sortedPosts.map((post) => post.series).filter(Boolean)).size;

        setText('#posts-count', postCount);
        setText('#notes-count', noteCount);
        setText('#series-count', seriesCount);
        setText('#hero-posts-count', postCount);
        setText('#hero-notes-count', noteCount);
        setText('#hero-series-count', seriesCount);

        if (langToggleButton) {
            const targetLabel = lang === 'eng' ? '한국어로 전환' : 'Switch to English';
            langToggleButton.textContent = lang === 'eng' ? '가' : 'A';
            langToggleButton.setAttribute('aria-label', targetLabel);
            langToggleButton.setAttribute('title', targetLabel);
        }
    }

    function getFeaturedPost() {
        const featuredPostId = siteData.blogHome?.featuredPostId;
        if (featuredPostId && postById.has(featuredPostId)) {
            return postById.get(featuredPostId);
        }

        return sortedPosts.find((post) => post.category === 'post') || sortedPosts[0];
    }

    function updatePostPreview(card, post, lang) {
        const title = getPostTitle(post, lang);
        const subtitleText = getPostSubtitle(post, lang);
        const url = getPostUrl(post, lang);
        const coverLink = card.querySelector('.post-card-cover');
        const titleElement = card.querySelector('.post-title');
        const titleLink = card.querySelector('.post-title a');
        const coverImage = card.querySelector('img[data-blog-cover]');
        const subtitle = card.querySelector('.post-subtitle');
        const series = card.querySelector('[data-post-series]');
        const time = card.querySelector('.post-meta time');

        if (coverLink) {
            coverLink.href = url;
            coverLink.setAttribute('aria-label', `${copy(lang, 'readPost')}: ${title}`);
        }
        if (titleLink) {
            titleLink.href = url;
            titleLink.textContent = title;
        }
        if (titleElement) {
            titleElement.lang = getTextLang(title, lang);
        }
        if (coverImage) {
            coverImage.alt = `${title} cover image`;
        }
        if (subtitle) {
            subtitle.textContent = subtitleText;
            subtitle.hidden = !subtitleText;
            subtitle.lang = getTextLang(subtitleText, lang);
        }
        if (series) {
            series.textContent = getSeriesTitle(post, lang);
        }
        if (time) {
            time.dateTime = post.date;
            time.textContent = formatDate(post.date, lang);
        }
    }

    function updateFeaturedPost(lang) {
        const card = featureContainer?.querySelector('.blog-feature-card[data-post-id]');
        const post = card ? postById.get(card.dataset.postId) : null;
        if (!card || !post) {
            return;
        }

        const title = getPostTitle(post, lang);
        const subtitle = getPostSubtitle(post, lang);
        const url = getPostUrl(post, lang);
        const coverLink = card.querySelector('.blog-feature-cover');
        const titleElement = card.querySelector('.blog-feature-copy h2');
        const titleLink = card.querySelector('.blog-feature-copy h2 a');
        const readLink = card.querySelector('.blog-feature-read');
        const coverImage = card.querySelector('img[data-blog-cover]');
        const subtitleElement = card.querySelector('.blog-feature-subtitle');
        const time = card.querySelector('.blog-feature-date');

        card.querySelector('[data-feature-label]')?.replaceChildren(copy(lang, 'featuredLabel'));
        card.querySelector('[data-feature-series]')?.replaceChildren(getSeriesTitle(post, lang));
        card.querySelector('[data-feature-read-label]')?.replaceChildren(copy(lang, 'readPost'));

        [coverLink, titleLink, readLink].forEach((link) => {
            if (link) {
                link.href = url;
            }
        });
        if (coverLink) {
            coverLink.setAttribute('aria-label', `${copy(lang, 'readPost')}: ${title}`);
        }
        if (titleLink) {
            titleLink.textContent = title;
        }
        if (titleElement) {
            titleElement.lang = getTextLang(title, lang);
        }
        if (readLink) {
            readLink.setAttribute('aria-label', `${copy(lang, 'readPost')}: ${title}`);
        }
        if (coverImage) {
            coverImage.alt = `${title} cover image`;
        }
        if (subtitleElement) {
            subtitleElement.textContent = subtitle;
            subtitleElement.hidden = !subtitle;
            subtitleElement.lang = getTextLang(subtitle, lang);
        }
        if (time) {
            time.dateTime = post.date;
            time.textContent = formatDate(post.date, lang);
        }
        scheduleFeaturedCopyFit();
    }

    function updateSeriesGroups(lang) {
        seriesContainer?.querySelectorAll('.series-group[data-series-id]').forEach((group) => {
            const posts = sortedPosts.filter((post) => post.series === group.dataset.seriesId);
            const latestPost = posts[0];
            const seriesTitle = siteData.series[group.dataset.seriesId]?.[lang]
                || siteData.series[group.dataset.seriesId]?.eng
                || 'Series';

            const seriesTitleElement = group.querySelector('[data-series-title]');
            seriesTitleElement?.replaceChildren(seriesTitle);
            if (seriesTitleElement) {
                seriesTitleElement.closest('.series-title')?.setAttribute('lang', getTextLang(seriesTitle, lang));
            }
            group.querySelector('[data-series-count]')?.replaceChildren(`${posts.length} ${copy(lang, 'items')}`);
            group.querySelector('[data-series-latest-label]')?.replaceChildren(copy(lang, 'latest'));

            const latestTime = group.querySelector('.series-card-meta time');
            if (latestTime && latestPost) {
                latestTime.dateTime = latestPost.date;
                latestTime.textContent = formatShortDate(latestPost.date, lang);
            }

            group.querySelectorAll('.series-post-list [data-series-post-id]').forEach((item) => {
                const post = postById.get(item.dataset.seriesPostId);
                const link = item.querySelector('a');
                const time = item.querySelector('time');
                if (!post || !link) {
                    return;
                }
                const postTitle = getPostTitle(post, lang);
                link.href = getPostUrl(post, lang);
                link.textContent = postTitle;
                link.lang = getTextLang(postTitle, lang);
                if (time) {
                    time.dateTime = post.date;
                    time.textContent = formatShortDate(post.date, lang);
                }
            });
        });
    }

    function updateRenderedContent(lang) {
        updateStaticCopy(lang);
        updateFeaturedPost(lang);
        document.querySelectorAll('.post-preview[data-post-id]').forEach((card) => {
            const post = postById.get(card.dataset.postId);
            if (post) {
                updatePostPreview(card, post, lang);
            }
        });
        updateSeriesGroups(lang);
    }

    if (langToggleButton) {
        langToggleButton.addEventListener('click', function() {
            const currentLang = localStorage.getItem('language') || 'eng';
            const nextLang = currentLang === 'eng' ? 'kor' : 'eng';
            localStorage.setItem('language', nextLang);
            updateRenderedContent(nextLang);
        });
    }

    updateRenderedContent(localStorage.getItem('language') || 'eng');
    window.addEventListener('resize', scheduleFeaturedCopyFit, { passive: true });
    document.fonts?.ready.then(scheduleFeaturedCopyFit);
    coverMedia?.initializeBlogCoverMedia(document);
});
