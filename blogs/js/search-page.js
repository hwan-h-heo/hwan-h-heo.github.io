document.addEventListener('DOMContentLoaded', async function() {
    const resultsContainer = document.getElementById('search-results-container');
    const resultCount = document.getElementById('search-result-count');
    const searchInput = document.getElementById('search-input');
    const languageButton = document.getElementById('lang-toggle-main');
    const urlParams = new URLSearchParams(window.location.search);
    const rawSearchTerm = urlParams.get('q')?.trim() || '';
    const coverMedia = window.blogCoverMedia;
    const search = window.blogSearch;
    let searchIndex;
    let searchFailed = false;
    let archivedTags = new Set();
    let siteData;

    const copy = {
        eng: {
            kicker: 'Search Index',
            title: 'Search the archive',
            intro: 'Find research notes by title, topic, tag, or anything mentioned in the article.',
            results: 'Results',
            searching: 'Searching the archive...',
            prompt: 'Enter a search term to browse the archive.',
            noResults: 'No results found for',
            failed: 'Search could not be loaded.',
            retry: 'Try again',
            result: 'result',
            resultsCount: 'results'
        },
        kor: {
            kicker: 'Search Index',
            title: '아카이브 검색',
            intro: '제목, 주제, 태그와 본문에 포함된 내용으로 글을 찾아보세요.',
            results: '검색 결과',
            searching: '아카이브를 검색하고 있습니다...',
            prompt: '검색어를 입력해 글을 찾아보세요.',
            noResults: '검색 결과가 없습니다:',
            failed: '검색을 불러오지 못했습니다.',
            retry: '다시 시도',
            result: '개 결과',
            resultsCount: '개 결과'
        }
    };

    function escapeHtml(value) {
        return String(value ?? '')
            .replace(/&/g, '&amp;')
            .replace(/</g, '&lt;')
            .replace(/>/g, '&gt;')
            .replace(/"/g, '&quot;')
            .replace(/'/g, '&#39;');
    }

    function getStoredLanguage() {
        try {
            return localStorage.getItem('language') === 'kor' ? 'kor' : 'eng';
        } catch (error) {
            return 'eng';
        }
    }

    function highlightText(value) {
        return search.splitHighlights(value, rawSearchTerm).map((part) => (
            part.match ? `<mark>${escapeHtml(part.text)}</mark>` : escapeHtml(part.text)
        )).join('');
    }

    function getTextLanguage(value) {
        return /[\u1100-\u11ff\u3130-\u318f\uac00-\ud7af]/.test(value) ? 'ko' : 'en';
    }

    function storeLanguage(language) {
        try {
            localStorage.setItem('language', language);
        } catch (error) {}
    }

    function formatDate(date, language) {
        return new Date(`${date}T00:00:00`).toLocaleDateString(
            language === 'kor' ? 'ko-KR' : 'en-US',
            { year: 'numeric', month: language === 'kor' ? 'numeric' : 'long', day: 'numeric' }
        );
    }

    function getSeriesTitle(post, language) {
        const series = siteData?.series?.[post.series];
        return series?.[language] || series?.eng || post.series || '';
    }

    function renderTags(post) {
        return (post.tags || []).map((tag) => {
            return archivedTags.has(tag)
                ? `<a class="post-tag" href="${escapeHtml(window.siteDataClient.getTagUrl(tag))}">${escapeHtml(tag)}</a>`
                : `<span class="post-tag">${escapeHtml(tag)}</span>`;
        }).join('<span class="post-tag-separator" aria-hidden="true">·</span>');
    }

    function renderSearchResults(language) {
        if (!resultsContainer) {
            return;
        }

        const languageCopy = copy[language];
        if (!rawSearchTerm) {
            resultCount.textContent = '';
            resultsContainer.innerHTML = `<p class="blog-search-state">${languageCopy.prompt}</p>`;
            return;
        }
        if (searchFailed) {
            resultCount.textContent = '';
            resultsContainer.innerHTML = `<p class="blog-search-state">${languageCopy.failed} <button class="blog-search-retry" type="button">${languageCopy.retry}</button></p>`;
            return;
        }
        if (!siteData || !searchIndex) {
            resultsContainer.innerHTML = `<p class="blog-search-state">${languageCopy.searching}</p>`;
            return;
        }

        const results = search.searchPosts(siteData, searchIndex, rawSearchTerm, language);

        const countLabel = language === 'kor'
            ? `${results.length}${languageCopy.resultsCount}`
            : `${results.length} ${results.length === 1 ? languageCopy.result : languageCopy.resultsCount}`;
        resultCount.textContent = countLabel;

        if (results.length === 0) {
            resultsContainer.innerHTML = `
                <p class="blog-search-state">
                    ${escapeHtml(languageCopy.noResults)} <strong>“${escapeHtml(rawSearchTerm)}”</strong>
                </p>
            `;
            return;
        }

        resultsContainer.innerHTML = results.map(({ post, excerpt }) => {
            const title = window.siteDataClient.getPostTitle(post, language);
            const subtitle = window.siteDataClient.getPostSubtitle(post, language);
            const description = excerpt || subtitle;
            const url = window.siteDataClient.getPostUrl(post, language);
            const seriesTitle = getSeriesTitle(post, language);
            const languageNotice = window.siteDataClient.getPostLanguageNotice(post, language);
            const source = post.cover || '/assets/blog_bg.jpeg';
            const keepAnimated = post.animatedPreview === true && coverMedia?.isAnimatedCover(source);
            const preview = coverMedia?.getBlogCoverPreviewUrl(post) || source;
            const autoplaySource = keepAnimated ? ` data-autoplay-src="${escapeHtml(source)}"` : '';
            const animatedSource = !keepAnimated && coverMedia?.isAnimatedCover(source)
                ? ` data-animated-src="${escapeHtml(source)}"`
                : '';
            return `
                <article class="post-preview post-preview-media-right" data-post-id="${escapeHtml(post.id)}" data-post-category="${escapeHtml(post.category)}">
                    <div class="post-card-link">
                        <a href="${escapeHtml(url)}" class="post-card-cover" aria-label="Read ${escapeHtml(title)}">
                            <img src="${escapeHtml(preview)}" data-blog-cover data-preview-src="${escapeHtml(preview)}"${autoplaySource}${animatedSource} alt="${escapeHtml(title)} cover image" loading="lazy" decoding="async">
                        </a>
                        <div class="post-card-body">
                            <div class="post-card-eyebrow post-series-context">
                                <span class="post-series-context-label">SERIES</span>
                                <span class="post-series-context-separator" aria-hidden="true">/</span>
                                <span class="post-series-context-value">${escapeHtml(seriesTitle)}</span>
                            </div>
                            <h3 class="post-title" lang="${getTextLanguage(title)}"><a href="${escapeHtml(url)}">${highlightText(title)}</a></h3>
                            ${description ? `<p class="post-subtitle${excerpt ? ' post-search-excerpt' : ''}" lang="${getTextLanguage(description)}">${highlightText(description)}</p>` : ''}
                            <p class="post-meta"><time datetime="${escapeHtml(post.date)}">${escapeHtml(formatDate(post.date, language))}</time>${languageNotice ? `<span class="post-language-note">${escapeHtml(languageNotice)}</span>` : ''}</p>
                            ${post.tags?.length ? `<div class="post-tag-row">${renderTags(post)}</div>` : ''}
                        </div>
                    </div>
                </article>
            `;
        }).join('');

        coverMedia?.initializeBlogCoverMedia(resultsContainer);
    }

    function applyLanguage(language) {
        document.documentElement.lang = language === 'kor' ? 'ko' : 'en';
        if (languageButton) {
            const targetLabel = language === 'eng' ? '한국어로 전환' : 'Switch to English';
            languageButton.textContent = language === 'eng' ? '가' : 'A';
            languageButton.setAttribute('aria-label', targetLabel);
            languageButton.setAttribute('title', targetLabel);
        }
        document.querySelectorAll('[data-search-copy]').forEach((element) => {
            const key = element.dataset.searchCopy;
            if (copy[language][key]) {
                element.textContent = copy[language][key];
            }
        });
        if (searchInput) {
            searchInput.placeholder = 'Search...';
        }
        renderSearchResults(language);
    }

    if (searchInput) {
        searchInput.value = rawSearchTerm;
    }

    if (!resultsContainer) {
        return;
    }

    const initialLanguage = getStoredLanguage();
    applyLanguage(initialLanguage);

    languageButton?.addEventListener('click', () => {
        const nextLanguage = getStoredLanguage() === 'eng' ? 'kor' : 'eng';
        storeLanguage(nextLanguage);
        applyLanguage(nextLanguage);
    });

    resultsContainer.addEventListener('click', (event) => {
        if (event.target.closest('.blog-search-retry')) {
            window.location.reload();
        }
    });
    if (!rawSearchTerm) {
        return;
    }
    try {
        const indexUrl = document.querySelector('meta[name="blog-search-index"]')?.content;
        if (!indexUrl) {
            throw new Error('Missing search index URL.');
        }
        [siteData, searchIndex] = await Promise.all([
            window.siteDataClient.loadSiteData(),
            fetch(indexUrl, { cache: 'force-cache' }).then((response) => {
                if (!response.ok) {
                    throw new Error(`Search index returned HTTP ${response.status}.`);
                }
                return response.json();
            })
        ]);
        if (searchIndex.version !== 1 || !searchIndex.posts) {
            throw new Error('Invalid search index.');
        }
        archivedTags = window.siteDataClient.getArchivedTagSet(siteData);
    } catch (error) {
        console.warn('Failed to load search:', error);
        searchFailed = true;
    }

    applyLanguage(getStoredLanguage());
});
