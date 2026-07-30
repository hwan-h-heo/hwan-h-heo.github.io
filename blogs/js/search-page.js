document.addEventListener('DOMContentLoaded', async function() {
    const resultsContainer = document.getElementById('search-results-container');
    const resultCount = document.getElementById('search-result-count');
    const searchInput = document.getElementById('search-input');
    const languageButton = document.getElementById('lang-toggle-main');
    const urlParams = new URLSearchParams(window.location.search);
    const rawSearchTerm = urlParams.get('q')?.trim() || '';
    const searchTerm = rawSearchTerm.toLowerCase();
    const coverMedia = window.blogCoverMedia;
    let filteredPosts = [];
    let siteData;

    const copy = {
        eng: {
            kicker: 'Search',
            title: 'Search the archive',
            intro: 'Find research notes by title, topic, tag, or anything mentioned in the article.',
            eyebrow: 'Archive index',
            results: 'Results',
            searching: 'Searching the archive...',
            prompt: 'Enter a search term to browse the archive.',
            noResults: 'No results found for',
            result: 'result',
            resultsCount: 'results',
            post: 'Post',
            note: 'Note',
            languages: 'Languages',
            english: 'English',
            korean: 'Korean'
        },
        kor: {
            kicker: '검색',
            title: '아카이브 검색',
            intro: '제목, 주제, 태그와 본문에 포함된 내용으로 글을 찾아보세요.',
            eyebrow: '아카이브 인덱스',
            results: '검색 결과',
            searching: '아카이브를 검색하고 있습니다...',
            prompt: '검색어를 입력해 글을 찾아보세요.',
            noResults: '검색 결과가 없습니다:',
            result: '개 결과',
            resultsCount: '개 결과',
            post: '포스트',
            note: '노트',
            languages: '언어',
            english: '영문',
            korean: '국문'
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

    function storeLanguage(language) {
        try {
            localStorage.setItem('language', language);
        } catch (error) {}
    }

    function tagSlug(tag) {
        return String(tag)
            .toLowerCase()
            .replace(/[^a-z0-9]+/g, '-')
            .replace(/^-|-$/g, '');
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

    function renderLanguageLinks(post, language) {
        return post.languages.map((postLanguage) => {
            const label = postLanguage === 'kor' ? copy[language].korean : copy[language].english;
            const url = window.siteDataClient.getPostUrl(post, postLanguage);
            return `<a href="${escapeHtml(url)}" hreflang="${postLanguage === 'kor' ? 'ko' : 'en'}">${escapeHtml(label)}</a>`;
        }).join(', ');
    }

    function renderTags(post) {
        return (post.tags || []).map((tag) => {
            return `<a class="post-tag" href="/blogs/tags/${escapeHtml(tagSlug(tag))}/">${escapeHtml(tag)}</a>`;
        }).join('');
    }

    function renderSearchResults(language) {
        if (!resultsContainer || !siteData) {
            return;
        }

        const languageCopy = copy[language];
        if (!searchTerm) {
            resultCount.textContent = '';
            resultsContainer.innerHTML = `<p class="blog-search-state">${languageCopy.prompt}</p>`;
            return;
        }

        const countLabel = language === 'kor'
            ? `${filteredPosts.length}${languageCopy.resultsCount}`
            : `${filteredPosts.length} ${filteredPosts.length === 1 ? languageCopy.result : languageCopy.resultsCount}`;
        resultCount.textContent = countLabel;

        if (filteredPosts.length === 0) {
            resultsContainer.innerHTML = `
                <p class="blog-search-state">
                    ${escapeHtml(languageCopy.noResults)} <strong>“${escapeHtml(rawSearchTerm)}”</strong>
                </p>
            `;
            return;
        }

        resultsContainer.innerHTML = filteredPosts.map((post) => {
            const title = window.siteDataClient.getPostTitle(post, language);
            const description = window.siteDataClient.getPostDescription(post, language);
            const url = window.siteDataClient.getPostUrl(post, language);
            const seriesTitle = getSeriesTitle(post, language);
            const categoryLabel = post.category === 'note' ? languageCopy.note : languageCopy.post;
            const source = post.cover || '/assets/blog_bg.jpeg';
            const keepAnimated = post.animatedPreview === true && coverMedia?.isAnimatedCover(source);
            const preview = coverMedia?.getBlogCoverPreviewUrl(post.id) || source;
            const autoplaySource = keepAnimated ? ` data-autoplay-src="${escapeHtml(source)}"` : '';
            const animatedSource = !keepAnimated && coverMedia?.isAnimatedCover(source)
                ? ` data-animated-src="${escapeHtml(source)}"`
                : '';

            return `
                <article class="post-preview" data-post-id="${escapeHtml(post.id)}" data-post-category="${escapeHtml(post.category)}">
                    <div class="post-card-link">
                        <a href="${escapeHtml(url)}" class="post-card-cover" aria-label="Read ${escapeHtml(title)}">
                            <img src="${escapeHtml(preview)}" data-blog-cover data-preview-src="${escapeHtml(preview)}"${autoplaySource}${animatedSource} alt="${escapeHtml(title)} cover image" loading="lazy" decoding="async">
                        </a>
                        <div class="post-card-body">
                            <div class="post-card-eyebrow">
                                <span>${escapeHtml(categoryLabel)}</span>
                                <span>${escapeHtml(seriesTitle)}</span>
                            </div>
                            <h3 class="post-title"><a href="${escapeHtml(url)}">${escapeHtml(title)}</a></h3>
                            ${description ? `<p class="post-subtitle">${escapeHtml(description)}</p>` : ''}
                            ${post.tags?.length ? `<div class="post-tag-row">${renderTags(post)}</div>` : ''}
                        </div>
                    </div>
                    <p class="post-meta">
                        ${escapeHtml(seriesTitle)} / <time datetime="${escapeHtml(post.date)}">${escapeHtml(formatDate(post.date, language))}</time> /
                        ${escapeHtml(languageCopy.languages)}: ${renderLanguageLinks(post, language)}
                    </p>
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
            searchInput.placeholder = language === 'kor' ? '검색...' : 'Search...';
        }
        if (!siteData) {
            const loadingState = resultsContainer?.querySelector('.blog-search-state');
            if (loadingState) {
                loadingState.textContent = copy[language].searching;
            }
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

    const { loadSiteData } = window.siteDataClient;
    siteData = await loadSiteData();

    if (searchTerm) {
        const postContentCache = {};
        await Promise.all(siteData.posts.map(async (post) => {
            try {
                const contents = await Promise.all(post.languages.map(async (language) => {
                    const response = await fetch(`../posts/${post.id}/content-${language}.md`);
                    if (!response.ok) {
                        throw new Error(`HTTP ${response.status} for ${language}`);
                    }
                    return response.text();
                }));
                postContentCache[post.id] = contents.join('\n').toLowerCase();
            } catch (error) {
                console.warn(`Failed to fetch markdown for ${post.id}:`, error);
                postContentCache[post.id] = '';
            }
        }));

        filteredPosts = siteData.posts.filter((post) => {
            const metadataMatch = Object.entries(post).some(([key, value]) => {
                return (
                    (key.startsWith('title_') || key.startsWith('subtitle_') || key.startsWith('description_'))
                    && String(value).toLowerCase().includes(searchTerm)
                );
            });
            const tagMatch = (post.tags || []).some((tag) => tag.toLowerCase().includes(searchTerm));
            const contentMatch = postContentCache[post.id]?.includes(searchTerm);
            return metadataMatch || tagMatch || contentMatch;
        });
    }

    applyLanguage(getStoredLanguage());
});
