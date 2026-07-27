document.addEventListener('DOMContentLoaded', async function() {
    const postsContainer = document.querySelector('#posts-tab');
    const seriesContainer = document.querySelector('#series-tab');
    const notesContainer = document.querySelector('#notes-tab');
    const featureContainer = document.querySelector('#blog-home-feature');
    const langToggleButton = document.getElementById('lang-toggle-main');

    const { loadSiteData, getPostTitle, getPostDescription, getPostUrl } = window.siteDataClient;
    const siteData = await loadSiteData();
    const sortedPosts = [...siteData.posts].sort((a, b) => new Date(b.date) - new Date(a.date));

    const labels = {
        eng: {
            heroKicker: "Hwan's Blog",
            heroTitle: 'Research notes for 3D AI systems',
            heroIntro: '3D generation, computer vision, graphics, CUDA inference, and the implementation details that usually stay between commits.',
            searchPlaceholder: 'Search...',
            archiveTitle: 'Archive',
            tabPosts: 'Posts',
            tabNotes: 'Notes',
            tabSeries: 'Series',
            featuredLabel: 'Featured',
            latestPost: 'Latest Post',
            readPost: 'Read post',
            post: 'Post',
            note: 'Note',
            items: 'items',
            latest: 'Latest',
            noPosts: 'No posts yet.',
            noNotes: 'No notes yet.',
            noSeries: 'No series yet.'
        },
        kor: {
            heroKicker: "Hwan's Blog",
            heroTitle: '3D AI 시스템을 만드는 과정의 기록',
            heroIntro: '3D 생성, 컴퓨터 비전, 그래픽스, CUDA 추론, 그리고 보통 커밋 사이에 묻히는 구현 디테일을 정리합니다.',
            searchPlaceholder: '검색...',
            archiveTitle: '글 목록',
            tabPosts: 'Posts',
            tabNotes: 'Notes',
            tabSeries: 'Series',
            featuredLabel: 'Featured',
            latestPost: 'Latest Post',
            readPost: '글 읽기',
            post: 'Post',
            note: 'Note',
            items: 'items',
            latest: 'Latest',
            noPosts: '아직 게시글이 없습니다.',
            noNotes: '아직 노트가 없습니다.',
            noSeries: '아직 시리즈가 없습니다.'
        }
    };

    function copy(lang, key) {
        return labels[lang]?.[key] || labels.eng[key] || '';
    }

    function escapeHtml(value) {
        return String(value || '')
            .replace(/&/g, '&amp;')
            .replace(/</g, '&lt;')
            .replace(/>/g, '&gt;')
            .replace(/"/g, '&quot;')
            .replace(/'/g, '&#39;');
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
            month: 'short',
            day: 'numeric'
        });
    }

    function getSeriesTitle(post, lang) {
        return siteData.series[post.series]?.[lang] || siteData.series[post.series]?.eng || 'Series';
    }

    function getCategoryLabel(post, lang) {
        return post.category === 'note' ? copy(lang, 'note') : copy(lang, 'post');
    }

    function renderTags(post) {
        return (post.tags || [])
            .map((tag) => `<span class="post-tag">${escapeHtml(tag)}</span>`)
            .join('');
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

        const featuredPost = getFeaturedPost();
        const postCount = sortedPosts
            .filter((post) => post.category === 'post' && post.id !== featuredPost?.id)
            .length;
        const noteCount = sortedPosts.filter((post) => post.category === 'note').length;
        const seriesCount = new Set(sortedPosts.map((post) => post.series).filter(Boolean)).size;

        setText('#posts-count', postCount);
        setText('#notes-count', noteCount);
        setText('#series-count', seriesCount);

        if (langToggleButton) {
            langToggleButton.textContent = lang === 'eng' ? 'KOR' : 'ENG';
        }
    }

    function getFeaturedPost() {
        return sortedPosts.find((post) => post.category === 'post') || sortedPosts[0];
    }

    function createPostPreviewHTML(post, lang) {
        const title = getPostTitle(post, lang);
        const subtitle = getPostDescription(post, lang);
        const seriesTitle = getSeriesTitle(post, lang);
        const tagsHtml = renderTags(post);

        if (!title) {
            return '';
        }

        return `
        <article class="post-preview">
            <a href="${escapeHtml(getPostUrl(post, lang))}" class="post-card-link">
                <div class="post-card-cover" role="img" aria-label="${escapeHtml(title)}" style="background-image: url('${escapeHtml(post.cover || '/assets/blog_bg.jpeg')}')">
                </div>
                <div class="post-card-body">
                    <div class="post-card-eyebrow">
                        <span>${escapeHtml(getCategoryLabel(post, lang))}</span>
                        <span>${escapeHtml(seriesTitle)}</span>
                    </div>
                    <h3 class="post-title">${escapeHtml(title)}</h3>
                    ${subtitle ? `<p class="post-subtitle">${escapeHtml(subtitle)}</p>` : ''}
                    ${tagsHtml ? `<div class="post-tag-row">${tagsHtml}</div>` : ''}
                </div>
            </a>
            <p class="post-meta">
                ${escapeHtml(seriesTitle)} / ${escapeHtml(formatDate(post.date, lang))}
            </p>
        </article>
        `;
    }

    function renderFeatured(lang) {
        if (!featureContainer) {
            return;
        }

        const featuredPost = getFeaturedPost();
        if (!featuredPost) {
            featureContainer.innerHTML = '';
            return;
        }

        const title = getPostTitle(featuredPost, lang);
        const subtitle = getPostDescription(featuredPost, lang);
        const seriesTitle = getSeriesTitle(featuredPost, lang);
        const tagsHtml = renderTags(featuredPost);
        const url = getPostUrl(featuredPost, lang);

        featureContainer.innerHTML = `
            <article class="blog-feature-card">
                <div class="blog-feature-label">
                    <span>${escapeHtml(copy(lang, 'featuredLabel'))}</span>
                    <span>${escapeHtml(copy(lang, 'latestPost'))}</span>
                </div>
                <a class="blog-feature-cover" href="${escapeHtml(url)}" role="img" aria-label="${escapeHtml(title)}" style="background-image: url('${escapeHtml(featuredPost.cover || '/assets/blog_bg.jpeg')}')">
                </a>
                <div class="blog-feature-copy">
                    <div class="blog-feature-meta">
                        <span>${escapeHtml(seriesTitle)}</span>
                        <time datetime="${escapeHtml(featuredPost.date)}">${escapeHtml(formatDate(featuredPost.date, lang))}</time>
                    </div>
                    <h2><a href="${escapeHtml(url)}">${escapeHtml(title)}</a></h2>
                    ${subtitle ? `<p>${escapeHtml(subtitle)}</p>` : ''}
                    ${tagsHtml ? `<div class="post-tag-row">${tagsHtml}</div>` : ''}
                    <a class="blog-feature-read" href="${escapeHtml(url)}">
                        <span>${escapeHtml(copy(lang, 'readPost'))}</span>
                        <i class="bi bi-arrow-right" aria-hidden="true"></i>
                    </a>
                </div>
            </article>
        `;
    }

    function renderAllPosts(lang) {
        if (!postsContainer) {
            return;
        }

        const allPostsHTML = sortedPosts
            .filter((post) => post.category === 'post' && post.id !== getFeaturedPost()?.id)
            .map((post) => createPostPreviewHTML(post, lang))
            .join('');

        postsContainer.innerHTML = allPostsHTML || `<p class="blog-home-empty">${escapeHtml(copy(lang, 'noPosts'))}</p>`;
    }

    function renderNotes(lang) {
        if (!notesContainer) {
            return;
        }

        const notesHTML = sortedPosts
            .filter((post) => post.category === 'note')
            .map((post) => createPostPreviewHTML(post, lang))
            .join('');

        notesContainer.innerHTML = notesHTML || `<p class="blog-home-empty">${escapeHtml(copy(lang, 'noNotes'))}</p>`;
    }

    function renderSeries(lang) {
        if (!seriesContainer) {
            return;
        }

        const postsBySeries = {};
        sortedPosts.forEach((post) => {
            if (!post.series) {
                return;
            }
            if (!postsBySeries[post.series]) {
                postsBySeries[post.series] = [];
            }
            postsBySeries[post.series].push(post);
        });

        const seriesHTML = Object.entries(postsBySeries)
            .sort(([, firstPosts], [, secondPosts]) => new Date(secondPosts[0].date) - new Date(firstPosts[0].date))
            .map(([seriesId, posts]) => {
                const seriesTitle = siteData.series[seriesId]?.[lang] || siteData.series[seriesId]?.eng || 'Series';
                const latestPost = posts[0];
                const itemsHtml = posts.map((post) => {
                    const title = getPostTitle(post, lang);
                    if (!title) {
                        return '';
                    }

                    return `
                        <li>
                            <a href="${escapeHtml(getPostUrl(post, lang))}">${escapeHtml(title)}</a>
                            <span class="post-meta-sm">${escapeHtml(formatShortDate(post.date, lang))}</span>
                        </li>
                    `;
                }).join('');

                return `
                    <article class="series-group">
                        <div class="series-card-header">
                            <div>
                                <span class="series-card-kicker">Series</span>
                                <h3 class="series-title">${escapeHtml(seriesTitle)}</h3>
                            </div>
                            <div class="series-card-meta">
                                <span>${posts.length} ${escapeHtml(copy(lang, 'items'))}</span>
                                <span>${escapeHtml(copy(lang, 'latest'))} ${escapeHtml(formatShortDate(latestPost.date, lang))}</span>
                            </div>
                        </div>
                        <ol class="series-post-list">
                            ${itemsHtml}
                        </ol>
                    </article>
                `;
            }).join('');

        seriesContainer.innerHTML = seriesHTML || `<p class="blog-home-empty">${escapeHtml(copy(lang, 'noSeries'))}</p>`;
    }

    function renderAllTabs(lang) {
        updateStaticCopy(lang);
        renderFeatured(lang);
        renderAllPosts(lang);
        renderNotes(lang);
        renderSeries(lang);
    }

    if (langToggleButton) {
        langToggleButton.addEventListener('click', function() {
            const currentLang = localStorage.getItem('language') || 'eng';
            const nextLang = currentLang === 'eng' ? 'kor' : 'eng';
            localStorage.setItem('language', nextLang);
            renderAllTabs(nextLang);
        });
    }

    renderAllTabs(localStorage.getItem('language') || 'eng');
});
