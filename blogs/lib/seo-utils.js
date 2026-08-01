const { SITE_URL } = require('./site-config');
const { render: renderSiteIcon } = require('../../assets/js/site-icons');
const {
    getPostRoute,
    getSeriesRoute,
    getTagRoute,
    listTagArchiveEntries
} = require('./site-routes');
const {
    getBlogCoverPreviewUrl,
    isAnimatedCover
} = require('../js/blog-cover-media');

const LANGUAGE_META = {
    eng: {
        htmlLang: 'en',
        hreflang: 'en',
        label: 'English',
        shortLabel: 'ENG',
        locale: 'en-US'
    },
    kor: {
        htmlLang: 'ko',
        hreflang: 'ko',
        label: 'Korean',
        shortLabel: 'KOR',
        locale: 'ko-KR'
    }
};

function escapeHtml(value) {
    return String(value || '')
        .replace(/&/g, '&amp;')
        .replace(/</g, '&lt;')
        .replace(/>/g, '&gt;')
        .replace(/"/g, '&quot;')
        .replace(/'/g, '&#39;');
}

function escapeXml(value) {
    return escapeHtml(value);
}

function stripHtml(value) {
    return String(value || '')
        .replace(/<br\s*\/?>/gi, ' ')
        .replace(/<[^>]+>/g, ' ')
        .replace(/&amp;/g, '&')
        .replace(/&quot;/g, '"')
        .replace(/&#39;/g, "'")
        .replace(/\s+/g, ' ')
        .trim();
}

function truncateText(value, maxLength = 160) {
    const text = stripHtml(value);
    if (text.length <= maxLength) {
        return text;
    }
    const truncated = text.slice(0, maxLength + 1).replace(/\s+\S*$/, '').trim();
    return truncated || text.slice(0, maxLength).trim();
}

function getLanguageMeta(lang) {
    return LANGUAGE_META[lang] || LANGUAGE_META.eng;
}

function getTextHtmlLang(value, lang = 'eng') {
    return lang === 'kor' && /[\u1100-\u11ff\u3130-\u318f\uac00-\ud7af]/.test(String(value || '')) ? 'ko' : 'en';
}

function getPostTitle(post, lang = 'eng') {
    return post[`title_${lang}`] || post.title_eng || post.id;
}

function getPostDescription(post, lang = 'eng') {
    return truncateText(post[`description_${lang}`] || post[`subtitle_${lang}`] || post.description_eng || post.subtitle_eng || '');
}

function getPostLanguageRoute(post, lang = 'eng') {
    const resolvedLang = post.languages.includes(lang) ? lang : 'eng';
    return getPostRoute(post, resolvedLang);
}

function getPostCanonicalUrl(post, lang = 'eng') {
    return `${SITE_URL}${getPostLanguageRoute(post, lang)}`;
}

function getAbsoluteUrl(value) {
    const text = String(value || '').trim();
    if (!text) {
        return '';
    }
    if (/^https?:\/\//i.test(text)) {
        return text;
    }
    return `${SITE_URL}/${text.replace(/^\/+/, '')}`;
}

function isValidDateString(value) {
    return typeof value === 'string'
        && /^\d{4}-\d{2}-\d{2}$/.test(value)
        && !Number.isNaN(new Date(`${value}T00:00:00Z`).getTime());
}

function formatDate(value, lang = 'eng', options = {}) {
    const date = new Date(`${value}T00:00:00`);
    if (Number.isNaN(date.getTime())) {
        return '';
    }

    return date.toLocaleDateString(getLanguageMeta(lang).locale, {
        year: options.year || 'numeric',
        month: options.month || 'long',
        day: options.day || 'numeric'
    });
}

function formatShortDate(value, lang = 'eng') {
    const date = new Date(`${value}T00:00:00`);
    if (Number.isNaN(date.getTime())) {
        return '';
    }

    return date.toLocaleDateString(getLanguageMeta(lang).locale, {
        year: 'numeric',
        month: 'short'
    });
}

function getSeriesTitle(siteData, seriesId, lang = 'eng') {
    return siteData.series?.[seriesId]?.[lang] || siteData.series?.[seriesId]?.eng || 'Series';
}

function getLanguageSummary(post) {
    return post.languages
        .map((lang) => getLanguageMeta(lang).label)
        .join(', ');
}

function renderLanguageLinks(post) {
    return post.languages
        .map((lang) => {
            const meta = getLanguageMeta(lang);
            return `<a href="${escapeHtml(getPostLanguageRoute(post, lang))}" hreflang="${meta.hreflang}">${escapeHtml(meta.label)}</a>`;
        })
        .join(', ');
}

function getArchivedTagSet(siteData) {
    return new Set(listTagArchiveEntries(siteData).map((entry) => entry.tag));
}

function renderTags(post, siteData = null) {
    const archivedTags = siteData ? getArchivedTagSet(siteData) : new Set();
    return (post.tags || [])
        .map((tag) => {
            if (archivedTags.has(tag)) {
                return `<a class="post-tag" href="${escapeHtml(getTagRoute(tag))}">${escapeHtml(tag)}</a>`;
            }
            return `<span class="post-tag">${escapeHtml(tag)}</span>`;
        })
        .join('<span class="post-tag-separator" aria-hidden="true">·</span>');
}

function getFeaturedPost(siteData) {
    return siteData.posts.find((post) => post.category === 'post') || siteData.posts[0] || null;
}

function renderBlogCoverImage(post, title, options = {}) {
    const source = post.cover || '/assets/blog_bg.jpeg';
    const keepAnimated = post.animatedPreview === true && isAnimatedCover(source);
    const preview = getBlogCoverPreviewUrl(post.id, options.variant);
    const autoplaySource = keepAnimated
        ? ` data-autoplay-src="${escapeHtml(source)}"`
        : '';
    const animatedSource = !keepAnimated && isAnimatedCover(source)
        ? ` data-animated-src="${escapeHtml(source)}"`
        : '';
    const loading = options.eager ? 'eager' : 'lazy';
    const fetchPriority = options.eager ? ' fetchpriority="high"' : '';
    const alt = options.alt || `${title} cover image`;

    return `<img src="${escapeHtml(preview)}" data-blog-cover data-preview-src="${escapeHtml(preview)}"${autoplaySource}${animatedSource} alt="${escapeHtml(alt)}" loading="${loading}" decoding="async"${fetchPriority}>`;
}

function renderPostPreview(post, lang, siteData) {
    const title = getPostTitle(post, lang);
    const description = getPostDescription(post, lang);
    const seriesTitle = getSeriesTitle(siteData, post.series, lang);
    const url = getPostLanguageRoute(post, lang);
    const tagsHtml = renderTags(post, siteData);

    return `
        <article class="post-preview" data-post-id="${escapeHtml(post.id)}" data-post-category="${escapeHtml(post.category)}">
            <div class="post-card-link">
                <a href="${escapeHtml(url)}" class="post-card-cover" aria-label="Read ${escapeHtml(title)}">
                    ${renderBlogCoverImage(post, title)}
                </a>
                <div class="post-card-body">
                    <div class="post-card-eyebrow">
                        <span>${escapeHtml(seriesTitle)}</span>
                    </div>
                    <h3 class="post-title" lang="${escapeHtml(getTextHtmlLang(title, lang))}"><a href="${escapeHtml(url)}">${escapeHtml(title)}</a></h3>
                    ${description ? `<p class="post-subtitle" lang="${escapeHtml(getTextHtmlLang(description, lang))}">${escapeHtml(description)}</p>` : ''}
                    ${tagsHtml ? `<div class="post-tag-row">${tagsHtml}</div>` : ''}
                    <p class="post-meta">
                        <time datetime="${escapeHtml(post.date)}">${escapeHtml(formatDate(post.date, lang))}</time>
                    </p>
                </div>
            </div>
            <nav class="visually-hidden" aria-label="Available languages">
                ${renderLanguageLinks(post)}
            </nav>
        </article>
    `;
}

function renderFeaturedPost(siteData, lang = 'eng') {
    const featuredPost = getFeaturedPost(siteData);
    if (!featuredPost) {
        return '';
    }

    const title = getPostTitle(featuredPost, lang);
    const description = getPostDescription(featuredPost, lang);
    const seriesTitle = getSeriesTitle(siteData, featuredPost.series, lang);
    const url = getPostLanguageRoute(featuredPost, lang);
    const tagsHtml = renderTags(featuredPost, siteData);

    return `
            <article class="blog-feature-card" data-post-id="${escapeHtml(featuredPost.id)}">
                <div class="blog-feature-label">
                    <span data-feature-label>Featured</span>
                </div>
                <a class="blog-feature-cover" href="${escapeHtml(url)}" aria-label="Read ${escapeHtml(title)}">
                    ${renderBlogCoverImage(featuredPost, title, { eager: true })}
                </a>
                <div class="blog-feature-copy">
                    <div class="blog-feature-meta">
                        <span data-feature-series>${escapeHtml(seriesTitle)}</span>
                    </div>
                    <h2 lang="${escapeHtml(getTextHtmlLang(title, lang))}"><a href="${escapeHtml(url)}">${escapeHtml(title)}</a></h2>
                    ${description ? `<p lang="${escapeHtml(getTextHtmlLang(description, lang))}">${escapeHtml(description)}</p>` : ''}
                    <time class="blog-feature-date" datetime="${escapeHtml(featuredPost.date)}">${escapeHtml(formatDate(featuredPost.date, lang))}</time>
                    ${tagsHtml ? `<div class="post-tag-row">${tagsHtml}</div>` : ''}
                    <a class="blog-feature-read" href="${escapeHtml(url)}" aria-label="Read ${escapeHtml(title)}">
                        <span data-feature-read-label>Read post</span>
                        ${renderSiteIcon('arrow-right')}
                    </a>
                </div>
                <nav class="visually-hidden" aria-label="Available languages">
                    ${renderLanguageLinks(featuredPost)}
                </nav>
            </article>
    `;
}

function renderSeriesGroups(siteData, lang = 'eng') {
    const postsBySeries = {};
    siteData.posts.forEach((post) => {
        if (!post.series || !post.languages.includes(lang)) {
            return;
        }
        if (!postsBySeries[post.series]) {
            postsBySeries[post.series] = [];
        }
        postsBySeries[post.series].push(post);
    });

    return Object.entries(postsBySeries)
        .sort(([, firstPosts], [, secondPosts]) => new Date(secondPosts[0].date) - new Date(firstPosts[0].date))
        .map(([seriesId, posts]) => {
            const seriesTitle = getSeriesTitle(siteData, seriesId, lang);
            const latestPost = posts[0];
            const itemsHtml = posts.map((post) => {
                const title = getPostTitle(post, lang);
                return `
                        <li data-series-post-id="${escapeHtml(post.id)}">
                            <a href="${escapeHtml(getPostLanguageRoute(post, lang))}" lang="${escapeHtml(getTextHtmlLang(title, lang))}">${escapeHtml(title)}</a>
                            <span class="post-meta-sm"><time datetime="${escapeHtml(post.date)}">${escapeHtml(formatShortDate(post.date, lang))}</time></span>
                        </li>
                `;
            }).join('');
            const seriesRoute = getSeriesRoute(seriesId);

            return `
                    <article class="series-group" data-series-id="${escapeHtml(seriesId)}">
                        <div class="series-card-header">
                            <span class="series-card-kicker">Series</span>
                            <h3 class="series-title" lang="${escapeHtml(getTextHtmlLang(seriesTitle, lang))}">
                                <a href="${escapeHtml(seriesRoute)}">
                                    <span data-series-title>${escapeHtml(seriesTitle)}</span>
                                    ${renderSiteIcon('arrow-up-right')}
                                </a>
                            </h3>
                            <div class="series-card-meta">
                                <span data-series-count>${posts.length} items</span>
                                <span><span data-series-latest-label>Latest</span> <time datetime="${escapeHtml(latestPost.date)}">${escapeHtml(formatShortDate(latestPost.date, lang))}</time></span>
                            </div>
                        </div>
                        <ol class="series-post-list">
                            ${itemsHtml}
                        </ol>
                    </article>
            `;
        }).join('');
}

function getPostAlternates(post) {
    const alternates = post.languages.map((lang) => ({
        lang,
        hreflang: getLanguageMeta(lang).hreflang,
        href: getPostCanonicalUrl(post, lang)
    }));

    if (post.languages.includes('eng')) {
        alternates.push({
            lang: 'x-default',
            hreflang: 'x-default',
            href: getPostCanonicalUrl(post, 'eng')
        });
    }

    return alternates;
}

function getPostsForLanguage(siteData, lang) {
    return siteData.posts.filter((post) => post.languages.includes(lang));
}

function getAdjacentLanguagePosts(siteData, post, lang) {
    const posts = getPostsForLanguage(siteData, lang);
    const currentIndex = posts.findIndex((entry) => entry.id === post.id);
    return {
        newer: currentIndex > 0 ? posts[currentIndex - 1] : null,
        older: currentIndex >= 0 && currentIndex < posts.length - 1 ? posts[currentIndex + 1] : null
    };
}

function getAutomaticRelatedItems(siteData, post, lang = 'eng', limit = 4) {
    return siteData.posts
        .filter((entry) => entry.id !== post.id && entry.languages.includes(lang))
        .map((entry) => {
            const sharedTags = (entry.tags || []).filter((tag) => (post.tags || []).includes(tag)).length;
            const score = (entry.series && entry.series === post.series ? 4 : 0)
                + (sharedTags * 2)
                + (entry.category === post.category ? 1 : 0);
            return { entry, score };
        })
        .filter((item) => item.score > 0)
        .sort((a, b) => b.score - a.score || new Date(b.entry.date) - new Date(a.entry.date))
        .slice(0, limit)
        .map((item) => ({
            type: 'post',
            post: item.entry
        }));
}

function getExplicitRelatedItems(siteData, post, lang = 'eng') {
    const relatedItems = post.relatedByLanguage?.[lang] || post.relatedByLanguage?.eng || [];
    const seenPostIds = new Set([post.id]);
    const seenUrls = new Set();

    return relatedItems
        .map((item) => {
            if (item.type === 'post') {
                const relatedPost = siteData.postById[item.postId];
                if (!relatedPost || seenPostIds.has(relatedPost.id)) {
                    return null;
                }
                seenPostIds.add(relatedPost.id);
                return {
                    type: 'post',
                    post: relatedPost
                };
            }

            if (item.type === 'external' && item.url && item.title) {
                const url = getAbsoluteUrl(item.url);
                if (seenUrls.has(url)) {
                    return null;
                }
                seenUrls.add(url);
                return {
                    type: 'external',
                    title: item.title,
                    url
                };
            }

            return null;
        })
        .filter(Boolean);
}

function getRelatedItems(siteData, post, lang = 'eng', limit = 4) {
    const explicitItems = getExplicitRelatedItems(siteData, post, lang);
    if (explicitItems.length > 0) {
        return explicitItems;
    }

    return getAutomaticRelatedItems(siteData, post, lang, limit);
}

function truncateNavTitle(title) {
    return title.length > 58 ? `${title.slice(0, 55)}...` : title;
}

function renderPostNavCard({ href, type, label, title }) {
    const isPrevious = type === 'older';
    const directionIcon = renderSiteIcon(isPrevious ? 'arrow-left' : 'arrow-right');
    const kicker = isPrevious
        ? `${directionIcon} ${escapeHtml(label)}`
        : `${escapeHtml(label)} ${directionIcon}`;

    return `
                <a class="post-nav-card is-${escapeHtml(type)}" href="${escapeHtml(href)}" aria-label="${escapeHtml(`${label}: ${title}`)}">
                    <span class="post-nav-kicker">
                        ${kicker}
                    </span>
                    <strong>${escapeHtml(truncateNavTitle(title))}</strong>
                </a>
    `;
}

function renderSeriesPostNavigation(siteData, post, lang) {
    if (!post.series) {
        return '';
    }

    const postsInSeries = siteData.posts
        .filter((entry) => entry.series === post.series && entry.languages.includes(lang))
        .sort((a, b) => new Date(b.date) - new Date(a.date));
    const currentIndex = postsInSeries.findIndex((entry) => entry.id === post.id);

    if (postsInSeries.length <= 1 || currentIndex < 0) {
        return '';
    }

    const olderPost = postsInSeries[currentIndex + 1] || null;
    const newerPost = postsInSeries[currentIndex - 1] || null;
    let navHtml = '<nav class="post-nav-grid" aria-label="Post navigation">';

    if (olderPost) {
        navHtml += renderPostNavCard({
            href: getPostLanguageRoute(olderPost, lang),
            type: 'older',
            label: 'Previous Post',
            title: getPostTitle(olderPost, lang)
        });
    } else if (newerPost) {
        navHtml += '<span class="post-nav-spacer" aria-hidden="true"></span>';
    }

    if (newerPost) {
        navHtml += renderPostNavCard({
            href: getPostLanguageRoute(newerPost, lang),
            type: 'next',
            label: 'Next Post',
            title: getPostTitle(newerPost, lang)
        });
    } else {
        const recommendation = Object.keys(siteData.series || {})
            .filter((seriesId) => seriesId !== post.series)
            .map((seriesId) => ({
                seriesId,
                post: siteData.posts
                    .filter((entry) => entry.series === seriesId && entry.languages.includes(lang))
                    .sort((a, b) => new Date(b.date) - new Date(a.date))[0] || null
            }))
            .filter((entry) => entry.post)
            .sort((a, b) => new Date(b.post.date) - new Date(a.post.date))[0];

        if (recommendation) {
            navHtml += renderPostNavCard({
                href: getPostLanguageRoute(recommendation.post, lang),
                type: 'explore',
                label: 'Explore Series',
                title: getSeriesTitle(siteData, recommendation.seriesId, lang)
            });
        } else if (olderPost) {
            navHtml += '<span class="post-nav-spacer" aria-hidden="true"></span>';
        }
    }

    navHtml += '</nav>';
    return navHtml;
}

function renderChronologicalPostNavigation(siteData, post, lang = 'eng') {
    const seriesNavigation = renderSeriesPostNavigation(siteData, post, lang);
    if (seriesNavigation) {
        return seriesNavigation;
    }

    const adjacent = getAdjacentLanguagePosts(siteData, post, lang);
    const labels = lang === 'kor'
        ? { older: 'Previous Post', newer: 'Next Post' }
        : { older: 'Previous Post', newer: 'Next Post' };

    let navHtml = '<nav class="post-nav-grid" aria-label="Post navigation">';
    if (adjacent.older) {
        navHtml += renderPostNavCard({
            href: getPostLanguageRoute(adjacent.older, lang),
            type: 'older',
            label: labels.older,
            title: getPostTitle(adjacent.older, lang)
        });
    } else if (adjacent.newer) {
        navHtml += '<span class="post-nav-spacer" aria-hidden="true"></span>';
    }

    if (adjacent.newer) {
        navHtml += renderPostNavCard({
            href: getPostLanguageRoute(adjacent.newer, lang),
            type: 'next',
            label: labels.newer,
            title: getPostTitle(adjacent.newer, lang)
        });
    } else if (adjacent.older) {
        navHtml += '<span class="post-nav-spacer" aria-hidden="true"></span>';
    }

    if (!adjacent.older && !adjacent.newer) {
        navHtml += renderPostNavCard({
            href: '/blogs/',
            type: 'explore',
            label: 'Blog',
            title: 'All posts'
        });
    }

    navHtml += '</nav>';
    return navHtml;
}

function renderSeriesNavigation(siteData, post, lang = 'eng') {
    if (!post.series) {
        return '';
    }

    const postsInSeries = siteData.posts
        .filter((entry) => entry.series === post.series && entry.languages.includes(lang))
        .sort((a, b) => new Date(b.date) - new Date(a.date));

    if (postsInSeries.length <= 1) {
        return '';
    }

    const seriesTitle = getSeriesTitle(siteData, post.series, lang);
    const currentIndex = postsInSeries.findIndex((entry) => entry.id === post.id);
    const listItems = postsInSeries.map((entry, index) => {
        const title = getPostTitle(entry, lang);
        const date = entry.date ? entry.date.replace(/-/g, '.') : '';
        const number = String(index + 1).padStart(2, '0');

        if (entry.id === post.id) {
            return `
                    <li class="series-post is-current" aria-current="page">
                        <span class="series-post-index">${number}</span>
                        <span class="series-post-copy">
                            <strong>${escapeHtml(title)}</strong>
                            ${date ? `<span>${escapeHtml(date)}</span>` : ''}
                        </span>
                    </li>
            `;
        }

        return `
                <li class="series-post">
                    <a href="${escapeHtml(getPostLanguageRoute(entry, lang))}">
                        <span class="series-post-index">${number}</span>
                        <span class="series-post-copy">
                            <strong>${escapeHtml(title)}</strong>
                            ${date ? `<span>${escapeHtml(date)}</span>` : ''}
                        </span>
                    </a>
                </li>
        `;
    }).join('');

    return `
            <details class="series-card">
                <summary class="series-summary">
                    <span class="series-icon" aria-hidden="true">${renderSiteIcon('collection')}</span>
                    <span class="series-summary-copy">
                        <span class="series-kicker">Series</span>
                        <strong>${escapeHtml(seriesTitle)}</strong>
                    </span>
                    <span class="series-meta">${currentIndex + 1} / ${postsInSeries.length}</span>
                    <span class="series-toggle">
                        <span>${postsInSeries.length} posts</span>
                        ${renderSiteIcon('chevron-down')}
                    </span>
                </summary>
                <div class="series-body">
                    <ol class="series-list">${listItems}</ol>
                </div>
            </details>
    `;
}

function renderRelatedPosts(siteData, post, lang = 'eng') {
    const relatedItems = getRelatedItems(siteData, post, lang, 4);
    if (relatedItems.length === 0) {
        return '';
    }

    const items = relatedItems.map((item) => {
        if (item.type === 'external') {
            const title = item.title;
            return `
                            <article class="related-post-card">
                                <a href="${escapeHtml(item.url)}" target="_blank" rel="noopener noreferrer" aria-label="Open ${escapeHtml(title)}">
                                    <strong>${escapeHtml(title)}</strong>
                                    <span>External</span>
                                </a>
                            </article>
            `;
        }

        const entry = item.post;
        const title = getPostTitle(entry, lang);
        return `
                            <article class="related-post-card">
                                <a href="${escapeHtml(getPostLanguageRoute(entry, lang))}" aria-label="Read ${escapeHtml(title)}">
                                    <strong>${escapeHtml(title)}</strong>
                                    <time datetime="${escapeHtml(entry.date)}">${escapeHtml(formatShortDate(entry.date, lang))}</time>
                                </a>
                            </article>
        `;
    }).join('');

    return `
                    <section class="related-posts" aria-labelledby="related-posts-heading">
                        <h2 id="related-posts-heading">Related Posts</h2>
                        <div class="related-post-grid">
                            ${items}
                        </div>
                    </section>
    `;
}

function serializeStructuredData(value) {
    return JSON.stringify(value).replace(/</g, '\\u003c');
}

module.exports = {
    LANGUAGE_META,
    escapeHtml,
    escapeXml,
    formatDate,
    formatShortDate,
    getAbsoluteUrl,
    getFeaturedPost,
    getLanguageMeta,
    getPostAlternates,
    getPostCanonicalUrl,
    getPostDescription,
    getPostLanguageRoute,
    getPostTitle,
    getSeriesTitle,
    isValidDateString,
    renderChronologicalPostNavigation,
    renderFeaturedPost,
    renderLanguageLinks,
    renderPostPreview,
    renderRelatedPosts,
    renderSeriesGroups,
    renderSeriesNavigation,
    renderTags,
    serializeStructuredData,
    stripHtml,
    truncateText
};
