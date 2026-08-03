const { SITE_URL } = require('./site-config');
const { isFromArchive } = require('./site-data');
const { render: renderSiteIcon } = require('../../assets/js/site-icons');
const {
    escapeHtml,
    renderPostPreview,
    serializeStructuredData
} = require('./seo-utils');

function renderArchiveEntries(posts, siteData) {
    let archiveStarted = false;

    return posts.map((post) => {
        const fromArchive = isFromArchive(post, siteData);
        const fromArchiveHeading = fromArchive && !archiveStarted
            ? `
                        <div class="blog-home-era-break">
                            <h3>
                                <span class="blog-home-era-index" aria-hidden="true">02</span>
                                <span>From the Archive</span>
                            </h3>
                            <span class="blog-home-era-rule" aria-hidden="true"></span>
                        </div>`
            : '';

        if (fromArchive) {
            archiveStarted = true;
        }

        return `${fromArchiveHeading}${renderPostPreview(post, 'eng', siteData, {
            mediaSide: fromArchive ? 'left' : 'right'
        })}`;
    }).join('');
}

function renderArchivePage({ title, description, canonicalPath, posts, siteData, archiveKind = 'tag' }) {
    const canonicalUrl = `${SITE_URL}${canonicalPath}`;
    const structuredData = {
        '@context': 'https://schema.org',
        '@type': 'CollectionPage',
        name: title,
        description,
        url: canonicalUrl,
        isPartOf: {
            '@type': 'Blog',
            name: "Hwan Heo's Blog",
            url: `${SITE_URL}/blogs/`
        }
    };

    const postHtml = renderArchiveEntries(posts, siteData);
    const archiveContext = archiveKind === 'series' ? 'Series Index' : 'Topic Index';

    return `<!DOCTYPE html>
<html lang="en">
    <head>
        <meta charset="utf-8" />
        <meta name="viewport" content="width=device-width, initial-scale=1, shrink-to-fit=no" />
        <meta name="description" content="${escapeHtml(description)}" />
        <meta name="author" content="Hwan Heo" />
        <title>${escapeHtml(title)} | Hwan Heo</title>
        <link rel="canonical" href="${canonicalUrl}" />
        <meta property="og:type" content="website" />
        <meta property="og:title" content="${escapeHtml(title)}" />
        <meta property="og:description" content="${escapeHtml(description)}" />
        <meta property="og:url" content="${canonicalUrl}" />
        <meta property="og:image" content="${SITE_URL}/assets/image_fx_.jpg" />
        <meta name="twitter:card" content="summary_large_image" />
        <meta name="twitter:title" content="${escapeHtml(title)}" />
        <meta name="twitter:description" content="${escapeHtml(description)}" />
        <meta name="twitter:image" content="${SITE_URL}/assets/image_fx_.jpg" />
        <script type="application/ld+json">${serializeStructuredData(structuredData)}</script>
        <link rel="icon" type="image/x-icon" href="/assets/favicon.ico" />
        <link href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500&family=Inter:wght@400;500;600&family=Manrope:wght@500;600;700;800&family=Noto+Sans+KR:wght@400;500;600;700&display=swap" rel="stylesheet" />
        <link href="/blogs/css/blog.css" rel="stylesheet" />
        <link href="/blogs/css/sidebar.css" rel="stylesheet" />
        <link href="/blogs/css/typography.css" rel="stylesheet" />
        <link href="/assets/css/site-icons.css" rel="stylesheet" />
        <script src="/assets/js/site-icons.js"></script>
        <script>
            (function() {
                try {
                    const storedTheme = localStorage.getItem('blog-theme');
                    const theme = storedTheme === 'dark' || storedTheme === 'light' ? storedTheme : 'light';
                    document.documentElement.dataset.theme = theme;
                } catch (error) {}
            })();
        </script>
    </head>
    <body class="blog-home-page blog-archive-page">
        <main class="main blog-home-main">
            <nav class="blog-home-topbar" id="blogHomeTopbar" aria-label="Blog utilities">
                <a class="blog-home-back" href="/blogs/" aria-label="Back to blog home">
                    ${renderSiteIcon('arrow-left')}
                    <span>Blog</span>
                </a>
                <div class="blog-home-toolbar-actions">
                    <form id="blog-home-search-form" class="blog-home-search" role="search" data-collapsible-search>
                        <label class="visually-hidden" for="blog-home-search-input">Search posts</label>
                        <input type="text" id="blog-home-search-input" placeholder="Search...">
                        <button type="submit" aria-label="Search">
                            ${renderSiteIcon('search')}
                        </button>
                    </form>
                    <button class="blog-theme-toggle" type="button" data-theme-toggle aria-label="Toggle color theme" aria-pressed="false">
                        ${renderSiteIcon('moon-stars', { className: 'theme-toggle-icon' })}
                    </button>
                    <button id="lang-toggle-main" class="blog-home-language" type="button" aria-label="한국어로 전환" title="한국어로 전환">가</button>
                </div>
            </nav>

            <header class="masthead blog-home-hero blog-editorial-utility-hero">
                <div class="blog-shell blog-hero-shell">
                    <div class="blog-editorial-utility-copy">
                        <div class="blog-editorial-imprint">
                            <div class="blog-editorial-brandline" lang="en">
                                <span class="blog-editorial-index">00</span>
                                <span class="blog-editorial-separator" aria-hidden="true">/</span>
                                <span class="blog-home-kicker">${escapeHtml(archiveContext)}</span>
                            </div>
                            <span class="blog-editorial-edition" lang="en">Hwan's Blog</span>
                        </div>
                        <h1>${escapeHtml(title)}</h1>
                        <p>${escapeHtml(description)}</p>
                    </div>
                </div>
            </header>

            <div class="blog-shell blog-home-container">
                <section class="blog-home-archive blog-editorial-archive" aria-labelledby="archive-posts-heading">
                    <div class="blog-home-archive-head blog-editorial-section-head">
                        <h2 id="archive-posts-heading">
                            <span class="blog-home-archive-index" aria-hidden="true">01</span>
                            <span>Articles</span>
                        </h2>
                        <span class="blog-home-archive-rule" aria-hidden="true"></span>
                        <span class="blog-archive-count">${posts.length} ${posts.length === 1 ? 'article' : 'articles'}</span>
                    </div>
                    <div class="blog-home-tab-content">
                        ${postHtml}
                    </div>
                </section>
            </div>
        </main>
        <script src="/blogs/js/site-data-client.js"></script>
        <script src="/blogs/js/blog-cover-media.js"></script>
        <script src="/blogs/js/blog-shell.js"></script>
        <script src="/blogs/js/theme-toggle.js"></script>
        <script>
            initBlogShell({ formSelector: '#blog-home-search-form', inputSelector: '#blog-home-search-input' });
            (function() {
                const button = document.getElementById('lang-toggle-main');
                if (!button) {
                    return;
                }

                function currentLanguage() {
                    try {
                        return localStorage.getItem('language') === 'kor' ? 'kor' : 'eng';
                    } catch (error) {
                        return 'eng';
                    }
                }

                function setLanguage(language) {
                    const targetLabel = language === 'eng' ? '한국어로 전환' : 'Switch to English';
                    document.documentElement.lang = language === 'kor' ? 'ko' : 'en';
                    button.textContent = language === 'eng' ? '가' : 'A';
                    button.setAttribute('aria-label', targetLabel);
                    button.setAttribute('title', targetLabel);
                    updateArchiveCards(language);
                }

                function formatArchiveDate(value, language) {
                    return new Date(value + 'T00:00:00').toLocaleDateString(
                        language === 'kor' ? 'ko-KR' : 'en-US',
                        {
                            year: 'numeric',
                            month: language === 'kor' ? 'numeric' : 'long',
                            day: 'numeric'
                        }
                    );
                }

                async function updateArchiveCards(language) {
                    if (!window.siteDataClient) {
                        return;
                    }

                    try {
                        const siteData = await window.siteDataClient.loadSiteData();
                        document.querySelectorAll('.post-preview[data-post-id]').forEach(function(card) {
                            const post = siteData.postById[card.dataset.postId];
                            if (!post) {
                                return;
                            }

                            const title = window.siteDataClient.getPostTitle(post, language);
                            const subtitleText = window.siteDataClient.getPostSubtitle(post, language);
                            const url = window.siteDataClient.getPostUrl(post, language);
                            const titleLink = card.querySelector('.post-title a');
                            const coverLink = card.querySelector('.post-card-cover');
                            const coverImage = card.querySelector('img[data-blog-cover]');
                            const subtitle = card.querySelector('.post-subtitle');
                            const series = card.querySelector('[data-post-series]');
                            const time = card.querySelector('.post-meta time');
                            const textLanguage = language === 'kor' ? 'ko' : 'en';

                            if (titleLink) {
                                titleLink.textContent = title;
                                titleLink.href = url;
                                titleLink.closest('.post-title')?.setAttribute('lang', textLanguage);
                            }

                            if (coverLink) {
                                coverLink.href = url;
                                coverLink.setAttribute('aria-label', 'Read ' + title);
                            }

                            if (coverImage) {
                                coverImage.alt = title + ' cover image';
                            }

                            if (subtitle) {
                                subtitle.textContent = subtitleText;
                                subtitle.setAttribute('lang', textLanguage);
                            }

                            if (series) {
                                series.textContent = siteData.series[post.series]?.[language]
                                    || siteData.series[post.series]?.eng
                                    || 'Series';
                            }

                            if (time) {
                                time.textContent = formatArchiveDate(post.date, language);
                            }

                        });
                    } catch (error) {
                        console.warn('Failed to update archive language:', error);
                    }
                }

                setLanguage(currentLanguage());
                button.addEventListener('click', function() {
                    const nextLanguage = currentLanguage() === 'eng' ? 'kor' : 'eng';
                    try {
                        localStorage.setItem('language', nextLanguage);
                    } catch (error) {}
                    setLanguage(nextLanguage);
                });
            })();
        </script>
    </body>
</html>`;
}

module.exports = {
    renderArchivePage
};
