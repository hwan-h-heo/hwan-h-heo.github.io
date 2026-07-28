const { SITE_URL } = require('./site-config');
const {
    escapeHtml,
    renderPostPreview,
    serializeStructuredData
} = require('./seo-utils');

function renderArchivePage({ title, description, canonicalPath, posts, siteData }) {
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

    const postHtml = posts.map((post) => renderPostPreview(post, 'eng', siteData)).join('');

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
        <link href="/blogs/css/used.css" rel="stylesheet" />
        <link href="/blogs/css/sidebar.css" rel="stylesheet" />
        <link href="/blogs/css/typography.css" rel="stylesheet" />
        <link href="/assets/vendor/bootstrap-icons/bootstrap-icons.css" rel="stylesheet" />
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
                    <i class="bi bi-arrow-left" aria-hidden="true"></i>
                    <span>Blog Home</span>
                </a>
                <div class="blog-home-toolbar-actions">
                    <form id="blog-home-search-form" class="blog-home-search" role="search">
                        <label class="visually-hidden" for="blog-home-search-input">Search posts</label>
                        <input type="text" id="blog-home-search-input" placeholder="Search...">
                        <button type="submit" aria-label="Search">
                            <i class="bi bi-search" aria-hidden="true"></i>
                        </button>
                    </form>
                    <button class="blog-theme-toggle" type="button" data-theme-toggle aria-label="Toggle color theme" aria-pressed="false">
                        <i class="bi bi-moon-stars" aria-hidden="true"></i>
                    </button>
                    <button id="lang-toggle-main" class="blog-home-language" type="button" aria-label="Switch language preference">KOR</button>
                </div>
            </nav>

            <header class="masthead blog-home-hero" style="background-image: url('/assets/image_fx_.jpg'); background-position: center 34%; background-size: cover;">
                <div class="container position-relative px-4 px-lg-5">
                    <div class="blog-home-hero-grid">
                        <div class="blog-home-hero-copy">
                            <span class="blog-home-kicker">Archive</span>
                            <h1>${escapeHtml(title)}</h1>
                            <p>${escapeHtml(description)}</p>
                        </div>
                    </div>
                </div>
            </header>

            <div class="container px-4 px-lg-5 blog-home-container">
                <section class="blog-home-archive" aria-labelledby="archive-posts-heading">
                    <div class="blog-home-archive-head">
                        <div>
                            <h2 id="archive-posts-heading">Articles</h2>
                        </div>
                    </div>
                    <div class="blog-home-tab-content">
                        ${postHtml}
                    </div>
                </section>
            </div>
        </main>
        <script src="/blogs/js/site-data-client.js"></script>
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
                    document.documentElement.lang = language === 'kor' ? 'ko' : 'en';
                    button.textContent = language === 'eng' ? 'KOR' : 'ENG';
                    updateArchiveCards(language);
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
                            const description = window.siteDataClient.getPostDescription(post, language);
                            const url = window.siteDataClient.getPostUrl(post, language);
                            const titleLink = card.querySelector('.post-title a');
                            const coverLink = card.querySelector('.post-card-cover');
                            const subtitle = card.querySelector('.post-subtitle');

                            if (titleLink) {
                                titleLink.textContent = title;
                                titleLink.href = url;
                            }

                            if (coverLink) {
                                coverLink.href = url;
                                coverLink.setAttribute('aria-label', 'Read ' + title);
                            }

                            if (subtitle) {
                                subtitle.textContent = description;
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
