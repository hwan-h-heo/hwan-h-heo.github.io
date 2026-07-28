const { SITE_URL } = require('./site-config');
const {
    escapeHtml,
    getAbsoluteUrl,
    getPostAlternates,
    getPostCanonicalUrl,
    getPostDescription,
    getPostLanguageRoute,
    getPostTitle,
    renderBreadcrumbs,
    renderChronologicalPostNavigation,
    renderRelatedPosts,
    renderSeriesNavigation,
    renderTags,
    serializeStructuredData
} = require('./seo-utils');

function serializeForScript(value) {
    return JSON.stringify(value, null, 2).replace(/</g, '\\u003c');
}

function resolveOgImage(post, featuredPortfolioPosts) {
    const configuredImage = post.socialImage || post.cover || '';
    if (configuredImage && !/\.svg(?:[?#]|$)/i.test(configuredImage)) {
        return getAbsoluteUrl(configuredImage);
    }

    const featured = featuredPortfolioPosts.find((item) => item.id === post.id);
    if (featured?.teaserImage && !/\.svg(?:[?#]|$)/i.test(featured.teaserImage)) {
        return getAbsoluteUrl(featured.teaserImage);
    }

    return `${SITE_URL}/assets/image_fx_.jpg`;
}

function renderPostPage({ post, lang, contentHtml, metaDescription, readingTime, siteData }) {
    const title = getPostTitle(post, lang);
    const seoTitle = post[`seoTitle_${lang}`] || post.seoTitle || title;
    const defaultEnglishSeoTitle = post.seoTitle_eng || post.seoTitle || post.title_eng;
    const metadataTitle = lang !== 'eng' && seoTitle === defaultEnglishSeoTitle
        ? `${seoTitle} (Korean)`
        : seoTitle;
    const pageTitle = `${metadataTitle} | Hwan Heo`;
    const date = new Date(post.date).toLocaleDateString('en-US', {
        year: 'numeric',
        month: 'long',
        day: 'numeric'
    });
    const isoDate = `${post.date}T00:00:00.000Z`;
    const updatedIsoDate = `${post.updated || post.date}T00:00:00.000Z`;
    const slug = lang === 'eng' ? post.slug : `${post.slug}-kor`;
    const canonicalUrl = getPostCanonicalUrl(post, lang);
    const ogImage = resolveOgImage(post, siteData.featuredPortfolioPosts);
    const alternateLang = lang === 'eng' ? 'kor' : 'eng';
    const hasAlternateLang = post.languages.includes(alternateLang);
    const alternateHref = hasAlternateLang
        ? getPostLanguageRoute(post, alternateLang)
        : null;
    const alternateLinksHtml = getPostAlternates(post)
        .map((alternate) => `    <link rel="alternate" hreflang="${alternate.hreflang}" href="${alternate.href}" />`)
        .join('\n');
    const breadcrumbsHtml = renderBreadcrumbs(siteData, post, lang);
    const staticSeriesNavigation = renderSeriesNavigation(siteData, post, lang);
    const staticPostNavigation = renderChronologicalPostNavigation(siteData, post, lang);
    const relatedPostsHtml = renderRelatedPosts(siteData, post, lang);

    const keywords = [
        post.category || 'blog',
        post.series || 'article',
        ...(post.tags || []),
        '3D graphics',
        'computer vision',
        'machine learning'
    ].join(', ');
    const coverImage = post.cover || '/assets/image_fx_.jpg';
    const tagHtml = (post.tags || []).length
        ? `<div class="post-tags">${renderTags(post, siteData)}</div>`
        : '';

    const structuredData = {
        '@context': 'https://schema.org',
        '@type': 'BlogPosting',
        headline: title,
        description: metaDescription,
        url: canonicalUrl,
        mainEntityOfPage: canonicalUrl,
        image: [ogImage],
        author: {
            '@type': 'Person',
            name: 'Hwan Heo',
            url: SITE_URL
        },
        publisher: {
            '@type': 'Person',
            name: 'Hwan Heo',
        },
        datePublished: post.date,
        dateModified: post.updated || post.date,
        inLanguage: lang === 'eng' ? 'en' : 'ko'
    };

    const clientSiteData = {
        posts: siteData.posts.map((entry) => ({
            id: entry.id,
            title_eng: entry.title_eng,
            title_kor: entry.title_kor,
            subtitle_eng: entry.subtitle_eng,
            subtitle_kor: entry.subtitle_kor,
            description_eng: entry.description_eng,
            description_kor: entry.description_kor,
            tags: entry.tags || [],
            cover: entry.cover,
            status: entry.status,
            updated: entry.updated,
            date: entry.date,
            category: entry.category,
            series: entry.series,
            languages: entry.languages,
            slug: entry.slug
        })),
        series: siteData.series
    };

    return `<!DOCTYPE html>
<html lang="${lang === 'eng' ? 'en' : 'ko'}">
<head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1, shrink-to-fit=no" />
    <meta name="description" content="${escapeHtml(metaDescription)}" />
    <meta name="keywords" content="${escapeHtml(keywords)}" />
    <meta name="author" content="Hwan Heo" />
    <link rel="canonical" href="${canonicalUrl}" />
${alternateLinksHtml}
    <meta property="og:type" content="article" />
    <meta property="og:url" content="${canonicalUrl}" />
    <meta property="og:title" content="${escapeHtml(metadataTitle)}" />
    <meta property="og:description" content="${escapeHtml(metaDescription)}" />
    <meta property="og:image" content="${ogImage}" />
    <meta property="og:site_name" content="HwanHeo's Blog" />
    <meta property="article:published_time" content="${isoDate}" />
    <meta property="article:modified_time" content="${updatedIsoDate}" />
    <meta property="article:author" content="Hwan Heo" />
    <meta name="twitter:card" content="summary_large_image" />
    <meta name="twitter:url" content="${canonicalUrl}" />
    <meta name="twitter:title" content="${escapeHtml(metadataTitle)}" />
    <meta name="twitter:description" content="${escapeHtml(metaDescription)}" />
    <meta name="twitter:image" content="${ogImage}" />
    <script type="application/ld+json">${serializeStructuredData(structuredData)}</script>

    <title>${escapeHtml(pageTitle)}</title>
    <link rel="icon" type="image/x-icon" href="/assets/favicon.ico" />

    <script type="importmap">
        {
            "imports": {
                "three": "/vendor/three/build/three.module.js",
                "three/addons/": "/vendor/three/examples/jsm/",
                "GaussianSplats3D": "https://unpkg.com/@mkkellogg/gaussian-splats-3d@0.4.0/build/gaussian-splats-3d.module.js"
            }
        }
    </script>

    <script src="/vendor/tween/tween.umd.js"></script>
    <link href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500&family=Inter:wght@400;500;600&family=Manrope:wght@500;600;700;800&family=Noto+Sans+KR:wght@400;500;600;700&display=swap" rel="stylesheet" />
    <link href="https://cdnjs.cloudflare.com/ajax/libs/prism/1.28.0/themes/prism.min.css" rel="stylesheet" />
    <link href="/blogs/css/used.css" rel="stylesheet" />
    <link href="/blogs/css/typography.css" rel="stylesheet" />
    <link href="/blogs/css/blog_post_specific.css" rel="stylesheet" />
    <link href="/blogs/css/code-copy.css" rel="stylesheet" />
    <link href="/blogs/css/scroll-progress.css" rel="stylesheet" />
    <link href="/assets/vendor/bootstrap-icons/bootstrap-icons.css" rel="stylesheet" />
    <script type="module" src="https://unpkg.com/@google/model-viewer/dist/model-viewer.min.js"></script>
    <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/katex@0.16.9/dist/katex.min.css">
    <script defer src="https://cdn.jsdelivr.net/npm/katex@0.16.9/dist/katex.min.js"></script>
    <script defer src="https://cdn.jsdelivr.net/npm/katex@0.16.9/dist/contrib/auto-render.min.js"></script>

    <style>
        .post-tags { display: flex; flex-wrap: wrap; gap: 0.4rem; margin-top: 1rem; }
        .post-tags span { display: inline-flex; align-items: center; border: 1px solid rgba(255,255,255,0.6); color: #fff; border-radius: 999px; padding: 0.16rem 0.55rem; font-size: 0.72rem; background: rgba(0,0,0,0.18); }
    </style>

    <script async src="https://www.googletagmanager.com/gtag/js?id=G-RF7ETSKPK9"></script>
    <script>
        window.dataLayer = window.dataLayer || [];
        function gtag(){dataLayer.push(arguments);}
        gtag('js', new Date());
        gtag('config', 'G-RF7ETSKPK9');
    </script>
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
<body>
    <nav class="navbar navbar-expand-lg navbar-light" id="mainNav">
        <div class="container px-4 px-lg-5">
            <a class="navbar-brand" href="/blogs/">Hwan's Blog</a>
            <button class="navbar-toggler" type="button" data-bs-toggle="collapse" data-bs-target="#navbarResponsive">
                Menu <i class="bi bi-list"></i>
            </button>
            <div class="collapse navbar-collapse" id="navbarResponsive">
                <ul class="navbar-nav ms-auto py-4 py-lg-0">
                    <li class="nav-item"><button class="btn nav-link px-lg-3 py-3 py-lg-4 blog-theme-toggle" type="button" data-theme-toggle aria-label="Toggle color theme" aria-pressed="false"><i class="bi bi-moon-stars" aria-hidden="true"></i></button></li>
                    ${alternateHref ? `<li class="nav-item"><a href="${alternateHref}" class="btn nav-link px-lg-3 py-3 py-lg-4" data-language-target="${alternateLang}" style="font-size:0.7rem">${alternateLang === 'eng' ? 'ENG' : 'KOR'}</a></li>` : ''}
                </ul>
            </div>
        </div>
    </nav>

    <header class="masthead" style="background-image: url('${coverImage}')">
        <div class="container position-relative px-4 px-lg-5">
            <div class="row gx-4 gx-lg-5 justify-content-center">
                <div class="col-md-10 col-lg-8 col-xl-7">
                    <div class="post-heading">
                        <br/>
                        <h1>${escapeHtml(title)}</h1>
                        <span class="meta">
                            Posted on <time datetime="${escapeHtml(post.date)}">${date}</time>
                            <span style="margin: 0 8px;">•</span>
                            <i class="bi bi-clock" style="margin-right: 4px;"></i>${readingTime.text}
                        </span>
                        ${tagHtml}
                        <hr/>
                    </div>
                </div>
            </div>
        </div>
    </header>

    <article class="mb-4">
        <div class="container px-4 px-lg-5">
            <div class="row gx-4 gx-lg-5 justify-content-center">
                <div class="col-md-10 col-lg-8 col-xl-7">
                    ${breadcrumbsHtml}
                    <div id="series-container">${staticSeriesNavigation}</div>
                </div>
                <div class="col-md-10 col-lg-8 col-xl-7 main-content">
                    ${contentHtml}
                </div>
                <div class="col-md-10 col-lg-8 col-xl-7">
                    ${relatedPostsHtml}
                    <div id="post-navigation" class="mt-4">${staticPostNavigation}</div>
                </div>
            </div>
        </div>
    </article>

    <footer class="border-top">
        <div class="container px-4 px-lg-5">
            <div class="row gx-4 gx-lg-5 justify-content-center">
                <div class="col-md-10 col-lg-8 col-xl-7">
                    <div class="small text-center text-muted fst-italic">Copyright © Hwan Heo</div>
                </div>
            </div>
        </div>
    </footer>

    <script>
        window.siteData = ${serializeForScript(clientSiteData)};
        window.blogPostPageConfig = ${serializeForScript({
            postId: post.id,
            lang,
            alternateLang,
            alternateHref
        })};
    </script>
    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.2.3/dist/js/bootstrap.bundle.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/prismjs@1.28.0/components/prism-core.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/prismjs@1.28.0/plugins/autoloader/prism-autoloader.min.js"></script>
    <script src="/blogs/js/code-copy.js"></script>
    <script src="/blogs/js/theme-toggle.js"></script>
    <script src="/blogs/js/scroll-progress.js"></script>
    <script src="/blogs/js/post-page.js"></script>
</body>
</html>`;
}

module.exports = {
    renderPostPage
};
