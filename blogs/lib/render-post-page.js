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

function renderImportMap(runtimeFeatures) {
    const entries = [];

    if (runtimeFeatures.three) {
        entries.push('                "three": "/vendor/three/build/three.module.js"');
        entries.push('                "three/addons/": "/vendor/three/examples/jsm/"');
    }

    if (runtimeFeatures.gaussianSplats) {
        entries.push('                "GaussianSplats3D": "https://unpkg.com/@mkkellogg/gaussian-splats-3d@0.4.0/build/gaussian-splats-3d.module.js"');
    }

    if (entries.length === 0) {
        return '';
    }

    return `    <script type="importmap">
        {
            "imports": {
${entries.join(',\n')}
            }
        }
    </script>`;
}

function renderConditionalHeadAssets(runtimeFeatures) {
    const assets = [];
    const importMap = renderImportMap(runtimeFeatures);

    if (importMap) {
        assets.push(importMap);
    }

    if (runtimeFeatures.prism) {
        assets.push('    <link href="https://cdnjs.cloudflare.com/ajax/libs/prism/1.28.0/themes/prism.min.css" rel="stylesheet" />');
        assets.push('    <link href="/blogs/css/code-copy.css" rel="stylesheet" />');
    }

    if (runtimeFeatures.modelViewer) {
        assets.push('    <script type="module" src="https://unpkg.com/@google/model-viewer/dist/model-viewer.min.js"></script>');
    }

    if (runtimeFeatures.katex) {
        assets.push('    <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/katex@0.16.9/dist/katex.min.css">');
        assets.push('    <script defer src="https://cdn.jsdelivr.net/npm/katex@0.16.9/dist/katex.min.js"></script>');
        assets.push('    <script defer src="https://cdn.jsdelivr.net/npm/katex@0.16.9/dist/contrib/auto-render.min.js"></script>');
    }

    return assets.join('\n');
}

function renderConditionalBodyScripts(runtimeFeatures) {
    const scripts = [];

    if (runtimeFeatures.bootstrap) {
        scripts.push('    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.2.3/dist/js/bootstrap.bundle.min.js"></script>');
    }

    if (runtimeFeatures.prism) {
        scripts.push('    <script src="https://cdn.jsdelivr.net/npm/prismjs@1.28.0/components/prism-core.min.js"></script>');
        scripts.push('    <script src="https://cdn.jsdelivr.net/npm/prismjs@1.28.0/plugins/autoloader/prism-autoloader.min.js"></script>');
        scripts.push('    <script src="/blogs/js/code-copy.js"></script>');
    }

    if (runtimeFeatures.modelViewerTextureToggle) {
        scripts.push('    <script src="/blogs/js/model-viewer-texture-toggle.js"></script>');
    }

    if (runtimeFeatures.tween) {
        scripts.push('    <script src="/vendor/tween/tween.umd.js"></script>');
    }

    return scripts.join('\n');
}

function renderPostPage({ post, lang, contentHtml, metaDescription, readingTime, runtimeFeatures = {}, siteData }) {
    const activeRuntimeFeatures = {
        katex: Boolean(runtimeFeatures.katex),
        prism: Boolean(runtimeFeatures.prism),
        bootstrap: Boolean(runtimeFeatures.bootstrap),
        modelViewer: Boolean(runtimeFeatures.modelViewer),
        modelViewerTextureToggle: Boolean(runtimeFeatures.modelViewerTextureToggle),
        three: Boolean(runtimeFeatures.three),
        tween: Boolean(runtimeFeatures.tween),
        simpleModelViewer: Boolean(runtimeFeatures.simpleModelViewer),
        gaussianSplats: Boolean(runtimeFeatures.gaussianSplats)
    };
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
    const searchLabel = lang === 'kor' ? '블로그 글 검색' : 'Search blog posts';
    const searchPlaceholder = 'Search...';
    const searchButtonLabel = lang === 'kor' ? '검색' : 'Search';
    const openMenuLabel = lang === 'kor' ? '메뉴 열기' : 'Open menu';
    const closeMenuLabel = lang === 'kor' ? '메뉴 닫기' : 'Close menu';
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
    <link href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500&family=Inter:wght@400;500;600&family=Manrope:wght@500;600;700;800&family=Noto+Sans+KR:wght@400;500;600;700&display=swap" rel="stylesheet" />
    <link href="/blogs/css/blog.css" rel="stylesheet" />
    <link href="/blogs/css/typography.css" rel="stylesheet" />
    <link href="/blogs/css/post.css" rel="stylesheet" />
    <link href="/blogs/css/scroll-progress.css" rel="stylesheet" />
    <link href="/assets/vendor/bootstrap-icons/bootstrap-icons.min.css" rel="stylesheet" />
${renderConditionalHeadAssets(activeRuntimeFeatures)}

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
            <div class="post-nav-brand-group">
                <a class="navbar-brand" href="/blogs/">Hwan's Blog</a>
                <a class="post-nav-portfolio-link" href="/">
                    Portfolio <i class="bi bi-box-arrow-up-right" aria-hidden="true"></i>
                </a>
            </div>
            <button class="navbar-toggler" type="button" data-nav-toggle data-open-label="${openMenuLabel}" data-close-label="${closeMenuLabel}" aria-controls="navbarResponsive" aria-expanded="false" aria-label="${openMenuLabel}">
                <span class="post-nav-menu-icon" aria-hidden="true"></span>
            </button>
            <div class="collapse navbar-collapse" id="navbarResponsive">
                <ul class="navbar-nav ms-auto py-4 py-lg-0">
                    <li class="nav-item post-nav-search-item">
                        <form id="post-nav-search-form" class="post-nav-search" role="search">
                            <label class="visually-hidden" for="post-nav-search-input">${searchLabel}</label>
                            <input id="post-nav-search-input" type="search" placeholder="${searchPlaceholder}" enterkeyhint="search" />
                            <button type="submit" aria-label="${searchButtonLabel}">
                                <i class="bi bi-search" aria-hidden="true"></i>
                            </button>
                        </form>
                    </li>
                    <li class="nav-item nav-theme-item"><button class="btn nav-link blog-theme-toggle post-nav-theme-toggle" type="button" data-theme-toggle aria-label="Toggle color theme" aria-pressed="false"><i class="bi bi-moon-stars" aria-hidden="true"></i></button></li>
                    ${alternateHref ? `<li class="nav-item post-nav-language-item"><a href="${alternateHref}" class="nav-link post-nav-language-link" data-language-target="${alternateLang}" aria-label="${alternateLang === 'eng' ? 'Switch to English' : '한국어로 전환'}" title="${alternateLang === 'eng' ? 'Switch to English' : '한국어로 전환'}">${alternateLang === 'eng' ? 'A' : '가'}</a></li>` : ''}
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
            alternateHref,
            runtimeFeatures: activeRuntimeFeatures
        })};
    </script>
${renderConditionalBodyScripts(activeRuntimeFeatures)}
    <script src="/blogs/js/theme-toggle.js"></script>
    <script src="/blogs/js/scroll-progress.js"></script>
    <script src="/blogs/js/blog-shell.js"></script>
    <script src="/blogs/js/post-page.js"></script>
    <script>
        initBlogShell({
            formSelector: '#post-nav-search-form',
            inputSelector: '#post-nav-search-input'
        });
    </script>
</body>
</html>`;
}

module.exports = {
    renderPostPage
};
