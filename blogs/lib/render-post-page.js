const { SITE_URL } = require('./site-config');
const { getSeriesRoute } = require('./site-routes');
const { render: renderSiteIcon } = require('../../assets/js/site-icons');
const {
    escapeHtml,
    getAbsoluteUrl,
    getPostAlternates,
    getPostCanonicalUrl,
    getPostDescription,
    getPostLanguageRoute,
    getPostTitle,
    getSeriesTitle,
    renderChronologicalPostNavigation,
    renderRelatedPosts,
    renderTags,
    serializeStructuredData
} = require('./seo-utils');

function serializeForScript(value) {
    return JSON.stringify(value, null, 2).replace(/</g, '\\u003c');
}

function resolveOgImage(post, featuredPortfolioPosts) {
    const configuredImage = post.socialImage || post.previewImage || post.cover || '';
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

function renderPostSidebar() {
    return `    <header id="header" class="header blog-sidebar dark-background">
        <div class="profile-img">
            <img src="/assets/icon.webp" alt="Portrait illustration of Hwan Heo">
        </div>

        <a href="/blogs/" class="logo">
            <span class="sitename">Hwan's Blog</span>
        </a>

        <div class="social-links">
            <a href="https://github.com/hwanhuh" aria-label="GitHub">${renderSiteIcon('github')}</a>
            <a href="https://www.linkedin.com/in/hwan-heo-0905korea/" aria-label="LinkedIn">${renderSiteIcon('linkedin')}</a>
            <a href="https://scholar.google.com/citations?user=RulvYTkAAAAJ" aria-label="Google Scholar">${renderSiteIcon('mortarboard-fill')}</a>
            <a href="mailto:hwan.heo.ai@gmail.com" aria-label="Email">${renderSiteIcon('envelope-fill')}</a>
        </div>

        <nav id="navmenu" class="navmenu" aria-label="Blog navigation">
            <ul>
                <li>
                    <a href="/blogs/" class="active" aria-current="location">
                        ${renderSiteIcon('house', { className: 'navicon' })}
                        <span>Blog Home</span>
                    </a>
                </li>
                <li>
                    <a href="/" class="sidebar-external-link">
                        ${renderSiteIcon('briefcase', { className: 'navicon' })}
                        <span>Portfolio</span>
                        ${renderSiteIcon('box-arrow-up-right', { className: 'sidebar-external-icon' })}
                    </a>
                </li>
            </ul>
        </nav>

        <div class="sidebar-labs">
            <details class="sidebar-labs-menu">
                <summary>
                    ${renderSiteIcon('tools', { className: 'navicon' })}
                    <span>Labs</span>
                    ${renderSiteIcon('chevron-down', { className: 'sidebar-labs-chevron' })}
                </summary>
                <div class="sidebar-labs-panel">
                    <a href="/blogs/3DViewer/">${renderSiteIcon('box', { className: 'navicon' })}<span>3D Viewer</span></a>
                    <a href="/blogs/editor/">${renderSiteIcon('pencil-square', { className: 'navicon' })}<span>Editor</span></a>
                </div>
            </details>
        </div>
    </header>`;
}

function renderPostPage({ post, lang, contentHtml, metaDescription, readingTime, runtimeFeatures = {}, siteData }) {
    const activeRuntimeFeatures = {
        katex: Boolean(runtimeFeatures.katex),
        prism: Boolean(runtimeFeatures.prism),
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
    const alternateLinksHtml = getPostAlternates(post)
        .map((alternate) => `    <link rel="alternate" hreflang="${alternate.hreflang}" href="${alternate.href}" />`)
        .join('\n');
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
    const seriesTitle = post.series ? getSeriesTitle(siteData, post.series, lang) : 'Technical Writing';
    const seriesHref = post.series ? getSeriesRoute(post.series) : '/blogs/';
    const heroSummary = post[`subtitle_${lang}`] || post[`description_${lang}`] || metaDescription;
    const tagHtml = (post.tags || []).length
        ? `<div class="post-tags">${renderTags(post, siteData)}</div>`
        : '';
    const authorCopy = lang === 'kor'
        ? {
            description: '대규모 3D 생성 모델부터 CUDA 추론 최적화와 그래픽스 파이프라인까지, 실제 제품으로 이어지는 3D AI 시스템을 만듭니다.'
        }
        : {
            description: 'Lead 3D AI Research Engineer building production systems across large-scale 3D generation, CUDA inference, and graphics pipelines.'
        };
    const detailLabels = lang === 'kor'
        ? { details: '글 정보', author: '작성자', published: '발행', reading: '읽는 시간', topics: '주제' }
        : { details: 'Article details', author: 'Author', published: 'Published', reading: 'Reading', topics: 'Topics' };

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
    <link href="/blogs/css/sidebar.css" rel="stylesheet" />
    <link href="/css/sidebar-nav.css" rel="stylesheet" />
    <link href="/blogs/css/typography.css" rel="stylesheet" />
    <link href="/blogs/css/post.css" rel="stylesheet" />
    <link href="/assets/css/site-icons.css" rel="stylesheet" />
    <script src="/assets/js/site-icons.js"></script>
    <script src="/js/sidebar-controller.js"></script>
${renderConditionalHeadAssets(activeRuntimeFeatures)}

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
<body class="blog-post-page">
${renderPostSidebar()}

    <main class="main blog-post-main">
        <nav class="post-site-nav" id="mainNav" aria-label="Post utilities">
            <div class="blog-shell">
                <div class="post-reading-row">
                    <div class="post-reading-column post-utility-column">
                        <a class="post-nav-home" href="/blogs/" aria-label="Back to Blog Home">
                            ${renderSiteIcon('arrow-left')}
                            <span>Blog Home</span>
                        </a>
                        <ul class="post-nav-actions">
                            <li class="post-nav-action post-nav-search-item">
                                <form id="post-nav-search-form" class="post-nav-search" role="search" data-collapsible-search>
                                    <label class="visually-hidden" for="post-nav-search-input">${searchLabel}</label>
                                    <input id="post-nav-search-input" type="search" placeholder="${searchPlaceholder}" enterkeyhint="search" />
                                    <button type="submit" aria-label="${searchButtonLabel}" aria-controls="post-nav-search-input" aria-expanded="false">
                                        ${renderSiteIcon('search')}
                                    </button>
                                </form>
                            </li>
                            <li class="post-nav-action"><button class="blog-theme-toggle post-nav-theme-toggle" type="button" data-theme-toggle aria-label="Toggle color theme" aria-pressed="false">${renderSiteIcon('moon-stars', { className: 'theme-toggle-icon' })}</button></li>
                            ${alternateHref ? `<li class="post-nav-action"><a href="${alternateHref}" class="post-nav-language-link" data-language-target="${alternateLang}" aria-label="${alternateLang === 'eng' ? 'Switch to English' : '한국어로 전환'}" title="${alternateLang === 'eng' ? 'Switch to English' : '한국어로 전환'}">${alternateLang === 'eng' ? 'A' : '가'}</a></li>` : ''}
                        </ul>
                    </div>
                </div>
            </div>
        </nav>

        <header class="masthead post-masthead">
            <div class="blog-shell post-hero-shell">
                <div class="post-reading-row">
                    <div class="post-reading-column post-hero-column">
                        <div class="post-heading">
                            <div class="post-hero-series">
                                <span class="post-hero-series-label">Series</span>
                                <span class="post-hero-series-separator" aria-hidden="true">/</span>
                                <a href="${escapeHtml(seriesHref)}">${escapeHtml(seriesTitle)}</a>
                            </div>
                            <h1>${escapeHtml(title)}</h1>
                        </div>
                    </div>
                </div>
            </div>
        </header>

        <article class="post-article">
            <div class="blog-shell">
                <div class="post-reading-row">
                    <div class="post-reading-column post-hero-column post-intro-column">
                        <div class="post-intro-layout">
                            <div class="post-intro-copy">
                                <p class="post-intro-deck">${escapeHtml(heroSummary)}</p>
                                ${tagHtml ? `<div class="post-intro-topics" aria-label="${detailLabels.topics}">${tagHtml}</div>` : ''}
                            </div>
                            <aside class="post-article-info" aria-label="${detailLabels.details}">
                                <ul>
                                    <li data-post-detail="author">
                                        <strong>${detailLabels.author}</strong>
                                        <span>Hwan Heo</span>
                                    </li>
                                    <li data-post-detail="published">
                                        <strong>${detailLabels.published}</strong>
                                        <time datetime="${escapeHtml(post.date)}">${date}</time>
                                    </li>
                                    <li data-post-detail="reading">
                                        <strong>${detailLabels.reading}</strong>
                                        <span class="post-detail-reading">${readingTime.text}</span>
                                    </li>
                                </ul>
                            </aside>
                        </div>
                    </div>
                    <div class="post-reading-column main-content">
                        ${contentHtml}
                    </div>
                    <div class="post-reading-column post-end-matter">
                        <section class="post-author-note" aria-labelledby="post-author-name">
                            <img class="post-author-portrait" src="/assets/profile4-author.png" alt="Hwan Heo" loading="lazy">
                            <div class="post-author-copy">
                                <span class="post-author-kicker">Written by</span>
                                <h2 id="post-author-name">Hwan Heo</h2>
                                <p>${authorCopy.description}</p>
                                <div class="post-author-links" aria-label="Author links">
                                    <a href="mailto:hwan.heo.ai@gmail.com">Email</a>
                                    <a href="https://www.linkedin.com/in/hwan-heo-0905korea/" target="_blank" rel="noopener noreferrer">LinkedIn</a>
                                </div>
                            </div>
                        </section>
                        ${relatedPostsHtml}
                        <div id="post-navigation" class="post-navigation-space">${staticPostNavigation}</div>
                    </div>
                </div>
            </div>
        </article>

        <button id="scroll-top" class="scroll-top button_top post-scroll-top" type="button" aria-label="Back to top">
            ${renderSiteIcon('arrow-up')}
        </button>

        <footer class="blog-post-footer">
            <div class="blog-shell">
                <div class="post-reading-row">
                    <div class="post-reading-column">
                        <div class="blog-footer-note">Copyright © Hwan Heo</div>
                    </div>
                </div>
            </div>
        </footer>
    </main>

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
