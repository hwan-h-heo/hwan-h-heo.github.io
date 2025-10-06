const fs = require('fs');
const path = require('path');
const { marked } = require('marked');

// Import posts data
const postsDataContent = fs.readFileSync('./js/posts-data.js', 'utf-8');
// Remove const declarations so variables are global for eval
const cleanedPostsData = postsDataContent
    .replace(/^const\s+/gm, '')
    .replace(/^let\s+/gm, '');
eval(cleanedPostsData);

// Slug generation function
function generateSlug(title) {
    return title
        .toLowerCase()
        .replace(/[^\w\s-]/g, '') // Remove special characters
        .replace(/\s+/g, '-')      // Replace spaces with hyphens
        .replace(/-+/g, '-')       // Replace multiple hyphens with single
        .trim();
}

// TOC generation function (regex-based, similar to client-side version)
function generateTOC(htmlContent) {
    const headingRegex = /<h([23])([^>]*)>(.*?)<\/h\1>/gi;
    const headings = [];
    let headingCounter = 0;

    // First pass: ensure all headings have IDs
    let modifiedContent = htmlContent.replace(headingRegex, (fullMatch, level, attrs, text) => {
        const idMatch = attrs.match(/id="([^"]+)"/);
        let id = idMatch ? idMatch[1] : null;

        if (!id) {
            id = `toc-heading-${headingCounter++}`;
            return `<h${level} id="${id}"${attrs}>${text}</h${level}>`;
        }
        return fullMatch;
    });

    // Second pass: collect headings
    modifiedContent.replace(/<h([23])[^>]*id="([^"]+)"[^>]*>(.*?)<\/h\1>/gi, (match, level, id, text) => {
        headings.push({ level: parseInt(level), id, text });
        return match;
    });

    if (headings.length === 0) {
        return { tocHtml: '', contentHtml: htmlContent };
    }

    // Generate TOC HTML
    let tocHTML = '<ul>';
    let currentH2 = null;

    headings.forEach(heading => {
        if (heading.level === 2) {
            if (currentH2) tocHTML += '</ul></li>';
            tocHTML += `<li><a href="#${heading.id}">${heading.text}</a><ul>`;
            currentH2 = heading;
        } else if (heading.level === 3 && currentH2) {
            tocHTML += `<li><a href="#${heading.id}">${heading.text}</a></li>`;
        }
    });

    if (currentH2) tocHTML += '</ul></li>';
    tocHTML += '</ul>';

    return { tocHtml: tocHTML, contentHtml: modifiedContent };
}

// Create dist directory
const distDir = path.join(__dirname, 'dist');
if (!fs.existsSync(distDir)) {
    fs.mkdirSync(distDir, { recursive: true });
}

// Copy static assets
function copyRecursiveSync(src, dest) {
    const exists = fs.existsSync(src);
    const stats = exists && fs.statSync(src);
    const isDirectory = exists && stats.isDirectory();

    if (isDirectory) {
        if (!fs.existsSync(dest)) {
            fs.mkdirSync(dest, { recursive: true });
        }
        fs.readdirSync(src).forEach(childItemName => {
            copyRecursiveSync(
                path.join(src, childItemName),
                path.join(dest, childItemName)
            );
        });
    } else {
        fs.copyFileSync(src, dest);
    }
}

// Copy blog assets
['css', 'js', '3DViewer', 'editor', 'search'].forEach(dir => {
    const srcPath = path.join(__dirname, dir);
    const destPath = path.join(distDir, 'blogs', dir);
    if (fs.existsSync(srcPath)) {
        copyRecursiveSync(srcPath, destPath);
    }
});

// Copy posts folder (for images and other assets)
const postsPath = path.join(__dirname, 'posts');
const distPostsPath = path.join(distDir, 'blogs', 'posts');
if (fs.existsSync(postsPath)) {
    copyRecursiveSync(postsPath, distPostsPath);
}

// Copy parent-level assets (shared with portfolio site)
const parentAssetsPath = path.join(__dirname, '..', 'assets');
const distAssetsPath = path.join(distDir, 'assets');
if (fs.existsSync(parentAssetsPath)) {
    copyRecursiveSync(parentAssetsPath, distAssetsPath);
}

// Copy parent-level CSS and JS if they exist
['css', 'js'].forEach(dir => {
    const parentPath = path.join(__dirname, '..', dir);
    const destPath = path.join(distDir, dir);
    if (fs.existsSync(parentPath)) {
        copyRecursiveSync(parentPath, destPath);
    }
});

// Copy portfolio index.html to root
const portfolioIndexPath = path.join(__dirname, '..', 'index.html');
const distPortfolioIndexPath = path.join(distDir, 'index.html');
if (fs.existsSync(portfolioIndexPath)) {
    fs.copyFileSync(portfolioIndexPath, distPortfolioIndexPath);
}

// Copy projects folder if exists
const projectsPath = path.join(__dirname, '..', 'projects');
const distProjectsPath = path.join(distDir, 'projects');
if (fs.existsSync(projectsPath)) {
    copyRecursiveSync(projectsPath, distProjectsPath);
}

// Generate slug mapping
const slugMapping = {};
postsData.forEach(post => {
    const titleEng = post.title_eng || '';
    const slug = generateSlug(titleEng);
    slugMapping[post.id] = slug;
    post.slug = slug; // Add slug to post object
});

// Save slug mapping
fs.mkdirSync(path.join(distDir, 'blogs', 'js'), { recursive: true });
fs.writeFileSync(
    path.join(distDir, 'blogs', 'js', 'slug-mapping.js'),
    `const slugMapping = ${JSON.stringify(slugMapping, null, 2)};\nconst slugToId = ${JSON.stringify(Object.fromEntries(Object.entries(slugMapping).map(([k, v]) => [v, k])), null, 2)};`
);

// Template for post pages
function getPostTemplate(post, lang, content, metaDescription) {
    const title = post[`title_${lang}`] || post.title_eng;
    const subtitle = post[`subtitle_${lang}`] || post.subtitle_eng || '';
    const date = new Date(post.date).toLocaleDateString('en-US', { year: 'numeric', month: 'long', day: 'numeric' });
    const isoDate = new Date(post.date).toISOString();
    const slug = lang === 'eng' ? post.slug : `${post.slug}-kor`;
    const canonicalUrl = `https://hwan-h-heo.io/blogs/posts/${slug}/`;
    const ogImage = `https://hwan-h-heo.io/assets/image_fx_.jpg`; // Default OG image

    // Generate keywords from post data
    const keywords = [
        post.category || 'blog',
        post.series || 'article',
        '3D graphics',
        'computer vision',
        'machine learning'
    ].join(', ');

    // JSON-LD structured data for SEO
    const structuredData = {
        "@context": "https://schema.org",
        "@type": "BlogPosting",
        "headline": title,
        "description": metaDescription,
        "image": ogImage,
        "author": {
            "@type": "Person",
            "name": "Hwan Heo",
            "url": "https://hwan-h-heo.io"
        },
        "publisher": {
            "@type": "Person",
            "name": "Hwan Heo",
            "logo": {
                "@type": "ImageObject",
                "url": "https://hwan-h-heo.io/assets/favicon.ico"
            }
        },
        "datePublished": isoDate,
        "dateModified": isoDate,
        "mainEntityOfPage": {
            "@type": "WebPage",
            "@id": canonicalUrl
        },
        "inLanguage": lang === 'eng' ? 'en' : 'ko'
    };

    return `<!DOCTYPE html>
<html lang="${lang === 'eng' ? 'en' : 'ko'}">
<head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1, shrink-to-fit=no" />
    <meta name="description" content="${metaDescription}" />
    <meta name="keywords" content="${keywords}" />
    <meta name="author" content="Hwan Heo" />

    <!-- Canonical URL -->
    <link rel="canonical" href="${canonicalUrl}" />

    <!-- Open Graph / Facebook -->
    <meta property="og:type" content="article" />
    <meta property="og:url" content="${canonicalUrl}" />
    <meta property="og:title" content="${title}" />
    <meta property="og:description" content="${subtitle}" />
    <meta property="og:image" content="${ogImage}" />
    <meta property="og:site_name" content="HwanHeo's Blog" />
    <meta property="article:published_time" content="${isoDate}" />
    <meta property="article:author" content="Hwan Heo" />

    <!-- Twitter -->
    <meta name="twitter:card" content="summary_large_image" />
    <meta name="twitter:url" content="${canonicalUrl}" />
    <meta name="twitter:title" content="${title}" />
    <meta name="twitter:description" content="${subtitle}" />
    <meta name="twitter:image" content="${ogImage}" />

    <!-- JSON-LD Structured Data -->
    <script type="application/ld+json">
    ${JSON.stringify(structuredData, null, 2)}
    </script>

    <title>${title} - HwanHeo's Blog</title>
    <link rel="icon" type="image/x-icon" href="/assets/favicon.ico" />

    <!-- Import Maps for three.js -->
    <script type="importmap">
        {
            "imports": {
                "three": "https://cdn.jsdelivr.net/npm/three@0.150.0/build/three.module.js",
                "three/addons/": "https://cdn.jsdelivr.net/npm/three@0.150.0/examples/jsm/",
                "GaussianSplats3D": "https://unpkg.com/@mkkellogg/gaussian-splats-3d@0.4.0/build/gaussian-splats-3d.module.js"
            }
        }
    </script>

    <script src="https://cdnjs.cloudflare.com/ajax/libs/tween.js/25.0.0/tween.umd.js"></script>
    <script src="https://use.fontawesome.com/releases/v6.5.0/js/all.js" crossorigin="anonymous"></script>
    <link href="https://fonts.googleapis.com/css?family=Lora:400,700,400italic,700italic" rel="stylesheet" type="text/css" />
    <link href="https://fonts.googleapis.com/css?family=Open+Sans:300italic,400italic,600italic,700italic,800italic,400,300,600,700,800" rel="stylesheet" type="text/css" />
    <link href="https://cdnjs.cloudflare.com/ajax/libs/prism/1.28.0/themes/prism.min.css" rel="stylesheet" />
    <link href="/blogs/css/used.css" rel="stylesheet" />
    <link href="/blogs/css/blog_post_specific.css" rel="stylesheet" />
    <link href="/blogs/css/code-copy.css" rel="stylesheet" />
    <link rel="stylesheet" href="https://cdn.jsdelivr.net/gh/jpswalsh/academicons@1.9.1/css/academicons.min.css">
    <script type="module" src="https://unpkg.com/@google/model-viewer/dist/model-viewer.min.js"></script>

    <style>
        pre { background-color: #f4f4f4; padding: 10px; border-radius: 5px; overflow-x: auto; font-size: 0.9rem !important; }
        pre code { font-size: 0.9rem !important; }
        .main-content a { color: #15B886; text-decoration: none; }
        .main-content a:hover { color: #11926b; }
        table { width: 100%; border-collapse: collapse; }
        th, td { border: 1px solid #ddd; padding: 10px; text-align: center; vertical-align: middle; }
        th { width: 15%; background-color: #f2f2f2; }
    </style>

    <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/katex@0.16.9/dist/katex.min.css">
    <script defer src="https://cdn.jsdelivr.net/npm/katex@0.16.9/dist/katex.min.js"></script>
    <script defer src="https://cdn.jsdelivr.net/npm/katex@0.16.9/dist/contrib/auto-render.min.js"></script>
    <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bootstrap-icons@1.7.2/font/bootstrap-icons.css">

    <script async src="https://www.googletagmanager.com/gtag/js?id=G-RF7ETSKPK9"></script>
    <script>
        window.dataLayer = window.dataLayer || [];
        function gtag(){dataLayer.push(arguments);}
        gtag('js', new Date());
        gtag('config', 'G-RF7ETSKPK9');
    </script>
</head>
<body>
    <nav class="navbar navbar-expand-lg navbar-light" id="mainNav">
        <div class="container px-4 px-lg-5">
            <a class="navbar-brand" href="/blogs/">Blog Home</a>
            <button class="navbar-toggler" type="button" data-bs-toggle="collapse" data-bs-target="#navbarResponsive">
                Menu <i class="fas fa-bars"></i>
            </button>
            <div class="collapse navbar-collapse" id="navbarResponsive">
                <ul class="navbar-nav ms-auto py-4 py-lg-0">
                    <li class="nav-item">
                        <a href="${lang === 'eng' ? '../' + post.slug + '-kor/' : '../' + post.slug + '/'}" class="btn nav-link px-lg-3 py-3 py-lg-4" style="font-size:0.7rem">
                            ${lang === 'eng' ? 'KOR' : 'ENG'}
                        </a>
                    </li>
                </ul>
            </div>
        </div>
    </nav>

    <header class="masthead" style="background-image: url('/assets/image_fx_.jpg')">
        <div class="container position-relative px-4 px-lg-5">
            <div class="row gx-4 gx-lg-5 justify-content-center">
                <div class="col-md-10 col-lg-8 col-xl-7">
                    <div class="post-heading">
                        <br/>
                        <h2>${title}</h2>
                        <span class="meta">Posted on ${date}</span>
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
                    <!-- Series Navigation -->
                    <div id="series-container"></div>
                </div>
                <div class="col-md-10 col-lg-8 col-xl-7 main-content">
                    ${content}
                </div>
                <div class="col-md-10 col-lg-8 col-xl-7">
                    <!-- Post Navigation -->
                    <div id="post-navigation" class="mt-4"></div>
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

    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.2.3/dist/js/bootstrap.bundle.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/prismjs@1.28.0/components/prism-core.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/prismjs@1.28.0/plugins/autoloader/prism-autoloader.min.js"></script>
    <script src="/blogs/js/code-copy.js"></script>
    <script src="/blogs/js/posts-data.js"></script>
    <script>
        // Math rendering
        document.addEventListener('DOMContentLoaded', function() {
            if (typeof renderMathInElement !== 'undefined') {
                renderMathInElement(document.body, {
                    delimiters: [
                        {left: '$$', right: '$$', display: true},
                        {left: '$', right: '$', display: false},
                        {left: '\\\\(', right: '\\\\)', display: false},
                        {left: '\\\\[', right: '\\\\]', display: true}
                    ],
                    throwOnError: false
                });
            }

            // Share button functionality
            const copyButton = document.getElementById('copyButton');
            const shareModal = document.getElementById('myshare_modal');
            const closeModal = shareModal ? shareModal.querySelector('.share_modal_close') : null;
            const indicator = document.getElementById('share_modalIndicator');

            if (copyButton && shareModal) {
                let animationId;

                function updateShareButtonVisibility() {
                    var headerHeight = document.querySelector('.masthead').offsetHeight;
                    if (window.innerWidth > 1280 && window.scrollY > headerHeight) {
                        copyButton.style.display = 'block';
                    } else {
                        copyButton.style.display = 'none';
                    }
                }

                function animateIndicator() {
                    let startTime = null;
                    const duration = 1500;

                    function step(timestamp) {
                        if (!startTime) startTime = timestamp;
                        const progress = Math.min((timestamp - startTime) / duration, 1);
                        indicator.style.width = \`\${(1 - progress) * 100}%\`;

                        if (progress < 1) {
                            animationId = requestAnimationFrame(step);
                        } else {
                            shareModal.style.display = 'none';
                        }
                    }

                    cancelAnimationFrame(animationId);
                    animationId = requestAnimationFrame(step);
                }

                function closeShareModal() {
                    cancelAnimationFrame(animationId);
                    shareModal.style.display = 'none';
                    indicator.style.width = '0%';
                }

                // Scroll event for share button visibility
                document.addEventListener('scroll', updateShareButtonVisibility);
                window.addEventListener('resize', updateShareButtonVisibility);

                copyButton.addEventListener('click', function() {
                    const url = new URL(window.location.href);
                    url.hash = '';
                    const urlWithoutHash = url.href;

                    navigator.clipboard.writeText(urlWithoutHash).then(function() {
                        shareModal.style.display = 'block';
                        indicator.style.width = '100%';
                        animateIndicator();
                    }).catch(function(err) {
                        console.error('Link Copy Failed:', err);
                        alert('링크 복사에 실패했습니다.');
                    });
                });

                if (closeModal) {
                    closeModal.addEventListener('click', closeShareModal);
                }

                window.addEventListener('click', function(event) {
                    if (event.target === shareModal) {
                        closeShareModal();
                    }
                });

                updateShareButtonVisibility();
            }

            // Series navigation
            const postId = '${post.id}';
            const lang = '${lang}';
            renderSeriesNavigation(postId, lang);

            // TOC scroll tracking
            initializeTOC();

            // Custom script for specific posts with 3D viewers
            if (postId === '240917_3djs') {
                import('/blogs/3DViewer/js/gaussian_viewer.js')
                    .then(module => {
                        module.initGaussianViewer();
                    })
                    .catch(err => {
                        console.error("Failed to load the Gaussian viewer script:", err);
                    });
            }

            if (postId === '250310_model_viewer') {
                import('/js/simple-model-viewer.js')
                    .then(module => {
                        module.initGaussianViewer();
                    })
                    .catch(err => {
                        console.error("Failed to load the Gaussian viewer script:", err);
                    });
            }
        });

        function initializeTOC() {
            const toc = document.querySelector('.toc');
            if (!toc) return;

            let tocItems = [];

            // Scroll에 따라 TOC 표시/숨김
            function updateTocVisibility() {
                const headerHeight = document.querySelector('.masthead')?.offsetHeight || 300;
                if (window.scrollY > headerHeight) {
                    toc.style.display = 'block';
                } else {
                    toc.style.display = 'none';
                }
            }

            document.addEventListener('scroll', updateTocVisibility);

            // 초기 상태 설정
            const headerHeight = document.querySelector('.masthead')?.offsetHeight || 300;
            toc.style.display = window.scrollY > headerHeight ? 'block' : 'none';

            // 스크롤에 따라 현재 섹션 활성화
            const TOP_MARGIN = 0.1;
            const BOTTOM_MARGIN = 0.2;

            function initTocItems() {
                tocItems = Array.from(toc.querySelectorAll('li')).map(function(item) {
                    const anchor = item.querySelector('a');
                    if (!anchor) return null;
                    const href = anchor.getAttribute('href');
                    if (!href || href === '#') return null;
                    const target = document.getElementById(href.slice(1));
                    return { listItem: item, anchor: anchor, target: target };
                }).filter(item => item && item.target);
            }

            function syncToc() {
                const windowHeight = window.innerHeight;
                let currentSection = null;

                tocItems.forEach(function(item) {
                    const targetBounds = item.target.getBoundingClientRect();
                    if (targetBounds.top <= windowHeight * (1 - BOTTOM_MARGIN)) {
                        currentSection = item;
                    }
                });

                tocItems.forEach(function(item) {
                    if (item === currentSection) {
                        item.listItem.classList.add('active');
                    } else {
                        item.listItem.classList.remove('active');
                    }
                });
            }

            initTocItems();
            syncToc();
            window.addEventListener('scroll', syncToc, false);
        }

        function getPostTitle(post, lang) {
            return post['title_' + lang] || post.title_eng;
        }

        function renderSeriesNavigation(postId, lang) {
            const seriesContainer = document.getElementById('series-container');
            const navContainer = document.getElementById('post-navigation');

            if (!postsData || !seriesInfo || !seriesContainer) return;

            const currentPost = postsData.find(p => p.id === postId);
            if (!currentPost || !currentPost.series) return;

            const seriesId = currentPost.series;
            const postsInSeries = postsData
                .filter(p => p.series === seriesId)
                .sort((a, b) => new Date(b.date) - new Date(a.date)); // 최신순 정렬

            if (postsInSeries.length <= 1) return;

            const seriesTitle = seriesInfo[seriesId]?.[lang] || seriesInfo[seriesId]?.['eng'] || 'Series';

            // Render series accordion
            const listItems = postsInSeries.map(post => {
                const title = getPostTitle(post, lang);
                const slug = post.slug || post.id;
                const postLang = lang === 'eng' ? slug : slug + '-kor';
                if (post.id === postId) {
                    return '<li><strong>' + title + '</strong></li>';
                } else {
                    return '<li><a href="/blogs/posts/' + postLang + '/">' + title + '</a></li>';
                }
            }).join('');

            seriesContainer.innerHTML = \`
                <div class="accordion mb-4" id="accordionExample">
                    <div class="accordion-item">
                        <h2 class="accordion-header">
                            <button class="accordion-button collapsed" type="button" data-bs-toggle="collapse" data-bs-target="#collapseOne" aria-expanded="false" aria-controls="collapseOne">
                                <strong>\${seriesTitle}</strong>
                            </button>
                        </h2>
                        <div id="collapseOne" class="accordion-collapse collapse" data-bs-parent="#accordionExample">
                            <div class="accordion-body">
                                <ol>\${listItems}</ol>
                            </div>
                        </div>
                    </div>
                </div>
            \`;

            // Render post navigation (prev/next)
            const currentIndex = postsInSeries.findIndex(p => p.id === postId);
            const olderPost = postsInSeries[currentIndex + 1];
            const nextPost = postsInSeries[currentIndex - 1];

            if (olderPost || nextPost) {
                let navHtml = '<div class="d-flex justify-content-between mb-4">';

                if (olderPost) {
                    const olderTitle = getPostTitle(olderPost, lang);
                    const truncatedTitle = olderTitle.length > 25 ? olderTitle.substring(0, 25) + '...' : olderTitle;
                    const olderSlug = olderPost.slug || olderPost.id;
                    const olderLang = lang === 'eng' ? olderSlug : olderSlug + '-kor';
                    navHtml += \`<a class="btn btn-light text-uppercase" href="/blogs/posts/\${olderLang}/" style="width: 40%; text-align: center;">← Older Post<br><small style="font-size: 0.7rem; text-transform: none;">\${truncatedTitle}</small></a>\`;
                } else {
                    if (nextPost) {
                        navHtml += '<div></div>';
                    }
                }

                if (nextPost) {
                    const nextTitle = getPostTitle(nextPost, lang);
                    const truncatedTitle = nextTitle.length > 25 ? nextTitle.substring(0, 25) + '...' : nextTitle;
                    const nextSlug = nextPost.slug || nextPost.id;
                    const nextLang = lang === 'eng' ? nextSlug : nextSlug + '-kor';
                    navHtml += \`<a class="btn btn-light text-uppercase" href="/blogs/posts/\${nextLang}/" style="width: 40%; text-align: center;">Next Post →<br><small style="font-size: 0.7rem; text-transform: none;">\${truncatedTitle}</small></a>\`;
                } else {
                    const allSeriesIds = Object.keys(seriesInfo);
                    const otherSeriesIds = allSeriesIds.filter(id => id !== currentPost.series);

                    if (otherSeriesIds.length > 0) {
                        const randomSeriesId = otherSeriesIds[Math.floor(Math.random() * otherSeriesIds.length)];
                        const latestPostInRandomSeries = postsData
                            .filter(p => p.series === randomSeriesId)
                            .sort((a, b) => new Date(a.date) - new Date(b.date))[0];

                        if (latestPostInRandomSeries) {
                            const seriesTitle = seriesInfo[randomSeriesId][lang] || seriesInfo[randomSeriesId]['eng'];
                            const recSlug = latestPostInRandomSeries.slug || latestPostInRandomSeries.id;
                            const recLang = lang === 'eng' ? recSlug : recSlug + '-kor';
                            navHtml += \`<a class="btn btn-outline-secondary text-uppercase" href="/blogs/posts/\${recLang}/" style="width: 40%; text-align: center;">Explore Series<br><small style="font-size: 0.7rem; text-transform: none;">\${seriesTitle}</small></a>\`;
                        }
                    } else {
                        navHtml += '<div></div>';
                    }
                }

                navHtml += '</div>';
                navContainer.innerHTML = navHtml;
            }
        }
    </script>
</body>
</html>`;
}

// Generate blog home page
function generateBlogHomePage() {
    const homeTemplate = fs.readFileSync('./index.html', 'utf-8');

    // Update links in home template to use new slug-based URLs
    let updatedHome = homeTemplate;
    postsData.forEach(post => {
        const oldLink = `./posts/?id=${post.id}`;
        const newLink = `./posts/${post.slug}/`;
        updatedHome = updatedHome.replace(new RegExp(oldLink.replace(/[.*+?^${}()|[\]\\]/g, '\\$&'), 'g'), newLink);
    });

    // Ensure blogs directory exists
    fs.mkdirSync(path.join(distDir, 'blogs'), { recursive: true });
    fs.writeFileSync(path.join(distDir, 'blogs', 'index.html'), updatedHome);
}

// Update portfolio index.html with new blog links
function updatePortfolioLinks() {
    const portfolioPath = path.join(distDir, 'index.html');
    if (!fs.existsSync(portfolioPath)) return;

    let portfolioHtml = fs.readFileSync(portfolioPath, 'utf-8');

    // Update blog post links in portfolio
    postsData.forEach(post => {
        const oldLink = `blogs/posts/?id=${post.id}`;
        const newLink = `blogs/posts/${post.slug}/`;
        portfolioHtml = portfolioHtml.replace(new RegExp(oldLink.replace(/[.*+?^${}()|[\]\\]/g, '\\$&'), 'g'), newLink);
    });

    fs.writeFileSync(portfolioPath, portfolioHtml);
}

// Update JS files to use slugs
function updateJsFiles() {
    // Update posts-data.js with slug information
    const updatedPostsData = `const postsData = ${JSON.stringify(postsData, null, 2)};\n\nconst seriesInfo = ${JSON.stringify(seriesInfo, null, 2)};`;
    fs.writeFileSync(path.join(distDir, 'blogs', 'js', 'posts-data.js'), updatedPostsData);

    // Update main-list.js to use slugs
    let mainListJs = fs.readFileSync('./js/main-list.js', 'utf-8');
    mainListJs = mainListJs.replace(
        /href="\.\/posts\/\?id=\$\{post\.id\}"/g,
        'href="./posts/${post.slug}/"'
    );
    fs.writeFileSync(path.join(distDir, 'blogs', 'js', 'main-list.js'), mainListJs);
}

// Generate static HTML for each post
postsData.forEach(post => {
    ['eng', 'kor'].forEach(lang => {
        // Check if this language exists for the post
        const hasLang = post.languages && post.languages.includes(lang);
        if (!hasLang && lang === 'kor') return; // Skip Korean if not available

        const mdFileName = `content-${lang}.md`;
        const mdPath = path.join(__dirname, 'posts', post.id, mdFileName);

        if (!fs.existsSync(mdPath)) {
            if (lang === 'eng') {
                console.warn(`Missing file: ${mdPath}`);
            }
            return;
        }

        const mdContent = fs.readFileSync(mdPath, 'utf-8');
        const parts = mdContent.split('--- 여기부터 실제 콘텐츠 ---');
        const content = parts.length > 1 ? parts[1].trim() : mdContent;

        // Parse markdown to HTML
        let htmlContent = marked.parse(content);

        // Add share button if not already present
        const shareButtonHtml = `<button id="copyButton">
    <i class="bi bi-share-fill"></i>
</button>

<div id="myshare_modal" class="share_modal">
    <div class="share_modal-content">
        <span class="share_modal_close">×</span>
        <p><strong>Link Copied!</strong></p>
        <div class="copy_indicator-container">
        <div class="copy_indicator" id="share_modalIndicator"></div>
        </div>
    </div>
</div>

`;

        if (!content.includes('id="copyButton"') && !content.includes('id="myshare_modal"')) {
            htmlContent = shareButtonHtml + htmlContent;
        }

        // Generate TOC if not already present
        if (!content.includes('<nav class="toc">')) {
            const { tocHtml, contentHtml } = generateTOC(htmlContent);
            if (tocHtml) {
                htmlContent = `<nav class="toc">${tocHtml}</nav>` + contentHtml;
            } else {
                htmlContent = contentHtml;
            }
        }

        // Fix asset paths in the HTML content
        // Replace relative paths like ./240805_gs/assets/ with /blogs/posts/240805_gs/assets/
        htmlContent = htmlContent.replace(
            new RegExp(`\\./${post.id}/assets/`, 'g'),
            `/blogs/posts/${post.id}/assets/`
        );

        // Also fix any other common relative asset paths
        htmlContent = htmlContent.replace(
            /src="\.\/assets\//g,
            `/blogs/posts/${post.id}/assets/`
        );

        // Extract meta description from subtitle
        const metaDescription = (post[`subtitle_${lang}`] || post.subtitle_eng || '').substring(0, 160);

        // Generate HTML page
        const html = getPostTemplate(post, lang, htmlContent, metaDescription);

        // Create directory structure (under /blogs/posts/)
        const postDir = path.join(distDir, 'blogs', 'posts', lang === 'eng' ? post.slug : `${post.slug}-kor`);
        fs.mkdirSync(postDir, { recursive: true });

        // Write HTML file
        fs.writeFileSync(path.join(postDir, 'index.html'), html);

        console.log(`Generated: /blogs/posts/${lang === 'eng' ? post.slug : `${post.slug}-kor`}/index.html`);
    });
});

// Generate blog home page
generateBlogHomePage();

// Update portfolio links
updatePortfolioLinks();

// Update JS files
updateJsFiles();

// Generate sitemap.xml for SEO
function generateSitemap() {
    const baseUrl = 'https://hwan-h-heo.io';
    let sitemap = '<?xml version="1.0" encoding="UTF-8"?>\n';
    sitemap += '<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">\n';

    // Add homepage
    sitemap += `  <url>\n`;
    sitemap += `    <loc>${baseUrl}/</loc>\n`;
    sitemap += `    <changefreq>weekly</changefreq>\n`;
    sitemap += `    <priority>1.0</priority>\n`;
    sitemap += `  </url>\n`;

    // Add blog home
    sitemap += `  <url>\n`;
    sitemap += `    <loc>${baseUrl}/blogs/</loc>\n`;
    sitemap += `    <changefreq>weekly</changefreq>\n`;
    sitemap += `    <priority>0.9</priority>\n`;
    sitemap += `  </url>\n`;

    // Add all blog posts
    postsData.forEach(post => {
        ['eng', 'kor'].forEach(lang => {
            const hasLang = post.languages && post.languages.includes(lang);
            if (!hasLang && lang === 'kor') return;

            const slug = lang === 'eng' ? post.slug : `${post.slug}-kor`;
            const lastmod = new Date(post.date).toISOString().split('T')[0];

            sitemap += `  <url>\n`;
            sitemap += `    <loc>${baseUrl}/blogs/posts/${slug}/</loc>\n`;
            sitemap += `    <lastmod>${lastmod}</lastmod>\n`;
            sitemap += `    <changefreq>monthly</changefreq>\n`;
            sitemap += `    <priority>0.8</priority>\n`;
            sitemap += `  </url>\n`;
        });
    });

    sitemap += '</urlset>';

    fs.writeFileSync(path.join(distDir, 'sitemap.xml'), sitemap);
    console.log('✅ Generated sitemap.xml');
}

// Generate robots.txt for SEO
function generateRobotsTxt() {
    const baseUrl = 'https://hwan-h-heo.io';
    const robotsTxt = `User-agent: *
Allow: /

Sitemap: ${baseUrl}/sitemap.xml
`;

    fs.writeFileSync(path.join(distDir, 'robots.txt'), robotsTxt);
    console.log('✅ Generated robots.txt');
}

// Copy redirect pages
const redirectOldSite = fs.readFileSync('./redirect-old-site.html', 'utf-8');

// Create /hwan-h-heo.io redirect directory
const oldSiteRedirectDir = path.join(distDir, 'hwan-h-heo.io');
fs.mkdirSync(oldSiteRedirectDir, { recursive: true });
fs.writeFileSync(path.join(oldSiteRedirectDir, 'index.html'), redirectOldSite);
console.log('✅ Created redirect for /hwan-h-heo.io');

// Note: Legacy blog post URL redirect is handled directly in /blogs/posts/index.html
console.log('✅ Legacy blog post URL redirect is integrated in /blogs/posts/index.html');

// Create .nojekyll file for GitHub Pages
fs.writeFileSync(path.join(distDir, '.nojekyll'), '');

// Create CNAME if needed (uncomment and set your domain)
// fs.writeFileSync(path.join(distDir, 'CNAME'), 'yourdomain.com');

// Generate SEO files
generateSitemap();
generateRobotsTxt();

console.log('\n✅ Build completed successfully!');
console.log(`📝 Total posts generated: ${postsData.length}`);
console.log(`📂 Output directory: ${distDir}`);
console.log(`🔗 Portfolio integrated with blog successfully!`);
