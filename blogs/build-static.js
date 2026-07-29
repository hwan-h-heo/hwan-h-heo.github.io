const fs = require('fs');
const path = require('path');
const { marked } = require('marked');
const { parseMarkdownWithMath } = require('./js/markdown-with-math');

const { copyRecursiveSync, ensureDirSync } = require('./lib/fs-utils');
const { generateBlogCoverPreviews } = require('./lib/blog-cover-assets');
const { normalizeContentImageAccessibility } = require('./lib/content-image-accessibility');
const { loadSiteData } = require('./lib/site-data');
const { SITE_URL } = require('./lib/site-config');
const { renderPostPage } = require('./lib/render-post-page');
const { renderStaticBlogIndex } = require('./lib/render-blog-index');
const { renderArchivePage } = require('./lib/render-archive-page');
const { loadLegacyRedirects, validateLegacyRedirects } = require('./lib/legacy-redirects');
const {
    inferPostRuntimeFeatures,
    parsePostMarkdownSource
} = require('./lib/post-runtime-dependencies');
const { parseProjectMarkdown } = require('./lib/project-markdown');
const { renderProjectPage } = require('./lib/render-project-page');
const {
    buildSitemapEntries,
    getPostRoute,
    listProjectEntries,
    listSeriesArchiveEntries,
    listTagArchiveEntries
} = require('./lib/site-routes');
const { escapeXml, getPostAlternates, truncateText } = require('./lib/seo-utils');
const { analyzeChangedFiles, getChangedFiles, serializeChangedFiles } = require('../scripts/lib/change-impact');
const { renderBlock: renderPortfolioBlock } = require('../js/portfolio-blocks');
const {
    renderProject: renderPortfolioProject,
    renderPublication,
    renderTalk
} = require('../js/portfolio-content');

const siteData = loadSiteData();
const legacyRedirects = loadLegacyRedirects();
const distDir = path.join(__dirname, 'dist');
const repoRoot = path.join(__dirname, '..');

function parseArguments(argv) {
    const options = {
        changedFiles: [],
        incremental: false
    };

    argv.forEach((argument) => {
        if (argument === '--incremental') {
            options.incremental = true;
            return;
        }

        if (argument.startsWith('--changed=')) {
            options.changedFiles.push(...argument.slice('--changed='.length).split(',').filter(Boolean));
        }
    });

    return options;
}

function resetDistDir() {
    if (fs.existsSync(distDir)) {
        fs.rmSync(distDir, { recursive: true, force: true });
    }
    ensureDirSync(distDir);
}

function copyStaticAssets() {
    const blogDirs = ['css', 'js', '3DViewer', 'search', 'data'];
    blogDirs.forEach((dir) => {
        const srcPath = path.join(__dirname, dir);
        const destPath = path.join(distDir, 'blogs', dir);
        if (fs.existsSync(srcPath)) {
            copyRecursiveSync(srcPath, destPath);
        }
    });

    const editorPath = path.join(__dirname, 'editor');
    const privateEditorDirs = new Set(['drafts', 'draft-assets', 'project-snapshots']);
    if (fs.existsSync(editorPath)) {
        copyRecursiveSync(editorPath, path.join(distDir, 'blogs', 'editor'), {
            shouldCopy: (sourcePath) => {
                const relativePath = path.relative(editorPath, sourcePath);
                const topLevelName = relativePath.split(path.sep)[0];
                return !privateEditorDirs.has(topLevelName);
            }
        });
    }

    const blogFiles = ['index.html', 'redirect-old-site.html', 'redirect-legacy-posts.html'];
    blogFiles.forEach((fileName) => {
        const srcPath = path.join(__dirname, fileName);
        const destPath = path.join(distDir, 'blogs', fileName);
        if (fs.existsSync(srcPath)) {
            copyRecursiveSync(srcPath, destPath);
        }
    });

    const postsPath = path.join(__dirname, 'posts');
    if (fs.existsSync(postsPath)) {
        copyRecursiveSync(postsPath, path.join(distDir, 'blogs', 'posts'));
    }

    const sharedDirs = ['assets', 'css', 'js', 'content'];
    sharedDirs.forEach((dir) => {
        const srcPath = path.join(__dirname, '..', dir);
        const destPath = path.join(distDir, dir);
        if (fs.existsSync(srcPath)) {
            copyRecursiveSync(srcPath, destPath);
        }
    });

    copyProjectAssets();
    copyRuntimeDependencies();
    generatePortfolioIndex();
}

function copyPostSource(postId) {
    const srcPath = path.join(__dirname, 'posts', postId);
    const destPath = path.join(distDir, 'blogs', 'posts', postId);

    fs.rmSync(destPath, { recursive: true, force: true });

    if (!fs.existsSync(srcPath)) {
        return;
    }

    copyRecursiveSync(srcPath, destPath);
    console.log(`Copied post source: blogs/posts/${postId}`);
}

function getIncrementalStaticDestination(filePath) {
    const normalizedPath = filePath.replace(/\\/g, '/').replace(/^\.\/+/, '');
    if (normalizedPath.startsWith('blogs/')) {
        return path.join(distDir, normalizedPath);
    }

    return path.join(distDir, normalizedPath);
}

function copyIncrementalStaticFiles(filePaths) {
    filePaths.forEach((filePath) => {
        const srcPath = path.join(repoRoot, filePath);
        const destPath = getIncrementalStaticDestination(filePath);

        if (!fs.existsSync(srcPath)) {
            fs.rmSync(destPath, { recursive: true, force: true });
            console.log(`Removed static file: ${filePath}`);
            return;
        }

        copyRecursiveSync(srcPath, destPath);
        console.log(`Copied static file: ${filePath}`);
    });
}

function generatePortfolioIndex() {
    const indexPath = path.join(__dirname, '..', 'index.html');
    const contentPath = path.join(__dirname, '..', 'content', 'portfolio', 'home.json');
    const portfolioContent = JSON.parse(fs.readFileSync(contentPath, 'utf8'));
    let html = fs.readFileSync(indexPath, 'utf8');

    (portfolioContent.blocks || []).forEach((block) => {
        const pattern = new RegExp(`(<section\\b[^>]*data-portfolio-block=["']${escapeRegExp(block.id)}["'][^>]*>)[\\s\\S]*?(</section>)`);
        html = html.replace(pattern, (match, openingTag, closingTag) => {
            return `${openingTag}${renderPortfolioBlock(block)}${closingTag}`;
        });
    });

    html = html
        .replace(
            '<!-- portfolio-projects-static -->',
            (siteData.portfolioProjects || []).map(renderPortfolioProject).join('')
        )
        .replace(
            '<!-- portfolio-publications-static -->',
            (siteData.publications || []).map(renderPublication).join('')
        )
        .replace(
            '<!-- portfolio-talks-static -->',
            (siteData.talks || []).map(renderTalk).join('')
        )
        .replace(
            '</head>',
            `  <script type="application/ld+json">${serializeStructuredData(buildPortfolioStructuredData())}</script>\n</head>`
        );

    fs.writeFileSync(path.join(distDir, 'index.html'), html);
}

function copyRuntimeDependencies() {
    const threeRoot = path.join(__dirname, 'node_modules', 'three');
    const runtimePaths = [
        ['build/three.module.js', 'vendor/three/build/three.module.js']
    ];
    const threeExampleRuntimePaths = [
        ['examples/jsm/controls', 'vendor/three/examples/jsm/controls'],
        ['examples/jsm/loaders', 'vendor/three/examples/jsm/loaders'],
        ['examples/jsm/curves/NURBSCurve.js', 'vendor/three/examples/jsm/curves/NURBSCurve.js'],
        ['examples/jsm/curves/NURBSUtils.js', 'vendor/three/examples/jsm/curves/NURBSUtils.js'],
        ['examples/jsm/libs/fflate.module.js', 'vendor/three/examples/jsm/libs/fflate.module.js'],
        ['examples/jsm/libs/draco/gltf', 'vendor/three/examples/jsm/libs/draco/gltf'],
        ['examples/jsm/utils/BufferGeometryUtils.js', 'vendor/three/examples/jsm/utils/BufferGeometryUtils.js']
    ];

    runtimePaths.forEach(([source, destination]) => {
        const sourcePath = path.join(threeRoot, source);
        if (!fs.existsSync(sourcePath)) {
            throw new Error(`Missing runtime dependency: ${sourcePath}. Run npm install first.`);
        }
        copyRecursiveSync(sourcePath, path.join(distDir, destination));
    });

    threeExampleRuntimePaths.forEach(([source, destination]) => {
        const sourcePath = path.join(threeRoot, source);
        if (!fs.existsSync(sourcePath)) {
            throw new Error(`Missing Three.js example runtime dependency: ${sourcePath}. Run npm install first.`);
        }
        copyRecursiveSync(sourcePath, path.join(distDir, destination));
    });

    const tweenPath = path.join(__dirname, 'node_modules', '@tweenjs', 'tween.js', 'dist', 'tween.umd.js');
    if (!fs.existsSync(tweenPath)) {
        throw new Error(`Missing runtime dependency: ${tweenPath}. Run npm install first.`);
    }
    copyRecursiveSync(tweenPath, path.join(distDir, 'vendor', 'tween', 'tween.umd.js'));
}

function copyProjectAssets() {
    const srcPath = path.join(__dirname, '..', 'projects');
    const destPath = path.join(distDir, 'projects');

    if (!fs.existsSync(srcPath)) {
        return;
    }

    copyProjectAssetRecursive(srcPath, destPath);
}

function copyProjectAssetRecursive(src, dest) {
    const stat = fs.statSync(src);
    const baseName = path.basename(src);

    if (stat.isDirectory()) {
        ensureDirSync(dest);
        fs.readdirSync(src).forEach((child) => {
            copyProjectAssetRecursive(path.join(src, child), path.join(dest, child));
        });
        return;
    }

    if (baseName === 'content.md' || baseName === 'project.json') {
        return;
    }

    ensureDirSync(path.dirname(dest));
    fs.copyFileSync(src, dest);
}

function calculateReadingTime(content) {
    const textContent = content
        .replace(/<[^>]*>/g, '')
        .replace(/```[\s\S]*?```/g, '')
        .replace(/`[^`]*`/g, '')
        .trim();

    const words = textContent.split(/\s+/).filter(Boolean).length;
    const minutes = Math.max(1, Math.ceil(words / 200));

    return {
        words,
        minutes,
        text: minutes === 1 ? '1 min read' : `${minutes} min read`
    };
}

function escapeRegExp(value) {
    return value.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
}

function stripHtml(value) {
    return String(value || '')
        .replace(/<br\s*\/?>/gi, ' ')
        .replace(/<[^>]+>/g, '')
        .replace(/&amp;/g, '&')
        .replace(/&quot;/g, '"')
        .replace(/&#39;/g, "'")
        .replace(/\s+/g, ' ')
        .trim();
}

function escapeHtml(value) {
    return String(value || '')
        .replace(/&/g, '&amp;')
        .replace(/</g, '&lt;')
        .replace(/>/g, '&gt;')
        .replace(/"/g, '&quot;')
        .replace(/'/g, '&#39;');
}

function normalizeTocLabel(label) {
    return label.replace(/\s+/g, ' ').trim();
}

function serializeStructuredData(value) {
    return JSON.stringify(value).replace(/</g, '\\u003c');
}

function buildPortfolioStructuredData() {
    const personId = `${SITE_URL}/#person`;
    const publications = (siteData.publications || []).map((publication) => {
        const paperLink = (publication.links || []).find((link) => link.label === 'paper');
        const authors = stripHtml(publication.authorsHtml)
            .split(',')
            .map((name) => name.trim())
            .filter(Boolean)
            .map((name) => ({
                '@type': 'Person',
                name
            }));

        return {
            '@type': 'ScholarlyArticle',
            name: publication.title,
            author: authors,
            url: paperLink?.url || `${SITE_URL}/#portfolio`,
            isPartOf: {
                '@type': 'CreativeWork',
                name: stripHtml(publication.venueHtml)
            }
        };
    });

    return {
        '@context': 'https://schema.org',
        '@graph': [
            {
                '@type': 'Person',
                '@id': personId,
                name: 'Hwan Heo',
                alternateName: '허환',
                url: SITE_URL,
                jobTitle: 'Lead 3D AI Research Engineer',
                worksFor: {
                    '@type': 'Organization',
                    name: 'NC AI'
                },
                alumniOf: {
                    '@type': 'CollegeOrUniversity',
                    name: 'Korea University'
                },
                sameAs: [
                    'https://scholar.google.com/citations?user=RulvYTkAAAAJ',
                    'https://github.com/hwanhuh',
                    'https://www.linkedin.com/in/hwan-heo-0905korea/'
                ],
                knowsAbout: [
                    '3D generative AI',
                    'Neural rendering',
                    'CUDA inference optimization',
                    'Computer graphics'
                ]
            },
            ...publications
        ]
    };
}

function generateTOC(htmlContent, lang = 'eng') {
    const headingRegex = /<h([23])([^>]*)>(.*?)<\/h\1>/gi;
    const headings = [];
    let headingCounter = 0;

    const modifiedContent = htmlContent.replace(headingRegex, (fullMatch, level, attrs, text) => {
        const idMatch = attrs.match(/id="([^"]+)"/);
        if (idMatch) {
            return fullMatch;
        }

        const id = `toc-heading-${headingCounter++}`;
        return `<h${level} id="${id}"${attrs}>${text}</h${level}>`;
    });

    modifiedContent.replace(/<h([23])[^>]*id="([^"]+)"[^>]*>(.*?)<\/h\1>/gi, (match, level, id, text) => {
        const plainText = stripHtml(text);
        if (plainText) {
            headings.push({
                level: Number(level),
                id,
                text: plainText,
                label: normalizeTocLabel(plainText)
            });
        }
        return match;
    });

    if (headings.length === 0) {
        return { tocHtml: '', contentHtml: modifiedContent };
    }

    const tocSections = [];
    let currentSection = null;

    headings.forEach((heading) => {
        if (heading.level === 2) {
            currentSection = { ...heading, children: [] };
            tocSections.push(currentSection);
        } else if (currentSection) {
            currentSection.children.push(heading);
        }
    });

    if (tocSections.length === 0) {
        return { tocHtml: '', contentHtml: modifiedContent };
    }

    const renderTocLink = (heading) => {
        const fullText = escapeHtml(heading.text);
        const label = escapeHtml(heading.label);
        return `<a class="toc-link" href="#${heading.id}" title="${fullText}" aria-label="${fullText}"><span class="toc-link-text">${label}</span></a>`;
    };

    const tocTitle = lang === 'kor' ? '목차' : 'On This Page';
    const tocItems = tocSections.map((section) => {
        const childHtml = section.children.length
            ? `<ol class="toc-sublist">${section.children.map((child) => `<li class="toc-item toc-item-level-3" data-level="3">${renderTocLink(child)}</li>`).join('')}</ol>`
            : '';
        return `<li class="toc-item toc-item-level-2" data-level="2">${renderTocLink(section)}${childHtml}</li>`;
    }).join('');

    const tocHtml = `<div class="toc-title">${tocTitle}</div><ol class="toc-list">${tocItems}</ol>`;

    return { tocHtml, contentHtml: modifiedContent };
}

function getLegacyPostTarget(postId, lang = 'eng', hash = '') {
    const targetPost = siteData.postById[postId];
    if (targetPost && lang === 'kor' && targetPost.languages.includes('kor')) {
        return `${getPostRoute(targetPost, 'kor')}${hash || ''}`;
    }
    return legacyRedirects[postId] ? `${legacyRedirects[postId]}${hash || ''}` : '';
}

function deriveDescriptionFromHtml(htmlContent) {
    const contentWithoutCode = String(htmlContent || '')
        .replace(/<pre\b[^>]*>[\s\S]*?<\/pre>/gi, ' ')
        .replace(/<code\b[^>]*>[\s\S]*?<\/code>/gi, ' ')
        .replace(/<script\b[^>]*>[\s\S]*?<\/script>/gi, ' ')
        .replace(/<style\b[^>]*>[\s\S]*?<\/style>/gi, ' ')
        .replace(/\$\$[\s\S]*?\$\$/g, ' ')
        .replace(/\$[^$]*\$/g, ' ');
    const paragraphMatches = contentWithoutCode.match(/<p\b[^>]*>[\s\S]*?<\/p>/gi) || [];

    for (const paragraph of paragraphMatches) {
        const text = stripHtml(paragraph);
        if (text.length >= 60 && !/^Posted on\b/i.test(text)) {
            return truncateText(text, 160);
        }
    }

    return '';
}

function normalizeContentHeadingHierarchy(htmlContent) {
    return String(htmlContent || '').replace(/<h1(\s[^>]*)?>([\s\S]*?)<\/h1>/gi, (fullMatch, attrs = '', text) => {
        return `<h2${attrs}>${text}</h2>`;
    });
}

function replaceLegacyPostLinks(htmlContent, lang = 'eng') {
    let updatedHtmlContent = htmlContent.replace(/\/blogs\/posts\/(?:\?id=|id\?=)([A-Za-z0-9_]+)\/?(#[A-Za-z0-9_.-]+)?/g, (fullMatch, postId, hash = '') => {
        return getLegacyPostTarget(postId, lang, hash) || fullMatch;
    });

    updatedHtmlContent = updatedHtmlContent.replace(/(href=["'])\.\/\?id=([A-Za-z0-9_]+)\/?(#[^"']*)?/g, (fullMatch, prefix, postId, hash = '') => {
        const target = getLegacyPostTarget(postId, lang, hash);
        return target ? `${prefix}${target}` : fullMatch;
    });

    updatedHtmlContent = updatedHtmlContent.replace(/(href=["'])\.\.\/([A-Za-z0-9_]+)\/?(#[^"']*)?/g, (fullMatch, prefix, postId, hash = '') => {
        const target = getLegacyPostTarget(postId, lang, hash);
        return target ? `${prefix}${target}` : fullMatch;
    });

    return updatedHtmlContent;
}

function generateBlogIndex() {
    const sourcePath = path.join(__dirname, 'index.html');
    const html = renderStaticBlogIndex(fs.readFileSync(sourcePath, 'utf8'), siteData);
    fs.writeFileSync(path.join(distDir, 'blogs', 'index.html'), html);
    console.log('Generated static blog index: /blogs/index.html');
}

function renderLegacyRedirectPage(redirects) {
    const serializedRedirects = serializeStructuredData(redirects);
    return `<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <meta name="robots" content="noindex, follow">
    <link rel="canonical" href="${SITE_URL}/blogs/posts/">
    <title>Legacy Blog URL Redirect | Hwan Heo</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            display: flex;
            justify-content: center;
            align-items: center;
            min-height: 100vh;
            margin: 0;
            background-color: #f5f5f5;
        }
        .message {
            max-width: 38rem;
            text-align: center;
            padding: 20px;
        }
    </style>
</head>
<body>
    <main class="message">
        <h1>Legacy blog URL</h1>
        <p id="legacy-redirect-message">Checking the legacy post identifier...</p>
        <p><a href="/blogs/">Return to the blog archive</a></p>
    </main>

    <script>
        (function() {
            const redirects = ${serializedRedirects};
            const params = new URLSearchParams(window.location.search);
            const legacyId = params.get('id') || '';
            const target = redirects[legacyId];
            const message = document.getElementById('legacy-redirect-message');

            if (target) {
                let canonical = document.querySelector('link[rel="canonical"]');
                if (!canonical) {
                    canonical = document.createElement('link');
                    canonical.rel = 'canonical';
                    document.head.appendChild(canonical);
                }
                canonical.href = new URL(target, window.location.origin).href;

                const currentPath = window.location.pathname.replace(/\\/+$/, '/');
                if (currentPath !== target) {
                    window.location.replace(target);
                    return;
                }
            }

            if (message) {
                message.textContent = legacyId
                    ? 'No published post mapping exists for this legacy identifier.'
                    : 'No legacy post identifier was provided.';
            }
        })();
    </script>
</body>
</html>
`;
}

function generateLegacyRedirectPages() {
    const html = renderLegacyRedirectPage(legacyRedirects);
    const postsDir = path.join(distDir, 'blogs', 'posts');
    ensureDirSync(postsDir);
    fs.writeFileSync(path.join(postsDir, 'index.html'), html);
    fs.writeFileSync(path.join(distDir, 'blogs', 'redirect-legacy-posts.html'), html);
    console.log('Generated legacy redirect fallback pages');
}

function writeHtmlRoute(routePath, html) {
    const outputDir = path.join(distDir, routePath.replace(/^\/+|\/+$/g, ''));
    ensureDirSync(outputDir);
    fs.writeFileSync(path.join(outputDir, 'index.html'), html);
}

function generateArchivePages() {
    listSeriesArchiveEntries(siteData).forEach((entry) => {
        const description = `Articles in the ${entry.title} series by Hwan Heo.`;
        writeHtmlRoute(entry.path, renderArchivePage({
            title: `${entry.title} Series`,
            description,
            canonicalPath: entry.path,
            posts: entry.posts,
            siteData
        }));
        console.log(`Generated series archive: ${entry.path}`);
    });

    listTagArchiveEntries(siteData).forEach((entry) => {
        const description = `Technical articles tagged ${entry.title}, covering implementation details and research notes by Hwan Heo.`;
        writeHtmlRoute(entry.path, renderArchivePage({
            title: `${entry.title} Articles`,
            description,
            canonicalPath: entry.path,
            posts: entry.posts,
            siteData
        }));
        console.log(`Generated tag archive: ${entry.path}`);
    });
}

function normalizePostContent(post, content, htmlContent, lang) {
    let updatedHtmlContent = normalizeContentHeadingHierarchy(htmlContent);
    const postTitle = post[`title_${lang}`] || post.title_eng || post.id;
    const shareLabels = lang === 'kor'
        ? {
            copied: '링크 복사됨',
            ready: '글 주소를 바로 공유할 수 있습니다.'
        }
        : {
            copied: 'Link copied',
            ready: 'Post URL is ready to share.'
        };

    const shareButtonHtml = `<button id="copyButton" type="button" aria-label="Copy post link">
    <i class="bi bi-link-45deg" aria-hidden="true"></i>
</button>

<div id="myshare_modal" class="share_modal" role="status" aria-live="polite" aria-hidden="true">
    <div class="share_modal-content">
        <button class="share_modal_close" type="button" aria-label="Dismiss">
            <i class="bi bi-x-lg" aria-hidden="true"></i>
        </button>
        <span class="share_modal-icon" aria-hidden="true">
            <i class="bi bi-check2"></i>
        </span>
        <span class="share_modal-message">
            <strong>${shareLabels.copied}</strong>
            <small>${shareLabels.ready}</small>
        </span>
        <div class="copy_indicator-container">
            <div class="copy_indicator" id="share_modalIndicator"></div>
        </div>
    </div>
</div>

`;

    if (!content.includes('id="copyButton"') && !content.includes('id="myshare_modal"')) {
        updatedHtmlContent = shareButtonHtml + updatedHtmlContent;
    }

    if (!content.includes('<nav class="toc">')) {
        const { tocHtml, contentHtml } = generateTOC(updatedHtmlContent, lang);
        updatedHtmlContent = tocHtml ? `<nav class="toc" aria-label="Table of contents">${tocHtml}</nav>${contentHtml}` : contentHtml;
    }

    updatedHtmlContent = updatedHtmlContent.replace(
        new RegExp(`\\./${escapeRegExp(post.id)}/assets/`, 'g'),
        `/blogs/posts/${post.id}/assets/`
    );

    updatedHtmlContent = updatedHtmlContent.replace(
        /src=(["'])\.\/assets\//g,
        `src=$1/blogs/posts/${post.id}/assets/`
    );

    updatedHtmlContent = updatedHtmlContent.replace(
        /src=(["'])\.\/([0-9]{6}_[A-Za-z0-9_]+)\/assets\//g,
        'src=$1/blogs/posts/$2/assets/'
    );

    updatedHtmlContent = replaceLegacyPostLinks(updatedHtmlContent, lang);
    updatedHtmlContent = normalizeContentImageAccessibility(updatedHtmlContent, {
        title: postTitle
    });
    return updatedHtmlContent;
}

function generatePostPage(post, lang) {
    const mdPath = path.join(__dirname, 'posts', post.id, `content-${lang}.md`);
    if (!fs.existsSync(mdPath)) {
        throw new Error(`Missing content file: ${mdPath}`);
    }

    const mdContent = fs.readFileSync(mdPath, 'utf8');
    const { content, frontmatter } = parsePostMarkdownSource(mdContent);

    const parsedHtml = parseMarkdownWithMath(content, (source) => marked.parse(source));
    const normalizedHtml = normalizePostContent(post, content, parsedHtml, lang);
    const runtimeFeatures = inferPostRuntimeFeatures({
        post,
        contentSource: content,
        contentHtml: normalizedHtml,
        frontmatter
    });
    const explicitDescription = post[`description_${lang}`] || post[`subtitle_${lang}`] || post.description_eng || post.subtitle_eng || '';
    const derivedDescription = deriveDescriptionFromHtml(normalizedHtml);
    const descriptionSource = explicitDescription && explicitDescription.length < 50 && derivedDescription
        ? `${explicitDescription}. ${derivedDescription}`
        : explicitDescription || derivedDescription || post[`title_${lang}`] || post.title_eng;
    const metaDescription = truncateText(descriptionSource, 160);
    const readingTime = calculateReadingTime(normalizedHtml);
    const html = renderPostPage({
        post,
        lang,
        contentHtml: normalizedHtml,
        metaDescription,
        readingTime,
        runtimeFeatures,
        siteData
    });

    const slugDir = lang === 'eng' ? post.slug : `${post.slug}-kor`;
    const postDir = path.join(distDir, 'blogs', 'posts', slugDir);
    ensureDirSync(postDir);
    fs.writeFileSync(path.join(postDir, 'index.html'), html);
    console.log(`Generated: /blogs/posts/${slugDir}/index.html`);
    return getPostRoute(post, lang);
}

function getTargetLanguages(post, postTargets) {
    if (!postTargets) {
        return post.languages;
    }

    const languages = postTargets.get(post.id);
    if (!languages) {
        return [];
    }

    return post.languages.filter((lang) => languages.has(lang));
}

function generatePostPages(postTargets = null) {
    const routes = [];

    siteData.posts.forEach((post) => {
        getTargetLanguages(post, postTargets).forEach((lang) => {
            routes.push(generatePostPage(post, lang));
        });
    });

    return routes;
}

function validateContentFiles() {
    const missingFiles = [];

    siteData.posts.forEach((post) => {
        post.languages.forEach((lang) => {
            const mdPath = path.join(__dirname, 'posts', post.id, `content-${lang}.md`);
            if (!fs.existsSync(mdPath)) {
                missingFiles.push(mdPath);
            }
        });
    });

    if (missingFiles.length > 0) {
        throw new Error(`Missing content files:\n${missingFiles.join('\n')}`);
    }
}

function getProjectSlugFromUrl(url) {
    const match = String(url || '').match(/^projects\/([^/]+)\/?$/);
    return match ? match[1] : '';
}

function getProjectTitleLabel(project) {
    return String(project.title || project.heroTitle || 'Project')
        .replace(/<br\s*\/?>/gi, ' ')
        .replace(/<[^>]*>/g, '')
        .replace(/\s+/g, ' ')
        .trim();
}

function buildProjectNavItems(projectEntries) {
    const bySlug = new Map(projectEntries.map((entry) => [entry.slug, entry]));
    const ordered = [];
    const seen = new Set();

    (siteData.portfolioProjects || []).forEach((portfolioProject) => {
        const slug = getProjectSlugFromUrl(portfolioProject.url);
        if (!slug || !bySlug.has(slug) || seen.has(slug)) {
            return;
        }

        const entry = bySlug.get(slug);
        ordered.push({
            slug,
            label: getProjectTitleLabel(portfolioProject) || getProjectTitleLabel(entry.project)
        });
        seen.add(slug);
    });

    return ordered;
}

function getProjectNav(projectNavItems, currentSlug) {
    const currentIndex = projectNavItems.findIndex((item) => item.slug === currentSlug);
    if (currentIndex === -1 || projectNavItems.length < 2) {
        return null;
    }

    return {
        currentSlug,
        items: projectNavItems,
        previous: projectNavItems[(currentIndex - 1 + projectNavItems.length) % projectNavItems.length],
        next: projectNavItems[(currentIndex + 1) % projectNavItems.length]
    };
}

function generateProjectPages() {
    const projectEntries = listProjectEntries();

    const projectNavItems = buildProjectNavItems(projectEntries);

    projectEntries.forEach((entry) => {
        const { slug, projectDir, contentPath, project } = entry;
        const markdown = fs.readFileSync(contentPath, 'utf8');
        const contentHtml = markdown.trimStart().startsWith('<')
            ? markdown
            : parseProjectMarkdown(markdown, (source) => marked.parse(source));
        const html = renderProjectPage({
            project,
            contentHtml: normalizeContentImageAccessibility(contentHtml, {
                title: project.title || slug
            }),
            projectNav: getProjectNav(projectNavItems, slug)
        });

        fs.writeFileSync(path.join(projectDir, 'index.html'), html);
        console.log(`Generated project: /projects/${slug}/index.html`);
    });
}

function generateSitemap() {
    const urls = buildSitemapEntries(siteData).map((entry) => ({
        ...entry,
        loc: `${SITE_URL}${entry.path}`
    }));

    const lines = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        '<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9" xmlns:xhtml="http://www.w3.org/1999/xhtml">'
    ];
    urls.forEach((url) => {
        lines.push('  <url>');
        lines.push(`    <loc>${escapeXml(url.loc)}</loc>`);
        if (url.type === 'post' && url.id && siteData.postById[url.id]) {
            getPostAlternates(siteData.postById[url.id]).forEach((alternate) => {
                lines.push(`    <xhtml:link rel="alternate" hreflang="${alternate.hreflang}" href="${escapeXml(alternate.href)}" />`);
            });
        }
        if (url.lastmod) {
            lines.push(`    <lastmod>${escapeXml(url.lastmod)}</lastmod>`);
        }
        lines.push(`    <changefreq>${url.changefreq}</changefreq>`);
        lines.push(`    <priority>${url.priority}</priority>`);
        lines.push('  </url>');
    });
    lines.push('</urlset>');

    fs.writeFileSync(path.join(distDir, 'sitemap.xml'), `${lines.join('\n')}\n`);
    console.log('Generated sitemap.xml');
}

function generateFeed() {
    const feedItems = siteData.posts
        .filter((post) => post.languages.includes('eng'))
        .slice(0, 30)
        .map((post) => {
            const route = getPostRoute(post, 'eng');
            const url = `${SITE_URL}${route}`;
            const description = post.description_eng || post.subtitle_eng || post.title_eng;
            const pubDate = new Date(`${post.date}T00:00:00Z`).toUTCString();
            return [
                '    <item>',
                `      <title>${escapeXml(post.title_eng)}</title>`,
                `      <link>${escapeXml(url)}</link>`,
                `      <guid isPermaLink="true">${escapeXml(url)}</guid>`,
                `      <pubDate>${escapeXml(pubDate)}</pubDate>`,
                '      <language>en</language>',
                `      <description>${escapeXml(description)}</description>`,
                '    </item>'
            ].join('\n');
        }).join('\n');

    const feed = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        '<rss version="2.0">',
        '  <channel>',
        "    <title>Hwan Heo's Blog</title>",
        `    <link>${SITE_URL}/blogs/</link>`,
        '    <description>Technical articles on sparse 3D generation, neural rendering, mesh processing, CUDA kernels, and production inference optimization.</description>',
        '    <language>en</language>',
        feedItems,
        '  </channel>',
        '</rss>'
    ].join('\n');

    const feedDir = path.join(distDir, 'blogs');
    ensureDirSync(feedDir);
    fs.writeFileSync(path.join(feedDir, 'feed.xml'), `${feed}\n`);
    console.log('Generated RSS feed: /blogs/feed.xml');
}

function generateRobotsTxt() {
    const robotsTxt = `User-agent: *
Allow: /

Sitemap: ${SITE_URL}/sitemap.xml
`;

    fs.writeFileSync(path.join(distDir, 'robots.txt'), robotsTxt);
    console.log('Generated robots.txt');
}

function generateSupportFiles() {
    const redirectOldSiteHtml = fs.readFileSync(path.join(__dirname, 'redirect-old-site.html'), 'utf8');
    const oldSiteRedirectDir = path.join(distDir, 'hwan-h-heo.io');
    ensureDirSync(oldSiteRedirectDir);
    fs.writeFileSync(path.join(oldSiteRedirectDir, 'index.html'), redirectOldSiteHtml);
    fs.writeFileSync(path.join(distDir, '.nojekyll'), '');
}

async function buildSite() {
    resetDistDir();
    generateProjectPages();
    copyStaticAssets();
    await generateBlogCoverPreviews({ siteData, repoRoot, distDir });
    validateContentFiles();
    validateLegacyRedirects(siteData, legacyRedirects);
    generateBlogIndex();
    generateLegacyRedirectPages();
    generateArchivePages();
    generatePostPages();
    generateSitemap();
    generateFeed();
    generateRobotsTxt();
    generateSupportFiles();

    console.log('\nBuild completed successfully.');
    console.log(`Total posts generated: ${siteData.posts.length}`);
    console.log(`Output directory: ${distDir}`);
}

async function buildIncremental(options = {}) {
    const changedFiles = getChangedFiles({
        changedFiles: options.changedFiles,
        repoRoot
    });
    const impact = analyzeChangedFiles(changedFiles, siteData);

    if (!fs.existsSync(path.join(distDir, 'index.html'))) {
        console.log('Incremental build needs an existing blogs/dist. Running full build.');
        await buildSite();
        return;
    }

    if (impact.strategy !== 'incremental') {
        console.log('Incremental build fell back to a full build.');
        impact.reasons.forEach((reason) => console.log(`- ${reason}`));
        await buildSite();
        return;
    }

    validateContentFiles();
    validateLegacyRedirects(siteData, legacyRedirects);
    copyIncrementalStaticFiles(impact.staticFiles || []);
    impact.postTargets.forEach((languages, postId) => {
        copyPostSource(postId);
    });
    await generateBlogCoverPreviews({ siteData, repoRoot, distDir });
    const generatedRoutes = generatePostPages(impact.postTargets);
    generateBlogIndex();
    generateLegacyRedirectPages();
    generateArchivePages();
    generateSitemap();
    generateFeed();
    generateRobotsTxt();
    generateSupportFiles();

    console.log('\nIncremental build completed successfully.');
    console.log(`Changed files: ${serializeChangedFiles(changedFiles) || '(none)'}`);
    console.log(`Static files copied: ${(impact.staticFiles || []).length}`);
    console.log(`Post pages generated: ${generatedRoutes.length}`);
    generatedRoutes.forEach((route) => console.log(`- ${route}`));
}

const options = parseArguments(process.argv.slice(2));
const buildPromise = options.incremental
    ? buildIncremental(options)
    : buildSite();

buildPromise.catch((error) => {
    console.error(error);
    process.exit(1);
});
