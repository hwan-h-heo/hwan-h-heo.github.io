const fs = require('fs');
const path = require('path');
const { marked } = require('marked');
const { parseMarkdownWithMath } = require('./js/markdown-with-math');

const { copyRecursiveSync, ensureDirSync } = require('./lib/fs-utils');
const { loadSiteData } = require('./lib/site-data');
const { SITE_URL } = require('./lib/site-config');
const { renderPostPage } = require('./lib/render-post-page');
const { parseProjectMarkdown } = require('./lib/project-markdown');
const { renderProjectPage } = require('./lib/render-project-page');
const { buildSitemapEntries, getPostRoute, listProjectEntries } = require('./lib/site-routes');
const { analyzeChangedFiles, getChangedFiles, serializeChangedFiles } = require('../scripts/lib/change-impact');
const { renderBlock: renderPortfolioBlock } = require('../js/portfolio-blocks');
const {
    renderProject: renderPortfolioProject,
    renderPublication,
    renderTalk
} = require('../js/portfolio-content');

const siteData = loadSiteData();
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
        ['build/three.module.js', 'vendor/three/build/three.module.js'],
        ['examples/jsm', 'vendor/three/examples/jsm']
    ];

    runtimePaths.forEach(([source, destination]) => {
        const sourcePath = path.join(threeRoot, source);
        if (!fs.existsSync(sourcePath)) {
            throw new Error(`Missing runtime dependency: ${sourcePath}. Run npm install first.`);
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

function truncateTocLabel(label, maxLength) {
    const normalized = label.replace(/\s+/g, ' ').trim();
    if (normalized.length <= maxLength) {
        return {
            text: normalized,
            truncated: false
        };
    }

    const prefixMatch = normalized.match(/^(\d+(?:\.\d+)*\.?\s+)/);
    const prefix = prefixMatch ? prefixMatch[1] : '';
    const available = Math.max(18, maxLength - prefix.length - 1);
    const body = normalized.slice(prefix.length, prefix.length + available).trimEnd();
    return {
        text: `${prefix}${body}`,
        truncated: true
    };
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
            const label = truncateTocLabel(plainText, Number(level) === 2 ? 38 : 34);
            headings.push({
                level: Number(level),
                id,
                text: plainText,
                label: label.text,
                truncated: label.truncated
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
        const truncatedClass = heading.truncated ? ' is-truncated' : '';
        const ellipsis = heading.truncated ? '<span class="toc-ellipsis" aria-hidden="true">...</span>' : '';
        return `<a class="toc-link${truncatedClass}" href="#${heading.id}" title="${fullText}" aria-label="${fullText}"><span class="toc-link-text">${label}</span>${ellipsis}</a>`;
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

function replaceLegacyPostLinks(htmlContent) {
    return htmlContent.replace(/\/blogs\/posts\/\?id=([A-Za-z0-9_]+)\/?/g, (fullMatch, postId) => {
        const slug = siteData.slugMapping[postId];
        return slug ? `/blogs/posts/${slug}/` : fullMatch;
    });
}

function normalizePostContent(post, content, htmlContent, lang) {
    let updatedHtmlContent = htmlContent;
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

    updatedHtmlContent = replaceLegacyPostLinks(updatedHtmlContent);
    return updatedHtmlContent;
}

function generatePostPage(post, lang) {
    const mdPath = path.join(__dirname, 'posts', post.id, `content-${lang}.md`);
    if (!fs.existsSync(mdPath)) {
        throw new Error(`Missing content file: ${mdPath}`);
    }

    const mdContent = fs.readFileSync(mdPath, 'utf8');
    const parts = mdContent.split('--- 여기부터 실제 콘텐츠 ---');
    const content = parts.length > 1 ? parts[1].trim() : mdContent;

    const parsedHtml = parseMarkdownWithMath(content, (source) => marked.parse(source));
    const normalizedHtml = normalizePostContent(post, content, parsedHtml, lang);
    const metaDescription = (post[`description_${lang}`] || post[`subtitle_${lang}`] || post.description_eng || post.subtitle_eng || '').substring(0, 160);
    const readingTime = calculateReadingTime(normalizedHtml);
    const html = renderPostPage({
        post,
        lang,
        contentHtml: normalizedHtml,
        metaDescription,
        readingTime,
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
        const backupPath = project.sourceBackup
            ? path.join(__dirname, '..', project.sourceBackup)
            : '';
        const legacyHtml = backupPath && fs.existsSync(backupPath)
            ? fs.readFileSync(backupPath, 'utf8')
            : '';
        const html = renderProjectPage({
            project,
            contentHtml,
            legacyHtml,
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

    const lines = ['<?xml version="1.0" encoding="UTF-8"?>', '<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">'];
    urls.forEach((url) => {
        lines.push('  <url>');
        lines.push(`    <loc>${url.loc}</loc>`);
        if (url.lastmod) {
            lines.push(`    <lastmod>${url.lastmod}</lastmod>`);
        }
        lines.push(`    <changefreq>${url.changefreq}</changefreq>`);
        lines.push(`    <priority>${url.priority}</priority>`);
        lines.push('  </url>');
    });
    lines.push('</urlset>');

    fs.writeFileSync(path.join(distDir, 'sitemap.xml'), `${lines.join('\n')}\n`);
    console.log('Generated sitemap.xml');
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

function buildSite() {
    resetDistDir();
    generateProjectPages();
    copyStaticAssets();
    validateContentFiles();
    generatePostPages();
    generateSitemap();
    generateRobotsTxt();
    generateSupportFiles();

    console.log('\nBuild completed successfully.');
    console.log(`Total posts generated: ${siteData.posts.length}`);
    console.log(`Output directory: ${distDir}`);
}

function buildIncremental(options = {}) {
    const changedFiles = getChangedFiles({
        changedFiles: options.changedFiles,
        repoRoot
    });
    const impact = analyzeChangedFiles(changedFiles, siteData);

    if (!fs.existsSync(path.join(distDir, 'index.html'))) {
        console.log('Incremental build needs an existing blogs/dist. Running full build.');
        buildSite();
        return;
    }

    if (impact.strategy !== 'incremental') {
        console.log('Incremental build fell back to a full build.');
        impact.reasons.forEach((reason) => console.log(`- ${reason}`));
        buildSite();
        return;
    }

    validateContentFiles();
    copyIncrementalStaticFiles(impact.staticFiles || []);
    impact.postTargets.forEach((languages, postId) => {
        copyPostSource(postId);
    });
    const generatedRoutes = generatePostPages(impact.postTargets);
    generateSitemap();
    generateRobotsTxt();
    generateSupportFiles();

    console.log('\nIncremental build completed successfully.');
    console.log(`Changed files: ${serializeChangedFiles(changedFiles) || '(none)'}`);
    console.log(`Static files copied: ${(impact.staticFiles || []).length}`);
    console.log(`Post pages generated: ${generatedRoutes.length}`);
    generatedRoutes.forEach((route) => console.log(`- ${route}`));
}

const options = parseArguments(process.argv.slice(2));
if (options.incremental) {
    buildIncremental(options);
} else {
    buildSite();
}
