const fs = require('fs');
const path = require('path');
const { marked } = require('marked');
const { parseMarkdownWithMath } = require('./js/markdown-with-math');

const { copyRecursiveSync, ensureDirSync } = require('./lib/fs-utils');
const { loadSiteData } = require('./lib/site-data');
const { renderPostPage } = require('./lib/render-post-page');
const { parseProjectMarkdown } = require('./lib/project-markdown');
const { renderProjectPage } = require('./lib/render-project-page');
const { buildSitemapEntries, listProjectEntries } = require('./lib/site-routes');
const { renderBlock: renderPortfolioBlock } = require('../js/portfolio-blocks');

const siteData = loadSiteData();
const distDir = path.join(__dirname, 'dist');

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

function generateTOC(htmlContent) {
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
        headings.push({ level: Number(level), id, text });
        return match;
    });

    if (headings.length === 0) {
        return { tocHtml: '', contentHtml: modifiedContent };
    }

    let tocHtml = '<ul>';
    let currentH2 = false;

    headings.forEach((heading) => {
        if (heading.level === 2) {
            if (currentH2) {
                tocHtml += '</ul></li>';
            }
            tocHtml += `<li><a href="#${heading.id}">${heading.text}</a><ul>`;
            currentH2 = true;
        } else if (currentH2) {
            tocHtml += `<li><a href="#${heading.id}">${heading.text}</a></li>`;
        }
    });

    if (currentH2) {
        tocHtml += '</ul></li>';
    }
    tocHtml += '</ul>';

    return { tocHtml, contentHtml: modifiedContent };
}

function replaceLegacyPostLinks(htmlContent) {
    return htmlContent.replace(/\/blogs\/posts\/\?id=([A-Za-z0-9_]+)\/?/g, (fullMatch, postId) => {
        const slug = siteData.slugMapping[postId];
        return slug ? `/blogs/posts/${slug}/` : fullMatch;
    });
}

function normalizePostContent(post, content, htmlContent) {
    let updatedHtmlContent = htmlContent;

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
        updatedHtmlContent = shareButtonHtml + updatedHtmlContent;
    }

    if (!content.includes('<nav class="toc">')) {
        const { tocHtml, contentHtml } = generateTOC(updatedHtmlContent);
        updatedHtmlContent = tocHtml ? `<nav class="toc">${tocHtml}</nav>${contentHtml}` : contentHtml;
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

function generatePostPages() {
    siteData.posts.forEach((post) => {
        post.languages.forEach((lang) => {
            const mdPath = path.join(__dirname, 'posts', post.id, `content-${lang}.md`);
            if (!fs.existsSync(mdPath)) {
                throw new Error(`Missing content file: ${mdPath}`);
            }

            const mdContent = fs.readFileSync(mdPath, 'utf8');
            const parts = mdContent.split('--- 여기부터 실제 콘텐츠 ---');
            const content = parts.length > 1 ? parts[1].trim() : mdContent;

            const parsedHtml = parseMarkdownWithMath(content, (source) => marked.parse(source));
            const normalizedHtml = normalizePostContent(post, content, parsedHtml);
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
        });
    });
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
    const baseUrl = 'https://hwan-h-heo.io';
    const urls = buildSitemapEntries(siteData).map((entry) => ({
        ...entry,
        loc: `${baseUrl}${entry.path}`
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

Sitemap: https://hwan-h-heo.io/sitemap.xml
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

buildSite();
