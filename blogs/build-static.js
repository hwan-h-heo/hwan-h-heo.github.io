const fs = require('fs');
const path = require('path');
const { marked } = require('marked');
const { parseMarkdownWithMath } = require('./js/markdown-with-math');

const { copyRecursiveSync, ensureDirSync } = require('./lib/fs-utils');
const { loadSiteData } = require('./lib/site-data');
const { renderPostPage } = require('./lib/render-post-page');
const { parseProjectMarkdown } = require('./lib/project-markdown');
const { renderProjectPage } = require('./lib/render-project-page');

const siteData = loadSiteData();
const distDir = path.join(__dirname, 'dist');

function resetDistDir() {
    if (fs.existsSync(distDir)) {
        fs.rmSync(distDir, { recursive: true, force: true });
    }
    ensureDirSync(distDir);
}

function copyStaticAssets() {
    const blogDirs = ['css', 'js', '3DViewer', 'editor', 'search', 'data'];
    blogDirs.forEach((dir) => {
        const srcPath = path.join(__dirname, dir);
        const destPath = path.join(distDir, 'blogs', dir);
        if (fs.existsSync(srcPath)) {
            copyRecursiveSync(srcPath, destPath);
        }
    });

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

    const sharedDirs = ['assets', 'css', 'js', 'projects'];
    sharedDirs.forEach((dir) => {
        const srcPath = path.join(__dirname, '..', dir);
        const destPath = path.join(distDir, dir);
        if (fs.existsSync(srcPath)) {
            copyRecursiveSync(srcPath, destPath);
        }
    });

    copyRecursiveSync(path.join(__dirname, '..', 'index.html'), path.join(distDir, 'index.html'));
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
        /src="\.\/assets\//g,
        `src="/blogs/posts/${post.id}/assets/`
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
            const metaDescription = (post[`subtitle_${lang}`] || post.subtitle_eng || '').substring(0, 160);
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

function generateProjectPages() {
    const projectsDir = path.join(__dirname, '..', 'projects');
    if (!fs.existsSync(projectsDir)) {
        return;
    }

    fs.readdirSync(projectsDir, { withFileTypes: true })
        .filter((entry) => entry.isDirectory())
        .forEach((entry) => {
            const projectDir = path.join(projectsDir, entry.name);
            const metadataPath = path.join(projectDir, 'project.json');
            const contentPath = path.join(projectDir, 'content.md');

            if (!fs.existsSync(metadataPath) || !fs.existsSync(contentPath)) {
                return;
            }

            const project = JSON.parse(fs.readFileSync(metadataPath, 'utf8'));
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
            const html = renderProjectPage({ project, contentHtml, legacyHtml });

            fs.writeFileSync(path.join(projectDir, 'index.html'), html);
            console.log(`Generated project: /projects/${entry.name}/index.html`);
        });
}

function generateSitemap() {
    const baseUrl = 'https://hwan-h-heo.io';
    const urls = [
        {
            loc: `${baseUrl}/`,
            changefreq: 'weekly',
            priority: '1.0'
        },
        {
            loc: `${baseUrl}/blogs/`,
            changefreq: 'weekly',
            priority: '0.9'
        }
    ];

    siteData.posts.forEach((post) => {
        post.languages.forEach((lang) => {
            const slug = lang === 'eng' ? post.slug : `${post.slug}-kor`;
            urls.push({
                loc: `${baseUrl}/blogs/posts/${slug}/`,
                lastmod: new Date(post.date).toISOString().split('T')[0],
                changefreq: 'monthly',
                priority: '0.8'
            });
        });
    });

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
