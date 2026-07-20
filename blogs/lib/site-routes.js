const fs = require('fs');
const path = require('path');

const DEFAULT_REPO_ROOT = path.join(__dirname, '..', '..');

function getPostRoute(post, language) {
    const slug = language === 'eng' ? post.slug : `${post.slug}-kor`;
    return `/blogs/posts/${slug}/`;
}

function listProjectEntries(repoRoot = DEFAULT_REPO_ROOT) {
    const projectsDir = path.join(repoRoot, 'projects');
    if (!fs.existsSync(projectsDir)) {
        return [];
    }

    return fs.readdirSync(projectsDir, { withFileTypes: true })
        .filter((entry) => entry.isDirectory())
        .map((entry) => {
            const projectDir = path.join(projectsDir, entry.name);
            const metadataPath = path.join(projectDir, 'project.json');
            const contentPath = path.join(projectDir, 'content.md');
            if (!fs.existsSync(metadataPath) || !fs.existsSync(contentPath)) {
                return null;
            }

            return {
                slug: entry.name,
                route: `/projects/${entry.name}/`,
                projectDir,
                metadataPath,
                contentPath,
                project: JSON.parse(fs.readFileSync(metadataPath, 'utf8'))
            };
        })
        .filter(Boolean)
        .sort((a, b) => a.slug.localeCompare(b.slug));
}

function buildPublicRoutes(siteData, repoRoot = DEFAULT_REPO_ROOT) {
    const routes = [
        { path: '/', type: 'portfolio' },
        { path: '/blogs/', type: 'blog-index' },
        { path: '/blogs/search/?q=3d', type: 'blog-search' },
        { path: '/blogs/3DViewer/', type: 'utility' },
        { path: '/blogs/editor/', type: 'editor' },
        { path: '/blogs/editor/portfolio.html', type: 'editor' }
    ];

    siteData.posts.forEach((post) => {
        post.languages.forEach((language) => {
            routes.push({
                path: getPostRoute(post, language),
                type: 'post',
                id: post.id,
                language
            });
        });
    });

    listProjectEntries(repoRoot).forEach((entry) => {
        routes.push({ path: entry.route, type: 'project', id: entry.slug });
    });

    return routes;
}

function buildSitemapEntries(siteData, repoRoot = DEFAULT_REPO_ROOT) {
    const entries = [
        { path: '/', changefreq: 'weekly', priority: '1.0' },
        { path: '/blogs/', changefreq: 'weekly', priority: '0.9' }
    ];

    siteData.posts.forEach((post) => {
        post.languages.forEach((language) => {
            entries.push({
                path: getPostRoute(post, language),
                lastmod: post.updated || post.date,
                changefreq: 'monthly',
                priority: '0.8'
            });
        });
    });

    listProjectEntries(repoRoot).forEach((entry) => {
        entries.push({
            path: entry.route,
            changefreq: 'monthly',
            priority: '0.7'
        });
    });

    return entries;
}

module.exports = {
    buildPublicRoutes,
    buildSitemapEntries,
    getPostRoute,
    listProjectEntries
};
