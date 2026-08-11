const fs = require('fs');
const path = require('path');

const DEFAULT_REPO_ROOT = path.join(__dirname, '..', '..');

function getPostRoute(post, language) {
    const slug = language === 'eng' ? post.slug : `${post.slug}-kor`;
    return `/blogs/posts/${slug}/`;
}

function createArchiveSlug(value) {
    return String(value || '')
        .toLowerCase()
        .replace(/&/g, 'and')
        .replace(/[^a-z0-9]+/g, '-')
        .replace(/-+/g, '-')
        .replace(/^-|-$/g, '');
}

function getTagRoute(tag) {
    return `/blogs/tags/${createArchiveSlug(tag)}/`;
}

function getSeriesRoute(seriesId) {
    return `/blogs/series/${createArchiveSlug(seriesId)}/`;
}

function listTagArchiveEntries(siteData, minimumPosts = 2) {
    const byTag = new Map();
    siteData.posts.forEach((post) => {
        (post.tags || []).forEach((tag) => {
            if (!byTag.has(tag)) {
                byTag.set(tag, []);
            }
            byTag.get(tag).push(post);
        });
    });

    return [...byTag.entries()]
        .filter(([, posts]) => posts.length >= minimumPosts)
        .map(([tag, posts]) => ({
            id: createArchiveSlug(tag),
            tag,
            title: tag,
            posts: [...posts].sort((a, b) => new Date(b.date) - new Date(a.date)),
            path: getTagRoute(tag),
            type: 'tag'
        }))
        .sort((a, b) => a.title.localeCompare(b.title));
}

function listSeriesArchiveEntries(siteData, minimumPosts = 1) {
    const bySeries = new Map();
    siteData.posts.forEach((post) => {
        if (!post.series) {
            return;
        }
        if (!bySeries.has(post.series)) {
            bySeries.set(post.series, []);
        }
        bySeries.get(post.series).push(post);
    });

    return [...bySeries.entries()]
        .filter(([, posts]) => posts.length >= minimumPosts)
        .map(([seriesId, posts]) => ({
            id: createArchiveSlug(seriesId),
            seriesId,
            title: siteData.series?.[seriesId]?.eng || seriesId,
            posts: [...posts].sort((a, b) => new Date(b.date) - new Date(a.date)),
            path: getSeriesRoute(seriesId),
            type: 'series'
        }))
        .sort((a, b) => a.title.localeCompare(b.title));
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
        { path: '/blogs/editor/', type: 'editor' }
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

    listSeriesArchiveEntries(siteData).forEach((entry) => {
        routes.push({ path: entry.path, type: 'series-archive', id: entry.seriesId });
    });

    listTagArchiveEntries(siteData).forEach((entry) => {
        routes.push({ path: entry.path, type: 'tag-archive', id: entry.tag });
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
                type: 'post',
                id: post.id,
                language,
                lastmod: post.updated || post.date,
                changefreq: 'monthly',
                priority: '0.8'
            });
        });
    });

    listSeriesArchiveEntries(siteData).forEach((entry) => {
        entries.push({
            path: entry.path,
            type: 'series-archive',
            id: entry.seriesId,
            lastmod: entry.posts[0]?.updated || entry.posts[0]?.date,
            changefreq: 'monthly',
            priority: '0.6'
        });
    });

    listTagArchiveEntries(siteData).forEach((entry) => {
        entries.push({
            path: entry.path,
            type: 'tag-archive',
            id: entry.tag,
            lastmod: entry.posts[0]?.updated || entry.posts[0]?.date,
            changefreq: 'monthly',
            priority: '0.6'
        });
    });

    listProjectEntries(repoRoot).forEach((entry) => {
        entries.push({
            path: entry.route,
            type: 'project',
            id: entry.slug,
            changefreq: 'monthly',
            priority: '0.7'
        });
    });

    return entries;
}

module.exports = {
    buildPublicRoutes,
    buildSitemapEntries,
    createArchiveSlug,
    getPostRoute,
    getSeriesRoute,
    getTagRoute,
    listSeriesArchiveEntries,
    listTagArchiveEntries,
    listProjectEntries
};
