const fs = require('fs');
const path = require('path');
const { SITE_URL } = require('./site-config');
const { getSeriesRoute } = require('./site-routes');
const { escapeXml } = require('./seo-utils');

const LEGACY_REDIRECTS_PATH = path.join(__dirname, '..', 'data', 'legacy-post-redirects.json');

function loadLegacyRedirects(filePath = LEGACY_REDIRECTS_PATH) {
    return JSON.parse(fs.readFileSync(filePath, 'utf8'));
}

function listSeriesRedirectEntries(siteData) {
    return Object.entries(siteData.seriesRedirects || {}).map(([source, target]) => ({
        path: getSeriesRoute(source),
        target: getSeriesRoute(target)
    }));
}

function renderSeriesRedirectPage(target) {
    const href = escapeXml(target);
    return `<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <meta name="robots" content="noindex, follow">
    <link rel="canonical" href="${escapeXml(SITE_URL)}${href}">
    <title>Series Redirect | Hwan Heo</title>
    <script>window.location.replace(${JSON.stringify(target)} + window.location.search + window.location.hash);</script>
    <meta http-equiv="refresh" content="0;url=${href}">
</head>
<body>
    <p>This series has moved. <a href="${href}">Continue to the series archive</a>.</p>
</body>
</html>
`;
}

function validateLegacyRedirects(siteData, redirects = loadLegacyRedirects()) {
    const errors = [];
    const posts = siteData.routablePosts || siteData.posts;
    const postIds = new Set(posts.map((post) => post.id));
    const canonicalTargets = new Set();

    posts.forEach((post) => {
        const expectedTarget = `/blogs/posts/${post.slug}/`;
        if (!redirects[post.id]) {
            errors.push(`Missing legacy redirect mapping for "${post.id}".`);
        } else if (redirects[post.id] !== expectedTarget) {
            errors.push(`Legacy redirect target mismatch for "${post.id}": expected "${expectedTarget}", found "${redirects[post.id]}".`);
        }
    });

    Object.entries(redirects).forEach(([legacyId, target]) => {
        if (!postIds.has(legacyId)) {
            errors.push(`Orphaned legacy redirect id "${legacyId}".`);
        }

        if (typeof target !== 'string' || !/^\/blogs\/posts\/[a-z0-9-]+\/$/.test(target)) {
            errors.push(`Invalid legacy redirect target for "${legacyId}": "${target}".`);
            return;
        }

        if (target === '/blogs/posts/') {
            errors.push(`Legacy redirect loop for "${legacyId}".`);
        }

        if (canonicalTargets.has(target)) {
            errors.push(`Duplicate legacy redirect target "${target}".`);
        }
        canonicalTargets.add(target);
    });

    if (errors.length > 0) {
        throw new Error(`Legacy redirect validation failed:\n${errors.join('\n')}`);
    }

    return redirects;
}

module.exports = {
    LEGACY_REDIRECTS_PATH,
    listSeriesRedirectEntries,
    loadLegacyRedirects,
    renderSeriesRedirectPage,
    validateLegacyRedirects
};
