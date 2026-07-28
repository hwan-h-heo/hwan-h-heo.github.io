const fs = require('fs');
const path = require('path');

const LEGACY_REDIRECTS_PATH = path.join(__dirname, '..', 'data', 'legacy-post-redirects.json');

function loadLegacyRedirects(filePath = LEGACY_REDIRECTS_PATH) {
    return JSON.parse(fs.readFileSync(filePath, 'utf8'));
}

function validateLegacyRedirects(siteData, redirects = loadLegacyRedirects()) {
    const errors = [];
    const postIds = new Set(siteData.posts.map((post) => post.id));
    const canonicalTargets = new Set();

    siteData.posts.forEach((post) => {
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
    loadLegacyRedirects,
    validateLegacyRedirects
};
