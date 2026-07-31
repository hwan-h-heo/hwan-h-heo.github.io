const fs = require('fs');
const path = require('path');
const sharp = require('sharp');

const PREVIEW_DIRECTORY = 'assets/generated/blog-covers';
const PREVIEW_WIDTH = 960;
const PREVIEW_HEIGHT = 600;

function sanitizeFilePart(value) {
    return String(value || '')
        .trim()
        .replace(/[^A-Za-z0-9_-]+/g, '-')
        .replace(/^-+|-+$/g, '');
}

function getBlogCoverPreviewUrl(postId, variant = 'cover') {
    const safePostId = sanitizeFilePart(postId);
    const suffix = variant === 'portfolio' ? '-portfolio' : '';
    return `/${PREVIEW_DIRECTORY}/${safePostId}${suffix}.webp`;
}

function resolveLocalAsset(repoRoot, assetUrl) {
    const cleanPath = String(assetUrl || '').split(/[?#]/, 1)[0];
    if (!cleanPath || /^https?:\/\//i.test(cleanPath)) {
        return '';
    }

    return path.join(repoRoot, cleanPath.replace(/^\/+/, ''));
}

async function writePreview(sourcePath, destinationPath) {
    fs.mkdirSync(path.dirname(destinationPath), { recursive: true });

    await sharp(sourcePath, {
        animated: false,
        pages: 1
    })
        .rotate()
        .resize(PREVIEW_WIDTH, PREVIEW_HEIGHT, {
            fit: 'cover',
            position: 'centre'
        })
        .webp({
            effort: 5,
            quality: 82,
            smartSubsample: true
        })
        .toFile(destinationPath);
}

async function generateBlogCoverPreviews({ siteData, repoRoot, distDir }) {
    const postsById = new Map(siteData.posts.map((post) => [post.id, post]));
    const entries = siteData.posts.map((post) => ({
        id: post.id,
        source: post.previewImage || post.cover,
        variant: 'cover'
    }));

    (siteData.featuredPortfolioPosts || []).forEach((item) => {
        const post = postsById.get(item.id);
        if (!post) {
            return;
        }

        entries.push({
            id: item.id,
            source: item.teaserImage || post.previewImage || post.cover,
            variant: 'portfolio'
        });
    });

    const uniqueEntries = new Map();
    entries.forEach((entry) => {
        uniqueEntries.set(`${entry.id}:${entry.variant}`, entry);
    });

    let generated = 0;
    let reused = 0;

    for (const entry of uniqueEntries.values()) {
        const sourcePath = resolveLocalAsset(repoRoot, entry.source);
        if (!sourcePath || !fs.existsSync(sourcePath)) {
            throw new Error(`Cannot generate blog cover preview; source is missing: ${entry.source}`);
        }

        const previewUrl = getBlogCoverPreviewUrl(entry.id, entry.variant);
        const destinationPath = path.join(distDir, previewUrl.replace(/^\/+/, ''));
        const sourceStat = fs.statSync(sourcePath);
        const destinationStat = fs.existsSync(destinationPath) ? fs.statSync(destinationPath) : null;

        if (destinationStat && destinationStat.mtimeMs >= sourceStat.mtimeMs) {
            reused += 1;
            continue;
        }

        await writePreview(sourcePath, destinationPath);
        generated += 1;
    }

    console.log(`Generated blog cover previews: ${generated} updated, ${reused} reused.`);
}

module.exports = {
    generateBlogCoverPreviews,
    getBlogCoverPreviewUrl
};
