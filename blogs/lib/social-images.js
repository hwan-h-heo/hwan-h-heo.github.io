const crypto = require('crypto');
const fs = require('fs');
const path = require('path');
const sharp = require('sharp');

const { SITE_URL } = require('./site-config');

const REPO_ROOT = path.join(__dirname, '../..');
const WIDTH = 1200;
const HEIGHT = 630;
const BACKGROUND = '#111113';
const RECIPE = `social-jpeg-v1:${WIDTH}:${HEIGHT}:${BACKGROUND}:contain:quality-88`;

function getSocialImage(source, repoRoot = REPO_ROOT) {
    const sourceUrl = new URL(source, `${SITE_URL}/`);
    if (sourceUrl.origin !== new URL(SITE_URL).origin) {
        throw new Error(`Social images must use a local source asset: ${source}`);
    }
    const sourcePath = path.resolve(repoRoot, `.${decodeURIComponent(sourceUrl.pathname)}`);
    if (!sourcePath.startsWith(`${repoRoot}${path.sep}`) || !fs.existsSync(sourcePath)) {
        throw new Error(`Social image source is missing or outside the repository: ${source}`);
    }
    const version = crypto.createHash('sha256')
        .update(RECIPE).update(fs.readFileSync(sourcePath)).digest('hex').slice(0, 16);
    const route = `/assets/generated/social/${version}.jpg`;

    return { sourcePath, route, url: `${SITE_URL}${route}`, width: WIDTH, height: HEIGHT };
}

function getPostSocialImage(post) {
    // previewImage selects a still for animation; a stale thumbnail must never
    // override a newly selected static cover. socialImage remains an explicit override.
    const source = post.socialImage
        || (/\.gif(?:[?#]|$)/i.test(post.cover) && post.previewImage)
        || post.cover;
    return getSocialImage(source);
}

function getProjectSocialImage(entry, siteData, fallback) {
    const preview = siteData.portfolioProjects.find((project) =>
        new URL(project.url, `${SITE_URL}/`).pathname === entry.route);
    const source = entry.project.socialImage
        ? new URL(entry.project.socialImage, `${SITE_URL}${entry.route}`).href
        : preview?.image;
    const poster = entry.project.overviewMedia?.poster;
    return getSocialImage(source || (poster ? new URL(poster, `${SITE_URL}${entry.route}`).href : fallback));
}

function renderSocialImageMetadata(image, alt) {
    const escapedAlt = String(alt).replace(/&/g, '&amp;').replace(/"/g, '&quot;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
    return `    <meta property="og:image" content="${image.url}" />
    <meta property="og:image:secure_url" content="${image.url}" />
    <meta property="og:image:type" content="image/jpeg" />
    <meta property="og:image:width" content="${image.width}" />
    <meta property="og:image:height" content="${image.height}" />
    <meta property="og:image:alt" content="${escapedAlt}" />
    <meta name="twitter:image" content="${image.url}" />
    <meta name="twitter:image:alt" content="${escapedAlt}" />`;
}

async function generateSocialImages({ images, distDir }) {
    const unique = new Map(images.map((image) => [image.route, image]));
    let generated = 0;
    for (const image of unique.values()) {
        const destination = path.join(distDir, image.route.slice(1));
        if (fs.existsSync(destination)) continue;
        fs.mkdirSync(path.dirname(destination), { recursive: true });
        await sharp(image.sourcePath, { animated: false, pages: 1 })
            .rotate()
            .resize(WIDTH, HEIGHT, { fit: 'contain', background: BACKGROUND })
            .flatten({ background: BACKGROUND })
            .jpeg({ quality: 88, mozjpeg: true })
            .toFile(destination);
        generated += 1;
    }
    console.log(`Generated social images: ${generated} updated, ${unique.size - generated} reused.`);
}

module.exports = {
    generateSocialImages,
    getPostSocialImage,
    getProjectSocialImage,
    getSocialImage,
    renderSocialImageMetadata
};
