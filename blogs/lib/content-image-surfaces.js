const path = require('path');
const sharp = require('sharp');

// Detect paper-backed figures during the build, without modifying source pixels
// or loading/analysing images in the reader's browser.
function createContentImageSurfaceNormalizer({ repoRoot }) {
    const results = new Map();
    const postsRoot = path.join(repoRoot, 'blogs', 'posts');

    async function hasPaperBackground(filePath) {
        try {
            const image = sharp(filePath);
            const metadata = await image.metadata();
            if ((metadata.pages || 1) > 1) return false;

            const { data, info } = await image
                .resize(96, 96, { fit: 'inside', withoutEnlargement: true, kernel: 'nearest' })
                .toColourspace('srgb')
                .ensureAlpha()
                .raw()
                .toBuffer({ resolveWithObject: true });
            const isWhite = (x, y) => {
                const offset = (y * info.width + x) * 4;
                return data[offset] >= 250 && data[offset + 1] >= 250 && data[offset + 2] >= 250;
            };
            let whitePixels = 0;
            for (let y = 0; y < info.height; y += 1) {
                for (let x = 0; x < info.width; x += 1) {
                    // Transparent cutouts already inherit the reading background.
                    if (data[(y * info.width + x) * 4 + 3] !== 255) return false;
                    if (isWhite(x, y)) whitePixels += 1;
                }
            }
            // Require substantial white space and consistently white corners;
            // do not tint ordinary photographs or dark/color-backed figures.
            if (whitePixels / (info.width * info.height) < 0.25) return false;
            for (const y of [0, info.height - 1]) {
                for (const x of [0, info.width - 1]) {
                    if (!isWhite(x, y)) return false;
                }
            }
            return true;
        } catch (error) {
            console.warn(`Could not inspect article image ${filePath}: ${error.message}`);
            return false;
        }
    }

    return async function normalizeContentImageSurfaces(html) {
        const images = [...html.matchAll(/<img\b[^>]*>/gi)];
        let output = '';
        let cursor = 0;
        for (const match of images) {
            let tag = match[0];
            const source = tag.match(/\ssrc\s*=\s*(["'])(.*?)\1/i)?.[2];
            if (source && !/\sdata-reading-surface\s*=/i.test(tag)) {
                let urlPath = '';
                try { urlPath = decodeURIComponent(source.split(/[?#]/, 1)[0]); } catch { /* Keep malformed URLs unchanged. */ }
                if (/^\/blogs\/posts\/.+\.(?:png|webp|jpe?g)$/i.test(urlPath)) {
                    const filePath = path.resolve(repoRoot, `.${urlPath}`);
                    if (filePath.startsWith(`${postsRoot}${path.sep}`)) {
                        if (!results.has(filePath)) results.set(filePath, hasPaperBackground(filePath));
                        if (await results.get(filePath)) {
                            tag = tag.replace(/\s*\/?>$/, ' data-reading-surface="paper">');
                        }
                    }
                }
            }
            output += html.slice(cursor, match.index) + tag;
            cursor = match.index + match[0].length;
        }
        return output + html.slice(cursor);
    };
}

module.exports = { createContentImageSurfaceNormalizer };
