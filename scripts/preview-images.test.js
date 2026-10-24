const assert = require('node:assert/strict');
const fs = require('node:fs');
const os = require('node:os');
const path = require('node:path');
const test = require('node:test');
const sharp = require('../blogs/node_modules/sharp');

const { generateBlogCoverPreviews } = require('../blogs/lib/blog-cover-assets');
const { generateSocialImages, getSocialImage } = require('../blogs/lib/social-images');
const { getBlogCoverPreviewUrl } = require('../blogs/js/blog-cover-media');

test('replacing a cover at the same path and mtime refreshes list and social image URLs', async () => {
    const repoRoot = fs.mkdtempSync(path.join(os.tmpdir(), 'portfolio-preview-'));
    try {
        const distDir = path.join(repoRoot, 'dist');
        const dataDir = path.join(distDir, 'blogs/data');
        fs.mkdirSync(dataDir, { recursive: true });
        const sourcePath = path.join(repoRoot, 'cover.png');
        const stalePreviewPath = path.join(repoRoot, 'old-preview.png');
        const post = { id: 'sample', cover: '/cover.png', previewImage: '/old-preview.png' };
        const siteData = { posts: [post], featuredPortfolioPosts: [{ id: post.id, post }] };
        fs.writeFileSync(path.join(dataDir, 'site-data.json'), JSON.stringify({ posts: [post] }));
        const makeImage = (background) => sharp({ create: { width: 40, height: 40, channels: 3, background } }).png().toBuffer();
        fs.writeFileSync(sourcePath, await makeImage('#ff0000'));
        fs.writeFileSync(stalePreviewPath, await makeImage('#00ff00'));
        const originalStat = fs.statSync(sourcePath);

        await generateBlogCoverPreviews({ siteData, repoRoot, distDir });
        const firstCoverUrl = post.coverPreviews.cover;
        const firstPortfolioUrl = post.coverPreviews.portfolio;
        const firstSocial = getSocialImage('/cover.png', repoRoot);
        await generateSocialImages({ images: [firstSocial], distDir });

        fs.writeFileSync(sourcePath, await makeImage('#0000ff'));
        fs.utimesSync(sourcePath, originalStat.atime, originalStat.mtime);
        await generateBlogCoverPreviews({ siteData, repoRoot, distDir });
        const nextSocial = getSocialImage('/cover.png', repoRoot);
        await generateSocialImages({ images: [nextSocial], distDir });

        assert.notEqual(post.coverPreviews.cover, firstCoverUrl);
        assert.notEqual(post.coverPreviews.portfolio, firstPortfolioUrl);
        assert.notEqual(nextSocial.url, firstSocial.url);
        assert.equal(getSocialImage('/cover.png', repoRoot).url, nextSocial.url, 'Unchanged input must retain its cache key.');
        const deployed = JSON.parse(fs.readFileSync(path.join(dataDir, 'site-data.json'), 'utf8')).posts[0];
        assert.equal(getBlogCoverPreviewUrl(deployed), post.coverPreviews.cover);
        assert.equal(getBlogCoverPreviewUrl(deployed, 'portfolio'), post.coverPreviews.portfolio);

        for (const route of [post.coverPreviews.cover, post.coverPreviews.portfolio, nextSocial.route]) {
            const filePath = path.join(distDir, route.slice(1));
            const metadata = await sharp(filePath).metadata();
            const pixel = await sharp(filePath).extract({ left: 400, top: 300, width: 1, height: 1 }).raw().toBuffer();
            assert(pixel[2] > 200 && pixel[0] < 30 && pixel[1] < 30, 'Generated image must use the latest cover, not the stale previewImage.');
            if (route === nextSocial.route) {
                assert.equal(metadata.format, 'jpeg');
                assert.equal(metadata.width, 1200);
                assert.equal(metadata.height, 630);
            }
        }
    } finally {
        fs.rmSync(repoRoot, { recursive: true, force: true });
    }
});
