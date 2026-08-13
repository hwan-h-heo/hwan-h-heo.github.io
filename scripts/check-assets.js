const fs = require('fs');
const path = require('path');

const { loadSiteData } = require('../blogs/lib/site-data');

const repoRoot = path.join(__dirname, '..');
const siteData = loadSiteData();
const routablePosts = siteData.routablePosts || siteData.posts;
const postIds = new Set(routablePosts.map((post) => post.id));
const postSlugs = new Set(routablePosts.flatMap((post) => post.languages.map((language) => language === 'eng' ? post.slug : `${post.slug}-kor`)));
const remoteRenderedImagePattern = /^https?:\/\//i;
const imageExtensionPattern = /\.(png|jpe?g|gif|webp|svg|avif)(\?[^\s)"']*)?$/i;
const largeMediaWarningBytes = 15 * 1024 * 1024;
const largeMediaPattern = /\.(gif|mp4|webm|mov)$/i;

function isIgnoredReference(value) {
    return !value
        || value.startsWith('#')
        || /^(mailto|tel|data|javascript):/i.test(value);
}

function resolveReference(baseFile, value) {
    const clean = String(value).split('#')[0].split('?')[0];
    if (isIgnoredReference(clean) || /^https?:\/\//i.test(clean)) {
        return null;
    }

    const legacyPostAsset = clean.match(/^\.\/([A-Za-z0-9_]+)\/assets\/(.+)$/);
    if (legacyPostAsset && postIds.has(legacyPostAsset[1])) {
        return path.join(repoRoot, 'blogs', 'posts', legacyPostAsset[1], 'assets', legacyPostAsset[2]);
    }

    if (clean.startsWith('/blogs/posts/')) {
        const slug = clean.replace(/^\/blogs\/posts\//, '').replace(/\/.*$/, '');
        return postSlugs.has(slug) ? null : path.join(repoRoot, clean.replace(/^\/+/, ''));
    }

    if (clean.startsWith('/')) {
        return path.join(repoRoot, clean.replace(/^\/+/, ''));
    }

    if (baseFile.endsWith(path.join('blogs', 'data', 'site-data.json')) || baseFile.endsWith(path.join('content', 'portfolio', 'home.json'))) {
        return path.join(repoRoot, clean);
    }

    return path.resolve(path.dirname(baseFile), clean);
}

function collectFiles(dir, predicate, files = []) {
    if (!fs.existsSync(dir)) {
        return files;
    }

    fs.readdirSync(dir, { withFileTypes: true }).forEach((entry) => {
        const filePath = path.join(dir, entry.name);
        if (entry.isDirectory()) {
            collectFiles(filePath, predicate, files);
        } else if (predicate(filePath)) {
            files.push(filePath);
        }
    });
    return files;
}

function stripNonRenderedSource(text) {
    return String(text || '')
        .replace(/<!--[^]*?-->/g, '')
        .replace(/```[^]*?```/g, '');
}

function collectReferences(filePath) {
    const text = stripNonRenderedSource(fs.readFileSync(filePath, 'utf8'));
    const refs = [];
    const patterns = [
        { kind: 'image', pattern: /!\[[^\]]*]\(([^)\s]+)(?:\s+["'][^"']*["'])?\)/g },
        { kind: 'image', pattern: /<img\b[^>]*\bsrc=["']([^"']+)["'][^>]*>/gi },
        { kind: 'source', pattern: /<source\b[^>]*\bsrc=["']([^"']+)["'][^>]*>/gi },
        { kind: 'video', pattern: /<video\b[^>]*\bsrc=["']([^"']+)["'][^>]*>/gi },
        { kind: 'image', pattern: /<video\b[^>]*\bposter=["']([^"']+)["'][^>]*>/gi },
        { kind: 'image', pattern: /(?:background-image|background)\s*:[^;]*url\(\s*["']?([^"')]+)["']?\s*\)/gi },
        { kind: 'runtime', pattern: /["'](\/assets\/[^"']+\.(?:avif|gif|glb|hdr|jpe?g|mov|mp4|png|svg|webm|webp))["']/gi }
    ];

    patterns.forEach(({ kind, pattern }) => {
        for (const match of text.matchAll(pattern)) {
            refs.push({ kind, value: match[1].trim() });
        }
    });
    return refs;
}

const errors = [];
const warnings = new Set();
const referencedFiles = new Set();
const files = [
    ...collectFiles(path.join(repoRoot, 'blogs', 'posts'), (file) => /content-(eng|kor)\.md$/i.test(file)),
    ...collectFiles(path.join(repoRoot, 'projects'), (file) => path.basename(file) === 'content.md'),
    path.join(repoRoot, 'blogs', '3DViewer', 'index.html'),
    path.join(repoRoot, 'js', 'simple-model-viewer.js')
];

function checkReference(sourceFile, reference) {
    const { kind, value } = reference;
    if (remoteRenderedImagePattern.test(value)) {
        if (kind === 'image' || imageExtensionPattern.test(value)) {
            errors.push(`${path.relative(repoRoot, sourceFile)} keeps a remote rendered image: ${value}`);
        }
        return;
    }

    const resolved = resolveReference(sourceFile, value);
    if (!resolved) {
        return;
    }

    referencedFiles.add(path.resolve(resolved));
    if (!fs.existsSync(resolved)) {
        errors.push(`${path.relative(repoRoot, sourceFile)} -> ${value}`);
        return;
    }

    if (largeMediaPattern.test(resolved)) {
        const size = fs.statSync(resolved).size;
        if (size > largeMediaWarningBytes) {
            const sizeMb = (size / (1024 * 1024)).toFixed(1);
            warnings.add(`${path.relative(repoRoot, resolved)} (${sizeMb} MiB); prefer compression for media over ~15 MiB.`);
        }
    }
}

files.forEach((file) => {
    collectReferences(file).forEach((reference) => checkReference(file, reference));
});

const metadataPath = path.join(repoRoot, 'blogs', 'data', 'site-data.json');
routablePosts.forEach((post) => {
    checkReference(metadataPath, { kind: 'image', value: post.cover });
    checkReference(metadataPath, { kind: 'image', value: post.previewImage });
});
siteData.featuredPortfolioPosts.forEach((item) => checkReference(metadataPath, { kind: 'image', value: item.teaserImage }));
siteData.portfolioProjects.forEach((project) => {
    ['image', 'gif', 'video', 'poster'].forEach((key) => {
        if (project[key]) {
            checkReference(metadataPath, { kind: key === 'video' ? 'video' : 'image', value: project[key] });
        }
    });
});

const portfolioContentPath = path.join(repoRoot, 'content', 'portfolio', 'home.json');
if (fs.existsSync(portfolioContentPath)) {
    const portfolioContent = JSON.parse(fs.readFileSync(portfolioContentPath, 'utf8'));
    (portfolioContent.blocks || []).forEach((block) => {
        if (block.image) {
            checkReference(portfolioContentPath, { kind: 'image', value: block.image });
        }
    });
}

const localizedAssets = [
    ...collectFiles(path.join(repoRoot, 'blogs', 'posts'), (file) => path.basename(file).startsWith('remote-')),
    ...collectFiles(path.join(repoRoot, 'projects'), (file) => path.basename(file).startsWith('remote-'))
];
localizedAssets.forEach((file) => {
    if (!referencedFiles.has(path.resolve(file))) {
        errors.push(`Localized asset is not referenced by rendered content: ${path.relative(repoRoot, file)}`);
    }
});

if (errors.length > 0) {
    console.error('Asset check failed.');
    errors.forEach((error) => console.error(`- ${error}`));
    process.exit(1);
}

if (warnings.size > 0) {
    console.warn('Asset check warnings:');
    [...warnings].sort().forEach((warning) => console.warn(`- ${warning}`));
}

console.log(`Asset check passed: ${files.length} content files, ${referencedFiles.size} local media references, ${localizedAssets.length} localized assets.`);
