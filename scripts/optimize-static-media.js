const fs = require('fs');
const path = require('path');

const sharp = require('../blogs/node_modules/sharp');

const repoRoot = path.join(__dirname, '..');
const mediaTargets = [
    {
        input: 'assets/image_fx_.jpg',
        output: 'assets/image_fx_.webp',
        width: 1408,
        quality: 78,
        maxBytes: 300 * 1024
    },
    {
        input: 'assets/profile4.png',
        output: 'assets/profile4.webp',
        width: 1049,
        quality: 82,
        maxBytes: 300 * 1024
    },
    {
        input: 'assets/thumbnails/varco3d_thumbnail.png',
        output: 'assets/thumbnails/varco3d_thumbnail.webp',
        width: 1280,
        quality: 80,
        maxBytes: 350 * 1024
    }
];

function formatBytes(bytes) {
    return `${Math.max(1, Math.round(bytes / 1024))} KB`;
}

async function optimizeTarget(target) {
    const inputPath = path.join(repoRoot, target.input);
    const outputPath = path.join(repoRoot, target.output);

    if (!fs.existsSync(inputPath)) {
        throw new Error(`Missing source media: ${target.input}`);
    }

    fs.mkdirSync(path.dirname(outputPath), { recursive: true });
    await sharp(inputPath)
        .rotate()
        .resize({
            width: target.width,
            fit: 'inside',
            withoutEnlargement: true
        })
        .webp({
            effort: 6,
            quality: target.quality,
            smartSubsample: true
        })
        .toFile(outputPath);

    const inputBytes = fs.statSync(inputPath).size;
    const outputBytes = fs.statSync(outputPath).size;
    if (outputBytes > target.maxBytes) {
        throw new Error(
            `${target.output} is ${formatBytes(outputBytes)}; budget is ${formatBytes(target.maxBytes)}.`
        );
    }

    console.log(`${target.output}: ${formatBytes(inputBytes)} -> ${formatBytes(outputBytes)}`);
}

async function main() {
    for (const target of mediaTargets) {
        await optimizeTarget(target);
    }
}

main().catch((error) => {
    console.error(error.message);
    process.exit(1);
});
