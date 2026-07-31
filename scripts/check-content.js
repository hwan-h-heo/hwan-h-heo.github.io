const fs = require('fs');
const path = require('path');

const { loadRawSiteData, loadSiteData, validateSiteData } = require('../blogs/lib/site-data');
const { renderBlock } = require('../js/portfolio-blocks');

const repoRoot = path.join(__dirname, '..');

function fail(message, errors) {
    console.error(message);
    errors.forEach((error) => console.error(`- ${error}`));
    process.exit(1);
}

function readJson(filePath) {
    return JSON.parse(fs.readFileSync(filePath, 'utf8'));
}

const errors = [];
const rawSiteData = loadRawSiteData();
validateSiteData(rawSiteData);
const siteData = loadSiteData();

siteData.posts.forEach((post) => {
    if (!post.cover || post.cover === '/assets/blog_bg.jpeg') {
        errors.push(`Post ${post.id} must define a post-specific cover image.`);
    }

    post.languages.forEach((language) => {
        const filePath = path.join(repoRoot, 'blogs', 'posts', post.id, `content-${language}.md`);
        if (!fs.existsSync(filePath)) {
            errors.push(`Missing post content: ${path.relative(repoRoot, filePath)}`);
            return;
        }

        const content = fs.readFileSync(filePath, 'utf8');
        if (/id=(["'])(?:copyButton|myshare_modal)\1/.test(content)) {
            errors.push(`Post source must not embed shared link controls: ${path.relative(repoRoot, filePath)}`);
        }
    });
});

const portfolioPath = path.join(repoRoot, 'content', 'portfolio', 'home.json');
const portfolio = readJson(portfolioPath);
if (!Array.isArray(portfolio.blocks) || portfolio.blocks.length === 0) {
    errors.push('content/portfolio/home.json must define a non-empty blocks array.');
}

const requiredBlockIds = ['home', 'about', 'resume'];
const blockIds = new Set((portfolio.blocks || []).map((block) => block.id));
if (blockIds.size !== (portfolio.blocks || []).length) {
    errors.push('content/portfolio/home.json contains duplicate block ids.');
}
requiredBlockIds.forEach((id) => {
    if (!blockIds.has(id)) {
        errors.push(`Missing portfolio block "${id}".`);
    }
});
(portfolio.blocks || []).forEach((block) => {
    if (!block.id || !block.type) {
        errors.push('Every portfolio block must define id and type.');
        return;
    }
    if (!renderBlock(block).trim()) {
        errors.push(`Portfolio block "${block.id}" uses unsupported type "${block.type}".`);
    }
});

const projectsDir = path.join(repoRoot, 'projects');
fs.readdirSync(projectsDir, { withFileTypes: true })
    .filter((entry) => entry.isDirectory())
    .forEach((entry) => {
        const projectDir = path.join(projectsDir, entry.name);
        const hasMetadata = fs.existsSync(path.join(projectDir, 'project.json'));
        const hasContent = fs.existsSync(path.join(projectDir, 'content.md'));
        if (hasMetadata !== hasContent) {
            errors.push(`Project ${entry.name} must have both project.json and content.md.`);
        }
    });

if (errors.length > 0) {
    fail('Content check failed.', errors);
}

console.log(`Content check passed: ${siteData.posts.length} published posts, ${portfolio.blocks.length} portfolio blocks.`);
