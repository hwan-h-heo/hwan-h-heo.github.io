const crypto = require('crypto');
const fs = require('fs');
const path = require('path');
const { load } = require('cheerio');
const { marked } = require('marked');
const { parseMarkdownWithMath } = require('../js/markdown-with-math');
const { parsePostMarkdownSource } = require('./post-runtime-dependencies');

function extractSearchText(markdown) {
    const { content } = parsePostMarkdownSource(markdown);
    const $ = load(parseMarkdownWithMath(content, (source) => marked.parse(source)));
    $('script,style,template,noscript,svg,canvas,iframe,nav,dialog,button,input,select,textarea,[hidden],[aria-hidden="true"]').remove();
    $('img[alt]').each((index, image) => {
        $(image).replaceWith($('<span>').text(` ${$(image).attr('alt')} `));
    });
    // Keep rendered block boundaries, while folding source-only soft line breaks.
    const paragraphBreak = '\u2029';
    $('br').replaceWith(paragraphBreak);
    $('h1,h2,h3,h4,h5,h6,p,li,figcaption,blockquote,pre,tr,div,section,figure,summary').append(paragraphBreak);
    return $('body').text()
        .normalize('NFC')
        .split(paragraphBreak)
        .map((line) => line.replace(/\s+/g, ' ').trim())
        .filter(Boolean)
        .join('\n');
}

function generateSearchIndex({ siteData, repoRoot, distDir }) {
    const posts = {};
    siteData.posts.filter((post) => (post.status || 'published') === 'published').forEach((post) => {
        posts[post.id] = Object.fromEntries(post.languages.map((language) => {
            const source = fs.readFileSync(path.join(repoRoot, 'blogs', 'posts', post.id, `content-${language}.md`), 'utf8');
            return [language, extractSearchText(source)];
        }));
    });
    const serialized = JSON.stringify({ version: 1, posts });
    const hash = crypto.createHash('sha256').update(serialized).digest('hex').slice(0, 12);
    const fileName = `search-index-${hash}.json`;
    const dataDir = path.join(distDir, 'blogs', 'data');
    fs.mkdirSync(dataDir, { recursive: true });
    fs.readdirSync(dataDir).forEach((file) => {
        if (/^search-index-[a-f0-9]{12}\.json$/.test(file) && file !== fileName) {
            fs.rmSync(path.join(dataDir, file));
        }
    });
    fs.writeFileSync(path.join(dataDir, fileName), serialized);

    const searchPage = path.join(distDir, 'blogs', 'search', 'index.html');
    const html = fs.readFileSync(searchPage, 'utf8');
    const marker = /<meta name="blog-search-index" content="[^"]*"\s*\/?>/;
    if (!marker.test(html)) {
        throw new Error('Search page is missing the blog-search-index meta element.');
    }
    const url = `/blogs/data/${fileName}`;
    fs.writeFileSync(searchPage, html.replace(marker, `<meta name="blog-search-index" content="${url}" />`));
    console.log(`Generated search index: ${Object.keys(posts).length} posts, ${Buffer.byteLength(serialized)} bytes.`);
    return url;
}

module.exports = { extractSearchText, generateSearchIndex };
