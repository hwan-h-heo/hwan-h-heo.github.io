const assert = require('node:assert/strict');
const fs = require('node:fs');
const os = require('node:os');
const path = require('node:path');
const test = require('node:test');
const { extractSearchText, generateSearchIndex } = require('../blogs/lib/search-index');
const { getQueryTerms, getSearchExcerpt, searchPosts, splitHighlights } = require('../blogs/js/blog-search');

function post(id, title, overrides = {}) {
    return { id, title_eng: title, date: '2026-01-01', languages: ['eng'], tags: [], ...overrides };
}

test('title and topic matches outrank recent body mentions, with stable date ties', () => {
    const posts = [
        post('body', 'Recent research', { date: '2026-10-25' }),
        post('tag', 'Implementation', { tags: ['CUDA'], date: '2026-10-22' }),
        post('series', 'GPU programming', { series: 'cuda' }),
        post('title', 'CUDA kernels', { date: '2024-01-01' }),
        post('older', 'Other research', { date: '2023-01-01' })
    ];
    const index = { posts: Object.fromEntries(posts.map((entry) => [entry.id, { eng: 'CUDA enables parallel execution.' }])) };
    const results = searchPosts({ posts, series: { cuda: { eng: 'CUDA' } } }, index, 'CUDA', 'eng');
    assert.deepEqual(results.map((entry) => entry.post.id), ['title', 'series', 'tag', 'body', 'older']);
});

test('all query terms may span title, topics and body, including Korean and technical names', () => {
    const posts = [post('match', 'CUDA kernels', { tags: ['Inference'] }), post('miss', 'CUDA kernels')];
    const index = { posts: { match: { eng: 'Use torch.compile for 추론 workloads.' }, miss: { eng: 'Only CUDA.' } } };
    const data = { posts };
    assert.deepEqual(searchPosts(data, index, '  CUDA   inference 추론 torch.compile ', 'eng').map((entry) => entry.post.id), ['match']);
    assert.deepEqual(getQueryTerms('C++ CUDA, torch.compile GPU-memory CUDA'), ['c++', 'cuda', 'torch.compile', 'gpu-memory']);
    assert.deepEqual(searchPosts(data, index, 'not-found-token', 'eng'), []);
    assert.deepEqual(searchPosts(data, index, '  ', 'eng'), []);
});

test('Korean search retains English-only metadata and body without searching the wrong bilingual body', () => {
    const posts = [
        post('english-only', 'Sora and 3D', { title_kor: 'Sora가 상상하는 세계' }),
        post('bilingual', 'Two languages', { title_kor: '두 언어', languages: ['eng', 'kor'] })
    ];
    const index = { posts: { 'english-only': { eng: 'Photogrammetry experiments.' }, bilingual: { eng: 'Photogrammetry.', kor: '한국어 본문입니다.' } } };
    assert.deepEqual(searchPosts({ posts }, index, 'photogrammetry', 'kor').map((entry) => entry.post.id), ['english-only']);
    assert.deepEqual(searchPosts({ posts }, index, '상상하는', 'kor').map((entry) => entry.post.id), ['english-only']);
    assert.deepEqual(searchPosts({ posts }, index, 'Sora and 3D', 'kor').map((entry) => entry.post.id), ['english-only']);
});

test('excerpts select matching context near the end of an article and favour coverage of multiple terms', () => {
    const body = `${'Unrelated introduction. '.repeat(80)}\nCUDA is mentioned alone.\n${'More background. '.repeat(40)}\nCUDA inference removes unnecessary memory traffic and improves latency.`;
    const excerpt = getSearchExcerpt(body, ['cuda', 'inference']);
    assert(excerpt.includes('CUDA inference'));
    assert(!excerpt.includes('introduction'));
    assert(excerpt.length <= 182);
    assert.equal(getSearchExcerpt(body, ['unmatched']), '');
});

test('highlighting preserves literal text and handles regular-expression characters safely', () => {
    const text = 'C++ and torch.compile use <script> as plain text.';
    const parts = splitHighlights(text, 'C++ torch.compile');
    assert.equal(parts.map((part) => part.text).join(''), text);
    assert.deepEqual(parts.filter((part) => part.match).map((part) => part.text), ['C++', 'torch.compile']);
    assert.deepEqual(splitHighlights('No query', ''), [{ text: 'No query', match: false }]);
});

test('excerpt boundaries keep both matches when they fit within the available context', () => {
    for (let gap = 1; gap < 16; gap += 1) {
        const source = `${'Background '.repeat(20)}CUDA ${'word '.repeat(gap)}inference ${'following '.repeat(30)}`;
        const excerpt = getSearchExcerpt(source, ['cuda', 'inference'], 100);
        assert(excerpt.includes('CUDA') && excerpt.includes('inference'), `Missing a term at gap ${gap}: ${excerpt}`);
        assert(excerpt.length <= 102);
    }
});

test('plain text contains visible prose, captions and code, excluding executable embeds, preambles and asset paths', () => {
    const source = `---\ndependencies: hidden-preamble\n---\n# CUDA & Memory\nReadable **explanation** with [reference](https://asset.invalid/private-path).\n![A helpful figure](./assets/private-file.webp)\n\n\`\`\`cpp\n__syncthreads();\n\`\`\`\n<style>.hidden-style { color: red; }</style><script>hiddenScript()</script><div hidden>hidden controls</div>`;
    const text = extractSearchText(source);
    assert(text.includes('CUDA & Memory'));
    assert(text.includes('Readable explanation with reference.'));
    assert(text.includes('A helpful figure'));
    assert(text.includes('__syncthreads();'));
    assert(!/hidden-preamble|asset.invalid|private-path|private-file|hidden-style|hiddenScript|hidden controls/.test(text));
    assert.equal(extractSearchText('A softly\nwrapped paragraph.\n\nA second paragraph.'), 'A softly wrapped paragraph.\nA second paragraph.');
});

test('index is published-only, fingerprinted, and refreshed by content edits without retaining stale files', (t) => {
    const root = fs.mkdtempSync(path.join(os.tmpdir(), 'blog-search-'));
    t.after(() => fs.rmSync(root, { recursive: true, force: true }));
    const distDir = path.join(root, 'dist');
    fs.mkdirSync(path.join(root, 'blogs/posts/public'), { recursive: true });
    fs.mkdirSync(path.join(distDir, 'blogs/search'), { recursive: true });
    fs.writeFileSync(path.join(distDir, 'blogs/search/index.html'), '<meta name="blog-search-index" content="" />');
    const source = path.join(root, 'blogs/posts/public/content-eng.md');
    fs.writeFileSync(source, 'Published first version.');
    const siteData = { posts: [post('public', 'Public'), post('draft', 'Draft secret', { status: 'draft' }), post('unlisted', 'Unlisted secret', { status: 'unlisted' })] };
    const first = generateSearchIndex({ siteData, repoRoot: root, distDir });
    const index = JSON.parse(fs.readFileSync(path.join(distDir, first), 'utf8'));
    assert.deepEqual(Object.keys(index.posts), ['public']);
    assert.equal(generateSearchIndex({ siteData, repoRoot: root, distDir }), first);
    fs.writeFileSync(source, 'Published second version.');
    const next = generateSearchIndex({ siteData, repoRoot: root, distDir });
    assert.notEqual(next, first);
    assert(!fs.existsSync(path.join(distDir, first)));
    assert(fs.readFileSync(path.join(distDir, 'blogs/search/index.html'), 'utf8').includes(next));
});
