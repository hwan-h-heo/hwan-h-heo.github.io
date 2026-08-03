const { SITE_URL } = require('./site-config');
const { isFromArchive } = require('./site-data');
const {
    escapeHtml,
    getFeaturedPost,
    renderFeaturedPost,
    renderPostPreview,
    renderSeriesGroups,
    serializeStructuredData
} = require('./seo-utils');

const BLOG_TITLE = '3D Generative AI and CUDA Engineering | Hwan Heo';
const BLOG_DESCRIPTION = 'Technical articles on sparse 3D generation, neural rendering, mesh processing, CUDA kernels, and production inference optimization.';

function replaceOrFail(html, pattern, replacement, label) {
    if (!pattern.test(html)) {
        throw new Error(`Could not render static blog index: missing ${label}.`);
    }
    return html.replace(pattern, replacement);
}

function renderBlogHeadMetadata() {
    const structuredData = {
        '@context': 'https://schema.org',
        '@type': 'Blog',
        name: "Hwan Heo's Blog",
        url: `${SITE_URL}/blogs/`,
        description: BLOG_DESCRIPTION,
        author: {
            '@type': 'Person',
            name: 'Hwan Heo',
            url: SITE_URL
        },
        inLanguage: 'en'
    };

    return [
        `        <link rel="canonical" href="${SITE_URL}/blogs/" />`,
        '        <link rel="alternate" type="application/rss+xml" title="Hwan Heo\'s Blog" href="/blogs/feed.xml" />',
        '        <meta property="og:type" content="website" />',
        `        <meta property="og:title" content="${escapeHtml(BLOG_TITLE)}" />`,
        `        <meta property="og:description" content="${escapeHtml(BLOG_DESCRIPTION)}" />`,
        `        <meta property="og:url" content="${SITE_URL}/blogs/" />`,
        `        <meta property="og:image" content="${SITE_URL}/assets/image_fx_.jpg" />`,
        '        <meta name="twitter:card" content="summary_large_image" />',
        `        <meta name="twitter:title" content="${escapeHtml(BLOG_TITLE)}" />`,
        `        <meta name="twitter:description" content="${escapeHtml(BLOG_DESCRIPTION)}" />`,
        `        <meta name="twitter:image" content="${SITE_URL}/assets/image_fx_.jpg" />`,
        `        <script type="application/ld+json">${serializeStructuredData(structuredData)}</script>`
    ].join('\n');
}

function renderArchiveEntries(posts, lang, siteData) {
    let archiveStarted = false;

    return posts.map((post) => {
        const fromArchive = isFromArchive(post, siteData);
        const fromArchiveHeading = fromArchive && !archiveStarted
            ? `
                        <div class="blog-home-era-break">
                            <h3>
                                <span class="blog-home-era-index" aria-hidden="true">03</span>
                                <span data-i18n="fromArchiveTitle">From the Archive</span>
                            </h3>
                            <span class="blog-home-era-rule" aria-hidden="true"></span>
                        </div>`
            : '';

        if (fromArchive) {
            archiveStarted = true;
        }

        return `${fromArchiveHeading}${renderPostPreview(post, lang, siteData, {
            mediaSide: fromArchive ? 'left' : 'right'
        })}`;
    }).join('');
}

function renderStaticBlogIndex(sourceHtml, siteData) {
    const lang = 'eng';
    const featuredPost = getFeaturedPost(siteData);
    const allPosts = siteData.posts.filter((post) => post.category === 'post');
    const regularPosts = allPosts.filter((post) => post.id !== featuredPost?.id);
    const notes = siteData.posts.filter((post) => post.category === 'note');
    const seriesCount = new Set(siteData.posts.map((post) => post.series).filter(Boolean)).size;
    const postsHtml = renderArchiveEntries(regularPosts, lang, siteData)
        || '<p class="blog-home-empty">No posts yet.</p>';
    const notesHtml = renderArchiveEntries(notes, lang, siteData)
        || '<p class="blog-home-empty">No notes yet.</p>';
    const seriesHtml = renderSeriesGroups(siteData, lang)
        || '<p class="blog-home-empty">No series yet.</p>';

    let html = sourceHtml;
    html = html.replace(
        /<meta name="description" content="[^"]*" \/>/,
        `<meta name="description" content="${escapeHtml(BLOG_DESCRIPTION)}" />`
    );
    html = html.replace(
        /<title>[^<]*<\/title>/,
        `<title>${escapeHtml(BLOG_TITLE)}</title>`
    );
    html = html.replace(
        '</head>',
        `${renderBlogHeadMetadata()}\n    </head>`
    );
    html = replaceOrFail(
        html,
        /<section id="blog-home-feature" class="blog-home-feature" aria-live="polite"><\/section>/,
        `<section id="blog-home-feature" class="blog-home-feature" aria-live="polite">${renderFeaturedPost(siteData, lang)}</section>`,
        'featured section'
    );
    html = replaceOrFail(
        html,
        /<span class="blog-home-tab-count" id="posts-count"><\/span>/,
        `<span class="blog-home-tab-count" id="posts-count">${allPosts.length}</span>`,
        'posts count'
    );
    html = replaceOrFail(
        html,
        /<span class="blog-home-tab-count" id="notes-count"><\/span>/,
        `<span class="blog-home-tab-count" id="notes-count">${notes.length}</span>`,
        'notes count'
    );
    html = replaceOrFail(
        html,
        /<span class="blog-home-tab-count" id="series-count"><\/span>/,
        `<span class="blog-home-tab-count" id="series-count">${seriesCount}</span>`,
        'series count'
    );
    html = replaceOrFail(
        html,
        /(<div class="blog-home-tab-panel is-active" id="posts-tab" role="tabpanel" aria-labelledby="posts-tab-control" tabindex="0">)\s*(<\/div>)/,
        `$1\n${postsHtml}\n                    $2`,
        'posts tab'
    );
    html = replaceOrFail(
        html,
        /(<div class="blog-home-tab-panel" id="notes-tab" role="tabpanel" aria-labelledby="notes-tab-control" tabindex="0" hidden>)\s*(<\/div>)/,
        `$1\n${notesHtml}\n                    $2`,
        'notes tab'
    );
    html = replaceOrFail(
        html,
        /(<div class="blog-home-tab-panel" id="series-tab" role="tabpanel" aria-labelledby="series-tab-control" tabindex="0" hidden>)\s*(<\/div>)/,
        `$1\n${seriesHtml}\n                    $2`,
        'series tab'
    );

    return html;
}

module.exports = {
    BLOG_DESCRIPTION,
    BLOG_TITLE,
    renderStaticBlogIndex
};
