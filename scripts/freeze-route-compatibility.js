const fs = require('fs');
const path = require('path');

const { loadSiteData } = require('../blogs/lib/site-data');
const {
    getPostRoute,
    listProjectEntries,
    listSeriesArchiveEntries,
    listTagArchiveEntries
} = require('../blogs/lib/site-routes');

const repoRoot = path.join(__dirname, '..');
const outputPath = path.join(repoRoot, 'docs', 'public-route-compatibility.md');

function buildMarkdown() {
    const siteData = loadSiteData();
    const lines = [
        '# Public Route Compatibility',
        '',
        'Generated from current source metadata. Preserve these routes unless a documented redirect or compatibility route is added.',
        '',
        '## Stable Shell Routes',
        '',
        '- `/`',
        '- `/#home`',
        '- `/#about`',
        '- `/#resume`',
        '- `/#portfolio`',
        '- `/#blog`',
        '- `/blogs/`',
        '- `/blogs/search/`',
        '- `/blogs/3DViewer/`',
        '- `/blogs/editor/` local editor UI, API only under `npm run edit`',
        '- `/hwan-h-heo.io/` old-site redirect support',
        '',
        '## Blog Post Routes',
        '',
        '| Post ID | Language | Route |',
        '| --- | --- | --- |'
    ];

    siteData.posts.forEach((post) => {
        post.languages.forEach((language) => {
            lines.push(`| \`${post.id}\` | ${language} | \`${getPostRoute(post, language)}\` |`);
        });
    });

    lines.push('', '## Legacy Blog ID Routes', '', '| Legacy ID | Canonical English Route |', '| --- | --- |');
    siteData.posts.forEach((post) => {
        lines.push(`| \`/blogs/posts/?id=${post.id}\` | \`${getPostRoute(post, 'eng')}\` |`);
    });

    lines.push('', '## Blog Archive Routes', '');
    lines.push('### Series Archives', '');
    listSeriesArchiveEntries(siteData).forEach((entry) => {
        lines.push(`- \`${entry.path}\` (${entry.posts.length} posts)`);
    });
    lines.push('', '### Tag Archives', '');
    listTagArchiveEntries(siteData).forEach((entry) => {
        lines.push(`- \`${entry.path}\` (${entry.posts.length} posts)`);
    });

    lines.push('', '## Project Routes', '');
    listProjectEntries(repoRoot).forEach((entry) => {
        lines.push(`- \`${entry.route}\``);
    });

    lines.push('', '## Compatibility Notes', '');
    lines.push('- Blog slugs are now pinned in `blogs/data/site-data.json` with the `slug` field.');
    lines.push('- English routes use `/blogs/posts/<slug>/`; Korean routes use `/blogs/posts/<slug>-kor/`.');
    lines.push('- Project routes are generated from `projects/<slug>/project.json` plus `content.md` and copied into `blogs/dist/projects/<slug>/`.');
    lines.push('- The static build must keep copying `index.html`, `assets/`, `css/`, `js/`, `content/`, and `blogs/data/site-data.json` into `blogs/dist`.');

    return `${lines.join('\n')}\n`;
}

fs.mkdirSync(path.dirname(outputPath), { recursive: true });
fs.writeFileSync(outputPath, buildMarkdown(), 'utf8');
console.log(`Wrote ${path.relative(repoRoot, outputPath)}`);
