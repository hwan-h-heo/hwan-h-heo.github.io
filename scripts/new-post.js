const fs = require('fs');
const path = require('path');

const {
    createSlug,
    loadRawSiteData,
    writeSiteData
} = require('../blogs/lib/site-data');

const repoRoot = path.join(__dirname, '..');
const postsDir = path.join(repoRoot, 'blogs', 'posts');

function todayIsoDate() {
    return new Date().toISOString().slice(0, 10);
}

function postIdFromTitle(title, date) {
    const datePart = date.slice(2).replace(/-/g, '');
    const slugPart = createSlug(title).replace(/-/g, '_') || 'untitled';
    return `${datePart}_${slugPart}`;
}

function main() {
    const title = process.argv.slice(2).join(' ').trim();
    if (!title) {
        throw new Error('Usage: npm run new:post -- "Post Title"');
    }

    const date = todayIsoDate();
    const id = postIdFromTitle(title, date);
    const postDir = path.join(postsDir, id);
    if (fs.existsSync(postDir)) {
        throw new Error(`Post directory already exists: ${path.relative(repoRoot, postDir)}`);
    }

    const siteData = loadRawSiteData();
    if (siteData.posts.some((post) => post.id === id)) {
        throw new Error(`Post metadata already exists: ${id}`);
    }

    fs.mkdirSync(path.join(postDir, 'assets'), { recursive: true });
    fs.writeFileSync(path.join(postDir, 'content-eng.md'), `## Introduction\n\nStart writing ${title} here.\n`, 'utf8');

    siteData.posts.push({
        id,
        title_eng: title,
        subtitle_eng: '',
        date,
        updated: date,
        category: 'post',
        series: '3d-generation',
        tags: ['3D Generation'],
        cover: '/assets/blog_bg.jpeg',
        status: 'draft',
        languages: ['eng'],
        slug: createSlug(title)
    });

    writeSiteData(siteData);
    console.log(`Created draft post ${id}`);
    console.log(`Edit blogs/posts/${id}/content-eng.md and publish by setting status to "published".`);
}

main();
