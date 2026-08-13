const http = require('http');
const fs = require('fs');
const path = require('path');
const url = require('url');
const { normalizeMathMarkdown } = require('./js/markdown-with-math');

const {
    POST_CATEGORIES,
    POST_LANGUAGES,
    POST_STATUSES,
    loadRawSiteData,
    validateSiteData,
    writeSiteData
} = require('./lib/site-data');

const PORT = 3030;
const ROOT_DIR = __dirname;
const SITE_ROOT_DIR = path.join(ROOT_DIR, '..');
const POSTS_DIR = path.join(ROOT_DIR, 'posts');
const DRAFTS_DIR = path.join(ROOT_DIR, 'editor', 'drafts');
const DRAFT_ASSETS_DIR = path.join(ROOT_DIR, 'editor', 'draft-assets');
const POST_ID_PATTERN = /^\d{6}_[A-Za-z0-9_]+$/;
const CONTENT_DELIMITER = '--- 여기부터 실제 콘텐츠 ---';

const mimeTypes = {
    '.html': 'text/html',
    '.js': 'text/javascript',
    '.css': 'text/css',
    '.json': 'application/json',
    '.png': 'image/png',
    '.jpg': 'image/jpg',
    '.jpeg': 'image/jpeg',
    '.gif': 'image/gif',
    '.webp': 'image/webp',
    '.svg': 'image/svg+xml',
    '.ico': 'image/x-icon',
    '.md': 'text/markdown'
};

ensureDirSync(DRAFTS_DIR);
ensureDirSync(DRAFT_ASSETS_DIR);

function ensureDirSync(dirPath) {
    if (!fs.existsSync(dirPath)) {
        fs.mkdirSync(dirPath, { recursive: true });
    }
}

function sendJson(res, statusCode, payload) {
    res.writeHead(statusCode, { 'Content-Type': 'application/json' });
    res.end(JSON.stringify(payload));
}

function serveStaticFile(filePath, res) {
    fs.readFile(filePath, (err, data) => {
        if (err) {
            sendJson(res, 404, { error: 'File not found' });
            return;
        }

        const ext = path.extname(filePath);
        res.writeHead(200, { 'Content-Type': mimeTypes[ext] || 'application/octet-stream' });
        res.end(data);
    });
}

function resolveInside(baseDir, targetPath) {
    const resolvedPath = path.resolve(baseDir, targetPath);
    const resolvedBase = path.resolve(baseDir);
    if (resolvedPath !== resolvedBase && !resolvedPath.startsWith(`${resolvedBase}${path.sep}`)) {
        throw new Error('Invalid path');
    }
    return resolvedPath;
}

function sanitizeFileName(fileName, defaultExtension = '') {
    const normalized = path.basename(String(fileName || '').trim());
    if (!normalized || normalized === '.' || normalized === '..') {
        throw new Error('Invalid file name');
    }
    return defaultExtension && !normalized.endsWith(defaultExtension)
        ? `${normalized}${defaultExtension}`
        : normalized;
}

function parseJsonBody(req) {
    return new Promise((resolve, reject) => {
        let body = '';

        req.on('data', (chunk) => {
            body += chunk.toString();
        });

        req.on('end', () => {
            if (!body) {
                resolve({});
                return;
            }

            try {
                resolve(JSON.parse(body));
            } catch (error) {
                reject(new Error('Invalid JSON'));
            }
        });

        req.on('error', reject);
    });
}

function parseMultipart(req, callback) {
    const contentType = req.headers['content-type'] || '';
    const boundary = contentType.split('boundary=')[1];
    if (!boundary) {
        callback(new Error('No boundary found'));
        return;
    }

    let data = Buffer.alloc(0);
    req.on('data', (chunk) => {
        data = Buffer.concat([data, chunk]);
    });

    req.on('end', () => {
        const parts = [];
        const boundaryBuffer = Buffer.from(`--${boundary}`);
        let start = 0;

        while (true) {
            const boundaryIndex = data.indexOf(boundaryBuffer, start);
            if (boundaryIndex === -1) {
                break;
            }

            const nextBoundaryIndex = data.indexOf(boundaryBuffer, boundaryIndex + boundaryBuffer.length);
            if (nextBoundaryIndex === -1) {
                break;
            }

            const partData = data.slice(boundaryIndex + boundaryBuffer.length, nextBoundaryIndex);
            const headerEnd = partData.indexOf(Buffer.from('\r\n\r\n'));
            if (headerEnd !== -1) {
                const headers = partData.slice(0, headerEnd).toString();
                const content = partData.slice(headerEnd + 4, partData.length - 2);
                const nameMatch = headers.match(/name="([^"]+)"/);
                const filenameMatch = headers.match(/filename="([^"]+)"/);
                const contentTypeMatch = headers.match(/content-type:\s*([^\r\n]+)/i);

                if (nameMatch) {
                    parts.push({
                        name: nameMatch[1],
                        filename: filenameMatch ? path.basename(filenameMatch[1]) : null,
                        contentType: contentTypeMatch ? contentTypeMatch[1].trim() : '',
                        data: content
                    });
                }
            }

            start = nextBoundaryIndex;
        }

        callback(null, parts);
    });
}

function isValidIsoDate(value) {
    if (!/^\d{4}-\d{2}-\d{2}$/.test(value)) {
        return false;
    }

    const date = new Date(`${value}T00:00:00Z`);
    return !Number.isNaN(date.getTime()) && date.toISOString().startsWith(value);
}

function sortPostsByDate(posts) {
    return [...posts].sort((a, b) => new Date(b.date) - new Date(a.date));
}

function stripLegacyContentPreamble(content) {
    const text = typeof content === 'string' ? content : '';
    const parts = text.split(CONTENT_DELIMITER);
    return parts.length > 1 ? parts.slice(1).join(CONTENT_DELIMITER).trim() : text;
}

function buildBootstrapPayload() {
    const rawSiteData = loadRawSiteData();
    validateSiteData(rawSiteData);

    const featuredMap = new Map(
        (rawSiteData.featuredPortfolioPosts || []).map((item, index) => [item.id, { ...item, order: index }])
    );

    return {
        categories: POST_CATEGORIES,
        languages: POST_LANGUAGES,
        statuses: POST_STATUSES,
        blogHome: { ...rawSiteData.blogHome },
        series: rawSiteData.series,
        featuredPortfolioPosts: rawSiteData.featuredPortfolioPosts || [],
        posts: sortPostsByDate(rawSiteData.posts).map((post) => ({
            id: post.id,
            title_eng: post.title_eng,
            title_kor: post.title_kor || '',
            subtitle_eng: post.subtitle_eng || '',
            subtitle_kor: post.subtitle_kor || '',
            tags: Array.isArray(post.tags) ? post.tags : [],
            cover: post.cover || '',
            status: post.status || 'published',
            updated: post.updated || post.date,
            slug: post.slug || '',
            date: post.date,
            category: post.category,
            series: post.series,
            languages: post.languages,
            featured: featuredMap.has(post.id)
        }))
    };
}

function saveBlogHomeSettings(payload) {
    const rawSiteData = loadRawSiteData();
    validateSiteData(rawSiteData);

    const featuredPostId = String(payload?.featuredPostId || '').trim();
    const featuredPost = rawSiteData.posts.find((post) => post.id === featuredPostId);
    const errors = [];

    if (!featuredPostId) {
        errors.push('Select a featured post for the blog home.');
    } else if (!featuredPost) {
        errors.push(`Post "${featuredPostId}" does not exist.`);
    } else {
        if ((featuredPost.status || 'published') !== 'published') {
            errors.push('The blog home featured post must be published.');
        }
        if (featuredPost.category !== 'post') {
            errors.push('The blog home featured post must use the post category.');
        }
    }

    if (errors.length > 0) {
        const validationError = new Error(errors.join('\n'));
        validationError.validationErrors = errors;
        throw validationError;
    }

    const nextSiteData = {
        ...rawSiteData,
        blogHome: {
            ...rawSiteData.blogHome,
            featuredPostId
        }
    };
    writeSiteData(nextSiteData);

    return { ...nextSiteData.blogHome };
}

function readPostBundle(postId) {
    const rawSiteData = loadRawSiteData();
    validateSiteData(rawSiteData);

    const post = rawSiteData.posts.find((item) => item.id === postId);
    if (!post) {
        return null;
    }

    const contents = {};
    POST_LANGUAGES.forEach((language) => {
        const filePath = path.join(POSTS_DIR, postId, `content-${language}.md`);
        if (fs.existsSync(filePath)) {
            contents[language] = stripLegacyContentPreamble(fs.readFileSync(filePath, 'utf8'));
        }
    });

    const featured = (rawSiteData.featuredPortfolioPosts || []).find((item) => item.id === postId) || null;
    const featuredOrder = featured
        ? (rawSiteData.featuredPortfolioPosts || []).findIndex((item) => item.id === postId)
        : null;

    return {
        post,
        featured: featured ? { ...featured, order: featuredOrder } : null,
        contents
    };
}

function sanitizePostInput(rawSiteData, payload) {
    const mode = payload.mode === 'update' ? 'update' : 'create';
    const errors = [];

    const originalId = String(payload.originalId || '').trim();
    const post = payload.post && typeof payload.post === 'object' ? payload.post : {};
    const contentsInput = payload.contents && typeof payload.contents === 'object' ? payload.contents : {};
    const featuredInput = payload.featured && typeof payload.featured === 'object' ? payload.featured : {};

    const sanitizedPost = {
        id: String(post.id || '').trim(),
        title_eng: String(post.title_eng || '').trim(),
        date: String(post.date || '').trim(),
        category: String(post.category || '').trim(),
        series: String(post.series || '').trim(),
        languages: Array.isArray(post.languages)
            ? [...new Set(post.languages.map((lang) => String(lang || '').trim()).filter(Boolean))]
            : []
    };

    const titleKor = String(post.title_kor || '').trim();
    const subtitleEng = String(post.subtitle_eng || '').trim();
    const subtitleKor = String(post.subtitle_kor || '').trim();
    const cover = String(post.cover || '').trim();
    const status = String(post.status || '').trim();
    const updated = String(post.updated || '').trim();
    const slug = String(post.slug || '').trim();
    if (titleKor) {
        sanitizedPost.title_kor = titleKor;
    }
    if (subtitleEng) {
        sanitizedPost.subtitle_eng = subtitleEng;
    }
    if (subtitleKor) {
        sanitizedPost.subtitle_kor = subtitleKor;
    }
    if (cover) {
        sanitizedPost.cover = cover;
    }
    if (status) {
        sanitizedPost.status = status;
    }
    if (updated) {
        sanitizedPost.updated = updated;
    }
    if (slug) {
        sanitizedPost.slug = slug;
    }
    if (Array.isArray(post.tags)) {
        sanitizedPost.tags = [...new Set(post.tags.map((tag) => String(tag || '').trim()).filter(Boolean))];
    }

    const existingPost = rawSiteData.posts.find((item) => item.id === sanitizedPost.id) || null;
    const originalPost = rawSiteData.posts.find((item) => item.id === originalId) || null;

    if (!sanitizedPost.id) {
        errors.push('Post ID is required.');
    } else if (!POST_ID_PATTERN.test(sanitizedPost.id)) {
        errors.push('Post ID must match YYMMDD_slug and use only letters, numbers, and underscores.');
    }

    if (!sanitizedPost.title_eng) {
        errors.push('English title is required.');
    }

    if (!isValidIsoDate(sanitizedPost.date)) {
        errors.push('Date must be a valid YYYY-MM-DD string.');
    }

    if (sanitizedPost.updated && !isValidIsoDate(sanitizedPost.updated)) {
        errors.push('Updated date must be a valid YYYY-MM-DD string.');
    }

    if (sanitizedPost.updated && sanitizedPost.date && sanitizedPost.updated < sanitizedPost.date) {
        errors.push('Updated date cannot precede the published date.');
    }

    if (sanitizedPost.slug && !/^[a-z0-9]+(?:-[a-z0-9]+)*$/.test(sanitizedPost.slug)) {
        errors.push('Slug must use lowercase words separated by hyphens.');
    }

    if (sanitizedPost.status && !POST_STATUSES.includes(sanitizedPost.status)) {
        errors.push(`Status must be one of: ${POST_STATUSES.join(', ')}.`);
    }

    if ((sanitizedPost.status || originalPost?.status || 'published') !== 'draft') {
        if (!(sanitizedPost.slug || originalPost?.slug)) {
            errors.push('Published and unlisted posts require a stable slug.');
        }
        if (!(sanitizedPost.subtitle_eng || originalPost?.subtitle_eng)) {
            errors.push('Published and unlisted posts require an English subtitle.');
        }
        if (
            sanitizedPost.languages.includes('kor')
            && !(sanitizedPost.subtitle_kor || originalPost?.subtitle_kor)
        ) {
            errors.push('Published and unlisted Korean posts require a Korean subtitle.');
        }
        if (!(sanitizedPost.cover || originalPost?.cover) || (sanitizedPost.cover || originalPost?.cover) === '/assets/blog_bg.jpeg') {
            errors.push('Published and unlisted posts require a post-specific cover image.');
        }
        if ((sanitizedPost.tags || originalPost?.tags || []).length === 0) {
            errors.push('Published and unlisted posts require at least one tag.');
        }
    }

    if (!POST_CATEGORIES.includes(sanitizedPost.category)) {
        errors.push(`Category must be one of: ${POST_CATEGORIES.join(', ')}.`);
    }

    if (!rawSiteData.series[sanitizedPost.series]) {
        errors.push('Series must be one of the configured site-data series.');
    }

    if (sanitizedPost.languages.length === 0) {
        errors.push('At least one language must be selected.');
    }

    sanitizedPost.languages.forEach((language) => {
        if (!POST_LANGUAGES.includes(language)) {
            errors.push(`Unsupported language "${language}".`);
        }
    });

    if (!sanitizedPost.languages.includes('eng')) {
        errors.push('English content is required for every post.');
    }

    if (sanitizedPost.languages.includes('kor') && !sanitizedPost.title_kor) {
        errors.push('Korean title is required when Korean content is enabled.');
    }

    const contentKeys = Object.keys(contentsInput);
    contentKeys.forEach((language) => {
        if (!sanitizedPost.languages.includes(language)) {
            errors.push(`Content for "${language}" was provided but that language is not selected.`);
        }
    });

    const sanitizedContents = {};
    sanitizedPost.languages.forEach((language) => {
        if (typeof contentsInput[language] !== 'string') {
            errors.push(`Content for "${language}" is required.`);
            return;
        }
        sanitizedContents[language] = normalizeMathMarkdown(
            stripLegacyContentPreamble(contentsInput[language])
        );
    });

    const featured = {
        enabled: Boolean(featuredInput.enabled),
        teaserImage: String(featuredInput.teaserImage || '').trim(),
        teaserAlt: String(featuredInput.teaserAlt || '').trim()
    };

    if (featured.enabled) {
        if (!featured.teaserImage) {
            errors.push('Teaser image is required when the post is featured on the portfolio.');
        }

        const requestedOrder = Number.parseInt(featuredInput.order, 10);
        featured.order = Number.isInteger(requestedOrder) ? requestedOrder : rawSiteData.featuredPortfolioPosts.length;
    }

    if (mode === 'create') {
        if (existingPost) {
            errors.push(`Post ID "${sanitizedPost.id}" already exists.`);
        }

        const postDir = path.join(POSTS_DIR, sanitizedPost.id);
        if (fs.existsSync(postDir)) {
            errors.push(`Post directory "${sanitizedPost.id}" already exists on disk.`);
        }
    }

    if (mode === 'update') {
        if (!originalId) {
            errors.push('originalId is required for updates.');
        } else if (!originalPost) {
            errors.push(`Existing post "${originalId}" was not found.`);
        } else if (originalId !== sanitizedPost.id) {
            errors.push('Renaming an existing post ID is not supported.');
        }

        if (originalPost) {
            originalPost.languages.forEach((language) => {
                if (!sanitizedPost.languages.includes(language)) {
                    errors.push(`Removing existing language "${language}" is not supported in the editor yet.`);
                }
            });
        }
    }

    if (errors.length > 0) {
        const validationError = new Error(errors.join('\n'));
        validationError.validationErrors = errors;
        throw validationError;
    }

    return {
        mode,
        originalId,
        existingPost,
        originalPost,
        post: sanitizedPost,
        contents: sanitizedContents,
        featured
    };
}

function buildNextSiteData(rawSiteData, sanitizedPayload) {
    const nextSiteData = {
        ...rawSiteData,
        posts: [...rawSiteData.posts],
        featuredPortfolioPosts: [...(rawSiteData.featuredPortfolioPosts || [])]
    };

    const nextPost = sanitizedPayload.originalPost
        ? { ...sanitizedPayload.originalPost, ...sanitizedPayload.post }
        : { ...sanitizedPayload.post };
    const existingIndex = nextSiteData.posts.findIndex((item) => item.id === nextPost.id);
    if (existingIndex >= 0) {
        nextSiteData.posts[existingIndex] = nextPost;
    } else {
        nextSiteData.posts.push(nextPost);
    }

    nextSiteData.posts = sortPostsByDate(nextSiteData.posts);

    nextSiteData.featuredPortfolioPosts = nextSiteData.featuredPortfolioPosts.filter((item) => item.id !== nextPost.id);
    if (sanitizedPayload.featured.enabled) {
        const nextFeaturedItem = {
            id: nextPost.id,
            teaserImage: sanitizedPayload.featured.teaserImage,
            teaserAlt: sanitizedPayload.featured.teaserAlt
        };

        const insertIndex = Math.max(
            0,
            Math.min(sanitizedPayload.featured.order, nextSiteData.featuredPortfolioPosts.length)
        );
        nextSiteData.featuredPortfolioPosts.splice(insertIndex, 0, nextFeaturedItem);
    }

    validateSiteData(nextSiteData);
    return nextSiteData;
}

function migrateDraftAssets(content, postId, migrationState) {
    const assetPattern = /\.\/draft-assets\/([^\s)]+)/g;
    const assetNames = [...content.matchAll(assetPattern)].map((match) => match[1]);
    if (assetNames.length === 0) {
        return { content, migrated: [] };
    }

    const postDir = resolveInside(POSTS_DIR, postId);
    const postAssetsDir = path.join(postDir, 'assets');
    ensureDirSync(postAssetsDir);

    const migrated = [];
    let updatedContent = content;

    [...new Set(assetNames)].forEach((assetName) => {
        const sourcePath = resolveInside(DRAFT_ASSETS_DIR, assetName);
        if (!fs.existsSync(sourcePath)) {
            throw new Error(`Draft asset "${assetName}" was not found. Re-upload the image and try again.`);
        }

        const destinationPath = resolveInside(postAssetsDir, assetName);
        if (!migrationState.copiedAssets.has(assetName)) {
            fs.copyFileSync(sourcePath, destinationPath);
            migrationState.copiedAssets.add(assetName);
            migrationState.createdAssets.push(destinationPath);
        }

        const sourceReference = `./draft-assets/${assetName}`;
        const targetReference = `./assets/${assetName}`;
        updatedContent = updatedContent.split(sourceReference).join(targetReference);
        migrated.push(assetName);
    });

    return { content: updatedContent, migrated };
}

function savePostBundle(payload) {
    const rawSiteData = loadRawSiteData();
    validateSiteData(rawSiteData);

    const sanitizedPayload = sanitizePostInput(rawSiteData, payload);
    const nextSiteData = buildNextSiteData(rawSiteData, sanitizedPayload);
    const postDir = resolveInside(POSTS_DIR, sanitizedPayload.post.id);

    ensureDirSync(postDir);

    const migrationState = {
        copiedAssets: new Set(),
        createdAssets: [],
        fileBackups: []
    };

    try {
        const savedFiles = [];
        sanitizedPayload.post.languages.forEach((language) => {
            const filePath = resolveInside(postDir, `content-${language}.md`);
            const previousContent = fs.existsSync(filePath) ? fs.readFileSync(filePath, 'utf8') : null;
            migrationState.fileBackups.push({ filePath, previousContent });

            const { content, migrated } = migrateDraftAssets(
                sanitizedPayload.contents[language],
                sanitizedPayload.post.id,
                migrationState
            );

            fs.writeFileSync(filePath, content, 'utf8');
            savedFiles.push({
                language,
                path: filePath,
                assetsMigrated: migrated.length
            });
        });

        writeSiteData(nextSiteData);

        return {
            mode: sanitizedPayload.mode,
            postId: sanitizedPayload.post.id,
            savedFiles,
            featured: sanitizedPayload.featured.enabled,
            featuredOrder: sanitizedPayload.featured.enabled ? sanitizedPayload.featured.order : null
        };
    } catch (error) {
        migrationState.fileBackups.reverse().forEach(({ filePath, previousContent }) => {
            if (previousContent === null) {
                if (fs.existsSync(filePath)) {
                    fs.unlinkSync(filePath);
                }
                return;
            }

            fs.writeFileSync(filePath, previousContent, 'utf8');
        });

        migrationState.createdAssets.reverse().forEach((assetPath) => {
            if (fs.existsSync(assetPath)) {
                fs.unlinkSync(assetPath);
            }
        });

        throw error;
    }
}

function handleDraftList(req, res) {
    fs.readdir(DRAFTS_DIR, (err, files) => {
        if (err) {
            sendJson(res, 500, { error: 'Failed to read drafts' });
            return;
        }

        const drafts = files.filter((fileName) => fileName.endsWith('.md')).sort();
        sendJson(res, 200, { drafts });
    });
}

function handleDraftRead(pathname, res) {
    try {
        const draftName = sanitizeFileName(decodeURIComponent(pathname.substring('/api/draft/'.length)), '.md');
        const draftPath = resolveInside(DRAFTS_DIR, draftName);
        const content = fs.readFileSync(draftPath, 'utf8');
        sendJson(res, 200, { content });
    } catch (error) {
        sendJson(res, 404, { error: 'Draft not found' });
    }
}

async function handleDraftSave(req, res) {
    try {
        const body = await parseJsonBody(req);
        const fileName = sanitizeFileName(body.filename, '.md');
        const content = typeof body.content === 'string' ? body.content : '';
        const draftPath = resolveInside(DRAFTS_DIR, fileName);

        fs.writeFileSync(draftPath, content, 'utf8');
        sendJson(res, 200, { success: true, filename: fileName });
    } catch (error) {
        sendJson(res, 400, { error: error.message || 'Failed to save draft' });
    }
}

function handleDraftDelete(pathname, res) {
    try {
        const draftName = sanitizeFileName(decodeURIComponent(pathname.substring('/api/draft/'.length)), '.md');
        const draftPath = resolveInside(DRAFTS_DIR, draftName);
        fs.unlinkSync(draftPath);
        sendJson(res, 200, { success: true });
    } catch (error) {
        sendJson(res, 404, { error: 'Draft not found' });
    }
}

function handleUploadImage(req, res) {
    parseMultipart(req, (err, parts) => {
        if (err) {
            sendJson(res, 400, { error: 'Invalid multipart data' });
            return;
        }

        const imagePart = parts.find((part) => part.name === 'image');
        if (!imagePart || !imagePart.filename) {
            sendJson(res, 400, { error: 'No image file found' });
            return;
        }

        try {
            const safeName = sanitizeFileName(imagePart.filename);
            const targetPath = resolveInside(DRAFT_ASSETS_DIR, safeName);
            fs.writeFileSync(targetPath, imagePart.data);

            sendJson(res, 200, {
                success: true,
                filename: safeName,
                serverPath: `/editor/draft-assets/${safeName}`,
                relativePath: `./draft-assets/${safeName}`
            });
        } catch (error) {
            sendJson(res, 500, { error: 'Failed to save image' });
        }
    });
}

function handleLegacySinglePostSave(body, res) {
    try {
        const payload = {
            mode: 'create',
            post: {
                id: body.postId,
                title_eng: body.postId,
                date: new Date().toISOString().slice(0, 10),
                category: 'post',
                series: Object.keys(loadRawSiteData().series)[0],
                languages: [body.language]
            },
            contents: {
                [body.language]: body.content
            },
            featured: {
                enabled: false
            }
        };

        savePostBundle(payload);
        sendJson(res, 200, { success: true });
    } catch (error) {
        sendJson(res, 400, {
            error: error.validationErrors ? error.validationErrors.join(' ') : error.message
        });
    }
}

function resolveStaticPath(pathname) {
    const normalizedPath = pathname === '/' ? '/editor/' : pathname;
    const cleanPath = path.posix.normalize(normalizedPath);
    const relativePath = cleanPath.startsWith('/') ? cleanPath.slice(1) : cleanPath;

    if (relativePath.startsWith('assets/') || relativePath.startsWith('projects/')) {
        return resolveInside(SITE_ROOT_DIR, relativePath);
    }

    if (relativePath === 'css/sidebar-nav.css') {
        return resolveInside(SITE_ROOT_DIR, relativePath);
    }

    let resolvedPath = resolveInside(ROOT_DIR, relativePath);

    if (fs.existsSync(resolvedPath) && fs.statSync(resolvedPath).isDirectory()) {
        resolvedPath = path.join(resolvedPath, 'index.html');
    }

    return resolvedPath;
}

const server = http.createServer(async (req, res) => {
    const parsedUrl = url.parse(req.url, true);
    const pathname = parsedUrl.pathname;

    res.setHeader('Access-Control-Allow-Origin', '*');
    res.setHeader('Access-Control-Allow-Methods', 'GET, POST, PUT, DELETE, OPTIONS');
    res.setHeader('Access-Control-Allow-Headers', 'Content-Type');

    if (req.method === 'OPTIONS') {
        res.writeHead(200);
        res.end();
        return;
    }

    if (pathname.startsWith('/api/')) {
        try {
            if (pathname === '/api/editor-bootstrap' && req.method === 'GET') {
                sendJson(res, 200, buildBootstrapPayload());
                return;
            }

            if (pathname === '/api/blog-home' && req.method === 'POST') {
                const body = await parseJsonBody(req);
                const blogHome = saveBlogHomeSettings(body);
                sendJson(res, 200, { success: true, blogHome });
                return;
            }

            if (pathname === '/api/drafts' && req.method === 'GET') {
                handleDraftList(req, res);
                return;
            }

            if (pathname.startsWith('/api/draft/') && req.method === 'GET') {
                handleDraftRead(pathname, res);
                return;
            }

            if (pathname === '/api/draft' && req.method === 'POST') {
                await handleDraftSave(req, res);
                return;
            }

            if (pathname.startsWith('/api/draft/') && req.method === 'DELETE') {
                handleDraftDelete(pathname, res);
                return;
            }

            if (pathname === '/api/upload-image' && req.method === 'POST') {
                handleUploadImage(req, res);
                return;
            }

            if (pathname.startsWith('/api/post-bundle/') && req.method === 'GET') {
                const postId = decodeURIComponent(pathname.substring('/api/post-bundle/'.length));
                const bundle = readPostBundle(postId);
                if (!bundle) {
                    sendJson(res, 404, { error: 'Post not found' });
                    return;
                }
                sendJson(res, 200, bundle);
                return;
            }

            if (pathname === '/api/post-bundle' && req.method === 'POST') {
                const body = await parseJsonBody(req);
                const result = savePostBundle(body);
                sendJson(res, 200, { success: true, ...result });
                return;
            }

            if (pathname.startsWith('/api/post/') && req.method === 'GET') {
                const parts = pathname.substring('/api/post/'.length).split('/');
                if (parts.length !== 2) {
                    sendJson(res, 400, { error: 'Invalid path format. Use /api/post/{postId}/{language}' });
                    return;
                }

                const [postId, language] = parts.map(decodeURIComponent);
                const filePath = resolveInside(path.join(POSTS_DIR, postId), `content-${language}.md`);
                if (!fs.existsSync(filePath)) {
                    sendJson(res, 404, { error: 'Post content not found' });
                    return;
                }
                const content = fs.readFileSync(filePath, 'utf8');
                sendJson(res, 200, { content });
                return;
            }

            if (pathname === '/api/post' && req.method === 'POST') {
                const body = await parseJsonBody(req);
                handleLegacySinglePostSave(body, res);
                return;
            }

            sendJson(res, 404, { error: 'API endpoint not found' });
        } catch (error) {
            const statusCode = error.validationErrors ? 400 : 500;
            sendJson(res, statusCode, {
                error: error.message || 'Internal server error',
                details: error.validationErrors || undefined
            });
        }
        return;
    }

    try {
        const filePath = resolveStaticPath(pathname);
        if (!fs.existsSync(filePath)) {
            sendJson(res, 404, { error: 'File not found' });
            return;
        }
        serveStaticFile(filePath, res);
    } catch (error) {
        sendJson(res, 404, { error: 'File not found' });
    }
});

server.listen(PORT, () => {
    console.log(`Editor server running at http://localhost:${PORT}/`);
    console.log(`Editor UI: http://localhost:${PORT}/editor/`);
    console.log(`Drafts directory: ${DRAFTS_DIR}`);
    console.log(`Posts directory: ${POSTS_DIR}`);
});
