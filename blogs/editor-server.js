const http = require('http');
const fs = require('fs');
const path = require('path');
const url = require('url');
const { marked } = require('marked');
const { parseMarkdownWithMath } = require('./js/markdown-with-math');
const { parseProjectMarkdown } = require('./lib/project-markdown');
const { renderProjectPage } = require('./lib/render-project-page');

const {
    POST_CATEGORIES,
    POST_LANGUAGES,
    PORTFOLIO_CATEGORIES,
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
const PROJECT_SNAPSHOTS_DIR = path.join(ROOT_DIR, 'editor', 'project-snapshots');
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
ensureDirSync(PROJECT_SNAPSHOTS_DIR);

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
        portfolioCategories: PORTFOLIO_CATEGORIES,
        series: rawSiteData.series,
        featuredPortfolioPosts: rawSiteData.featuredPortfolioPosts || [],
        posts: sortPostsByDate(rawSiteData.posts).map((post) => ({
            id: post.id,
            title_eng: post.title_eng,
            title_kor: post.title_kor || '',
            subtitle_eng: post.subtitle_eng || '',
            subtitle_kor: post.subtitle_kor || '',
            description_eng: post.description_eng || '',
            description_kor: post.description_kor || '',
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

function buildPortfolioBundle() {
    const rawSiteData = loadRawSiteData();
    validateSiteData(rawSiteData);

    return {
        portfolioCategories: PORTFOLIO_CATEGORIES,
        portfolioProjects: rawSiteData.portfolioProjects || [],
        publications: rawSiteData.publications || [],
        talks: rawSiteData.talks || []
    };
}

function normalizeProjectPagePath(rawPath) {
    const cleanPath = path.posix.normalize(`/${String(rawPath || '').trim()}`).slice(1);
    if (!cleanPath.startsWith('projects/') || !cleanPath.endsWith('/index.html')) {
        throw new Error('Project page path must match projects/<name>/index.html.');
    }
    return cleanPath;
}

function resolveProjectPagePath(rawPath) {
    return resolveInside(SITE_ROOT_DIR, normalizeProjectPagePath(rawPath));
}

function getProjectPagePathFromUrl(rawUrl) {
    const cleanUrl = String(rawUrl || '').split('#')[0].split('?')[0];
    if (!cleanUrl.startsWith('projects/')) {
        return null;
    }
    return cleanUrl.endsWith('/index.html') ? cleanUrl : `${cleanUrl.replace(/\/?$/, '/')}index.html`;
}

function getTimestampSlug() {
    return new Date().toISOString().replace(/[-:]/g, '').replace(/\..+/, '').replace('T', '-');
}

function getProjectSlugFromPagePath(pagePath) {
    return pagePath.split('/')[1];
}

function backupProjectPageSource(projectDir, pagePath, reason) {
    const slug = getProjectSlugFromPagePath(pagePath);
    const backupDir = path.join(PROJECT_SNAPSHOTS_DIR, `${getTimestampSlug()}-${reason}-${slug}`);
    ensureDirSync(backupDir);

    ['project.json', 'content.md', 'index.html'].forEach((fileName) => {
        const sourcePath = path.join(projectDir, fileName);
        if (fs.existsSync(sourcePath)) {
            fs.copyFileSync(sourcePath, path.join(backupDir, fileName));
        }
    });

    const assetsDir = path.join(projectDir, 'assets');
    if (fs.existsSync(assetsDir)) {
        fs.cpSync(assetsDir, path.join(backupDir, 'assets'), { recursive: true });
    }

    return backupDir;
}

function getAvailableAssetPath(projectDir, fileName) {
    const assetsDir = path.join(projectDir, 'assets');
    ensureDirSync(assetsDir);

    const extension = path.extname(fileName);
    const baseName = path.basename(fileName, extension);
    let candidateName = fileName;
    let index = 1;

    while (fs.existsSync(path.join(assetsDir, candidateName))) {
        candidateName = `${baseName}-${index}${extension}`;
        index += 1;
    }

    return {
        assetsDir,
        fileName: candidateName,
        targetPath: path.join(assetsDir, candidateName)
    };
}

function findPortfolioProjectByPagePath(rawSiteData, pagePath) {
    return (rawSiteData.portfolioProjects || []).find((project) => getProjectPagePathFromUrl(project.url) === pagePath) || null;
}

function buildProjectPagesPayload() {
    const rawSiteData = loadRawSiteData();
    validateSiteData(rawSiteData);

    const pagesByPath = new Map();
    (rawSiteData.portfolioProjects || []).forEach((project) => {
        const pagePath = getProjectPagePathFromUrl(project.url);
        if (!pagePath) {
            return;
        }

        const projectDir = path.dirname(resolveProjectPagePath(pagePath));
        const metadataPath = path.join(projectDir, 'project.json');
        const contentPath = path.join(projectDir, 'content.md');
        if (!fs.existsSync(metadataPath) || !fs.existsSync(contentPath)) {
            return;
        }

        const metadata = JSON.parse(fs.readFileSync(metadataPath, 'utf8'));
        pagesByPath.set(pagePath, {
            path: pagePath,
            title: metadata.title || project.title,
            projectId: project.id,
            source: 'portfolio'
        });
    });

    const projectsDir = path.join(SITE_ROOT_DIR, 'projects');
    if (fs.existsSync(projectsDir)) {
        fs.readdirSync(projectsDir, { withFileTypes: true })
            .filter((entry) => entry.isDirectory())
            .forEach((entry) => {
                const pagePath = `projects/${entry.name}/index.html`;
                const projectDir = path.dirname(resolveProjectPagePath(pagePath));
                const metadataPath = path.join(projectDir, 'project.json');
                const contentPath = path.join(projectDir, 'content.md');
                if (!fs.existsSync(metadataPath) || !fs.existsSync(contentPath) || pagesByPath.has(pagePath)) {
                    return;
                }
                const metadata = JSON.parse(fs.readFileSync(metadataPath, 'utf8'));
                pagesByPath.set(pagePath, {
                    path: pagePath,
                    title: metadata.title || entry.name,
                    projectId: '',
                    source: 'projects'
                });
            });
    }

    return {
        pages: [...pagesByPath.values()].sort((a, b) => a.path.localeCompare(b.path))
    };
}

function readProjectPage(rawPath) {
    const pagePath = normalizeProjectPagePath(rawPath);
    const projectDir = path.dirname(resolveProjectPagePath(pagePath));
    const metadataPath = path.join(projectDir, 'project.json');
    const contentPath = path.join(projectDir, 'content.md');
    if (!fs.existsSync(metadataPath) || !fs.existsSync(contentPath)) {
        throw new Error('Project page source files were not found.');
    }

    return {
        path: pagePath,
        metadata: JSON.parse(fs.readFileSync(metadataPath, 'utf8')),
        content: fs.readFileSync(contentPath, 'utf8')
    };
}

function saveProjectPage(payload) {
    const pagePath = normalizeProjectPagePath(payload.path);
    const indexPath = resolveProjectPagePath(pagePath);
    const projectDir = path.dirname(indexPath);
    const metadataPath = path.join(projectDir, 'project.json');
    const contentPath = path.join(projectDir, 'content.md');
    if (!fs.existsSync(metadataPath) || !fs.existsSync(contentPath)) {
        throw new Error('Project page source files were not found.');
    }

    const content = typeof payload.content === 'string' ? payload.content : '';
    if (!content.trim()) {
        throw new Error('Project page content cannot be empty.');
    }

    const metadataInput = payload.metadata && typeof payload.metadata === 'object' ? payload.metadata : {};
    const existingMetadata = JSON.parse(fs.readFileSync(metadataPath, 'utf8'));
    const metadata = {
        ...existingMetadata,
        title: String(metadataInput.title || '').trim(),
        heroTitle: String(metadataInput.heroTitle || '').trim(),
        subtitles: Array.isArray(metadataInput.subtitles)
            ? metadataInput.subtitles.map((subtitle) => String(subtitle || '').trim()).filter(Boolean)
            : [],
        description: String(metadataInput.description || '').trim(),
        keywords: String(metadataInput.keywords || '').trim()
    };

    if (!metadata.title || !metadata.heroTitle) {
        throw new Error('Project title and hero title are required.');
    }

    const backupDir = backupProjectPageSource(projectDir, pagePath, 'save');
    const metadataTempPath = `${metadataPath}.tmp`;
    const contentTempPath = `${contentPath}.tmp`;
    fs.writeFileSync(metadataTempPath, `${JSON.stringify(metadata, null, 2)}\n`, 'utf8');
    fs.writeFileSync(contentTempPath, content, 'utf8');
    fs.renameSync(metadataTempPath, metadataPath);
    fs.renameSync(contentTempPath, contentPath);

    const contentHtml = content.trimStart().startsWith('<')
        ? content
        : parseProjectMarkdown(content, (source) => marked.parse(source));
    fs.writeFileSync(indexPath, renderProjectPage({ project: metadata, contentHtml }), 'utf8');

    return {
        success: true,
        path: pagePath,
        bytes: Buffer.byteLength(content, 'utf8'),
        backup: path.relative(SITE_ROOT_DIR, backupDir)
    };
}

function createProjectPage(payload) {
    const requestedSlug = String(payload.slug || '').trim();
    const title = String(payload.title || '').trim() || 'New Project Page';
    const subtitle = String(payload.subtitle || '').trim() || 'Project';
    const createPortfolioCard = Boolean(payload.createPortfolioCard);
    const cardImage = String(payload.cardImage || '').trim();
    const projectsDir = path.join(SITE_ROOT_DIR, 'projects');
    ensureDirSync(projectsDir);

    if (createPortfolioCard && !cardImage) {
        throw new Error('Card image is required when creating a portfolio card.');
    }

    let slug = requestedSlug
        .toLowerCase()
        .replace(/[^a-z0-9_-]+/g, '_')
        .replace(/_+/g, '_')
        .replace(/^_+|_+$/g, '');

    if (!slug) {
        slug = 'new_project';
    }

    let candidateSlug = slug;
    let suffix = 1;
    while (fs.existsSync(path.join(projectsDir, candidateSlug))) {
        suffix += 1;
        candidateSlug = `${slug}_${suffix}`;
    }

    const projectDir = path.join(projectsDir, candidateSlug);
    ensureDirSync(projectDir);

    const metadata = {
        title,
        heroTitle: title,
        subtitles: [
            subtitle
        ],
        layout: 'case-study',
        overview: [
            'Add a concise overview of the work, your role, and the problem it addressed.'
        ],
        contributions: [],
        details: [
            {
                label: 'Role',
                value: 'Add your role'
            }
        ],
        description: '',
        keywords: ''
    };
    const content = `## Why It Mattered

Explain the context and why this work was important at the time.

## Technical Approach

Describe the key decisions and implementation.

## Outcome

Summarize the result and impact.
`;

    fs.writeFileSync(path.join(projectDir, 'project.json'), `${JSON.stringify(metadata, null, 2)}\n`, 'utf8');
    fs.writeFileSync(path.join(projectDir, 'content.md'), content, 'utf8');

    const contentHtml = parseProjectMarkdown(content, (source) => marked.parse(source));
    const indexPath = path.join(projectDir, 'index.html');
    fs.writeFileSync(indexPath, renderProjectPage({ project: metadata, contentHtml }), 'utf8');

    if (createPortfolioCard) {
        const rawSiteData = loadRawSiteData();
        validateSiteData(rawSiteData);
        rawSiteData.portfolioProjects = rawSiteData.portfolioProjects || [];
        rawSiteData.portfolioProjects.push({
            id: candidateSlug.replace(/_/g, '-'),
            title,
            summary: '',
            url: `projects/${candidateSlug}/`,
            categories: ['app'],
            image: cardImage,
            alt: `${title} teaser`
        });
        writeSiteData(rawSiteData);
    }

    return {
        success: true,
        page: {
            path: `projects/${candidateSlug}/index.html`,
            title,
            projectId: createPortfolioCard ? candidateSlug.replace(/_/g, '-') : '',
            source: createPortfolioCard ? 'portfolio' : 'projects'
        },
        portfolioCardCreated: createPortfolioCard
    };
}

function deleteProjectPage(payload) {
    const pagePath = normalizeProjectPagePath(payload.path);
    const rawSiteData = loadRawSiteData();
    validateSiteData(rawSiteData);

    const linkedProject = findPortfolioProjectByPagePath(rawSiteData, pagePath);
    if (linkedProject) {
        throw new Error(`Project page is linked to portfolio card "${linkedProject.id}". Remove the card first.`);
    }

    const indexPath = resolveProjectPagePath(pagePath);
    const projectDir = path.dirname(indexPath);
    const metadataPath = path.join(projectDir, 'project.json');
    const contentPath = path.join(projectDir, 'content.md');
    if (!fs.existsSync(metadataPath) || !fs.existsSync(contentPath)) {
        throw new Error('Project page source files were not found.');
    }

    const backupDir = backupProjectPageSource(projectDir, pagePath, 'delete');
    fs.rmSync(projectDir, { recursive: true, force: true });

    return {
        success: true,
        path: pagePath,
        backup: path.relative(SITE_ROOT_DIR, backupDir)
    };
}

function handleProjectAssetUpload(req, res) {
    parseMultipart(req, (err, parts) => {
        if (err) {
            sendJson(res, 400, { error: 'Invalid multipart data' });
            return;
        }

        const pathPart = parts.find((part) => part.name === 'path');
        const assetPart = parts.find((part) => part.name === 'asset');
        if (!pathPart || !assetPart || !assetPart.filename) {
            sendJson(res, 400, { error: 'Project path and asset file are required.' });
            return;
        }

        try {
            const pagePath = normalizeProjectPagePath(pathPart.data.toString().trim());
            const projectDir = path.dirname(resolveProjectPagePath(pagePath));
            const metadataPath = path.join(projectDir, 'project.json');
            const contentPath = path.join(projectDir, 'content.md');
            if (!fs.existsSync(metadataPath) || !fs.existsSync(contentPath)) {
                throw new Error('Project page source files were not found.');
            }

            const safeName = sanitizeFileName(assetPart.filename);
            const assetPath = getAvailableAssetPath(projectDir, safeName);
            fs.writeFileSync(assetPath.targetPath, assetPart.data);

            sendJson(res, 200, {
                success: true,
                filename: assetPath.fileName,
                relativePath: `assets/${assetPath.fileName}`,
                serverPath: `/${pagePath.replace(/index\.html$/, '')}assets/${assetPath.fileName}`,
                mimeType: assetPart.contentType || 'application/octet-stream'
            });
        } catch (error) {
            sendJson(res, 400, { error: error.message || 'Failed to save project asset.' });
        }
    });
}

function sanitizePortfolioBundle(payload) {
    const portfolioProjects = Array.isArray(payload.portfolioProjects) ? payload.portfolioProjects : [];
    const publications = Array.isArray(payload.publications) ? payload.publications : [];
    const talks = Array.isArray(payload.talks) ? payload.talks : [];

    return {
        portfolioProjects: portfolioProjects.map((project) => {
            const nextProject = {
                id: String(project.id || '').trim(),
                title: String(project.title || '').trim(),
                summary: String(project.summary || '').trim(),
                url: String(project.url || '').trim(),
                categories: Array.isArray(project.categories)
                    ? [...new Set(project.categories.map((category) => String(category || '').trim()).filter(Boolean))]
                    : [],
                tags: Array.isArray(project.tags)
                    ? [...new Set(project.tags.map((tag) => String(tag || '').trim()).filter(Boolean))]
                    : [],
                external: Boolean(project.external)
            };

            ['badge', 'image', 'gif', 'video', 'poster', 'alt'].forEach((key) => {
                const value = String(project[key] || '').trim();
                if (value) {
                    nextProject[key] = value;
                }
            });

            if (!nextProject.external) {
                delete nextProject.external;
            }

            return nextProject;
        }),
        publications: publications.map((publication) => ({
            title: String(publication.title || '').trim(),
            authorsHtml: String(publication.authorsHtml || '').trim(),
            venueHtml: String(publication.venueHtml || '').trim(),
            links: Array.isArray(publication.links)
                ? publication.links.map((link) => {
                    const nextLink = {
                        label: String(link.label || '').trim(),
                        url: String(link.url || '').trim()
                    };
                    const icon = String(link.icon || '').trim();
                    if (icon) {
                        nextLink.icon = icon;
                    }
                    return nextLink;
                })
                : []
        })),
        talks: talks.map((talk) => ({
            title: String(talk.title || '').trim(),
            venueHtml: String(talk.venueHtml || '').trim(),
            date: String(talk.date || '').trim()
        }))
    };
}

function savePortfolioBundle(payload) {
    const rawSiteData = loadRawSiteData();
    validateSiteData(rawSiteData);

    const sanitizedBundle = sanitizePortfolioBundle(payload);
    const nextSiteData = {
        ...rawSiteData,
        ...sanitizedBundle
    };

    validateSiteData(nextSiteData);
    writeSiteData(nextSiteData);

    return {
        success: true,
        projectCount: sanitizedBundle.portfolioProjects.length,
        publicationCount: sanitizedBundle.publications.length,
        talkCount: sanitizedBundle.talks.length
    };
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
    const descriptionEng = String(post.description_eng || '').trim();
    const descriptionKor = String(post.description_kor || '').trim();
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
    if (descriptionEng) {
        sanitizedPost.description_eng = descriptionEng;
    }
    if (descriptionKor) {
        sanitizedPost.description_kor = descriptionKor;
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

    if (sanitizedPost.status && !['published', 'draft'].includes(sanitizedPost.status)) {
        errors.push('Status must be published or draft.');
    }

    if ((sanitizedPost.status || originalPost?.status || 'published') === 'published') {
        if (!(sanitizedPost.slug || originalPost?.slug)) {
            errors.push('Published posts require a stable slug.');
        }
        if (!(sanitizedPost.description_eng || originalPost?.description_eng)) {
            errors.push('Published posts require an English description.');
        }
        if (!(sanitizedPost.cover || originalPost?.cover) || (sanitizedPost.cover || originalPost?.cover) === '/assets/blog_bg.jpeg') {
            errors.push('Published posts require a post-specific cover image.');
        }
        if ((sanitizedPost.tags || originalPost?.tags || []).length === 0) {
            errors.push('Published posts require at least one tag.');
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
        sanitizedContents[language] = contentsInput[language];
        sanitizedContents[language] = stripLegacyContentPreamble(sanitizedContents[language]);
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

            if (pathname === '/api/portfolio-bundle' && req.method === 'GET') {
                sendJson(res, 200, buildPortfolioBundle());
                return;
            }

            if (pathname === '/api/portfolio-bundle' && req.method === 'POST') {
                const body = await parseJsonBody(req);
                sendJson(res, 200, savePortfolioBundle(body));
                return;
            }

            if (pathname === '/api/project-pages' && req.method === 'GET') {
                sendJson(res, 200, buildProjectPagesPayload());
                return;
            }

            if (pathname === '/api/project-page' && req.method === 'GET') {
                sendJson(res, 200, readProjectPage(parsedUrl.query.path));
                return;
            }

            if (pathname === '/api/project-page' && req.method === 'POST') {
                const body = await parseJsonBody(req);
                sendJson(res, 200, saveProjectPage(body));
                return;
            }

            if (pathname === '/api/project-page-create' && req.method === 'POST') {
                const body = await parseJsonBody(req);
                sendJson(res, 200, createProjectPage(body));
                return;
            }

            if (pathname === '/api/project-page-delete' && req.method === 'POST') {
                const body = await parseJsonBody(req);
                sendJson(res, 200, deleteProjectPage(body));
                return;
            }

            if (pathname === '/api/project-asset-upload' && req.method === 'POST') {
                handleProjectAssetUpload(req, res);
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
    console.log(`Portfolio editor UI: http://localhost:${PORT}/editor/portfolio.html`);
    console.log(`Drafts directory: ${DRAFTS_DIR}`);
    console.log(`Posts directory: ${POSTS_DIR}`);
});
