const fs = require('fs');
const http = require('http');
const path = require('path');

const { chromium } = require('playwright');

const { SITE_URL } = require('../blogs/lib/site-config');
const { loadSiteData } = require('../blogs/lib/site-data');
const { buildPublicRoutes } = require('../blogs/lib/site-routes');

const repoRoot = path.join(__dirname, '..');
const distRoot = path.join(repoRoot, 'blogs', 'dist');
const defaultChromePath = '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome';
const contentTypes = {
    '.css': 'text/css; charset=utf-8',
    '.gif': 'image/gif',
    '.glb': 'model/gltf-binary',
    '.html': 'text/html; charset=utf-8',
    '.ico': 'image/x-icon',
    '.jpeg': 'image/jpeg',
    '.jpg': 'image/jpeg',
    '.js': 'text/javascript; charset=utf-8',
    '.json': 'application/json; charset=utf-8',
    '.m4v': 'video/x-m4v',
    '.md': 'text/markdown; charset=utf-8',
    '.mov': 'video/quicktime',
    '.mp4': 'video/mp4',
    '.png': 'image/png',
    '.svg': 'image/svg+xml',
    '.webm': 'video/webm',
    '.webp': 'image/webp',
    '.woff': 'font/woff',
    '.woff2': 'font/woff2',
    '.xml': 'application/xml; charset=utf-8'
};

function parseArguments(argv) {
    const options = {
        baseUrl: '',
        screenshotsDir: '',
        route: ''
    };

    argv.forEach((argument) => {
        if (argument === '--production') {
            options.baseUrl = SITE_URL;
        }
        if (argument.startsWith('--base-url=')) {
            options.baseUrl = argument.slice('--base-url='.length).replace(/\/+$/, '');
        }
        if (argument.startsWith('--screenshots-dir=')) {
            options.screenshotsDir = path.resolve(argument.slice('--screenshots-dir='.length));
        }
        if (argument.startsWith('--route=')) {
            options.route = argument.slice('--route='.length);
        }
    });

    return options;
}

function resolveStaticFile(urlPath) {
    let pathname;
    try {
        pathname = decodeURIComponent(urlPath);
    } catch (error) {
        return '';
    }

    const relativePath = pathname.replace(/^\/+/, '');
    const candidate = path.resolve(distRoot, relativePath || 'index.html');
    if (candidate !== distRoot && !candidate.startsWith(`${distRoot}${path.sep}`)) {
        return '';
    }

    if (fs.existsSync(candidate) && fs.statSync(candidate).isDirectory()) {
        return path.join(candidate, 'index.html');
    }
    return candidate;
}

function sendStaticFile(req, res) {
    const requestUrl = new URL(req.url, 'http://localhost');
    const filePath = resolveStaticFile(requestUrl.pathname);
    if (!filePath || !fs.existsSync(filePath) || !fs.statSync(filePath).isFile()) {
        res.writeHead(404, { 'Content-Type': 'text/plain; charset=utf-8' });
        res.end('Not found');
        return;
    }

    const stat = fs.statSync(filePath);
    const contentType = contentTypes[path.extname(filePath).toLowerCase()] || 'application/octet-stream';
    const range = req.headers.range && req.headers.range.match(/bytes=(\d*)-(\d*)/);
    let start = 0;
    let end = stat.size - 1;
    let status = 200;

    if (range) {
        start = range[1] ? Number.parseInt(range[1], 10) : 0;
        end = range[2] ? Number.parseInt(range[2], 10) : Math.min(stat.size - 1, start + (1024 * 1024));
        if (start >= stat.size || end < start) {
            res.writeHead(416, { 'Content-Range': `bytes */${stat.size}` });
            res.end();
            return;
        }
        end = Math.min(end, stat.size - 1);
        status = 206;
    }

    const headers = {
        'Accept-Ranges': 'bytes',
        'Cache-Control': 'public, max-age=3600',
        'Content-Length': end - start + 1,
        'Content-Type': contentType
    };
    if (status === 206) {
        headers['Content-Range'] = `bytes ${start}-${end}/${stat.size}`;
    }

    res.writeHead(status, headers);
    if (req.method === 'HEAD') {
        res.end();
        return;
    }
    fs.createReadStream(filePath, { start, end }).pipe(res);
}

function startStaticServer() {
    return new Promise((resolve, reject) => {
        const server = http.createServer(sendStaticFile);
        server.once('error', reject);
        server.listen(0, '127.0.0.1', () => {
            const address = server.address();
            resolve({
                server,
                baseUrl: `http://127.0.0.1:${address.port}`
            });
        });
    });
}

function getLaunchOptions() {
    const configuredPath = process.env.PLAYWRIGHT_CHROMIUM_EXECUTABLE_PATH;
    const executablePath = configuredPath || (fs.existsSync(defaultChromePath) ? defaultChromePath : '');
    return executablePath ? { headless: true, executablePath } : { headless: true };
}

async function waitForDynamicContent(page, route) {
    const checks = {
        portfolio: () => {
            return document.querySelectorAll('[data-portfolio-block] > *').length >= 3
                && document.querySelectorAll('#portfolio-projects .portfolio-box, #portfolio-projects .portfolio-project-link').length > 0
                && document.querySelectorAll('#portfolio-blog-posts .blog-preview-card, #portfolio-blog-posts .portfolio-blog-preview-item').length > 0;
        },
        'blog-index': () => document.querySelectorAll('.post-preview').length > 0,
        'blog-search': () => document.querySelectorAll('.post-preview').length > 0,
        post: () => (document.querySelector('.post-content, article')?.textContent || '').trim().length > 100,
        project: () => (document.querySelector('.portfolio-details')?.textContent || '').trim().length > 100,
        editor: () => Boolean(document.querySelector('.editor-shell')),
        utility: () => {
            const viewer = document.querySelector('simple-model-viewer');
            const canvas = viewer?.shadowRoot?.querySelector('canvas');
            return Boolean(viewer && canvas && canvas.width > 0 && canvas.height > 0);
        }
    };

    const check = checks[route.type];
    if (!check) {
        return;
    }
    await page.waitForFunction(check, null, { timeout: route.type === 'utility' ? 45000 : 15000 });
}

async function inspectRenderedMedia(page, pageOrigin) {
    return page.evaluate(async ({ origin }) => {
        const sleep = (milliseconds) => new Promise((resolve) => setTimeout(resolve, milliseconds));
        const withTimeout = (promise, milliseconds, fallback) => Promise.race([
            promise,
            sleep(milliseconds).then(() => fallback)
        ]);
        const normalizeLocalUrl = (value) => {
            try {
                const url = new URL(value, window.location.href);
                return url.origin === origin ? url.href : '';
            } catch (error) {
                return '';
            }
        };

        document.querySelectorAll('img[data-src]').forEach((image) => {
            if (!image.getAttribute('src')) {
                image.setAttribute('src', image.dataset.src);
            }
        });
        document.querySelectorAll('img').forEach((image) => {
            image.loading = 'eager';
        });
        window.scrollTo(0, document.documentElement.scrollHeight);
        await sleep(120);

        const media = [];
        const broken = [];
        const images = Array.from(document.images);
        await Promise.all(images.map(async (image) => {
            const source = normalizeLocalUrl(image.currentSrc || image.src);
            if (!source) {
                return;
            }
            media.push({ type: 'image', url: source });

            if (!image.complete) {
                await withTimeout(new Promise((resolve) => {
                    image.addEventListener('load', resolve, { once: true });
                    image.addEventListener('error', resolve, { once: true });
                }), 10000, null);
            }

            if (image.complete && image.naturalWidth > 0 && typeof image.decode === 'function') {
                await withTimeout(image.decode().catch(() => null), 10000, null);
            }
            if (!image.complete || image.naturalWidth === 0 || image.naturalHeight === 0) {
                broken.push({ type: 'image', url: source, reason: 'Image did not decode.' });
            }
        }));

        const backgroundUrls = new Set();
        Array.from(document.querySelectorAll('*')).forEach((element) => {
            const styles = window.getComputedStyle(element);
            [styles.backgroundImage, styles.borderImageSource, styles.listStyleImage].forEach((value) => {
                const matches = String(value || '').matchAll(/url\(["']?(.*?)["']?\)/g);
                for (const match of matches) {
                    const source = normalizeLocalUrl(match[1]);
                    if (source) {
                        backgroundUrls.add(source);
                    }
                }
            });
        });

        await Promise.all([...backgroundUrls].map(async (source) => {
            media.push({ type: 'background', url: source });
            const probe = new Image();
            probe.src = source;
            const loaded = await withTimeout(new Promise((resolve) => {
                probe.addEventListener('load', () => resolve(true), { once: true });
                probe.addEventListener('error', () => resolve(false), { once: true });
            }), 30000, false);
            if (loaded && typeof probe.decode === 'function') {
                await withTimeout(probe.decode().catch(() => null), 10000, null);
            }
            if (!loaded || probe.naturalWidth === 0 || probe.naturalHeight === 0) {
                broken.push({ type: 'background', url: source, reason: 'Background image did not decode.' });
            }
        }));

        const videoUrls = new Set();
        document.querySelectorAll('video').forEach((video) => {
            const currentSource = normalizeLocalUrl(video.currentSrc || video.src);
            if (currentSource) {
                videoUrls.add(currentSource);
            }
            video.querySelectorAll('source[src]').forEach((sourceElement) => {
                const source = normalizeLocalUrl(sourceElement.src);
                if (source) {
                    videoUrls.add(source);
                }
            });
        });

        await Promise.all([...videoUrls].map(async (source) => {
            media.push({ type: 'video', url: source });
            let response;
            try {
                response = await fetch(source, { headers: { Range: 'bytes=0-4095' } });
            } catch (error) {
                response = null;
            }
            if (!response || !response.ok) {
                broken.push({ type: 'video', url: source, reason: 'Video range request failed.' });
            }
        }));

        window.scrollTo(0, 0);
        return { media, broken };
    }, { origin: pageOrigin });
}

function routeScreenshotName(route) {
    if (route.path === '/') {
        return 'portfolio';
    }
    return route.path.replace(/[?&=]/g, '-').replace(/^\/+|\/+$/g, '').replace(/\//g, '-') || 'page';
}

async function inspect3DCanvas(page) {
    return page.evaluate(() => {
        const viewer = document.querySelector('simple-model-viewer');
        const canvas = viewer?.shadowRoot?.querySelector('canvas');
        if (!canvas) {
            return { ok: false, reason: 'Viewer canvas was not created.' };
        }

        if (!viewer.model) {
            const fileInputContainer = viewer.shadowRoot?.querySelector('#fileInputContainer');
            const fileInputVisible = fileInputContainer
                && getComputedStyle(fileInputContainer).display !== 'none';
            return {
                ok: Boolean(fileInputVisible && canvas.width > 0 && canvas.height > 0),
                reason: 'Empty viewer did not expose its model file input.'
            };
        }

        const sampleCanvas = document.createElement('canvas');
        sampleCanvas.width = 64;
        sampleCanvas.height = 64;
        const context = sampleCanvas.getContext('2d', { willReadFrequently: true });
        context.drawImage(canvas, 0, 0, sampleCanvas.width, sampleCanvas.height);
        const pixels = context.getImageData(0, 0, sampleCanvas.width, sampleCanvas.height).data;

        const colors = new Set();
        let opaquePixels = 0;
        for (let index = 0; index < pixels.length; index += 4) {
            if (pixels[index + 3] > 0) {
                opaquePixels += 1;
            }
            colors.add(`${pixels[index] >> 4}:${pixels[index + 1] >> 4}:${pixels[index + 2] >> 4}:${pixels[index + 3] >> 4}`);
        }

        return {
            ok: opaquePixels > 0 && colors.size > 2,
            reason: `Canvas sample has ${opaquePixels} opaque pixels and ${colors.size} color buckets.`
        };
    });
}

async function inspectPreviewLayout(page) {
    return page.evaluate(() => {
        const issues = [];
        const pairs = [
            ['#portfolio-projects .portfolio-box', '.aspect-ratio-box', '.polar_content'],
            ['#portfolio-projects .portfolio-project-link', '.portfolio-project-cover', '.portfolio-project-body'],
            ['#portfolio-blog-posts .blog-preview-card', '.aspect-ratio-box', '.polar_content'],
            ['#portfolio-blog-posts .portfolio-blog-preview-item', '.portfolio-blog-preview-cover', '.portfolio-blog-preview-body'],
            ['.post-preview .post-card-link', '.post-card-cover', '.post-card-body']
        ];

        const intersects = (first, second) => {
            const overlapWidth = Math.min(first.right, second.right) - Math.max(first.left, second.left);
            const overlapHeight = Math.min(first.bottom, second.bottom) - Math.max(first.top, second.top);
            return overlapWidth > 1 && overlapHeight > 1;
        };

        pairs.forEach(([containerSelector, mediaSelector, textSelector]) => {
            document.querySelectorAll(containerSelector).forEach((container, index) => {
                const media = container.querySelector(mediaSelector);
                const text = container.querySelector(textSelector);
                if (!media || !text) {
                    return;
                }
                if (intersects(media.getBoundingClientRect(), text.getBoundingClientRect())) {
                    issues.push(`${containerSelector}[${index}] media overlaps text`);
                }
            });
        });

        return issues;
    });
}

async function checkMobilePreviewLayout(context, baseUrl, route) {
    const page = await context.newPage();
    try {
        await page.setViewportSize({ width: 390, height: 844 });
        const response = await page.goto(`${baseUrl}${route.path}`, {
            waitUntil: 'domcontentloaded',
            timeout: 30000
        });
        if (!response || !response.ok()) {
            throw new Error(`Page returned ${response ? response.status() : 'no response'}.`);
        }
        await waitForDynamicContent(page, route);
        const layoutIssues = await inspectPreviewLayout(page);
        const horizontalOverflow = await page.evaluate(() => {
            return document.documentElement.scrollWidth > document.documentElement.clientWidth + 1;
        });
        if (horizontalOverflow) {
            layoutIssues.push('document has horizontal overflow at 390px');
        }
        if (layoutIssues.length) {
            throw new Error(layoutIssues.map((message) => `layout: ${message}`).join('\n'));
        }
        process.stdout.write(`PASS ${route.path} (390px preview layout)\n`);
    } finally {
        await page.close();
    }
}

async function checkRoute(context, baseUrl, route, options, totals) {
    const page = await context.newPage();
    const localFailures = [];
    const pageErrors = [];
    const pageUrl = `${baseUrl}${route.path}`;
    const origin = new URL(baseUrl).origin;
    const requestedMedia = new Set();

    page.on('requestfailed', (request) => {
        if (request.url().startsWith(origin)) {
            const errorText = request.failure()?.errorText || 'request failed';
            const isExpectedMediaCancellation = errorText === 'net::ERR_ABORTED'
                && ['media', 'fetch'].includes(request.resourceType());
            if (!isExpectedMediaCancellation) {
                localFailures.push(`${request.url()} (${errorText})`);
            }
        }
    });
    page.on('response', (response) => {
        if (response.url().startsWith(origin) && response.status() >= 400) {
            localFailures.push(`${response.url()} (HTTP ${response.status()})`);
        }
        if (response.url().startsWith(origin) && /\.(avif|gif|glb|hdr|jpe?g|mov|mp4|png|svg|webm|webp)(?:[?#]|$)/i.test(response.url())) {
            requestedMedia.add(response.url().split('#')[0]);
        }
    });
    page.on('pageerror', (error) => pageErrors.push(error.message));

    try {
        const response = await page.goto(pageUrl, { waitUntil: 'domcontentloaded', timeout: 30000 });
        if (!response || !response.ok()) {
            throw new Error(`Page returned ${response ? response.status() : 'no response'}.`);
        }
        await waitForDynamicContent(page, route);
        const inspection = await inspectRenderedMedia(page, origin);
        const layoutIssues = await inspectPreviewLayout(page);
        if (route.type === 'utility') {
            const canvasInspection = await inspect3DCanvas(page);
            if (!canvasInspection.ok) {
                throw new Error(canvasInspection.reason);
            }
        }

        if (options.screenshotsDir && ['portfolio', 'blog-index', 'blog-search'].includes(route.type)) {
            fs.mkdirSync(options.screenshotsDir, { recursive: true });
            await page.screenshot({
                path: path.join(options.screenshotsDir, `${routeScreenshotName(route)}.png`),
                fullPage: true
            });
        }

        if (localFailures.length || pageErrors.length || inspection.broken.length || layoutIssues.length) {
            const details = [
                ...localFailures.map((message) => `request: ${message}`),
                ...pageErrors.map((message) => `script: ${message}`),
                ...inspection.broken.map((item) => `${item.type}: ${item.url} (${item.reason})`),
                ...layoutIssues.map((message) => `layout: ${message}`)
            ];
            throw new Error(details.join('\n'));
        }

        inspection.media.forEach((item) => totals.media.add(`${item.type}:${item.url}`));
        requestedMedia.forEach((url) => totals.media.add(`request:${url}`));
        totals.pages += 1;
        process.stdout.write(`PASS ${route.path} (${inspection.media.length} local media)\n`);
    } finally {
        await page.close();
    }
}

async function main() {
    const options = parseArguments(process.argv.slice(2));
    if (!options.baseUrl && !fs.existsSync(path.join(distRoot, 'index.html'))) {
        throw new Error('blogs/dist is missing. Run npm run build before check:render.');
    }

    const localServer = options.baseUrl ? null : await startStaticServer();
    const baseUrl = options.baseUrl || localServer.baseUrl;
    const browser = await chromium.launch(getLaunchOptions());
    const context = await browser.newContext({
        serviceWorkers: 'block',
        viewport: { width: 1440, height: 900 }
    });
    const siteOrigin = new URL(baseUrl).origin;
    await context.route('**/*', async (route) => {
        const request = route.request();
        const requestUrl = new URL(request.url());
        const requestOrigin = requestUrl.origin;
        if (requestOrigin !== siteOrigin) {
            const isEditorDependency = request.resourceType() === 'script'
                && ['cdn.jsdelivr.net', 'cdnjs.cloudflare.com'].includes(requestUrl.hostname)
                && request.frame().url().includes('/blogs/editor/');
            if (isEditorDependency) {
                await route.continue();
                return;
            }
            await route.abort();
            return;
        }
        await route.continue();
    });

    const siteData = loadSiteData();
    const publicRoutes = buildPublicRoutes(siteData);
    const routes = options.route
        ? publicRoutes.filter((route) => route.path === options.route || route.path.split('?')[0] === options.route)
        : publicRoutes;
    if (routes.length === 0) {
        throw new Error(`No public route matches ${options.route}.`);
    }
    const totals = { pages: 0, media: new Set() };
    const failures = [];

    try {
        for (const route of routes) {
            try {
                await checkRoute(context, baseUrl, route, options, totals);
            } catch (error) {
                failures.push({ route: route.path, message: error.message });
                process.stderr.write(`FAIL ${route.path}\n${error.message}\n`);
            }
        }
        for (const route of routes.filter((entry) => ['portfolio', 'blog-index', 'blog-search'].includes(entry.type))) {
            try {
                await checkMobilePreviewLayout(context, baseUrl, route);
            } catch (error) {
                failures.push({ route: `${route.path} (390px)`, message: error.message });
                process.stderr.write(`FAIL ${route.path} (390px)\n${error.message}\n`);
            }
        }
    } finally {
        await context.close();
        await browser.close();
        if (localServer) {
            await new Promise((resolve) => localServer.server.close(resolve));
        }
    }

    if (failures.length > 0) {
        throw new Error(`Rendered site check failed on ${failures.length} of ${routes.length} routes.`);
    }

    console.log(`Rendered site check passed: ${totals.pages} routes, ${totals.media.size} unique local media resources.`);
}

main().catch((error) => {
    console.error(error.message);
    process.exit(1);
});
