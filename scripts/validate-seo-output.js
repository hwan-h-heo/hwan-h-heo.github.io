#!/usr/bin/env node

const fs = require('fs');
const path = require('path');
const { pathToFileURL } = require('url');
const cheerio = require('cheerio');

const { SITE_URL } = require('../blogs/lib/site-config');
const { loadRawSiteData, loadSiteData } = require('../blogs/lib/site-data');
const {
    getPostRoute,
    listSeriesArchiveEntries,
    listTagArchiveEntries
} = require('../blogs/lib/site-routes');
const { loadLegacyRedirects, validateLegacyRedirects } = require('../blogs/lib/legacy-redirects');
const { getPostTitle } = require('../blogs/lib/seo-utils');

const repoRoot = path.join(__dirname, '..');
const distRoot = path.join(repoRoot, 'blogs', 'dist');
const siteOrigin = new URL(SITE_URL).origin;

function readText(filePath) {
    return fs.readFileSync(filePath, 'utf8');
}

function loadHtml(relativePath) {
    const filePath = path.join(distRoot, relativePath);
    return {
        filePath,
        html: readText(filePath),
        $: cheerio.load(readText(filePath))
    };
}

function normalizeHref(value) {
    return String(value || '').trim();
}

function isIgnoredUrl(value) {
    return !value
        || value.startsWith('#')
        || /^(mailto|tel|javascript|data):/i.test(value);
}

function localPathToGeneratedFile(pathname) {
    const cleanPath = decodeURIComponent(pathname).replace(/^\/+/, '');
    if (!cleanPath) {
        return path.join(distRoot, 'index.html');
    }

    const withoutQuery = cleanPath.split('?')[0].split('#')[0];
    const candidate = path.join(distRoot, withoutQuery);
    if (withoutQuery.endsWith('/')) {
        return path.join(candidate, 'index.html');
    }
    if (fs.existsSync(candidate) && fs.statSync(candidate).isDirectory()) {
        return path.join(candidate, 'index.html');
    }
    return candidate;
}

function localUrlExists(value, baseRoute = '/') {
    if (isIgnoredUrl(value)) {
        return true;
    }

    let url;
    try {
        url = new URL(value, `${SITE_URL}${baseRoute}`);
    } catch (error) {
        return false;
    }

    if (url.origin !== siteOrigin) {
        return true;
    }

    return fs.existsSync(localPathToGeneratedFile(url.pathname));
}

function routeToDistRelative(route) {
    const clean = route.replace(/^\/+/, '');
    return clean ? path.join(clean, 'index.html') : 'index.html';
}

function getExpectedPostRoutes(siteData) {
    return siteData.posts.flatMap((post) => post.languages.map((language) => ({
        post,
        language,
        route: getPostRoute(post, language),
        absolute: `${SITE_URL}${getPostRoute(post, language)}`
    })));
}

function getIndexablePages(siteData) {
    return [
        { label: '/blogs/', route: '/blogs/', relativePath: path.join('blogs', 'index.html'), expectedCanonical: `${SITE_URL}/blogs/`, type: 'blog-index' },
        ...getExpectedPostRoutes(siteData).map(({ post, language, route, absolute }) => ({
            label: route,
            route,
            relativePath: routeToDistRelative(route),
            expectedCanonical: absolute,
            type: 'post',
            post,
            language
        })),
        ...listSeriesArchiveEntries(siteData).map((entry) => ({
            label: entry.path,
            route: entry.path,
            relativePath: routeToDistRelative(entry.path),
            expectedCanonical: `${SITE_URL}${entry.path}`,
            type: 'archive',
            archive: entry
        })),
        ...listTagArchiveEntries(siteData).map((entry) => ({
            label: entry.path,
            route: entry.path,
            relativePath: routeToDistRelative(entry.path),
            expectedCanonical: `${SITE_URL}${entry.path}`,
            type: 'archive',
            archive: entry
        }))
    ];
}

function collectSitemapUrls(errors) {
    const sitemapPath = path.join(distRoot, 'sitemap.xml');
    if (!fs.existsSync(sitemapPath)) {
        errors.push('Missing generated sitemap.xml.');
        return [];
    }

    const sitemapXml = readText(sitemapPath);
    const $ = cheerio.load(sitemapXml, { xmlMode: true });
    if ($('urlset').length !== 1) {
        errors.push('Malformed sitemap.xml: expected one <urlset> root.');
    }

    const urls = [];
    $('url').each((index, element) => {
        const loc = $(element).find('loc').first().text().trim();
        if (!loc) {
            errors.push(`Sitemap entry ${index} is missing <loc>.`);
            return;
        }
        const alternates = $(element).find('xhtml\\:link').map((alternateIndex, alternate) => ({
            hreflang: normalizeHref($(alternate).attr('hreflang')),
            href: normalizeHref($(alternate).attr('href'))
        })).get();
        urls.push({ loc, alternates });
    });

    return urls;
}

function checkStaticIndex(siteData, rawSiteData, errors) {
    const indexPath = path.join(distRoot, 'blogs', 'index.html');
    if (!fs.existsSync(indexPath)) {
        errors.push('Missing generated blog index at blogs/dist/blogs/index.html.');
        return;
    }

    const { $ } = loadHtml(path.join('blogs', 'index.html'));
    const hrefs = new Set($('a[href]').map((index, element) => normalizeHref($(element).attr('href'))).get());
    const postHrefs = [...hrefs].filter((href) => href.startsWith('/blogs/posts/') && !href.includes('?id='));

    if (postHrefs.length === 0) {
        errors.push('Blog index contains no static post links.');
    }

    getExpectedPostRoutes(siteData).forEach(({ post, language, route }) => {
        if (!hrefs.has(route)) {
            errors.push(`Published ${language} route for "${post.id}" is absent from the static blog index: ${route}`);
        }
    });

    (rawSiteData.posts || [])
        .filter((post) => post.status === 'draft')
        .forEach((post) => {
            if ($(`[data-post-id="${post.id}"]`).length || [...hrefs].some((href) => href.includes(post.slug || post.id))) {
                errors.push(`Draft post "${post.id}" appears in the generated blog index.`);
            }
        });

    const cardIds = $('[data-post-id]').map((index, element) => $(element).attr('data-post-id')).get();
    const duplicateCardIds = cardIds.filter((id, index) => cardIds.indexOf(id) !== index);
    if (duplicateCardIds.length > 0) {
        errors.push(`Featured/regular post cards are duplicated in the blog index: ${[...new Set(duplicateCardIds)].join(', ')}`);
    }
}

function checkSitemap(siteData, errors) {
    const urls = collectSitemapUrls(errors);
    const locs = urls.map((entry) => entry.loc);
    const locSet = new Set(locs);
    const sitemapAlternatesByLoc = new Map();

    urls.forEach(({ loc, alternates }) => {
        sitemapAlternatesByLoc.set(loc, alternates);
        if (!loc.startsWith(`${SITE_URL}/`) && loc !== SITE_URL) {
            errors.push(`Sitemap URL is not an absolute site HTTPS URL: ${loc}`);
        }
        if (loc.includes('?') || loc.includes('/blogs/posts/?id=')) {
            errors.push(`Sitemap contains noncanonical or legacy URL: ${loc}`);
        }
        if (loc.includes('/redirect-legacy-posts') || loc.endsWith('/blogs/posts/')) {
            errors.push(`Sitemap contains redirect-only URL: ${loc}`);
        }
        try {
            const url = new URL(loc);
            if (url.origin === siteOrigin && !fs.existsSync(localPathToGeneratedFile(url.pathname))) {
                errors.push(`Sitemap URL does not correspond to generated output: ${loc}`);
            }
        } catch (error) {
            errors.push(`Malformed sitemap URL: ${loc}`);
        }
    });

    if (locSet.size !== locs.length) {
        const duplicates = locs.filter((loc, index) => locs.indexOf(loc) !== index);
        errors.push(`Sitemap contains duplicate <loc> values: ${[...new Set(duplicates)].join(', ')}`);
    }

    if (!locSet.has(`${SITE_URL}/blogs/`)) {
        errors.push('Sitemap is missing the blog index URL.');
    }

    [...listSeriesArchiveEntries(siteData), ...listTagArchiveEntries(siteData)].forEach((entry) => {
        const absolute = `${SITE_URL}${entry.path}`;
        if (!locSet.has(absolute)) {
            errors.push(`Sitemap is missing archive URL: ${absolute}`);
        }
    });

    getExpectedPostRoutes(siteData).forEach(({ post, language, absolute }) => {
        if (!locSet.has(absolute)) {
            errors.push(`Sitemap is missing published ${language} post "${post.id}": ${absolute}`);
        }

        const expectedAlternates = post.languages.map((lang) => ({
            hreflang: lang === 'kor' ? 'ko' : 'en',
            href: `${SITE_URL}${getPostRoute(post, lang)}`
        }));
        expectedAlternates.push({ hreflang: 'x-default', href: `${SITE_URL}${getPostRoute(post, 'eng')}` });
        const actualAlternates = sitemapAlternatesByLoc.get(absolute) || [];
        expectedAlternates.forEach((expected) => {
            if (!actualAlternates.some((actual) => actual.hreflang === expected.hreflang && actual.href === expected.href)) {
                errors.push(`Sitemap alternates for ${absolute} are missing ${expected.hreflang} -> ${expected.href}.`);
            }
        });
    });
}

function checkCanonicals(siteData, errors) {
    const pages = getIndexablePages(siteData);
    const canonicalUrls = [];

    pages.forEach((page) => {
        const filePath = path.join(distRoot, page.relativePath);
        if (!fs.existsSync(filePath)) {
            errors.push(`Generated page is missing: ${page.label}`);
            return;
        }

        const { $ } = loadHtml(page.relativePath);
        const canonicals = $('link[rel="canonical"]').map((index, element) => normalizeHref($(element).attr('href'))).get();
        if (canonicals.length !== 1) {
            errors.push(`${page.label} must contain exactly one canonical link; found ${canonicals.length}.`);
            return;
        }
        if (canonicals[0] !== page.expectedCanonical) {
            errors.push(`${page.label} canonical does not match generated URL. Expected ${page.expectedCanonical}, found ${canonicals[0]}.`);
        }
        if (!canonicals[0].startsWith('https://')) {
            errors.push(`${page.label} canonical is not HTTPS: ${canonicals[0]}.`);
        }
        canonicalUrls.push(canonicals[0]);
    });

    const duplicateCanonicals = canonicalUrls.filter((url, index) => canonicalUrls.indexOf(url) !== index);
    if (duplicateCanonicals.length > 0) {
        errors.push(`Duplicate canonical URLs found: ${[...new Set(duplicateCanonicals)].join(', ')}`);
    }
}

function checkHeadingsAndTitles(siteData, errors) {
    const titleValues = [];

    getIndexablePages(siteData).forEach((page) => {
        const { $ } = loadHtml(page.relativePath);
        const title = $('title').first().text().trim();
        if (!title) {
            errors.push(`${page.label} is missing a <title>.`);
        } else {
            titleValues.push({ title, page: page.label });
        }

        if (page.type === 'blog-index' || page.type === 'archive') {
            if ($('h1').length !== 1) {
                errors.push(`${page.label} must contain exactly one primary <h1>.`);
            }
            return;
        }

        const h1s = $('h1').map((index, element) => $(element).text().replace(/\s+/g, ' ').trim()).get();
        if (h1s.length !== 1) {
            errors.push(`${page.label} must contain exactly one article <h1>; found ${h1s.length}.`);
        } else {
            const expectedTitle = getPostTitle(page.post, page.language);
            if (h1s[0] !== expectedTitle) {
                errors.push(`${page.label} <h1> does not match article title. Expected "${expectedTitle}", found "${h1s[0]}".`);
            }
        }

        if (!title.includes('Hwan Heo')) {
            errors.push(`${page.label} title tag does not identify the author.`);
        }
    });

    const seenTitles = new Map();
    titleValues.forEach(({ title, page }) => {
        if (seenTitles.has(title)) {
            errors.push(`Duplicate <title> value "${title}" on ${seenTitles.get(title)} and ${page}.`);
        }
        seenTitles.set(title, page);
    });
}

function checkDescriptions(siteData, errors) {
    const seenDescriptions = new Map();

    getIndexablePages(siteData).forEach((page) => {
        const { $ } = loadHtml(page.relativePath);
        const descriptions = $('meta[name="description"]').map((index, element) => normalizeHref($(element).attr('content'))).get();
        if (descriptions.length !== 1) {
            errors.push(`${page.label} must contain exactly one meta description; found ${descriptions.length}.`);
            return;
        }
        const description = descriptions[0].replace(/\s+/g, ' ').trim();
        if (description.length < 30) {
            errors.push(`${page.label} meta description is too short or empty.`);
        }
        if (/[#*_`]|<[^>]+>/.test(description)) {
            errors.push(`${page.label} meta description contains Markdown or HTML syntax.`);
        }
        if (seenDescriptions.has(description)) {
            errors.push(`Duplicate meta description on ${seenDescriptions.get(description)} and ${page.label}.`);
        }
        seenDescriptions.set(description, page.label);
    });
}

function checkHtmlLanguageAndDates(siteData, errors) {
    getExpectedPostRoutes(siteData).forEach(({ post, language, route }) => {
        const relativePath = routeToDistRelative(route);
        const { $ } = loadHtml(relativePath);
        const expectedHtmlLang = language === 'kor' ? 'ko' : 'en';
        const htmlLang = $('html').attr('lang');
        if (htmlLang !== expectedHtmlLang) {
            errors.push(`${route} html lang should be "${expectedHtmlLang}", found "${htmlLang}".`);
        }
        if ($(`time[datetime="${post.date}"]`).length === 0) {
            errors.push(`${route} does not render the publication date with <time datetime="${post.date}">.`);
        }
    });
}

function checkSocialMetadata(siteData, errors) {
    getIndexablePages(siteData).forEach((page) => {
        const { $ } = loadHtml(page.relativePath);
        const required = page.type === 'post'
            ? [
                ['meta[property="og:type"]', 'article'],
                ['meta[property="og:title"]'],
                ['meta[property="og:description"]'],
                ['meta[property="og:url"]', page.expectedCanonical],
                ['meta[property="og:image"]'],
                ['meta[name="twitter:card"]', 'summary_large_image'],
                ['meta[name="twitter:title"]'],
                ['meta[name="twitter:description"]'],
                ['meta[name="twitter:image"]']
            ]
            : [
                ['meta[property="og:type"]', 'website'],
                ['meta[property="og:title"]'],
                ['meta[property="og:description"]'],
                ['meta[property="og:url"]', page.expectedCanonical],
                ['meta[property="og:image"]'],
                ['meta[name="twitter:card"]', 'summary_large_image'],
                ['meta[name="twitter:title"]'],
                ['meta[name="twitter:description"]'],
                ['meta[name="twitter:image"]']
            ];

        required.forEach(([selector, expected]) => {
            const value = normalizeHref($(selector).attr('content'));
            if (!value) {
                errors.push(`${page.label} is missing ${selector}.`);
                return;
            }
            if (expected && value !== expected) {
                errors.push(`${page.label} ${selector} expected "${expected}", found "${value}".`);
            }
        });

        ['meta[property="og:image"]', 'meta[name="twitter:image"]'].forEach((selector) => {
            const imageUrl = normalizeHref($(selector).attr('content'));
            if (!imageUrl.startsWith('https://')) {
                errors.push(`${page.label} ${selector} is not an absolute HTTPS URL.`);
            }
            if (!localUrlExists(imageUrl, page.route)) {
                errors.push(`${page.label} ${selector} points to missing generated output: ${imageUrl}`);
            }
        });
    });
}

function hasEmptyStructuredValue(value) {
    if (value === '') {
        return true;
    }
    if (Array.isArray(value)) {
        return value.some(hasEmptyStructuredValue);
    }
    if (value && typeof value === 'object') {
        return Object.values(value).some(hasEmptyStructuredValue);
    }
    return false;
}

function isValidDateOnly(value) {
    return /^\d{4}-\d{2}-\d{2}$/.test(String(value || ''))
        && !Number.isNaN(new Date(`${value}T00:00:00Z`).getTime());
}

function checkStructuredData(siteData, errors) {
    getIndexablePages(siteData).forEach((page) => {
        const { $ } = loadHtml(page.relativePath);
        const blocks = $('script[type="application/ld+json"]').map((index, element) => $(element).contents().text().trim()).get();
        if (blocks.length === 0) {
            errors.push(`${page.label} has no JSON-LD block.`);
            return;
        }

        const parsedBlocks = [];
        blocks.forEach((block, index) => {
            try {
                parsedBlocks.push(JSON.parse(block));
            } catch (error) {
                errors.push(`${page.label} JSON-LD block ${index} is invalid JSON: ${error.message}`);
            }
        });

        if (page.type === 'blog-index' || page.type === 'archive') {
            const expectedTypes = page.type === 'blog-index' ? ['Blog', 'WebSite'] : ['CollectionPage'];
            if (!parsedBlocks.some((block) => expectedTypes.includes(block['@type']))) {
                errors.push(`${page.label} JSON-LD must include ${expectedTypes.join(' or ')} structured data.`);
            }
            return;
        }

        const blogPosting = parsedBlocks.find((block) => block['@type'] === 'BlogPosting');
        if (!blogPosting) {
            errors.push(`${page.label} JSON-LD must include BlogPosting.`);
            return;
        }

        ['headline', 'description', 'url', 'mainEntityOfPage', 'datePublished', 'inLanguage'].forEach((key) => {
            if (!blogPosting[key]) {
                errors.push(`${page.label} BlogPosting is missing "${key}".`);
            }
        });
        if (blogPosting.url !== page.expectedCanonical || blogPosting.mainEntityOfPage !== page.expectedCanonical) {
            errors.push(`${page.label} BlogPosting URL fields must match the canonical URL.`);
        }
        if (!isValidDateOnly(blogPosting.datePublished)) {
            errors.push(`${page.label} BlogPosting datePublished is invalid: ${blogPosting.datePublished}`);
        }
        if (blogPosting.dateModified && !isValidDateOnly(blogPosting.dateModified)) {
            errors.push(`${page.label} BlogPosting dateModified is invalid: ${blogPosting.dateModified}`);
        }
        if (!blogPosting.author?.name) {
            errors.push(`${page.label} BlogPosting author is missing a name.`);
        }
        if (!Array.isArray(blogPosting.image) || blogPosting.image.length === 0 || !blogPosting.image.every((url) => /^https:\/\//.test(url))) {
            errors.push(`${page.label} BlogPosting image must be a non-empty array of absolute HTTPS URLs.`);
        }
        if (hasEmptyStructuredValue(blogPosting)) {
            errors.push(`${page.label} BlogPosting contains empty string values.`);
        }
    });
}

function checkInternalLinks(siteData, errors) {
    const pageRoutes = getIndexablePages(siteData).map((page) => page.route);

    pageRoutes.forEach((route) => {
        const relativePath = routeToDistRelative(route);
        const filePath = path.join(distRoot, relativePath);
        if (!fs.existsSync(filePath)) {
            return;
        }

        const { $ } = loadHtml(relativePath);
        $('a[href]').each((index, element) => {
            const href = normalizeHref($(element).attr('href'));
            if (!localUrlExists(href, route)) {
                errors.push(`${route} links to missing generated output: ${href}`);
            }
        });
    });
}

function checkLanguageAlternates(siteData, errors) {
    getExpectedPostRoutes(siteData).forEach(({ post, language, route }) => {
        const relativePath = routeToDistRelative(route);
        const { $ } = loadHtml(relativePath);
        const alternates = $('link[rel="alternate"][hreflang]').map((index, element) => ({
            hreflang: normalizeHref($(element).attr('hreflang')),
            href: normalizeHref($(element).attr('href'))
        })).get();
        const expectedLanguages = post.languages.map((lang) => lang === 'kor' ? 'ko' : 'en');

        expectedLanguages.forEach((hreflang) => {
            if (!alternates.some((alternate) => alternate.hreflang === hreflang)) {
                errors.push(`${route} is missing hreflang="${hreflang}".`);
            }
        });

        alternates.forEach((alternate) => {
            if (!['en', 'ko', 'x-default'].includes(alternate.hreflang)) {
                errors.push(`${route} contains invalid hreflang "${alternate.hreflang}".`);
            }
            if (!alternate.href.startsWith(`${SITE_URL}/blogs/posts/`)) {
                errors.push(`${route} has noncanonical alternate href: ${alternate.href}`);
            }
            if (!localUrlExists(alternate.href, route)) {
                errors.push(`${route} alternate target is missing: ${alternate.href}`);
            }
        });

        if (language === 'eng' && post.languages.includes('kor')) {
            const koRoute = getPostRoute(post, 'kor');
            const koPath = routeToDistRelative(koRoute);
            const koPage = loadHtml(koPath).$;
            const koAlternates = koPage('link[rel="alternate"][hreflang]').map((index, element) => normalizeHref(koPage(element).attr('href'))).get();
            if (!koAlternates.includes(`${SITE_URL}${route}`)) {
                errors.push(`Language alternates are not reciprocal between ${route} and ${koRoute}.`);
            }
        }
    });
}

function checkRobots(errors) {
    const robotsPath = path.join(distRoot, 'robots.txt');
    if (!fs.existsSync(robotsPath)) {
        errors.push('Missing generated robots.txt.');
        return;
    }

    const robots = readText(robotsPath);
    if (!/User-agent:\s*\*/i.test(robots) || !/Allow:\s*\//i.test(robots)) {
        errors.push('robots.txt does not explicitly allow the public site.');
    }
    if (/Disallow:\s*\/blogs/i.test(robots)) {
        errors.push('robots.txt blocks /blogs/.');
    }
    if (!robots.includes(`Sitemap: ${SITE_URL}/sitemap.xml`)) {
        errors.push('robots.txt does not reference the absolute sitemap URL.');
    }
}

function checkLegacyRedirects(siteData, errors) {
    try {
        validateLegacyRedirects(siteData, loadLegacyRedirects());
    } catch (error) {
        errors.push(error.message);
    }

    const fallbackPath = path.join(distRoot, 'blogs', 'posts', 'index.html');
    if (!fs.existsSync(fallbackPath)) {
        errors.push('Legacy fallback page is missing at /blogs/posts/.');
        return;
    }

    const html = readText(fallbackPath);
    if (html.includes('site-data-client.js') || html.includes('fetch(')) {
        errors.push('Legacy fallback still depends on remote data fetches.');
    }
    if (!html.includes('location.replace(')) {
        errors.push('Legacy fallback must use location.replace().');
    }
    if (!html.includes('No published post mapping exists')) {
        errors.push('Legacy fallback does not expose a clear unknown-ID state.');
    }
}

function checkFeed(siteData, errors) {
    const feedPath = path.join(distRoot, 'blogs', 'feed.xml');
    if (!fs.existsSync(feedPath)) {
        errors.push('Missing RSS feed at /blogs/feed.xml.');
        return;
    }

    const feedXml = readText(feedPath);
    const $ = cheerio.load(feedXml, { xmlMode: true });
    if ($('rss').attr('version') !== '2.0' || $('channel').length !== 1) {
        errors.push('RSS feed is malformed or missing an RSS 2.0 channel.');
    }

    const itemLinks = $('item link').map((index, element) => $(element).text().trim()).get();
    if (itemLinks.length === 0) {
        errors.push('RSS feed contains no items.');
    }

    itemLinks.forEach((link) => {
        if (!link.startsWith(`${SITE_URL}/blogs/posts/`) || link.includes('?')) {
            errors.push(`RSS feed item link is noncanonical: ${link}`);
        }
        if (!localUrlExists(link, '/blogs/feed.xml')) {
            errors.push(`RSS feed item link points to missing generated output: ${link}`);
        }
    });

    const expectedEnglishLinks = siteData.posts
        .filter((post) => post.languages.includes('eng'))
        .map((post) => `${SITE_URL}${getPostRoute(post, 'eng')}`);
    expectedEnglishLinks.forEach((link) => {
        if (!itemLinks.includes(link)) {
            errors.push(`RSS feed is missing English post URL: ${link}`);
        }
    });

    const blogIndex = loadHtml(path.join('blogs', 'index.html')).$;
    const feedDiscovery = blogIndex('link[rel="alternate"][type="application/rss+xml"]').attr('href');
    if (feedDiscovery !== '/blogs/feed.xml') {
        errors.push('Blog index is missing RSS autodiscovery for /blogs/feed.xml.');
    }
}

async function checkCloudflareWorker(siteData, errors) {
    const workerPath = path.join(repoRoot, 'deploy', 'cloudflare-worker', 'src', 'index.js');
    if (!fs.existsSync(workerPath)) {
        errors.push('Cloudflare Worker redirect asset is missing.');
        return;
    }

    let workerModule;
    try {
        workerModule = await import(pathToFileURL(workerPath).href);
    } catch (error) {
        errors.push(`Cloudflare Worker module could not be imported for validation: ${error.message}`);
        return;
    }

    if (workerModule.REDIRECT_STATUS !== 308) {
        errors.push(`Cloudflare Worker must use redirect status 308; found ${workerModule.REDIRECT_STATUS}.`);
    }

    const redirects = loadLegacyRedirects();
    const samplePost = siteData.posts[0];
    const sampleRedirect = workerModule.getLegacyRedirectUrl(`https://example.com/blogs/posts/?id=${samplePost.id}`, {
        CANONICAL_ORIGIN: SITE_URL
    });
    if (sampleRedirect !== `${SITE_URL}${redirects[samplePost.id]}`) {
        errors.push(`Cloudflare Worker known-ID redirect mismatch: ${sampleRedirect}`);
    }

    const unknownRedirect = workerModule.getLegacyRedirectUrl('https://example.com/blogs/posts/?id=unknown', {
        CANONICAL_ORIGIN: SITE_URL
    });
    if (unknownRedirect) {
        errors.push('Cloudflare Worker must not redirect unknown legacy IDs.');
    }

    const unrelatedRedirect = workerModule.getLegacyRedirectUrl('https://example.com/blogs/', {
        CANONICAL_ORIGIN: SITE_URL
    });
    if (unrelatedRedirect) {
        errors.push('Cloudflare Worker must preserve unrelated URLs.');
    }
}

async function main() {
    const errors = [];
    if (!fs.existsSync(path.join(distRoot, 'index.html'))) {
        throw new Error('blogs/dist is missing. Run npm run build before npm run validate:seo.');
    }

    const rawSiteData = loadRawSiteData();
    const siteData = loadSiteData();

    checkStaticIndex(siteData, rawSiteData, errors);
    checkSitemap(siteData, errors);
    checkCanonicals(siteData, errors);
    checkHeadingsAndTitles(siteData, errors);
    checkDescriptions(siteData, errors);
    checkHtmlLanguageAndDates(siteData, errors);
    checkSocialMetadata(siteData, errors);
    checkStructuredData(siteData, errors);
    checkInternalLinks(siteData, errors);
    checkLanguageAlternates(siteData, errors);
    checkRobots(errors);
    checkLegacyRedirects(siteData, errors);
    await checkCloudflareWorker(siteData, errors);
    checkFeed(siteData, errors);

    if (errors.length > 0) {
        console.error('SEO validation failed.');
        errors.forEach((error) => console.error(`- ${error}`));
        process.exit(1);
    }

    console.log(`SEO validation passed: ${siteData.posts.length} published posts, ${getExpectedPostRoutes(siteData).length} post-language routes.`);
}

main().catch((error) => {
    console.error(error.message);
    process.exit(1);
});
