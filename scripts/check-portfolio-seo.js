#!/usr/bin/env node

const assert = require('node:assert/strict');
const { chromium } = require('playwright');

const { startStaticServer, getLaunchOptions } = require('./check-rendered-site');
const { getPortfolioProfile } = require('../blogs/lib/portfolio-profile');
const { loadSiteData } = require('../blogs/lib/site-data');
const { getPostRoute } = require('../blogs/lib/site-routes');

async function main() {
    const profile = getPortfolioProfile();
    const featured = loadSiteData().featuredPortfolioPosts.filter((item) => item.post?.status === 'published');
    const server = await startStaticServer();
    let browser;
    try {
        browser = await chromium.launch(getLaunchOptions());
        for (const width of [412, 1440]) {
            for (const mode of ['no-js', 'rendered', 'blocked-data']) {
                const context = await browser.newContext({
                    viewport: { width, height: 915 },
                    isMobile: width < 900,
                    javaScriptEnabled: mode !== 'no-js',
                    reducedMotion: 'reduce'
                });
                try {
                    await context.route('**/*', (route) => {
                        const url = new URL(route.request().url());
                        if (url.origin !== server.baseUrl || (mode === 'blocked-data' && url.pathname.endsWith('.json'))) {
                            return route.abort();
                        }
                        return route.continue();
                    });
                    await context.addInitScript(() => {
                        document.addEventListener('DOMContentLoaded', () => {
                            window.initialSeoTitle = document.title;
                            window.initialPreviewNodes = [...document.querySelectorAll('.portfolio-blog-preview-item')];
                        }, { once: true });
                    });
                    const page = await context.newPage();
                    const errors = [];
                    page.on('pageerror', (error) => errors.push(error.message));
                    const response = await page.goto(`${server.baseUrl}/#blog`, { waitUntil: 'load' });
                    assert(response.ok(), 'Portfolio must return successful HTML.');

                    if (mode !== 'no-js') {
                        await page.waitForFunction(() => window.portfolioChapters && document.getElementById('blog').dataset.sectionReady === 'true');
                        assert.equal(await page.evaluate(() => window.initialSeoTitle), profile.title, 'Initial title must identify the author and job.');
                        assert(await page.evaluate(() => window.initialPreviewNodes.every((node) => node.isConnected)), 'Hydration must preserve the initial preview links.');
                    }
                    assert.equal(await page.title(), profile.title, 'JavaScript must preserve the static SEO title.');
                    assert.equal(await page.locator('link[rel="canonical"]').getAttribute('href'), profile.canonical);
                    assert.equal(await page.locator('.portfolio-blog-preview-item').count(), featured.length);
                    assert(await page.locator('#blog').isVisible(), 'Blog must be readable without a content fetch or user reveal.');

                    for (const { id, post } of featured) {
                        const card = page.locator(`.portfolio-blog-preview-item[data-post-id="${id}"]`);
                        assert(await card.isVisible(), `${id} must remain visible in ${mode} mode.`);
                        assert.equal(await card.locator('.portfolio-blog-preview-title').textContent(), post.title_eng);
                        assert.equal(await card.locator('.portfolio-blog-preview-title-link').getAttribute('href'), getPostRoute(post, 'eng'));
                        const image = card.locator('img');
                        await image.scrollIntoViewIfNeeded();
                        await image.evaluate((node) => node.decode());
                        assert(await image.evaluate((node) => node.naturalWidth > 0), `${id} cover must load.`);
                    }

                    if (mode === 'rendered') {
                        // Exercise actual chapter controls: title stability must survive
                        // navigation as well as the initial JavaScript execution.
                        for (const chapter of ['about', 'portfolio', 'blog']) {
                            await page.locator(`.edition-nav a[href="#${chapter}"]`).click();
                            await page.waitForFunction((id) => document.body.dataset.chapter === id, chapter);
                            assert.equal(await page.title(), profile.title, `Chapter ${chapter} must preserve the document title.`);
                        }
                        if (width < 900) {
                            assert(await page.locator('.about-bio').isVisible(), 'Mobile profile content must exist without an interaction gate.');
                            assert(await page.locator('#portfolio-projects').isVisible(), 'Mobile projects must be available in normal document flow.');
                        }
                    }
                    assert.deepEqual(errors, [], 'Portfolio must not throw script errors.');
                    console.log(`PASS portfolio SEO: ${width}px, ${mode}, ${featured.length} static previews.`);
                } finally {
                    await context.close();
                }
            }
        }
    } finally {
        await browser?.close();
        await new Promise((resolve) => server.server.close(resolve));
    }
    console.log('Portfolio SEO rendering passed: stable metadata, static links/images, mobile content and data-fetch failure fallback.');
}

main().catch((error) => {
    console.error(error);
    process.exitCode = 1;
});
