const assert = require('node:assert/strict');
const { chromium } = require('playwright');
const { startStaticServer, getLaunchOptions } = require('./check-rendered-site');

async function checkThemePreferences(browser, base) {
    const context = await browser.newContext({ reducedMotion: 'reduce' });
    await context.route('**/*', route => new URL(route.request().url()).origin === base ? route.continue() : route.abort());
    const expectTheme = async (page, theme) => {
        await page.waitForFunction(value => document.documentElement.dataset.theme === value, theme);
        assert.equal(await page.evaluate(() => window.siteTheme.get()), theme);
    };
    const visit = async (page, path, theme) => {
        assert((await page.goto(base + path)).ok());
        await expectTheme(page, theme);
    };
    try {
        const portfolio = await context.newPage(), blog = await context.newPage();
        await visit(portfolio, '/#blog', 'dark');
        for (const path of ['/blogs/', '/blogs/search/?q=mesh', '/blogs/tags/graphics/', '/blogs/posts/gemm-on-nvidia-gpus/']) {
            await visit(blog, path, 'light');
        }
        await visit(portfolio, '/projects/varco3d/', 'dark');

        // The legacy shared setting belongs only to the Portfolio after migration.
        await portfolio.evaluate(() => {
            localStorage.clear();
            localStorage.setItem('site-theme', 'light');
            localStorage.setItem('blog-theme', 'dark');
        });
        await visit(portfolio, '/#blog', 'light');
        await visit(blog, '/blogs/', 'light');
        await blog.locator('[data-theme-toggle]').click();
        await expectTheme(blog, 'dark');
        await expectTheme(portfolio, 'light');
        await portfolio.locator('.edition-tone').click();
        await expectTheme(portfolio, 'dark');
        await portfolio.locator('.edition-tone').click();
        await expectTheme(portfolio, 'light');
        await expectTheme(blog, 'dark');

        const portfolioTab = await context.newPage(), blogTab = await context.newPage();
        await visit(portfolioTab, '/projects/varco3d/', 'light');
        await visit(blogTab, '/blogs/posts/gemm-on-nvidia-gpus/', 'dark');
        await blogTab.locator('[data-theme-toggle]').click();
        await expectTheme(blog, 'light');
        await portfolioTab.locator('.edition-tone').click();
        await expectTheme(portfolio, 'dark');
        await expectTheme(blog, 'light');
        await expectTheme(blogTab, 'light');
        await portfolio.reload(); await expectTheme(portfolio, 'dark');
        await blog.reload(); await expectTheme(blog, 'light');

        // A restored document picks up changes made elsewhere in its own area.
        await visit(portfolio, '/projects/varco3d/', 'dark');
        await portfolio.locator('.edition-tone').click();
        await portfolio.goBack(); await expectTheme(portfolio, 'light');
        await visit(blog, '/blogs/posts/gemm-on-nvidia-gpus/', 'light');
        await blog.locator('[data-theme-toggle]').click();
        await blog.goBack(); await expectTheme(blog, 'dark');
        await expectTheme(portfolio, 'light');
        await visit(blog, '/blogs/?theme=light', 'light');
        await visit(portfolio, '/?theme=dark#blog', 'dark');
        await expectTheme(blog, 'light');
        assert.deepEqual(await portfolio.evaluate(() => [localStorage.getItem('portfolio-theme'), localStorage.getItem('blog-reading-theme')]), ['dark', 'light']);
    } finally { await context.close(); }

    const blocked = await browser.newContext({ reducedMotion: 'reduce' });
    await blocked.route('**/*', route => new URL(route.request().url()).origin === base ? route.continue() : route.abort());
    await blocked.addInitScript(() => {
        for (const method of ['getItem', 'setItem']) Storage.prototype[method] = () => { throw new DOMException('Storage disabled', 'SecurityError'); };
    });
    try {
        const page = await blocked.newPage();
        await visit(page, '/blogs/?theme=dark', 'dark');
        await page.locator('[data-theme-toggle]').click(); await expectTheme(page, 'light');
        await visit(page, '/?theme=light#blog', 'light');
        await page.locator('.edition-tone').click(); await expectTheme(page, 'dark');
    } finally { await blocked.close(); }
    return { independentDefaults: true, legacyMigration: true, scopedTabSync: true, reloadAndBack: true, scopedOverrides: true, blockedStorage: true };
}

module.exports = { checkThemePreferences };
if (require.main === module) {
    (async () => {
        const server = await startStaticServer();
        const browser = await chromium.launch(getLaunchOptions());
        try { console.log('Theme preferences passed:', await checkThemePreferences(browser, server.baseUrl)); }
        finally { await browser.close(); server.server.close(); }
    })().catch(error => { console.error(error); process.exitCode = 1; });
}
