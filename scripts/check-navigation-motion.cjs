const assert = require('node:assert/strict');
const { chromium } = require('playwright');
const { startStaticServer } = require('./check-rendered-site');

(async () => {
    const server = await startStaticServer();
    // Playwright normally disables RenderDocument/PaintHolding and BFCache,
    // which prevents testing the browser's native document transitions.
    const browser = await chromium.launch({
        executablePath: process.env.PLAYWRIGHT_CHROMIUM_EXECUTABLE_PATH || chromium.executablePath(),
        args: ['--disable-features=OptimizationHints'],
        ignoreDefaultArgs: ['--disable-back-forward-cache']
    });
    try {
        const context = await browser.newContext({ viewport: { width: 1440, height: 960 } });
        const page = await context.newPage();
        const errors = [];
        page.on('pageerror', error => errors.push(error.message));
        await page.addInitScript(() => {
            addEventListener('pagereveal', event => {
                const report = window.navigationMotionCheck = { active: !!event.viewTransition, samples: [], done: false };
                if (!event.viewTransition) { report.done = true; return; }
                event.viewTransition.ready.then(() => {
                    const fades = document.getAnimations().filter(animation => /view-transition-fade/.test(animation.animationName || ''));
                    report.durations = fades.map(animation => animation.effect.getTiming().duration);
                    const sample = () => {
                        report.samples.push(fades.map(animation => animation.effect.getComputedTiming().progress));
                        if (!report.done) requestAnimationFrame(sample);
                    };
                    sample();
                }).catch(() => { report.skipped = true; });
                event.viewTransition.finished.finally(() => { report.done = true; });
            });
        });
        async function assertCrossfade(label) {
            await page.waitForFunction(() => window.navigationMotionCheck?.active && window.navigationMotionCheck.done, null, { timeout: 7000 }).catch(async error => {
                throw new Error(`${label}: ${JSON.stringify(await page.evaluate(() => window.navigationMotionCheck))}; ${error.message}`);
            });
            const report = await page.evaluate(() => window.navigationMotionCheck);
            assert(!report.skipped, `${label}: transition skipped`);
            assert.deepEqual(report.durations, [360, 360], `${label}: fade duration`);
            assert(report.samples.some(frame => frame.some(progress => progress > 0 && progress < 1)), `${label}: visible intermediate frames`);
            console.log(`PASS ${label}: native 360ms crossfade`);
        }

        await page.goto(server.baseUrl + '/#portfolio', { waitUntil: 'networkidle' });
        const entry = page.locator('a[href="projects/varco3d/"]').first();
        await entry.scrollIntoViewIfNeeded();
        const scroll = await page.locator('#portfolio').evaluate(node => node.scrollTop);
        await entry.click();
        await page.waitForURL('**/projects/varco3d/');
        await assertCrossfade('Projects → detail');
        await page.goBack({ waitUntil: 'commit' });
        await assertCrossfade('Detail → history return');
        assert.equal(await page.evaluate(() => document.body.dataset.chapter), 'portfolio');
        assert(Math.abs(await page.locator('#portfolio').evaluate(node => node.scrollTop) - scroll) < 4);

        await page.locator('.edition-nav a[href="#blog"]').click();
        await page.waitForFunction(() => document.body.dataset.chapter === 'blog');
        const [blog] = await Promise.all([
            context.waitForEvent('page'),
            page.locator('.portfolio-blog-all-link').click()
        ]);
        await blog.waitForLoadState('domcontentloaded');
        assert.equal(new URL(page.url()).hash, '#blog');
        assert.equal(await page.evaluate(() => document.documentElement.dataset.theme), 'dark');
        assert.equal(await blog.evaluate(() => document.documentElement.dataset.theme), 'light');
        await blog.close();
        console.log('PASS Blog opens independently in a new tab with its own theme');

        await page.goto(server.baseUrl + '/blogs/', { waitUntil: 'networkidle' });
        await page.locator('.blog-feature-copy h2 a').click();
        await page.waitForURL('**/blogs/posts/**');
        await assertCrossfade('Blog home → article');
        await page.locator('.post-nav-home').click();
        await page.waitForURL('**/blogs/');
        await assertCrossfade('Article → Blog Home');

        await page.emulateMedia({ reducedMotion: 'reduce' });
        await page.locator('.blog-feature-copy h2 a').click();
        await page.waitForURL('**/blogs/posts/**');
        await page.waitForFunction(() => window.navigationMotionCheck?.done);
        assert.equal(await page.evaluate(() => navigationMotionCheck.active), false);
        console.log('PASS reduced motion keeps immediate native navigation');
        assert.deepEqual(errors, []);
        await context.close();
    } finally {
        await browser.close();
        await new Promise(resolve => server.server.close(resolve));
    }
})().catch(error => { console.error(error); process.exitCode = 1; });
