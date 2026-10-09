const assert = require('node:assert/strict');
const { chromium } = require('playwright');
const { startStaticServer } = require('./check-rendered-site');

(async () => {
    const server = await startStaticServer();
    const browser = await chromium.launch({
        executablePath: process.env.PLAYWRIGHT_CHROMIUM_EXECUTABLE_PATH || chromium.executablePath(),
        args: ['--disable-features=OptimizationHints']
    });
    try {
        const context = await browser.newContext();
        const page = await context.newPage();
        const errors = [];
        page.on('pageerror', error => errors.push(error.message));
        await page.addInitScript(() => {
            window.themeFades = [];
            const start = document.startViewTransition.bind(document);
            document.startViewTransition = update => {
                const transition = start(update);
                const report = { samples: [], done: false };
                window.themeFades.push(report);
                transition.ready.then(() => {
                    const fades = document.getAnimations().filter(animation => /view-transition-fade/.test(animation.animationName || ''));
                    report.durations = fades.map(animation => animation.effect.getTiming().duration);
                    const sample = () => {
                        report.samples.push(fades.map(animation => animation.effect.getComputedTiming().progress));
                        if (!report.done) requestAnimationFrame(sample);
                    };
                    sample();
                }).catch(() => { report.skipped = true; });
                transition.finished.then(() => { report.done = true; });
                return transition;
            };
        });
        const button = '[data-theme-toggle],.edition-tone';
        const opposite = theme => theme === 'dark' ? 'light' : 'dark';
        const settled = () => page.waitForFunction(() => window.themeFades.at(-1)?.done && !document.documentElement.classList.contains('is-theme-changing'));
        for (const width of [1440, 390]) {
            await page.setViewportSize({ width, height: 960 });
            for (const route of ['/#portfolio', '/projects/varco3d/', '/blogs/', '/blogs/posts/sdf-and-eikonal-equation/']) {
                await page.goto(server.baseUrl + route, { waitUntil: 'networkidle' });
                await page.evaluate(() => document.fonts.ready);
                const initial = await page.evaluate(() => siteTheme.get());
                await page.locator(button).first().click();
                await settled();
                const report = await page.evaluate(() => themeFades.at(-1));
                assert(!report.skipped, `${route}: fade skipped`);
                assert.deepEqual(report.durations, [360, 360]);
                assert(report.samples.some(frame => frame.some(progress => progress > 0 && progress < 1)));
                assert.equal(await page.evaluate(() => document.documentElement.dataset.theme), opposite(initial));
                console.log(`PASS ${width}px ${route}: actual 360ms theme crossfade`);
            }
        }

        // Real pointer clicks must pass through the fade overlay, with the last
        // request winning even when earlier snapshot callbacks have not run yet.
        const initial = await page.evaluate(() => siteTheme.get());
        const count = await page.evaluate(() => themeFades.length);
        const rect = await page.locator(button).first().boundingBox();
        for (let i = 0; i < 3; i += 1) {
            await page.mouse.click(rect.x + rect.width / 2, rect.y + rect.height / 2);
            await page.waitForTimeout(35);
        }
        await settled();
        assert.equal(await page.evaluate(() => themeFades.length), count + 3);
        assert.deepEqual(await page.evaluate(() => [siteTheme.get(), document.documentElement.dataset.theme, localStorage.getItem('blog-reading-theme')]), Array(3).fill(opposite(initial)));
        console.log('PASS rapid clicks are received; latest theme wins with no stuck overlay');

        await page.evaluate(() => scrollTo({ top: 700, behavior: 'instant' }));
        const top = await page.evaluate(() => scrollY);
        await page.locator(button).first().evaluate(node => { node.focus({ preventScroll: true }); node.click(); });
        await settled();
        assert.equal(await page.evaluate(() => scrollY), top);
        assert(await page.locator(button).first().evaluate(node => document.activeElement === node));
        console.log('PASS article reading position and keyboard focus remain intact');

        const lastCount = await page.evaluate(() => themeFades.length);
        await page.emulateMedia({ reducedMotion: 'reduce' });
        await page.locator(button).first().evaluate(node => node.click());
        assert.equal(await page.evaluate(() => themeFades.length), lastCount);
        assert.equal(await page.evaluate(() => siteTheme.get()), await page.evaluate(() => document.documentElement.dataset.theme));
        await page.emulateMedia({ reducedMotion: 'no-preference' });
        await page.evaluate(() => { document.startViewTransition = undefined; });
        await page.locator(button).first().evaluate(node => node.click());
        assert.equal(await page.evaluate(() => siteTheme.get()), await page.evaluate(() => document.documentElement.dataset.theme));
        assert.equal(await page.evaluate(() => document.documentElement.classList.contains('is-theme-changing')), false);
        assert.deepEqual(errors, []);
        console.log('PASS reduced motion and unsupported browsers apply themes immediately');
        await context.close();
    } finally {
        await browser.close();
        await new Promise(resolve => server.server.close(resolve));
    }
})().catch(error => { console.error(error); process.exitCode = 1; });
