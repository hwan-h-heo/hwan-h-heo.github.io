const assert = require('node:assert/strict');
const fs = require('node:fs');
const http = require('node:http');
const path = require('node:path');
const { chromium } = require('playwright');

const dist = path.resolve(__dirname, '../blogs/dist');
const types = { '.html': 'text/html', '.css': 'text/css', '.js': 'text/javascript', '.json': 'application/json', '.svg': 'image/svg+xml', '.ttf': 'font/ttf', '.webp': 'image/webp', '.png': 'image/png' };

async function main() {
    const server = http.createServer((req, res) => {
        const file = path.resolve(dist, '.' + new URL(req.url, 'http://localhost').pathname);
        if (!file.startsWith(dist + '/') && file !== dist) { res.writeHead(403).end(); return; }
        const target = fs.existsSync(file) && fs.statSync(file).isDirectory() ? path.join(file, 'index.html') : file;
        if (!fs.existsSync(target) || !fs.statSync(target).isFile()) { res.writeHead(404).end(); return; }
        res.setHeader('Content-Type', types[path.extname(target)] || 'application/octet-stream');
        fs.createReadStream(target).pipe(res);
    });
    await new Promise(resolve => server.listen(0, '127.0.0.1', resolve));
    const base = `http://127.0.0.1:${server.address().port}`;
    const chrome = process.env.PLAYWRIGHT_CHROMIUM_EXECUTABLE_PATH || '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome';
    const browser = await chromium.launch({ headless: true, ...(fs.existsSync(chrome) ? { executablePath: chrome } : {}) });
    try {
        for (const width of [1440, 768, 390, 320]) {
            const height = width > 900 ? 1000 : width > 600 ? 1024 : width === 320 ? 640 : 844;
            const page = await browser.newPage({ viewport: { width, height } });
            const errors = [], retiredRequests = [];
            page.on('pageerror', error => errors.push(error.message));
            page.on('request', request => { if (/hero-wave|voxel-hierarchy|_workspace/.test(request.url())) retiredRequests.push(request.url()); });
            await page.goto(base);
            await page.waitForFunction(() => window.portfolioHero?.state().ready && portfolioHero.state().settled);
            await page.evaluate(() => document.fonts.ready);
            await page.waitForFunction(() => document.querySelector('.pbr-hero').clientWidth === innerWidth);
            const layout = await page.evaluate(() => {
                const hero = document.querySelector('.pbr-hero');
                const lines = [...hero.querySelectorAll('.pbr-lead-line')].map(el => {
                    const range = document.createRange(); range.selectNodeContents(el);
                    return [...range.getClientRects()].filter(rect => rect.width > 0).length;
                });
                const actions = [...hero.querySelectorAll('.pbr-actions a')].map(el => {
                    const r = el.getBoundingClientRect();
                    return { label: el.querySelector('.pbr-action-label').textContent, height: r.height, bottom: r.bottom, size: getComputedStyle(el).fontSize };
                });
                return { lines, actions, overflow: document.documentElement.scrollWidth > innerWidth,
                    leadSize: getComputedStyle(hero.querySelector('.pbr-lead')).fontSize,
                    about: hero.querySelector('.pbr-profile-link span').textContent.trim(), height: hero.clientHeight };
            });
            assert.deepEqual(layout.lines, [1, 1, 1]); assert(!layout.overflow);
            assert.equal(layout.leadSize, width <= 600 ? (height <= 720 ? '14px' : '15px') : '16px');
            assert.equal(layout.about, 'About me');
            assert.deepEqual(layout.actions.map(a => a.label), ['Projects', 'Blogs']);
            assert(layout.actions.every(a => a.height >= 44 && a.bottom < height && a.size === '12px'));

            const finished = await page.evaluate(() => portfolioHero.state());
            await page.evaluate(() => dispatchEvent(new WheelEvent('wheel',{deltaY:120})));
            await page.waitForTimeout(300);
            const afterGesture = await page.evaluate(() => portfolioHero.state());
            assert.equal(afterGesture.decoding.progress,1);
            assert.equal(afterGesture.decoding.plays,1);
            assert.equal(afterGesture.frames,finished.frames,'Gesture cues must not render or reverse geometry');
            assert.equal(await page.evaluate(() => scrollY),0);
            assert(afterGesture.tilts.every(t => t.x === 0 && t.y === 0 && t.z === 0));

            // Programmatic activation avoids scrolling the mobile footer into view.
            await page.evaluate(() => document.querySelector('.pbr-motion').click());
            const paused = await page.evaluate(() => portfolioHero.state());
            await page.evaluate(top => scrollTo({ top, behavior: 'instant' }), layout.height * .2);
            await page.waitForTimeout(300);
            assert.equal(await page.evaluate(() => portfolioHero.state().frames), paused.frames);
            if (width === 1440) {
                for (const selector of ['.pbr-actions a[href="#portfolio"]', '.pbr-actions a[href="#blog"]', '.pbr-profile-link']) {
                    await page.evaluate(() => scrollTo({ top: 0, behavior: 'instant' }));
                    const link = page.locator(selector), hash = await link.getAttribute('href');
                    await link.focus(); await page.keyboard.press('Enter');
                    await page.waitForFunction(hash => location.hash === hash && document.body.dataset.chapter === hash.slice(1),hash);
                    await page.keyboard.press('Escape');
                    await page.waitForFunction(() => document.body.dataset.chapter === 'home');
                }
            }
            await page.emulateMedia({ reducedMotion: 'reduce' });
            await page.goto(base);
            await page.waitForFunction(() => window.portfolioHero?.state().staticPreferred);
            assert.equal(await page.evaluate(() => portfolioHero.state().frames), 0);
            assert(await page.locator('.pbr-poster img').evaluate(el => el.complete && el.naturalWidth > 0));
            assert(await page.locator('.pbr-motion').isHidden());
            assert.deepEqual(errors, []); assert.deepEqual(retiredRequests, []);
            console.log(`Hero ${width}px: layout, fixed refinement, idle, pause and reduced motion passed.`);
            await page.close();
        }
        for (const javascriptEnabled of [false, true]) {
            const page = await browser.newPage({ javaScriptEnabled: javascriptEnabled });
            if (javascriptEnabled) await page.route('**/assets/js/hero-sculpture.js*', route => route.abort());
            await page.goto(base);
            assert(await page.locator('.pbr-name').isVisible());
            assert.equal(await page.locator('.pbr-actions a').count(), 2);
            assert(await page.locator('.pbr-profile-link').isVisible());
            assert(await page.locator('.pbr-motion').isHidden());
            assert(await page.locator('.pbr-poster img').evaluate(el => el.complete && el.naturalWidth > 0));
            await page.close();
        }
        console.log('Hero static fallback passed with JavaScript disabled and with the renderer download blocked.');
    } finally {
        // Some system Chrome versions leave their driver pipe open after exit.
        // Bound process cleanup after all assertions, then close our HTTP server.
        await Promise.race([browser.close(), new Promise(resolve => setTimeout(resolve, 5000).unref())]);
        server.closeAllConnections();
        await new Promise(resolve => server.close(resolve));
    }
}
main().then(() => process.exit(0)).catch(error => { console.error(error); process.exit(1); });
