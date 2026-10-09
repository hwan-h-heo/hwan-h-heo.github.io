const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const {chromium} = require('playwright');
const {startStaticServer,getLaunchOptions} = require('./check-rendered-site');
const {loadSiteData} = require('../blogs/lib/site-data');
const {getPostRoute} = require('../blogs/lib/site-routes');
const {checkThemePreferences} = require('./check-theme-preferences.cjs');
const out = path.resolve(__dirname,'../_workspace/2026-10-08-portfolio-editions/integration-review');
fs.mkdirSync(out,{recursive:true});
(async () => {
    const server = await startStaticServer(), base = server.baseUrl;
    const browser = await chromium.launch(getLaunchOptions());
    const report = {viewports:[],flows:[],fallbacks:[],errors:[]};
    const data = loadSiteData();
    const sdf = getPostRoute(data.posts.find(post => post.id === '250823_sdf'),'eng');
    async function goto(page, route, selector = 'h1') {
        const response = await page.goto(base+route,{waitUntil:'domcontentloaded'});
        assert(response?.ok(), `${route} must return a successful response`);
        await page.waitForSelector(selector,{state:'attached'});
        assert((await page.locator(selector).first().textContent()).trim().length > 0, `${route} is missing page content`);
    }
    async function chapter(page,id) {
        const cover = await page.evaluate(() => document.body.dataset.chapter === 'home');
        const selector = cover && ['portfolio','blog','about'].includes(id) ? '.pbr-intro' : '.edition-nav';
        await page.locator(`${selector} a[href="#${id}"]`).click();
        await page.waitForFunction(id => document.body.dataset.chapter === id,id);
    }
    try {
        // Native touch, nested reading scroll, route/reload/history and every project template.
        for (const [width,height] of [[320,740],[390,844],[844,390],[1440,1000],[2560,1440]]) {
            const context = await browser.newContext({viewport:{width,height},isMobile:width<900,hasTouch:width<900,reducedMotion:'reduce'});
            const page = await context.newPage();
            page.on('pageerror', error => report.errors.push({width,error:error.message}));
            await goto(page,'/#portfolio','#edition-title-portfolio');
            await page.waitForFunction(() => document.body.dataset.chapter === 'portfolio');
            await page.waitForFunction(() => document.querySelectorAll('.portfolio-blog-preview-item').length > 0);
            assert.equal(await page.evaluate(() => document.documentElement.dataset.theme),'dark');
            assert.equal(await page.locator('.pbr-scene canvas').count(),0);
            await page.locator('[data-portfolio-view="all"] button,button[data-portfolio-view="all"]').first().click();
            assert.equal(await page.locator('.portfolio-project-item:not([hidden])').count(),data.portfolioProjects.length);
            const disclosure = page.locator('#portfolio details').first();
            await disclosure.evaluate(node => {node.open = true;});
            const links = page.locator('.portfolio-project-item:not([hidden]) .portfolio-project-title-link');
            const selectedLink = links.last(), href = await selectedLink.getAttribute('href');
            await selectedLink.scrollIntoViewIfNeeded();
            const before = await page.locator('#portfolio').evaluate(node => node.scrollTop);
            await selectedLink.click();
            await page.waitForURL(url => url.pathname.includes('/projects/'));
            assert(await page.locator('.case-opening h1').count());
            assert.equal(await page.evaluate(() => document.documentElement.dataset.theme),'dark');
            await page.goBack();
            await page.waitForFunction(() => document.body.dataset.chapter === 'portfolio');
            assert.equal(await page.locator('#portfolio').getAttribute('data-portfolio-view'),'all');
            assert(await disclosure.evaluate(node => node.open));
            assert(Math.abs(await page.locator('#portfolio').evaluate(node => node.scrollTop)-before) < 4,'Back must restore project reading position');
            await page.reload();
            await page.waitForFunction(() => document.body.dataset.chapter === 'portfolio');
            assert.equal(await page.locator('#portfolio').getAttribute('data-portfolio-view'),'all');
            assert(Math.abs(await page.locator('#portfolio').evaluate(node => node.scrollTop)-before) < 4,'Reload must restore All view and scroll');
            // Leave through the actual project link again so the return control
            // can restore that departure's focus as well as scroll and filter.
            await selectedLink.click();
            await page.waitForURL(url => url.pathname.includes('/projects/'));
            await page.locator('.case-breadcrumb').click();
            await page.waitForFunction(() => document.body.dataset.chapter === 'portfolio');
            assert.equal(await page.locator('#portfolio').getAttribute('data-portfolio-view'),'all');
            assert(Math.abs(await page.locator('#portfolio').evaluate(node => node.scrollTop)-before) < 4,'Projects return link must restore reading position');
            assert.equal(await page.evaluate(() => document.activeElement.getAttribute('href')),href,'Projects return restores the departure link focus');
            for (const id of ['blog','about','portfolio']) {
                await chapter(page,id);
                assert.equal(await page.evaluate(() => scrollY),0,'Chapter scrolling must not move the document');
                const box = await page.locator('#'+id).boundingBox();
                assert(box.x>=0 && box.x+box.width<=width+1);
                assert.equal(await page.locator('.edition-chapter:not(.is-active)').evaluateAll(nodes => nodes.every(node => node.inert)),true);
                await page.screenshot({path:path.join(out,`integrated-${id}-${width}.png`)});
            }
            await page.goBack(); await page.waitForFunction(() => document.body.dataset.chapter === 'about');
            await page.goForward(); await page.waitForFunction(() => document.body.dataset.chapter === 'portfolio');
            await page.keyboard.press('Escape'); await page.waitForFunction(() => document.body.dataset.chapter === 'home');
            const actionBox = await page.locator('.pbr-actions').boundingBox();
            assert(actionBox.x>=0 && actionBox.x+actionBox.width<=width+1);
            if (height < 560) await page.locator('.pbr-actions').scrollIntoViewIfNeeded();
            assert(await page.locator('.pbr-actions a').first().isVisible());
            await page.screenshot({path:path.join(out,`integrated-cover-${width}.png`)});
            for (const slug of fs.readdirSync(path.resolve(__dirname,'../projects')).filter(slug => fs.existsSync(path.resolve(__dirname,`../projects/${slug}/project.json`)))) {
                await goto(page,`/projects/${slug}/`);
                assert.equal(await page.locator('canvas').count(),0,'Project details must not create a background 3D scene');
                await page.evaluate(() => document.fonts.ready);
                assert.equal(await page.evaluate(() => document.documentElement.scrollWidth),width,slug+' horizontal overflow');
                const media = await page.locator('.case-body .project-media').evaluateAll(nodes => nodes.map(node => ({type:node.tagName,width:node.getBoundingClientRect().width,rail:node.closest('.case-body').getBoundingClientRect().width})));
                assert(media.every(item => item.width<=item.rail+1),slug+' media escapes reading column');
            }
            await goto(page,sdf);
            assert.equal(await page.evaluate(() => document.documentElement.dataset.theme),'light');
            await page.screenshot({path:path.join(out,`integrated-post-${width}.png`)});
            await goto(page,'/blogs/');
            assert.equal(await page.evaluate(() => document.documentElement.scrollWidth),width);
            await page.screenshot({path:path.join(out,`integrated-blog-home-${width}.png`)});
            report.viewports.push({width,height,projects:7,nativeScroll:true,directReloadHistory:true,filterDisclosuresFocus:true,noOverflow:true});
            await context.close();
        }
        report.flows.push(await checkThemePreferences(browser,base));
        // Reduced-motion static entry can become live, always finished, without focus changes.
        const live = await browser.newContext({viewport:{width:1440,height:1000},reducedMotion:'reduce'}), livePage = await live.newPage();
        await goto(livePage,'/#about','#edition-title-about');
        assert.equal(await livePage.locator('.pbr-scene canvas').count(),0);
        await livePage.emulateMedia({reducedMotion:'no-preference'});
        assert.equal(await livePage.locator('.pbr-scene canvas').count(),0);
        await chapter(livePage,'home');
        await livePage.waitForFunction(() => window.portfolioHero.state().ready && window.portfolioHero.state().settled);
        assert.equal(await livePage.evaluate(() => portfolioHero.state().decoding.plays),0);
        assert.equal(await livePage.evaluate(() => portfolioHero.state().decoding.progress),1);
        await chapter(livePage,'portfolio');
        const resting = await livePage.evaluate(() => portfolioHero.state().frames);
        await livePage.waitForTimeout(700);
        assert.equal(await livePage.evaluate(() => portfolioHero.state().frames),resting);
        assert.equal(await livePage.evaluate(() => portfolioHero.state().framePending),false);
        await chapter(livePage,'home');
        await livePage.waitForFunction(() => portfolioHero.state().settled);
        await livePage.locator('.pbr-scene canvas').evaluate(canvas => canvas.getContext('webgl2').getExtension('WEBGL_lose_context').loseContext());
        await livePage.waitForFunction(() => portfolioHero.state().failed);
        assert(await livePage.locator('.pbr-hero').evaluate(node => node.classList.contains('pbr-static')));
        assert.equal(await livePage.evaluate(() => portfolioHero.state().framePending),false);
        report.fallbacks.push({reducedToLiveFinished:true,readingNoFrames:true,contextLossStatic:true});
        await live.close();
        // Delayed bundle response must not create WebGL after leaving the cover.
        const delayed = await browser.newContext({viewport:{width:1440,height:1000}}), delayedPage = await delayed.newPage();
        let release;
        const gate = new Promise(resolve => release=resolve);
        await delayedPage.route('**/assets/js/hero-sculpture.js*', async route => {await gate;await route.continue();});
        await goto(delayedPage,'/');
        await delayedPage.locator('.pbr-actions a[href="#portfolio"]').click();
        await delayedPage.waitForFunction(() => document.body.dataset.chapter==='portfolio');
        release(); await delayedPage.waitForFunction(() => Boolean(window.initializePortfolioHero));
        assert.equal(await delayedPage.locator('.pbr-scene canvas').count(),0);
        await chapter(delayedPage,'home'); await delayedPage.waitForFunction(() => portfolioHero.state().ready && portfolioHero.state().settled);
        assert.equal(await delayedPage.evaluate(() => portfolioHero.state().decoding.plays),0);
        report.fallbacks.push({bundleNavigationRace:true});
        await delayed.close();
        // Save-data and lost-context fallbacks retain readable content.
        const saver = await browser.newContext({viewport:{width:390,height:844}});
        await saver.addInitScript(() => Object.defineProperty(navigator,'connection',{value:{saveData:true},configurable:true}));
        const saverPage = await saver.newPage();await goto(saverPage,'/');
        assert.equal(await saverPage.locator('.pbr-scene canvas').count(),0);
        await saverPage.locator('.pbr-actions a[href="#portfolio"]').click(); await saverPage.waitForFunction(() => document.body.dataset.chapter==='portfolio');
        assert(await saverPage.locator('#portfolio h2').isVisible());
        report.fallbacks.push({saveDataStatic:true});await saver.close();
        assert.deepEqual(report.errors,[]);
        fs.writeFileSync(path.join(out,'integration-verification.json'),JSON.stringify(report,null,2));
        console.log('Integration passed: 5 viewports, 7 project templates, direct/reload/history/focus, All/disclosures, theme migration/new tabs, deferred/reduced/data-saving motion and loading race.');
    } finally {await browser.close();server.server.close();}
})().catch(error => {console.error(error);process.exit(1);});
