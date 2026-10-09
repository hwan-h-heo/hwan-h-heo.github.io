const {chromium} = require('playwright');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const {startStaticServer,getLaunchOptions} = require('./check-rendered-site');
const out = path.resolve(__dirname,'../_workspace/2026-10-08-portfolio-editions/integration-review');
fs.mkdirSync(out,{recursive:true});
(async()=>{
    const server = await startStaticServer(), base = server.baseUrl+'/';
    const browser=await chromium.launch(getLaunchOptions());
    const report={};
    try {
        const page=await browser.newPage({viewport:{width:1440,height:1000}});
        const errors=[];
        page.on('pageerror',e=>errors.push(e.message));
        page.on('console',m=>{if(m.type()==='error' && /shader|WebGL|GL_INVALID/.test(m.text()))errors.push(m.text());});
        await page.goto(base);
        await page.waitForFunction(()=>portfolioHero.state().decoding?.phase==='intro');
        await page.waitForFunction(()=>portfolioHero.state().decoding?.progress>.3);
        report.intro=await page.evaluate(()=>portfolioHero.state()); assert.equal(report.intro.decoding.plays,1);
        await page.waitForFunction(()=>portfolioHero.state().decoding?.phase==='static' && portfolioHero.state().settled);
        report.finished=await page.evaluate(()=>portfolioHero.state());
        await page.waitForTimeout(1000);
        assert.equal((await page.evaluate(()=>portfolioHero.state())).frames,report.finished.frames);
        assert(!report.finished.framePending && report.finished.decoding.progress===1);
        await page.screenshot({path:path.join(out,'v32-cover-rest.png')});
        await page.mouse.move(1320,250);
        await page.waitForFunction(()=>portfolioHero.state().settled && portfolioHero.state().px>.8);
        report.right=await page.evaluate(()=>portfolioHero.state());
        assert.deepEqual(report.right.tilts,[{x:0,y:0,z:0}]);
        assert.equal(report.right.cameraOffset.x,report.right.px*.22);
        assert.equal(report.right.cameraOffset.y,report.right.py*.12);
        assert.equal(report.right.work.geometryUpdates,report.finished.work.geometryUpdates);
        assert.equal(report.right.work.shadowUpdates,report.finished.work.shadowUpdates);
        assert.equal(report.right.work.stageBakes,report.finished.work.stageBakes);
        await page.screenshot({path:path.join(out,'v32-cover-right.png')});
        await page.mouse.move(110,690);
        await page.waitForFunction(()=>portfolioHero.state().settled && portfolioHero.state().px<-.8);
        report.left=await page.evaluate(()=>portfolioHero.state());
        assert.deepEqual(report.left.tilts,[{x:0,y:0,z:0}]);
        assert(report.left.cameraOffset.x<-.17 && report.left.cameraOffset.y<-.04);
        assert.equal(report.left.decoding.progress,1); assert.equal(report.left.decoding.plays,1);
        await page.screenshot({path:path.join(out,'v32-cover-left.png')});
        await page.waitForTimeout(700);
        assert.equal((await page.evaluate(()=>portfolioHero.state())).frames,report.left.frames);
        assert(!(await page.evaluate(()=>portfolioHero.state().framePending)));
        console.log('Entrance plays once; pointer moves the camera with original offsets and fixed geometry/shadows; rendering stops at rest.');

        await page.locator('.pbr-motion').click(); await page.mouse.move(700,500);
        const paused=await page.evaluate(()=>portfolioHero.state());
        // Sample actual pseudo-element geometry through the transition, with 3D
        // Motion off. This catches instantaneous flashes and wrong origins.
        report.outline=await page.evaluate(()=>new Promise(resolve=>{
            const link=document.querySelector('.pbr-actions a[href="#portfolio"]');
            getComputedStyle(link,'::after').transform;
            const start=performance.now(), samples=[];
            dispatchEvent(new WheelEvent('wheel',{deltaY:80}));
            function sample(){
                const style=getComputedStyle(link,'::after'), t=performance.now()-start;
                samples.push({t,scale:new DOMMatrix(style.transform).a,origin:style.transformOrigin,color:getComputedStyle(link).color,duration:style.transitionDuration});
                if(t<650) requestAnimationFrame(sample); else resolve(samples);
            }
            sample();
        }));
        assert(report.outline.some(s=>s.scale>.05 && s.scale<.95),'Underline must visibly sweep instead of appearing all at once');
        assert(report.outline.every(s=>s.origin.startsWith('0px')));
        assert.equal(report.outline.at(-1).scale,1);
        assert.equal(report.outline.at(-1).color,'rgb(190, 210, 211)');
        assert.equal(report.outline[0].color,'rgb(229, 229, 223)','Text color follows the line after a short delay');
        assert.equal((await page.evaluate(()=>portfolioHero.state())).frames,paused.frames);
        await page.screenshot({path:path.join(out,'v32-cover-hint.png')});
        await page.locator('.pbr-actions a[href="#blog"]').hover(); await page.waitForTimeout(350);
        assert(!(await page.evaluate(()=>editionPreview.state().navigationHint)));
        assert.equal(await page.locator('.pbr-actions a[href="#portfolio"]').evaluate(a=>getComputedStyle(a,'::after').transform),'matrix(0, 0, 0, 1, 0, 0)');
        await page.mouse.wheel(0,100);
        assert(!(await page.evaluate(()=>editionPreview.state().navigationHint)));
        console.log('Cover cue sweeps left-to-right in Motion off, with delayed subtle color and exclusive hover priority.');

        await page.locator('.pbr-actions a[href="#blog"]').click();
        await page.waitForFunction(()=>document.body.dataset.chapter==='blog'); await page.waitForTimeout(700);
        await page.mouse.move(700,500);
        for(const id of ['blog','portfolio','about']) {
            if(id!=='blog'){await page.locator(`.edition-nav a[href="#${id}"]`).click();await page.waitForFunction(id=>document.body.dataset.chapter===id,id);await page.waitForTimeout(650);}
            await page.locator('#'+id).evaluate(section=>section.scrollTop=section.scrollHeight);
            await page.mouse.move(700,500); await page.waitForTimeout(200);
            const top=await page.locator('#'+id).evaluate(section=>section.scrollTop);
            await page.mouse.wheel(0,-60);
            await page.waitForTimeout(550);
            assert.equal(await page.evaluate(()=>editionPreview.state().chapterHint),null);
            assert.equal(await page.evaluate(()=>editionPreview.state().chapter),id);
            assert((await page.locator('#'+id).evaluate(section=>section.scrollTop))<top,'Upward input must still scroll the chapter');
            assert.equal(await page.locator('#'+id+' .edition-next a.is-navigation-hint').count(),0);
            await page.screenshot({path:path.join(out,`v32-${id}-previous.png`)});
            await page.mouse.wheel(0,90);
            await page.waitForFunction(()=>editionPreview.state().chapterHint==='next'); await page.waitForTimeout(550);
            assert.equal(await page.locator('#'+id+' .edition-next a.is-navigation-hint').count(),1);
            assert.equal(await page.locator('#'+id+' .edition-next a.is-navigation-hint').getAttribute('rel'),'next');
            assert.equal(await page.locator('#'+id+' .edition-next a[rel="prev"]').evaluate(a=>getComputedStyle(a,'::after').transform),'matrix(0, 0, 0, 1, 0, 0)');
            await page.screenshot({path:path.join(out,`v32-${id}-next.png`)});
            await page.locator('#'+id+' .edition-next a[rel="prev"]').hover();
            assert.equal(await page.evaluate(()=>editionPreview.state().chapterHint),null);
            await page.mouse.wheel(0,90);
            assert.equal(await page.evaluate(()=>editionPreview.state().chapterHint),null,'Direct hover outranks a scroll cue');
        }
        await page.keyboard.press('Escape'); await page.waitForFunction(()=>document.body.dataset.chapter==='home');
        assert((await page.evaluate(()=>portfolioHero.state())).paused);
        await page.locator('.pbr-motion').click(); await page.mouse.move(720,500);
        await page.waitForFunction(()=>portfolioHero.state().settled);
        assert.equal(await page.evaluate(()=>portfolioHero.state().decoding.plays),1);
        assert.equal(await page.evaluate(()=>portfolioHero.state().decoding.progress),1);
        await page.emulateMedia({reducedMotion:'reduce'}); await page.mouse.move(1300,300);
        assert((await page.evaluate(()=>portfolioHero.state())).paused);
        assert.deepEqual(errors,[]); await page.close();
        console.log('Projects/Blogs/About: up → no cue, down → next, native scroll preserved, hover wins; returning never replays refine.');

        const direct=await browser.newPage({viewport:{width:1440,height:1000}});
        await direct.goto(base+'#about'); await direct.waitForFunction(()=>document.body.dataset.chapter==='about' && portfolioHero.state().deferred);
        assert.equal(await direct.locator('.pbr-scene canvas').count(),0);
        assert.equal(await direct.evaluate(()=>performance.getEntriesByType('resource').some(item=>item.name.includes('/assets/js/hero-sculpture.js'))),false);
        assert.equal(await direct.evaluate(()=>portfolioHero.state().decoding.phase),'static');
        await direct.locator('#about .edition-next a[rel="next"]').click(); await direct.waitForFunction(()=>document.body.dataset.chapter==='home');
        await direct.mouse.move(1200,250); await direct.waitForFunction(()=>portfolioHero.state().settled && portfolioHero.state().px>.5);
        assert.equal(await direct.evaluate(()=>portfolioHero.state().decoding.plays),0);
        await direct.close();

        const mobile=await browser.newPage({viewport:{width:390,height:844},isMobile:true,hasTouch:true});
        await mobile.goto(base); await mobile.waitForFunction(()=>portfolioHero.state().settled && portfolioHero.state().decoding?.phase==='static');
        const resting=await mobile.evaluate(()=>portfolioHero.state());
        await mobile.waitForTimeout(500); assert.equal((await mobile.evaluate(()=>portfolioHero.state())).frames,resting.frames);
        await mobile.screenshot({path:path.join(out,'v32-mobile-rest.png')});
        await mobile.locator('.pbr-actions a[href="#blog"]').click(); await mobile.waitForFunction(()=>document.body.dataset.chapter==='blog'); await mobile.waitForTimeout(700);
        const client=await mobile.context().newCDPSession(mobile);
        await client.send('Input.dispatchTouchEvent',{type:'touchStart',touchPoints:[{x:190,y:650}]});
        await client.send('Input.dispatchTouchEvent',{type:'touchMove',touchPoints:[{x:192,y:530}]});
        await client.send('Input.dispatchTouchEvent',{type:'touchEnd',touchPoints:[]});
        await mobile.waitForFunction(()=>editionPreview.state().chapterHint==='next');
        await mobile.waitForTimeout(500);
        await client.send('Input.dispatchTouchEvent',{type:'touchStart',touchPoints:[{x:190,y:450}]});
        await client.send('Input.dispatchTouchEvent',{type:'touchMove',touchPoints:[{x:192,y:570}]});
        await client.send('Input.dispatchTouchEvent',{type:'touchEnd',touchPoints:[]});
        await mobile.waitForFunction(()=>editionPreview.state().chapterHint===null);
        assert.equal(await mobile.evaluate(()=>portfolioHero.state().work.pointerUpdates),resting.work.pointerUpdates);
        await mobile.close();
        console.log('Direct chapter return has no intro; mobile stops after its entrance and uses directional touch cues without camera parallax.');
        fs.writeFileSync(path.join(out,'v32-motion-verification.json'),JSON.stringify(report,null,2));
    }finally{server.server.close(); await Promise.race([browser.close(),new Promise(r=>setTimeout(r,5000).unref())]);}
})().then(()=>process.exit(0)).catch(e=>{console.error(e);process.exit(1)});
