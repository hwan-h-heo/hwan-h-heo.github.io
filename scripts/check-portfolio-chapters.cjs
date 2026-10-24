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

        assert.equal(await page.evaluate(() => editionPreview.state().continuousScroll),false);
        assert.equal(await page.evaluate(() => getComputedStyle(document.documentElement).overflowY),'hidden');
        assert.equal(await page.locator('#home .pbr-foot').count(),1);
        assert.equal(await page.locator('#home .edition-cover-folio').textContent(),'00 — Cover');
        assert.equal(await page.locator('.edition-next').count(),3);
        assert.equal(await page.locator('.edition-colophon').count(),3);
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
        assert.deepEqual(errors,[]);
        console.log('Projects/Blog/About: up → no cue, down → next, native scroll preserved, hover wins; returning never replays refine.');

        const direct=await browser.newPage({viewport:{width:1440,height:1000}});
        await direct.goto(base+'#about');
        await direct.waitForFunction(()=>document.body.dataset.chapter==='about' && portfolioHero.state().deferred);
        assert.equal(await direct.locator('.pbr-scene canvas').count(),0);
        await direct.locator('.edition-nav a[href="#home"]').click();
        await direct.waitForFunction(()=>document.body.dataset.chapter==='home');
        await direct.waitForFunction(()=>portfolioHero.state().ready && portfolioHero.state().settled);
        assert.equal(await direct.evaluate(()=>portfolioHero.state().decoding.plays),0);
        await direct.close();

        const resizing=await browser.newPage({viewport:{width:1920,height:1080},reducedMotion:'reduce'});
        await resizing.goto(base+'#about');
        await resizing.waitForFunction(()=>window.editionPreview);
        await resizing.locator('.resume-disclosure').evaluate(node=>node.open=true);
        await resizing.waitForTimeout(100);
        await resizing.locator('#about').evaluate(node=>node.scrollTop=300);
        await resizing.waitForTimeout(250);
        await resizing.setViewportSize({width:900,height:900});
        await resizing.waitForFunction(()=>editionPreview.state().continuousScroll);
        await resizing.waitForTimeout(100);
        assert.equal(await resizing.evaluate(()=>document.body.dataset.chapter),'about');
        assert.equal(await resizing.locator('.resume-disclosure').evaluate(node=>node.open),true);
        const mobileOffset=await resizing.evaluate(()=>scrollY-editionPreview.state().chapters.find(range=>range.id==='about').start);
        assert(Math.abs(mobileOffset-300)<3);
        await resizing.setViewportSize({width:901,height:900});
        await resizing.waitForFunction(()=>!editionPreview.state().continuousScroll);
        assert.equal(await resizing.evaluate(()=>scrollY),0);
        assert(Math.abs(await resizing.locator('#about').evaluate(node=>node.scrollTop)-300)<3);
        assert.equal(await resizing.evaluate(()=>document.body.dataset.chapter),'about');
        await resizing.close();

        for (const [width,height] of [[320,740],[390,844],[844,390],[900,900]]) {
            const context=await browser.newContext({viewport:{width,height},isMobile:width<900,hasTouch:width<900});
            const mobile=await context.newPage();
            mobile.on('pageerror',e=>errors.push(e.message));
            await mobile.goto(base);
            await mobile.waitForFunction(()=>window.editionPreview);
            await mobile.evaluate(()=>document.fonts.ready);
            await mobile.waitForTimeout(100);
            const ranges=await mobile.evaluate(()=>editionPreview.state().chapters);
            if(width===390) {
                const client=await context.newCDPSession(mobile);
                for(let attempt=0; attempt<3 && await mobile.evaluate(()=>document.body.dataset.chapter==='home'); attempt++) {
                    await client.send('Input.dispatchTouchEvent',{type:'touchStart',touchPoints:[{x:190,y:640}]});
                    for(const y of [610,560,510,460,400,340,280,200]) {
                        await client.send('Input.dispatchTouchEvent',{type:'touchMove',touchPoints:[{x:190,y}]});
                        await mobile.waitForTimeout(35);
                    }
                    await client.send('Input.dispatchTouchEvent',{type:'touchEnd',touchPoints:[]});
                    await mobile.waitForTimeout(300);
                }
                await mobile.waitForFunction(()=>scrollY>100);
                await mobile.waitForFunction(()=>document.body.dataset.chapter==='portfolio');
                assert.equal(await mobile.evaluate(()=>editionPreview.state().navigationHint),false);
            }
            for (const range of ranges.slice(1)) {
                await mobile.evaluate(id=>scrollTo({top:editionPreview.state().chapters.find(range=>range.id===id).start,behavior:'instant'}),range.id);
                await mobile.waitForTimeout(100);
                assert.equal(await mobile.evaluate(()=>document.body.dataset.chapter),range.id);
                assert.equal(await mobile.evaluate(()=>document.documentElement.scrollWidth),width);
                assert.equal(await mobile.locator('#'+range.id).evaluate(section=>getComputedStyle(section).opacity),'1');
            }
            await mobile.screenshot({path:path.join(out,'continuous-about-'+width+'.png')});
            assert.equal(await mobile.locator('.edition-next').evaluateAll(nodes=>nodes.every(node=>getComputedStyle(node).display==='none')),true);
            assert.equal(await mobile.locator('.edition-colophon').evaluateAll(nodes=>nodes.filter(node=>getComputedStyle(node).display!=='none').length),1);
            await mobile.locator('.resume-disclosure').evaluate(node=>node.open=true);
            assert.equal(await mobile.locator('.resume-item h4').first().evaluate(node=>getComputedStyle(node).fontSize),'14px');
            assert.equal(await mobile.locator('.resume-entry-details li').first().evaluate(node=>getComputedStyle(node).fontSize),'13px');
            await mobile.emulateMedia({reducedMotion:'reduce'});
            await mobile.waitForTimeout(100);
            assert.equal(await mobile.evaluate(()=>document.body.classList.contains('edition-motion')),false);
            assert(await mobile.locator('.edition-chapter').evaluateAll(nodes=>nodes.every(node=>getComputedStyle(node).position==='relative' && getComputedStyle(node).opacity==='1')));
            await mobile.locator('.edition-nav a[href="#portfolio"]').click();
            await mobile.waitForFunction(()=>document.body.dataset.chapter==='portfolio');
            assert.equal(await mobile.locator('#edition-title-portfolio').evaluate(node=>node===document.activeElement),true);
            await mobile.emulateMedia({media:'print'});
            assert(await mobile.locator('.edition-chapter').evaluateAll(nodes=>nodes.every(node=>getComputedStyle(node).position==='relative' && getComputedStyle(node).opacity==='1')));
            assert.equal(await mobile.locator('.pbr-intro').evaluate(node=>getComputedStyle(node).opacity),'1');
            await context.close();
        }
        assert.deepEqual(errors,[]);
        console.log('PASS mobile touch, narrow/wide/landscape viewports, direct hashes and reduced motion.');
        fs.writeFileSync(path.join(out,'responsive-motion-verification.json'),JSON.stringify(report,null,2));
    }finally{server.server.close(); await Promise.race([browser.close(),new Promise(r=>setTimeout(r,5000).unref())]);}
})().then(()=>process.exit(0)).catch(e=>{console.error(e);process.exit(1);});
