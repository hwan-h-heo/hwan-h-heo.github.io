const {chromium} = require('playwright');
const fs = require('node:fs');
const path = require('node:path');
const {startStaticServer,getLaunchOptions} = require('./check-rendered-site');
const out = path.resolve(__dirname,'../_workspace/2026-10-08-portfolio-editions/integration-review');
const oldHtml = fs.readFileSync(path.resolve(__dirname,'../_workspace/2026-10-07-portfolio-design-review/tailnet-preview/portfolio-raster-v22.html'),'utf8');
(async () => {
    const server = await startStaticServer();
    const report = {device:'Apple M2 / 8 GB Mac',physicalPhone:false,notes:['v22 is the accepted frozen standalone; v32 uses the real built site. Network payload and document parsing are not directly comparable.','CPU throttle and touch viewport are browser emulation on this Mac, not phone measurements.'],runs:[]};
    try {
        for (const mode of ['headless','headed']) {
            const browser = await chromium.launch({...getLaunchOptions(),headless:mode==='headless',args:['--enable-precise-memory-info']});
            try {
                const system = await browser.newBrowserCDPSession();
                const info = await system.send('SystemInfo.getInfo');
                report[mode+'Gpu'] = {devices:info.gpu.devices,auxAttributes:info.gpu.auxAttributes,featureStatus:info.gpu.featureStatus};
                for (const version of ['v22','v32']) {
                    const page = await browser.newPage({viewport:{width:1440,height:1000}});
                    const session = await page.context().newCDPSession(page);
                    await session.send('Performance.enable');
                    await page.route('**/google-analytics.com/**',route=>route.abort());
                    await page.route('**/googletagmanager.com/**',route=>route.abort());
                    if (version==='v22') await page.route('**/baseline-v22.html',route=>route.fulfill({contentType:'text/html',body:oldHtml}));
                    await page.goto(server.baseUrl+(version==='v22'?'/baseline-v22.html':'/'));
                    if (version==='v22') await page.evaluate(()=>{window.portfolioHero=window.rasterStudy;});
                    await page.waitForFunction(()=>window.portfolioHero?.state().ready);
                    await page.waitForFunction(()=>portfolioHero.state().settled&&!portfolioHero.state().decoding.active,null,{timeout:30000});
                    const startState = await page.evaluate(()=>portfolioHero.state());
                    const gpu = await page.evaluate(()=>{
                        const gl=document.querySelector('.pbr-scene canvas').getContext('webgl2');
                        const ext=gl.getExtension('WEBGL_debug_renderer_info');
                        return {vendor:ext?gl.getParameter(ext.UNMASKED_VENDOR_WEBGL):gl.getParameter(gl.VENDOR),renderer:ext?gl.getParameter(ext.UNMASKED_RENDERER_WEBGL):gl.getParameter(gl.RENDERER)};
                    });
                    const before=(await session.send('Performance.getMetrics')).metrics;
                    await page.waitForTimeout(2000);
                    const after=(await session.send('Performance.getMetrics')).metrics;
                    const idleState=await page.evaluate(()=>portfolioHero.state());
                    const poses=[];
                    for(const [x,y] of [[1320,250],[110,690],[720,500]]){
                        const start=Date.now();await page.mouse.move(x,y);
                        await page.waitForFunction(([x,y])=>portfolioHero.state().settled&&Math.abs(portfolioHero.state().px-(x/1440*2-1))<.001&&Math.abs(portfolioHero.state().py-(1-y/1000*2))<.001,[x,y]);
                        poses.push({input:[x,y],settleMs:Date.now()-start,state:await page.evaluate(()=>({px:portfolioHero.state().px,py:portfolioHero.state().py,layout:portfolioHero.layout()}))});
                    }
                    const metrics=list=>Object.fromEntries(list.map(item=>[item.name,item.value]));
                    const a=metrics(before),b=metrics(after);
                    report.runs.push({mode,version,gpu,startState,idleFrames:idleState.frames-startState.frames,idlePending:idleState.framePending,idleTaskMs:(b.TaskDuration-a.TaskDuration)*1000,heapBytes:b.JSHeapUsedSize,poses});
                    await page.close();
                }
            } finally {await browser.close();}
        }
        const browser = await chromium.launch(getLaunchOptions());
        try {
            const page = await browser.newPage({viewport:{width:390,height:844},isMobile:true,hasTouch:true});
            const session = await page.context().newCDPSession(page);
            await session.send('Emulation.setCPUThrottlingRate',{rate:4});
            await page.goto(server.baseUrl+'/');
            await page.waitForFunction(()=>portfolioHero.state().ready);
            await page.waitForFunction(()=>portfolioHero.state().settled&&!portfolioHero.state().decoding.active,null,{timeout:45000});
            const state=await page.evaluate(()=>portfolioHero.state());await page.waitForTimeout(1000);
            report.cpuStress={viewport:[390,844],cpuRate:4,state,idleFrames:await page.evaluate(n=>portfolioHero.state().frames-n,state.frames)};
            await page.close();
        } finally {await browser.close();}
        for (const mode of ['headless','headed']) {
            const old=report.runs.find(run=>run.mode===mode&&run.version==='v22'),next=report.runs.find(run=>run.mode===mode&&run.version==='v32');
            const diff=old.poses.map((pose,i)=>Math.max(...['left','right','top','bottom'].map(key=>Math.abs(pose.state.layout[0][key]-next.poses[i].state.layout[0][key]))));
            report[mode+'ProjectedDifferences']=diff;
            if(diff.some(value=>value>.5)||next.idleFrames!==0||next.idlePending)throw new Error(`${mode}: pose or idle regression`);
        }
        fs.writeFileSync(path.join(out,'performance-verification.json'),JSON.stringify(report,null,2));
        console.log(JSON.stringify({runs:report.runs.map(run=>({mode:run.mode,version:run.version,gpu:run.gpu,firstFrameMs:run.startState.firstFrameMs,idleFrames:run.idleFrames,idleTaskMs:run.idleTaskMs,heapMB:run.heapBytes/1048576})),stress:report.cpuStress.state.performancePaused,headedProjectedDifferences:report.headedProjectedDifferences},null,2));
    } finally {server.server.close();}
})().catch(error=>{console.error(error);process.exit(1);});
