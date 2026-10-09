import * as THREE from 'three';
import { RectAreaLightUniformsLib } from 'three/addons/lights/RectAreaLightUniformsLib.js';
import { createCachedStage, movingStoneMaterial } from './hero-sculpture/cached-stage.js';
import { createMeshRefinement } from './hero-sculpture/intro-refinement.js';
import { makeStoneTexture, stoneMaterial } from './hero-sculpture/stone-material.js';
import { contactShadow } from './hero-sculpture/contact-shadow.js';
import { carvedGeometry, apertureGeometry, stoneBlock } from './hero-sculpture/stone-geometry.js';

// The static build already includes the cover. The core-ready event also
// supports a source checkout whose content is mounted after DOMContentLoaded.
function initializeHero() {
    const cover = document.querySelector('.pbr-hero');
    if (!cover?.querySelector('.pbr-scene') || cover.dataset.rendererStarted || ['portfolio','blog','about'].includes(location.hash.slice(1))) return;
    cover.dataset.rendererStarted = 'true';
    startHero();
}
function startHero() {
    const cover = document.querySelector('.pbr-hero'), host = cover.querySelector('.pbr-scene');
    const reduced = matchMedia('(prefers-reduced-motion: reduce)'), fine = matchMedia('(pointer: fine)');
    const capture = new URLSearchParams(location.search).has('capture');
    const staticPreferred = !capture && (reduced.matches || navigator.connection?.saveData);
    const motion = cover.querySelector('.pbr-motion');
    motion.hidden = false;
    let refinement;
    let renderer, scene, camera, sculptures, base, look, frame = 0, wakeTimer = 0;
    let visible = true, paused = reduced.matches, ready = false, failed = false, settled = false;
    let chapterActive = !['portfolio', 'blog', 'about'].includes(location.hash.slice(1));
    let manualPaused = false, lastDraw = -Infinity;
    try { manualPaused = localStorage.getItem('portfolio-motion') === 'off'; } catch {}
    paused = manualPaused || reduced.matches;
    const FRAME_MS = 1000 / 30;
    let scheduledInterval = FRAME_MS, viewDirty = true, stageDirty = true;
    let cachedStage, stillMaterial, movingMaterial, staticParts = [], supportTable;
    let movingSamples = 0, movingDuration = 0, slowSamples = 0, previousMoved = false, performancePaused = false;
    let viewWidth = 0, viewHeight = 0, introResolved = false;
    const work = {shadowUpdates:0,groundContactUpdates:0,formContactUpdates:0,geometryUpdates:0,stageBakes:0,fullSceneFrames:0,compositeFrames:0,pointerUpdates:0};
    let reducedApplied = reduced.matches;
    let tx = 0, ty = 0, px = 0, py = 0, frames = 0, triangleCount = 0, firstFrameMs = 0;
    const geometryCounts = {}, started = performance.now();
    let scrollTarget = 0, scrollProgress = 0, supportDirty = true;
    const supportedForms = [], contacts = [], poseMatrix = new THREE.Matrix4(), point = new THREE.Vector3();
    const supports = [{ x: 2.25, z: .15, halfX: .84, halfZ: .66, height: .86 }];
    const actionForms = [], actionWeights = [0, 0];
    const stageParts = {};

    function syncMotion() {
        motion.textContent = paused ? 'Motion off' : 'Motion on';
        motion.setAttribute('aria-pressed', String(!paused));
        motion.disabled = reduced.matches;
    }
    syncMotion();

    function add(geometry, material, name, x, y, z) {
        const mesh = new THREE.Mesh(geometry, material);
        mesh.name = name; mesh.position.set(x, y, z);
        mesh.castShadow = mesh.receiveShadow = true;
        sculptures.add(mesh);
        const count = geometry.index ? geometry.index.count / 3 : geometry.attributes.position.count / 3;
        triangleCount += count; geometryCounts[name] = count;
        return mesh;
    }

    // Linear HDR studio illumination, convolved once for the material roughness levels.
    // Panels only appear in reflections; they do not become visible background objects.
    function studioEnvironment() {
        const studio = new THREE.Scene();
        studio.background = new THREE.Color(.014, .014, .014);
        function panel(color, strength, w, h, position) {
            const mesh = new THREE.Mesh(new THREE.PlaneGeometry(w, h), new THREE.MeshBasicMaterial({
                color: new THREE.Color(color).multiplyScalar(strength), side: THREE.DoubleSide,
            }));
            mesh.position.set(...position); mesh.lookAt(1.5, 1.5, 0); studio.add(mesh);
        }
        panel('#ffffff', 1.6, 3, 4, [-3, 7, 4]);
        panel('#ffffff', .3, 6, 4, [3, 2, 7]);
        const pmrem = new THREE.PMREMGenerator(renderer);
        const target = pmrem.fromScene(studio, .02, .1, 30, { size: 256 });
        scene.environment = target.texture;
        scene.environmentIntensity = .20;
        studio.traverse(o => { o.geometry?.dispose(); o.material?.dispose(); });
        pmrem.dispose();
    }

    function area(color, intensity, w, h, position, target) {
        const light = new THREE.RectAreaLight(color, intensity, w, h);
        light.position.set(...position); light.lookAt(...target); scene.add(light);
    }

    function buildScene() {
        scene = new THREE.Scene(); scene.background = new THREE.Color('#111111');
        scene.fog = new THREE.Fog('#111111', 24, 60);
        studioEnvironment();
        sculptures = new THREE.Group(); scene.add(sculptures);
        const texture = makeStoneTexture(renderer);
        const mineral = stillMaterial = stoneMaterial(texture, { color: '#c2c2c2', scale: .68, relief: .012 });
        movingMaterial = movingStoneMaterial(texture);
        const wall = stoneMaterial(texture, { color: '#aaaaaa', scale: .30, relief: .018 });
        const ground = stoneMaterial(texture, { color: '#999999', scale: .32, relief: .012 });
        const main = add(carvedGeometry(), mineral, 'carved-form', 2.25, .87, .15);
        main.rotation.set(-.08, -.30, -.10);
        refinement = createMeshRefinement(main);
        refinement.applyMaterial(movingMaterial); stageParts.main = main;
        stageParts.mainPlinth = add(stoneBlock(1.62, .86, 1.28), wall, 'stone-block', 2.25, 0, .15);
        const aperture = add(apertureGeometry(), stoneMaterial(texture, { color: '#aaaaaa', scale: .30, relief: .018 }), 'aperture', 2.7, 2.95, -2.4);
        aperture.rotation.y = -.10;
        stageParts.aperture = aperture;
        const back = add(new THREE.PlaneGeometry(70, 30), stoneMaterial(texture, { color: '#353535', scale: .22, relief: .014 }), 'rear-wall', 0, 9, -7);
        back.castShadow = false;
        const floor = add(new THREE.PlaneGeometry(200, 200), ground, 'floor', 0, -.025, 0);
        floor.rotation.x = -Math.PI / 2; floor.castShadow = false;
        // A low foreground slab catches the raking light and anchors the scene's depth.
        stageParts.foreground = add(stoneBlock(12, .12, 2.2), ground, 'foreground-slab', 1.5, -.12, 4.5);
        // The floor's contact comes from the stationary architecture/plinth.
        // The form's separate contact is on top of the plinth, not the floor.
        const objects = sculptures.children.filter(o => o !== main && o !== floor && o !== back);
        contacts.push(
            contactShadow(renderer, objects, { x: 2.2, y: -.024, z: .15, size: 14, height: 1.4, opacity: .6 }),
            contactShadow(renderer, [main], { x: 2.25, y: .862, z: .15, size: 2.8, height: .8, opacity: .7 }),
        );
        sculptures.add(...contacts); triangleCount += 4; geometryCounts['contact-planes'] = 4;
        const pivot = new THREE.Group(); pivot.position.set(main.position.x, .87, main.position.z);
        main.position.sub(pivot.position); sculptures.add(pivot); pivot.add(main);
        supportedForms.push({ mesh: main, pivot, support: .87 });
        actionForms.push({ mesh: main, color: mineral.color.clone() },
            { mesh: aperture, color: aperture.material.color.clone(), background: true });
        area('#ffffff', .65, 4, 3, [0, 5.5, 5], [2.2, 2, -1]);
        area('#ffffff', .16, 7, 4, [2, 2, 8], [2, 2, 0]);
        const key = new THREE.SpotLight('#ffffff', 210, 35, .65, .65, 2);
        key.position.set(-1.8, 7.4, 5); key.target.position.set(2.5, 1.5, -1.5);
        key.castShadow = true; key.shadow.mapSize.setScalar(1024);
        key.shadow.camera.near = .5; key.shadow.camera.far = 28;
        key.shadow.bias = -.00012; key.shadow.normalBias = .016;
        key.shadow.radius = 5; key.shadow.blurSamples = 8;
        scene.add(key, key.target);
        scene.add(new THREE.HemisphereLight('#ffffff', '#242424', .12));
        camera = new THREE.PerspectiveCamera(36, 1, .1, 100);
        staticParts = sculptures.children.filter(object => object !== pivot);
        cachedStage = createCachedStage(renderer);
        // Fixed camera/orientation permits a one-time support lookup. Animation
        // never scans 4,825 vertices or recalculates normals on the main thread.
        pivot.updateMatrix(); main.updateMatrix();
        poseMatrix.multiplyMatrices(pivot.matrix, main.matrix);
        supportTable = new Float32Array(129);
        for (let sample = 0; sample < supportTable.length; sample++) {
            let correction = -Infinity;
            for (let i = 0; i < main.geometry.attributes.position.count; i++) {
                refinement.samplePosition(i, sample / 128, point).applyMatrix4(poseMatrix);
                let height = -.025;
                for (const block of supports) if (Math.abs(point.x - block.x) <= block.halfX && Math.abs(point.z - block.z) <= block.halfZ) height = Math.max(height, block.height);
                correction = Math.max(correction, height + .004 - point.y);
            }
            supportTable[sample] = correction;
        }
    }

    function positionCamera() {
        camera.position.copy(base); camera.position.x += px * .22; camera.position.y += py * .12;
        camera.lookAt(look); camera.updateMatrixWorld();
    }
    function readScroll() {
        // The cover is fixed. Wheel/touch gestures only reveal the
        // existing chapter link; refinement runs only on initial load.
        scrollTarget = scrollProgress = 0;
    }
    function fitSurfaceToSupport(geometryChanged = false, progress = refinement.state().surfaceProgress) {
        if (!supportDirty && !geometryChanged) return;
        const value = progress * 128, index = Math.min(127, Math.floor(value));
        supportedForms[0].pivot.position.y = supportedForms[0].support +
            THREE.MathUtils.lerp(supportTable[index], supportTable[index + 1], value - index);
        supportDirty = false;
    }
    function bakeStage() {
        const sampled = refinement.state().surfaceProgress;
        refinement.sampleForCache(1); fitSurfaceToSupport(true, 1);
        stageParts.main.material = stillMaterial;
        stageParts.main.receiveShadow = true;
        contacts[0].userData.update(); contacts[1].userData.update();
        work.groundContactUpdates++; work.formContactUpdates++; work.shadowUpdates++;
        cachedStage.bake(scene, camera, () => {renderer.shadowMap.needsUpdate = true;}, hidden => {stageParts.main.visible = !hidden;});
        work.stageBakes++; work.fullSceneFrames += 2;
        refinement.sampleForCache(sampled); fitSurfaceToSupport(true, sampled);
        stageDirty = false;
    }
    function resize() {
        if (!renderer || !camera) return;
        const w = cover.clientWidth, h = cover.clientHeight, mobile = w <= 900;
        if (w === viewWidth && h === viewHeight) return;
        viewWidth = w; viewHeight = h;
        const ratio = capture ? 1 : Math.min(devicePixelRatio, w <= 600 ? 1 : 1.25, Math.sqrt(1800000 / (w * h)));
        renderer.setPixelRatio(ratio);
        renderer.setSize(w, h); camera.aspect = w / h; camera.fov = mobile ? 42 : 36;
        camera.updateProjectionMatrix();
        base = mobile ? new THREE.Vector3(4.2, 4.8, 13.5) : new THREE.Vector3(4.0, 3.7, 14);
        look = mobile ? new THREE.Vector3(2.1, 4.25, 0) : new THREE.Vector3(.20, 2.15, -.4);
        sculptures.position.set(0, 0, 0);
        sculptures.scale.setScalar(mobile ? .76 : 1);
        stageParts.aperture.position.y = mobile ? 2.30 : 2.95;
        supportDirty = true;
        viewDirty = stageDirty = true; positionCamera(); readScroll();
        if (paused || !chapterActive) {
            if (chapterActive) { fitSurfaceToSupport(true); render(); viewDirty = false; }
        } else { settled = false; request(); }
    }
    function render() {
        if (stageDirty) bakeStage();
        const highQuality = paused || capture || !refinement.active;
        stageParts.main.material = highQuality ? stillMaterial : movingMaterial;
        stageParts.main.receiveShadow = highQuality;
        if (highQuality) {
            // One still render restores the full material/environment/self-shadow
            // path; there are no subsequent frames while paused.
            renderer.render(scene, camera); work.fullSceneFrames++;
        } else {
            for (const object of staticParts) object.visible = false;
            const environment = scene.environment; scene.environment = null;
            cachedStage.render(scene, camera); scene.environment = environment;
            for (const object of staticParts) object.visible = true;
            work.compositeFrames++;
        }
        frames++;
        if (!introResolved && refinement?.state().phase !== 'intro') {
            introResolved = true; cover.classList.remove('pbr-entering');
        }
        if (!firstFrameMs) {
            firstFrameMs = performance.now() - started;
            host.classList.add('is-ready'); cover.querySelector('.pbr-load')?.setAttribute('hidden', '');
        }
    }
    function draw(now) {
        frame = 0;
        if (!chapterActive || !visible || document.hidden || failed) return;
        // A media change can become observable before its change event is delivered.
        // Resolve it in the active frame as well, rather than leaving a coarse form frozen.
        if (reduced.matches !== reducedApplied) { syncReducedMotion(); return; }
        // Keep the entrance bounded at 30 fps. Pointer response follows the
        // production RAF cadence and .19 damping; stop all callbacks at rest.
        if (refinement.active && now - lastDraw < FRAME_MS - .5) { request(); return; }
        const frameDuration = now - lastDraw;
        lastDraw = now;
        const geometryChanged = !paused && !reduced.matches && (refinement?.update(now, scrollProgress) || false);
        if (geometryChanged) work.geometryUpdates++;
        const previousX = px, previousY = py;
        if (!refinement.active && !paused && !reduced.matches) {
            // Exact production pointer damping and camera offsets (js/hero.js).
            px += (tx - px) * .19; py += (ty - py) * .19;
            if (Math.abs(tx - px) + Math.abs(ty - py) < .002) {px = tx; py = ty;}
        }
        const pointerChanged = px !== previousX || py !== previousY;
        const pointerSettled = px === tx && py === ty;
        settled = paused || reduced.matches || (!refinement.active && pointerSettled);
        fitSurfaceToSupport(geometryChanged);
        if (pointerChanged) {positionCamera(); work.pointerUpdates++;}
        if (viewDirty || geometryChanged || pointerChanged) { render(); viewDirty = false; }
        // A sustained inability to reach even ~18 fps resolves to a finished,
        // still cover. A single delayed frame or endpoint hold cannot trip it.
        if (geometryChanged && previousMoved && !paused) {
            movingSamples++; movingDuration += frameDuration; if (frameDuration > 55) slowSamples++;
            if (movingSamples >= 48) {
                const slow = movingDuration / movingSamples > 55 && slowSamples >= 24;
                movingSamples = movingDuration = slowSamples = 0;
                if (slow) {
                    performancePaused = manualPaused = paused = true;
                    refinement.finish(); fitSurfaceToSupport(true); positionCamera(); render();
                    settled = true; syncMotion(); cancelScheduled();
                }
            }
        }
        previousMoved = geometryChanged;
        if (!settled) request();
    }
    function cancelScheduled() {
        cancelAnimationFrame(frame); clearTimeout(wakeTimer); frame = wakeTimer = 0;
    }
    function request(interval = refinement?.active ? FRAME_MS : 0) {
        if (staticPreferred || !chapterActive || frame || !visible || document.hidden || failed) return;
        if (wakeTimer) {
            if (interval >= scheduledInterval) return;
            clearTimeout(wakeTimer); wakeTimer = 0;
        }
        scheduledInterval = interval;
        // Sleep between display frames instead of waking on every 60/120 Hz RAF.
        // A small lead leaves time for the browser to align with the target paint.
        const delay = Math.max(0, lastDraw + interval - performance.now() - 8);
        if (delay > 1) wakeTimer = setTimeout(() => {
            wakeTimer = 0;
            if (chapterActive && visible && !document.hidden && !failed) frame = requestAnimationFrame(draw);
        }, delay);
        else frame = requestAnimationFrame(draw);
    }
    function fail(error) {
        failed = true; cancelScheduled();
        cover.classList.add('pbr-static'); cover.classList.remove('pbr-entering'); motion.hidden = true;
        cover.querySelector('.pbr-load')?.setAttribute('hidden', '');
        console.warn('Using the static portfolio cover:', error?.message || error);
    }
    if (staticPreferred) {
        settled = true; cover.classList.add('pbr-static'); motion.hidden = true;
        cover.querySelector('.pbr-load')?.setAttribute('hidden', '');
    } else try {
        RectAreaLightUniformsLib.init();
        THREE.ColorManagement.enabled = true;
        renderer = new THREE.WebGLRenderer({ antialias: true, powerPreference: 'low-power' });
        renderer.outputColorSpace = THREE.SRGBColorSpace;
        renderer.toneMapping = THREE.AgXToneMapping; renderer.toneMappingExposure = .95;
        renderer.shadowMap.enabled = true; renderer.shadowMap.type = THREE.VSMShadowMap;
        renderer.shadowMap.autoUpdate = false;
        host.appendChild(renderer.domElement);
        renderer.domElement.addEventListener('webglcontextlost', e => { e.preventDefault(); fail('WebGL context lost'); });
        buildScene(); ready = true;
        // Refine once on a cover load; direct reading links begin finished.
        if (!capture && !paused && chapterActive && cover.dataset.intro !== 'finished') { cover.classList.add('pbr-entering'); refinement.start(); }
        resize();
        if (!chapterActive || cover.dataset.intro === 'finished' || paused) { refinement.finish(); fitSurfaceToSupport(true); positionCamera(); render(); }
        new ResizeObserver(resize).observe(cover);
        new IntersectionObserver(([entry]) => {
            // Browser ratios can round .01 down; inspect actual visible pixels.
            visible = entry.isIntersecting && entry.intersectionRect.width > 0 && entry.intersectionRect.height > 0;
            if (visible) { if (chapterActive) refinement?.resume(); previousMoved = false; readScroll(); if (!settled) request(); }
            else if (!visible) cancelScheduled();
        }, { threshold: [0, .01] }).observe(cover);
    } catch (error) { fail(error); }

    function acceptsPointer(event) {
        return ready && !failed && !staticPreferred && chapterActive && !paused && !reduced.matches && fine.matches && event.pointerType !== 'touch';
    }
    cover.addEventListener('pointermove', event => {
        if (!acceptsPointer(event)) return;
        const x = THREE.MathUtils.clamp(event.clientX / viewWidth * 2 - 1, -1, 1);
        const y = THREE.MathUtils.clamp(1 - event.clientY / viewHeight * 2, -1, 1);
        if (Math.abs(x - tx) + Math.abs(y - ty) < .002) return;
        tx = x; ty = y; settled = false; request();
    });
    cover.addEventListener('pointerleave', event => {
        if (!acceptsPointer(event) || (tx === 0 && ty === 0)) return;
        tx = ty = 0; settled = false; request();
    });
    motion.addEventListener('click', () => {
        performancePaused = false; movingSamples = movingDuration = slowSamples = 0; previousMoved = false;
        manualPaused = !manualPaused;
        try { localStorage.setItem('portfolio-motion', manualPaused ? 'off' : 'on'); } catch {}
        paused = manualPaused || reduced.matches; syncMotion();
        // Pausing freezes the current camera instead of starting another transition.
        if (paused) {
            tx = px; ty = py; scrollTarget = scrollProgress; settled = true; cancelScheduled();
            if (refinement.active) {refinement.finish(); fitSurfaceToSupport(true); positionCamera();}
            render();
        }
        else { refinement?.resume(); tx = ty = 0; readScroll(); settled = false; request(); }
    });
    function syncReducedMotion() {
        if (reducedApplied === reduced.matches) return;
        reducedApplied = reduced.matches;
        paused = manualPaused || reducedApplied; syncMotion();
        if (paused) {
            tx = px; ty = py; scrollTarget = scrollProgress; settled = true; cancelScheduled();
            // Switching to reduced motion leaves the authored final surface in place.
            if (ready && !failed && refinement) { refinement.finish(); fitSurfaceToSupport(true); positionCamera(); render(); }
        }
        else { if (chapterActive) refinement?.resume(); readScroll(); settled = false; request(); }
    }
    reduced.addEventListener('change', syncReducedMotion);
    document.addEventListener('visibilitychange', () => {
        if (document.hidden) cancelScheduled();
        else if (chapterActive && !settled) { refinement?.resume(); previousMoved = false; request(); }
    });
    window.portfolioHero = {
        setChapterActive(active) {
            chapterActive = !!active;
            if (!chapterActive) {
                cancelScheduled();
                if (refinement?.active) { refinement.finish(); fitSurfaceToSupport(true); viewDirty = true; }
            }
            else if (!paused && !staticPreferred && !failed) {
                refinement?.resume(); previousMoved = false; settled = false; request();
            }
        },
        state: () => ({ engine: 'WebGL rasterization', ready, failed, staticPreferred, visible, paused, settled,
            suspended: !chapterActive,
            framePending: !!(frame || wakeTimer), frames, triangles: triangleCount, geometryCounts, px, py, firstFrameMs,
            work: {...work}, performancePaused, renderMode: 'intro-once-production-parallax', pixelRatio: renderer?.getPixelRatio(),
            cameraOffset:{x:px * .22,y:py * .12}, scrollProgress, scrollTarget, decoding: refinement?.state(), actionEmphasis: [...actionWeights],
            tilts: supportedForms.map(o => ({ x: o.pivot.rotation.x, y: o.pivot.rotation.y, z: o.pivot.rotation.z })) }),
        layout: () => {
            if (!ready || failed) return [];
            const bounds = host.getBoundingClientRect(), projected = new THREE.Vector3();
            return actionForms.filter(item => !item.background).map(({ mesh }) => {
                const result = { name: mesh.name, left: Infinity, right: -Infinity, top: Infinity, bottom: -Infinity };
                const positions = mesh.geometry.attributes.position;
                for (let i = 0; i < positions.count; i++) {
                    projected.fromBufferAttribute(positions, i).applyMatrix4(mesh.matrixWorld).project(camera);
                    const x = bounds.left + (projected.x + 1) * bounds.width / 2;
                    const y = bounds.top + (1 - projected.y) * bounds.height / 2;
                    result.left = Math.min(result.left, x); result.right = Math.max(result.right, x);
                    result.top = Math.min(result.top, y); result.bottom = Math.max(result.bottom, y);
                }
                return result;
            });
        },
        capture: () => { if (!ready || failed) return null; render(); return renderer.domElement.toDataURL('image/png'); },
    };

}
window.initializePortfolioHero = initializeHero;
if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', initializeHero, { once: true });
else initializeHero();
document.addEventListener('portfolio:core-ready', initializeHero, { once: true });
