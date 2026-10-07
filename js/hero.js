import * as THREE from 'three';
import { RectAreaLightUniformsLib } from 'three/addons/lights/RectAreaLightUniformsLib.js';
import { createMeshRefinement } from './hero-sculpture/mesh-refinement.js';
import { makeStoneTexture, stoneMaterial } from './hero-sculpture/stone-material.js';
import { contactShadow } from './hero-sculpture/contact-shadow.js';
import { carvedGeometry, apertureGeometry, stoneBlock } from './hero-sculpture/stone-geometry.js';

// The static build already includes the cover. The core-ready event also
// supports a source checkout whose content is mounted after DOMContentLoaded.
function initializeHero() {
    const cover = document.querySelector('.pbr-hero');
    if (!cover?.querySelector('.pbr-scene') || cover.dataset.rendererStarted) return;
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
    let renderer, scene, camera, sculptures, base, look, frame = 0;
    let visible = true, paused = reduced.matches, ready = false, failed = false, settled = false;
    let reducedApplied = reduced.matches;
    let tx = 0, ty = 0, px = 0, py = 0, frames = 0, triangleCount = 0, firstFrameMs = 0;
    const geometryCounts = {}, started = performance.now();
    let scrollTarget = 0, scrollProgress = 0, supportDirty = true;
    const supportedForms = [], contacts = [], poseMatrix = new THREE.Matrix4(), point = new THREE.Vector3();
    const supports = [{ x: 2.25, z: .15, halfX: .84, halfZ: .66, height: .86 }];
    const actionForms = [], actionWeights = [0, 0];
    let hoveredForm = -1, focusedForm = -1;
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
        const mineral = stoneMaterial(texture, { color: '#c2c2c2', scale: .68, relief: .012 });
        const wall = stoneMaterial(texture, { color: '#aaaaaa', scale: .30, relief: .018 });
        const ground = stoneMaterial(texture, { color: '#999999', scale: .32, relief: .012 });
        const main = add(carvedGeometry(), mineral, 'carved-form', 2.25, .87, .15);
        main.rotation.set(-.08, -.30, -.10);
        refinement = createMeshRefinement(main);
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
        const objects = sculptures.children.filter(o => o !== floor && o !== back);
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
        key.castShadow = true; key.shadow.mapSize.setScalar(cover.clientWidth <= 600 ? 1024 : 2048);
        key.shadow.camera.near = .5; key.shadow.camera.far = 28;
        key.shadow.bias = -.00012; key.shadow.normalBias = .016;
        key.shadow.radius = 10; key.shadow.blurSamples = 12;
        scene.add(key, key.target);
        scene.add(new THREE.HemisphereLight('#ffffff', '#242424', .12));
        camera = new THREE.PerspectiveCamera(36, 1, .1, 100);
    }

    function positionCamera() {
        camera.position.copy(base); camera.position.x += px * .22; camera.position.y += py * .12;
        camera.lookAt(look); camera.updateMatrixWorld();
    }
    function readScroll() {
        if (staticPreferred || paused || reduced.matches || failed) return;
        const rect = cover.getBoundingClientRect();
        scrollTarget = THREE.MathUtils.clamp(-rect.top / (rect.height * .72), 0, 1);
        if (Math.abs(scrollTarget - scrollProgress) > .0001) { settled = false; request(); }
    }
    function fitSurfaceToSupport(geometryChanged = false) {
        if (!supportDirty && !geometryChanged) return;
        for (const item of supportedForms) {
            item.pivot.position.y = item.support;
            // Keep the changing coarse/fine surface in contact with its plinth.
            // Its authored orientation stays fixed throughout refinement.
            {
                item.pivot.updateMatrix(); item.mesh.updateMatrix();
                poseMatrix.multiplyMatrices(item.pivot.matrix, item.mesh.matrix);
                const positions = item.mesh.geometry.attributes.position;
                let correction = -Infinity;
                // The finite stone block and ground constrain the actual surface.
                for (let i = 0; i < positions.count; i++) {
                    point.fromBufferAttribute(positions, i).applyMatrix4(poseMatrix);
                    let support = -.025;
                    for (const block of supports) {
                        if (Math.abs(point.x - block.x) <= block.halfX && Math.abs(point.z - block.z) <= block.halfZ) support = Math.max(support, block.height);
                    }
                    correction = Math.max(correction, support + .004 - point.y);
                }
                item.pivot.position.y += correction;
            }
        }
        for (const contact of contacts) contact.userData.update();
        renderer.shadowMap.needsUpdate = true;
        supportDirty = false;
    }
    function resize() {
        if (!renderer || !camera) return;
        const w = cover.clientWidth, h = cover.clientHeight, mobile = w <= 900;
        renderer.setSize(w, h); camera.aspect = w / h; camera.fov = mobile ? 42 : 36;
        camera.updateProjectionMatrix();
        base = mobile ? new THREE.Vector3(4.2, 4.8, 13.5) : new THREE.Vector3(4.0, 3.7, 14);
        look = mobile ? new THREE.Vector3(2.1, 4.25, 0) : new THREE.Vector3(.20, 2.15, -.4);
        sculptures.position.set(0, 0, 0);
        sculptures.scale.setScalar(mobile ? .76 : 1);
        stageParts.aperture.position.y = mobile ? 2.30 : 2.95;
        supportDirty = true;
        positionCamera(); readScroll(); renderer.shadowMap.needsUpdate = true;
        settled = false; request();
    }
    function render() {
        renderer.render(scene, camera); frames++;
        if (!refinement?.active) cover.classList.remove('pbr-entering');
        if (!firstFrameMs) firstFrameMs = performance.now() - started;
        host.classList.add('is-ready'); cover.querySelector('.pbr-load')?.setAttribute('hidden', '');
    }
    function draw(now) {
        frame = 0;
        if (!visible || document.hidden || failed) return;
        // A media change can become observable before its change event is delivered.
        // Resolve it in the active frame as well, rather than leaving a coarse form frozen.
        if (reduced.matches !== reducedApplied) { syncReducedMotion(); return; }
        px += (tx - px) * .19; py += (ty - py) * .19;
        scrollProgress += (scrollTarget - scrollProgress) * .16;
        const pointerSettled = Math.abs(tx - px) + Math.abs(ty - py) <= .002;
        const scrollSettled = Math.abs(scrollTarget - scrollProgress) <= .0005;
        if (pointerSettled) { px = tx; py = ty; }
        if (scrollSettled) scrollProgress = scrollTarget;
        const geometryChanged = !paused && !reduced.matches && (refinement?.update(now, scrollProgress) || false);
        let actionsSettled = true;
        if (!paused && !reduced.matches) {
            const selected = focusedForm >= 0 ? focusedForm : hoveredForm;
            actionForms.forEach(({ mesh, color }, index) => {
                const target = selected < 0 ? 0 : selected === index ? 1 : -.35;
                actionWeights[index] += (target - actionWeights[index]) * .19;
                if (Math.abs(target - actionWeights[index]) < .003) actionWeights[index] = target;
                else actionsSettled = false;
                // A small albedo response links the HTML action to its form while
                // retaining the same roughness, lights, silhouette and shadows.
                mesh.material.color.copy(color).multiplyScalar(1 + .12 * actionWeights[index]);
            });
        }
        settled = pointerSettled && scrollSettled && actionsSettled && (!refinement?.active || paused || reduced.matches);
        fitSurfaceToSupport(geometryChanged); positionCamera(); render();
        if (!settled) request();
    }
    function request() {
        if (!staticPreferred && !frame && visible && !document.hidden && !failed) frame = requestAnimationFrame(draw);
    }
    function fail(error) {
        failed = true; cancelAnimationFrame(frame); frame = 0;
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
        renderer.setPixelRatio(capture ? 1 : Math.min(devicePixelRatio, cover.clientWidth <= 600 ? 1 : 1.5));
        renderer.outputColorSpace = THREE.SRGBColorSpace;
        renderer.toneMapping = THREE.AgXToneMapping; renderer.toneMappingExposure = .95;
        renderer.shadowMap.enabled = true; renderer.shadowMap.type = THREE.VSMShadowMap;
        renderer.shadowMap.autoUpdate = false;
        host.appendChild(renderer.domElement);
        renderer.domElement.addEventListener('webglcontextlost', e => { e.preventDefault(); fail('WebGL context lost'); });
        buildScene(); ready = true;
        // Start coarse before the first visible frame, then refine once. Resize,
        // visibility changes resume it; scroll retracts the same nested surface.
        if (!capture && !paused) { cover.classList.add('pbr-entering'); refinement.start(); }
        resize();
        new ResizeObserver(resize).observe(cover);
        new IntersectionObserver(([entry]) => {
            // Browser ratios can round .01 down; inspect actual visible pixels.
            visible = entry.isIntersecting && entry.intersectionRect.width > 0 && entry.intersectionRect.height > 0;
            if (visible) { refinement?.resume(); readScroll(); if (!settled) request(); }
            else if (!visible) { cancelAnimationFrame(frame); frame = 0; }
        }, { threshold: [0, .01] }).observe(cover);
    } catch (error) { fail(error); }

    cover.addEventListener('pointermove', e => {
        if (staticPreferred || paused || reduced.matches || !fine.matches || e.pointerType === 'touch' || failed) return;
        const r = cover.getBoundingClientRect();
        tx = (e.clientX - r.left) / r.width * 2 - 1; ty = 1 - (e.clientY - r.top) / r.height * 2;
        settled = false; request();
    });
    cover.addEventListener('pointerleave', e => {
        if (staticPreferred || paused || e.pointerType === 'touch' || failed) return;
        tx = ty = 0; settled = false; request();
    });
    window.addEventListener('scroll', readScroll, { passive: true });
    for (const link of cover.querySelectorAll('.pbr-actions [data-form]')) {
        const index = Number(link.dataset.form);
        const refresh = () => {
            if (staticPreferred || paused || reduced.matches || failed) return;
            settled = false; request();
        };
        link.addEventListener('pointerenter', e => { if (e.pointerType === 'touch') return; hoveredForm = index; refresh(); });
        link.addEventListener('pointerleave', () => { hoveredForm = -1; refresh(); });
        link.addEventListener('focus', () => { focusedForm = index; refresh(); });
        link.addEventListener('blur', () => { focusedForm = -1; refresh(); });
    }
    motion.addEventListener('click', () => {
        paused = !paused; syncMotion();
        // Pausing freezes the current camera instead of starting another transition.
        if (paused) { tx = px; ty = py; scrollTarget = scrollProgress; settled = true; cancelAnimationFrame(frame); frame = 0; }
        else { refinement?.resume(); tx = ty = 0; readScroll(); settled = false; request(); }
    });
    function syncReducedMotion() {
        if (reducedApplied === reduced.matches) return;
        reducedApplied = reduced.matches;
        paused = reducedApplied; syncMotion();
        if (paused) {
            tx = px; ty = py; scrollTarget = scrollProgress; settled = true; cancelAnimationFrame(frame); frame = 0;
            // Switching to reduced motion leaves the authored final surface in place.
            if (ready && !failed && refinement) { refinement.finish(); fitSurfaceToSupport(true); render(); }
        }
        else { refinement?.resume(); readScroll(); settled = false; request(); }
    }
    reduced.addEventListener('change', syncReducedMotion);
    document.addEventListener('visibilitychange', () => {
        if (document.hidden) { cancelAnimationFrame(frame); frame = 0; }
        else if (!settled) { refinement?.resume(); request(); }
    });
    window.portfolioHero = {
        state: () => ({ engine: 'WebGL rasterization', ready, failed, staticPreferred, visible, paused, settled,
            framePending: !!frame, frames, triangles: triangleCount, geometryCounts, px, py, firstFrameMs,
            scrollProgress, scrollTarget, decoding: refinement?.state(), actionEmphasis: [...actionWeights],
            tilts: supportedForms.map(o => ({ x: o.pivot.rotation.x, z: o.pivot.rotation.z })) }),
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
if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', initializeHero, { once: true });
else initializeHero();
document.addEventListener('portfolio:core-ready', initializeHero, { once: true });
