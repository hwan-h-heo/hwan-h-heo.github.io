const aboutSection = document.getElementById('about');
const reducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)');
const finePointer = window.matchMedia('(hover: hover) and (pointer: fine)');
const portraitAspect = 1049 / 925;
const portraitHeight = 2;
const portraitWidth = portraitHeight * portraitAspect;
const flatImageOpacity = 1;
const flatPointOpacity = 0;

let preloadObserver = null;
let controller = null;

function supportsWebGL() {
    return 'WebGLRenderingContext' in window || 'WebGL2RenderingContext' in window;
}

function loadTexture(THREE, url) {
    return new Promise((resolve, reject) => {
        new THREE.TextureLoader().load(url, resolve, undefined, reject);
    });
}

function easeInOutCubic(value) {
    return value < 0.5
        ? 4 * (value ** 3)
        : 1 - (((-2 * value) + 2) ** 3) / 2;
}

function getPointOpacity(fanAmount, maximum = 1) {
    return maximum * Math.sqrt(Math.max(0, fanAmount));
}

function getUnderlayOpacity(fanAmount, minimum = 0) {
    const fadeStart = 0.2;
    const fadeProgress = Math.min(
        1,
        Math.max(0, (fanAmount - fadeStart) / (1 - fadeStart))
    );
    const faded = easeInOutCubic(fadeProgress);
    return minimum + ((1 - minimum) * (1 - faded));
}

function createPortraitGeometry(THREE, compact) {
    const columns = compact ? 176 : 256;
    const rows = Math.round(columns * (925 / 1049));
    const count = columns * rows;
    const positions = new Float32Array(count * 3);
    const uvs = new Float32Array(count * 2);
    let pointIndex = 0;
    let uvIndex = 0;

    for (let row = 0; row < rows; row += 1) {
        const v = (row + 0.5) / rows;
        for (let column = 0; column < columns; column += 1) {
            const u = (column + 0.5) / columns;
            positions[pointIndex] = (u - 0.5) * portraitWidth;
            positions[pointIndex + 1] = (v - 0.5) * portraitHeight;
            positions[pointIndex + 2] = 0;
            uvs[uvIndex] = u;
            uvs[uvIndex + 1] = v;
            pointIndex += 3;
            uvIndex += 2;
        }
    }

    const geometry = new THREE.BufferGeometry();
    geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
    geometry.setAttribute('uv', new THREE.BufferAttribute(uvs, 2));
    geometry.userData.columns = columns;
    return geometry;
}

function createPortraitMaterial(THREE, portraitTexture, depthTexture, compact) {
    return new THREE.ShaderMaterial({
        uniforms: {
            uPortrait: { value: portraitTexture },
            uDepth: { value: depthTexture },
            uDepthMix: { value: 0 },
            uOpacity: { value: 1 },
            uPointSize: { value: 1 },
            uPixelRatio: { value: 1 }
        },
        vertexShader: `
            uniform sampler2D uPortrait;
            uniform sampler2D uDepth;
            uniform float uDepthMix;
            uniform float uPointSize;
            uniform float uPixelRatio;

            varying vec3 vColor;
            varying float vAlpha;

            void main() {
                vec4 portrait = texture2D(uPortrait, uv);
                float depth = texture2D(uDepth, uv).r;
                float alpha = smoothstep(0.08, 0.56, portrait.a);
                vec3 transformed = position;

                transformed.z += (depth - 0.5) * 0.94 * uDepthMix;

                vec4 modelPosition = modelViewMatrix * vec4(transformed, 1.0);
                gl_Position = projectionMatrix * modelPosition;
                gl_PointSize = max(0.01, alpha * uPointSize * uPixelRatio);

                vColor = mix(portrait.rgb, vec3(0.055, 0.075, 0.11), 0.08);
                vAlpha = alpha;
            }
        `,
        fragmentShader: `
            uniform float uOpacity;
            varying vec3 vColor;
            varying float vAlpha;

            void main() {
                float radius = distance(gl_PointCoord, vec2(0.5));
                float pointAlpha = 1.0 - smoothstep(0.32, 0.5, radius);
                float alpha = pointAlpha * vAlpha * uOpacity;
                if (alpha < 0.012) discard;
                gl_FragColor = vec4(vColor, alpha);
            }
        `,
        transparent: true,
        depthTest: false,
        depthWrite: false,
        blending: THREE.NormalBlending
    });
}

async function createPortraitController(stage) {
    const image = stage.querySelector('.about-profile-image');
    const figure = stage.closest('.about-profile-note');
    const depthSource = stage.dataset.aboutDepth;
    if (!image || !depthSource) return null;

    const THREE = await import('/vendor/three/build/three.module.js');
    const portraitSource = new URL(image.currentSrc || image.src, document.baseURI).href;
    const depthUrl = new URL(depthSource, document.baseURI).href;
    const [portraitTexture, depthTexture] = await Promise.all([
        loadTexture(THREE, portraitSource),
        loadTexture(THREE, depthUrl)
    ]);

    portraitTexture.minFilter = THREE.LinearFilter;
    portraitTexture.magFilter = THREE.LinearFilter;
    depthTexture.minFilter = THREE.LinearFilter;
    depthTexture.magFilter = THREE.LinearFilter;

    const compact = window.matchMedia('(max-width: 767px)').matches
        || (navigator.hardwareConcurrency && navigator.hardwareConcurrency <= 4)
        || (navigator.deviceMemory && navigator.deviceMemory <= 4);
    const renderer = new THREE.WebGLRenderer({
        alpha: true,
        antialias: false,
        depth: false,
        powerPreference: 'low-power',
        premultipliedAlpha: false
    });
    const pixelRatioCap = compact ? 1.25 : 1.5;
    const scene = new THREE.Scene();
    const camera = new THREE.OrthographicCamera(
        -portraitWidth / 2,
        portraitWidth / 2,
        portraitHeight / 2,
        -portraitHeight / 2,
        0.1,
        20
    );
    const geometry = createPortraitGeometry(THREE, compact);
    const material = createPortraitMaterial(THREE, portraitTexture, depthTexture, compact);
    const points = new THREE.Points(geometry, material);

    camera.position.z = 3.8;
    scene.add(points);
    renderer.setClearColor(0xffffff, 0);
    renderer.domElement.className = 'about-portrait-canvas';
    renderer.domElement.setAttribute('aria-hidden', 'true');
    renderer.domElement.setAttribute('role', 'presentation');
    stage.append(renderer.domElement);

    let frame = 0;
    let lastTime = 0;
    let visible = false;
    let contextLost = false;
    let introElapsed = 0;
    let introStarted = false;
    let introComplete = false;
    let ambientTimer = 0;
    let ambientElapsed = 0;
    let ambientActive = false;
    let ambientPending = false;
    let hoverAmount = 0;
    let hoverTarget = 0;
    let pointerX = 0;
    let pointerY = 0;
    let pointerTargetX = 0;
    let pointerTargetY = 0;
    const introDuration = compact ? 4050 : 4500;
    const introPitch = compact ? 0.018 : 0.024;
    const introYawStart = compact ? -0.048 : -0.065;
    const introYawPeak = compact ? 0.035 : 0.047;
    const introPhotoHold = compact ? 0.15 : 0.18;
    const introOpenPoint = 0.24;
    const introTurnPoint = 0.62;
    const introBoundaryOffset = 200;
    const ambientDelay = 6500;
    const ambientDuration = 4500;
    const ambientYawStart = compact ? -0.043 : -0.055;
    const ambientYawPeak = compact ? 0.032 : 0.04;
    const ambientOpenPoint = 0.22;
    const ambientTurnPoint = 0.6;
    const ambientDepth = 0.82;
    const ambientPointOpacity = 0.85;

    function resize() {
        const width = Math.max(stage.clientWidth, 1);
        const height = Math.max(stage.clientHeight, 1);
        const pixelRatio = Math.min(window.devicePixelRatio || 1, pixelRatioCap);
        const viewWidth = portraitHeight * (width / height);
        camera.left = -viewWidth / 2;
        camera.right = viewWidth / 2;
        camera.top = portraitHeight / 2;
        camera.bottom = -portraitHeight / 2;
        camera.updateProjectionMatrix();
        renderer.setPixelRatio(pixelRatio);
        renderer.setSize(width, height, false);
        const pointSpacing = width / geometry.userData.columns;
        material.uniforms.uPointSize.value = Math.max(0.9, pointSpacing * 1.45);
        material.uniforms.uPixelRatio.value = pixelRatio;
        render();
    }

    function setLayerOpacity(canvasOpacity, imageOpacity) {
        stage.style.setProperty('--about-portrait-canvas-opacity', canvasOpacity.toFixed(3));
        stage.style.setProperty('--about-portrait-image-opacity', imageOpacity.toFixed(3));
    }

    function render() {
        renderer.render(scene, camera);
    }

    function clearAmbientPulse() {
        if (ambientTimer) window.clearTimeout(ambientTimer);
        ambientTimer = 0;
        ambientElapsed = 0;
        ambientActive = false;
        ambientPending = false;
    }

    function scheduleAmbientPulse(delay = ambientDelay) {
        if (!finePointer.matches || !introComplete || hoverTarget > 0 || contextLost) return;
        if (ambientTimer) window.clearTimeout(ambientTimer);
        ambientTimer = window.setTimeout(() => {
            ambientTimer = 0;
            if (!visible || document.hidden) {
                ambientPending = true;
                return;
            }
            ambientPending = false;
            ambientElapsed = 0;
            ambientActive = true;
            lastTime = 0;
            stage.dataset.aboutPortraitState = 'ambient';
            requestFrame();
        }, delay);
    }

    function settleToPlane() {
        material.uniforms.uDepthMix.value = 0;
        material.uniforms.uOpacity.value = flatPointOpacity;
        points.rotation.set(0, 0, 0);
        setLayerOpacity(1, flatImageOpacity);
        figure?.setAttribute('data-about-portrait-ready', 'true');
        stage.dataset.aboutPortraitState = 'flat';
        render();
        scheduleAmbientPulse();
    }

    function requestFrame() {
        if (!frame && visible && !document.hidden && !contextLost) {
            frame = window.requestAnimationFrame(tick);
        }
    }

    function tick(time) {
        frame = 0;
        if (!visible || document.hidden) {
            lastTime = 0;
            return;
        }

        if (!introComplete && !introStarted) {
            lastTime = 0;
            return;
        }

        const delta = lastTime ? Math.min(time - lastTime, 48) : 16;
        lastTime = time;

        if (!introComplete) {
            introElapsed += delta;
            const progress = Math.min(1, introElapsed / introDuration);
            const fanProgress = Math.max(
                0,
                (progress - introPhotoHold) / (1 - introPhotoHold)
            );
            const opened = easeInOutCubic(Math.min(1, fanProgress / introOpenPoint));
            const settled = fanProgress <= introTurnPoint
                ? 0
                : easeInOutCubic(
                    (fanProgress - introTurnPoint) / (1 - introTurnPoint)
                );
            const fanAmount = opened * (1 - settled);
            let introYaw = introYawPeak * (1 - settled);

            if (fanProgress < introOpenPoint) {
                introYaw = introYawStart * opened;
            } else if (fanProgress < introTurnPoint) {
                introYaw = introYawStart + (
                    (introYawPeak - introYawStart)
                    * easeInOutCubic(
                        (fanProgress - introOpenPoint)
                        / (introTurnPoint - introOpenPoint)
                    )
                );
            }

            material.uniforms.uDepthMix.value = fanAmount;
            material.uniforms.uOpacity.value = getPointOpacity(fanAmount);
            points.rotation.x = introPitch * fanAmount;
            points.rotation.y = introYaw;
            setLayerOpacity(1, flatImageOpacity * getUnderlayOpacity(fanAmount));
            stage.dataset.aboutPortraitState = 'intro';
            render();

            if (progress >= 1) {
                introComplete = true;
                settleToPlane();
                return;
            }
            requestFrame();
            return;
        }

        if (ambientActive && hoverTarget === 0) {
            ambientElapsed += delta;
            const progress = Math.min(1, ambientElapsed / ambientDuration);
            const opened = easeInOutCubic(Math.min(1, progress / ambientOpenPoint));
            const settled = progress <= ambientTurnPoint
                ? 0
                : easeInOutCubic(
                    (progress - ambientTurnPoint) / (1 - ambientTurnPoint)
                );
            const fanAmount = opened * (1 - settled);
            let ambientYaw = ambientYawPeak * (1 - settled);

            if (progress < ambientOpenPoint) {
                ambientYaw = ambientYawStart * opened;
            } else if (progress < ambientTurnPoint) {
                ambientYaw = ambientYawStart + (
                    (ambientYawPeak - ambientYawStart)
                    * easeInOutCubic(
                        (progress - ambientOpenPoint)
                        / (ambientTurnPoint - ambientOpenPoint)
                    )
                );
            }

            material.uniforms.uDepthMix.value = ambientDepth * fanAmount;
            material.uniforms.uOpacity.value = getPointOpacity(
                fanAmount,
                ambientPointOpacity
            );
            points.rotation.x = 0;
            points.rotation.y = ambientYaw;
            setLayerOpacity(
                1,
                flatImageOpacity * getUnderlayOpacity(fanAmount, 0.16)
            );
            stage.dataset.aboutPortraitState = 'ambient';
            render();

            if (progress >= 1) {
                ambientActive = false;
                settleToPlane();
                return;
            }
            requestFrame();
            return;
        }

        const depthResponse = 1 - Math.exp(-delta / (hoverTarget ? 360 : 460));
        const pointerResponse = 1 - Math.exp(-delta / 130);
        hoverAmount += (hoverTarget - hoverAmount) * depthResponse;
        pointerX += (pointerTargetX - pointerX) * pointerResponse;
        pointerY += (pointerTargetY - pointerY) * pointerResponse;

        material.uniforms.uDepthMix.value = hoverAmount;
        material.uniforms.uOpacity.value = getPointOpacity(hoverAmount);
        points.rotation.y = pointerX * hoverAmount;
        points.rotation.x = pointerY * hoverAmount;
        setLayerOpacity(1, flatImageOpacity * getUnderlayOpacity(hoverAmount));
        stage.dataset.aboutPortraitState = hoverAmount > 0.01 ? 'depth' : 'flat';
        render();

        const moving = Math.abs(hoverTarget - hoverAmount) > 0.002
            || Math.abs(pointerTargetX - pointerX) > 0.001
            || Math.abs(pointerTargetY - pointerY) > 0.001;
        if (moving) {
            requestFrame();
        } else if (hoverTarget === 0) {
            settleToPlane();
        }
    }

    function handlePointerMove(event) {
        const rect = stage.getBoundingClientRect();
        pointerTargetX = ((event.clientX - rect.left) / Math.max(rect.width, 1) - 0.5) * 0.24;
        pointerTargetY = ((event.clientY - rect.top) / Math.max(rect.height, 1) - 0.5) * -0.14;
        if (!introComplete) return;
        clearAmbientPulse();
        hoverTarget = 1;
        requestFrame();
    }

    function handlePointerEnter() {
        if (!introComplete) return;
        clearAmbientPulse();
        hoverTarget = 1;
        requestFrame();
    }

    function handlePointerLeave() {
        hoverTarget = 0;
        pointerTargetX = 0;
        pointerTargetY = 0;
        requestFrame();
    }

    function startIntro() {
        if (introStarted || introComplete) return;
        const rect = stage.getBoundingClientRect();
        visible = rect.bottom > 0
            && rect.top < window.innerHeight
            && rect.right > 0
            && rect.left < window.innerWidth;
        introStarted = true;
        lastTime = 0;
        material.uniforms.uDepthMix.value = 0;
        material.uniforms.uOpacity.value = flatPointOpacity;
        points.rotation.set(0, 0, 0);
        setLayerOpacity(1, flatImageOpacity);
        stage.dataset.aboutPortraitState = 'intro';
        render();
        requestFrame();
    }

    function handleContextLost(event) {
        event.preventDefault();
        clearAmbientPulse();
        contextLost = true;
        if (frame) window.cancelAnimationFrame(frame);
        frame = 0;
        setLayerOpacity(0, 1);
        figure?.removeAttribute('data-about-portrait-ready');
        stage.dataset.aboutPortraitState = 'fallback';
    }

    const visibilityObserver = new IntersectionObserver((entries) => {
        const entry = entries[0];
        visible = entry.isIntersecting;
        if (visible) {
            if (ambientPending && introComplete) scheduleAmbientPulse(1200);
            requestFrame();
        } else if (!visible && frame) {
            window.cancelAnimationFrame(frame);
            frame = 0;
            lastTime = 0;
        }
    }, { threshold: 0 });
    visibilityObserver.observe(stage);

    function handleIntroBoundary() {
        if (aboutSection.getBoundingClientRect().top > introBoundaryOffset) return;
        window.removeEventListener('scroll', handleIntroBoundary);
        startIntro();
    }
    window.addEventListener('scroll', handleIntroBoundary, { passive: true });
    handleIntroBoundary();

    const resizeObserver = new ResizeObserver(resize);
    resizeObserver.observe(stage);
    renderer.domElement.addEventListener('webglcontextlost', handleContextLost);
    if (finePointer.matches) {
        stage.addEventListener('pointerenter', handlePointerEnter);
        stage.addEventListener('pointermove', handlePointerMove);
        stage.addEventListener('pointerleave', handlePointerLeave);
    }
    function handleVisibilityChange() {
        if (!document.hidden && visible && ambientPending && introComplete) {
            scheduleAmbientPulse(1200);
        }
        requestFrame();
    }
    document.addEventListener('visibilitychange', handleVisibilityChange);

    material.uniforms.uDepthMix.value = 0;
    material.uniforms.uOpacity.value = flatPointOpacity;
    points.rotation.set(0, 0, 0);
    setLayerOpacity(1, flatImageOpacity);
    resize();
    stage.dataset.aboutPortraitState = 'intro-ready';

    return {
        destroy() {
            if (frame) window.cancelAnimationFrame(frame);
            clearAmbientPulse();
            visibilityObserver.disconnect();
            window.removeEventListener('scroll', handleIntroBoundary);
            resizeObserver.disconnect();
            renderer.domElement.removeEventListener('webglcontextlost', handleContextLost);
            stage.removeEventListener('pointerenter', handlePointerEnter);
            stage.removeEventListener('pointermove', handlePointerMove);
            stage.removeEventListener('pointerleave', handlePointerLeave);
            document.removeEventListener('visibilitychange', handleVisibilityChange);
            geometry.dispose();
            material.dispose();
            portraitTexture.dispose();
            depthTexture.dispose();
            renderer.dispose();
            renderer.domElement.remove();
            stage.style.removeProperty('--about-portrait-canvas-opacity');
            stage.style.removeProperty('--about-portrait-image-opacity');
            figure?.removeAttribute('data-about-portrait-ready');
            delete stage.dataset.aboutPortraitBound;
            delete stage.dataset.aboutPortraitState;
        }
    };
}

function initializePortrait() {
    if (
        !aboutSection
        || reducedMotion.matches
        || navigator.connection?.saveData
        || !supportsWebGL()
    ) {
        return;
    }

    const stage = aboutSection.querySelector('[data-about-portrait]');
    if (!stage || stage.dataset.aboutPortraitBound === 'true') return;
    stage.dataset.aboutPortraitBound = 'true';

    preloadObserver?.disconnect();
    preloadObserver = new IntersectionObserver((entries) => {
        if (!entries[0].isIntersecting) return;
        preloadObserver.disconnect();
        createPortraitController(stage)
            .then((result) => {
                if (reducedMotion.matches) {
                    result?.destroy();
                    return;
                }
                controller = result;
                if (!result) delete stage.dataset.aboutPortraitBound;
            })
            .catch((error) => {
                delete stage.dataset.aboutPortraitBound;
                stage.style.removeProperty('--about-portrait-canvas-opacity');
                stage.style.removeProperty('--about-portrait-image-opacity');
                console.warn('The About portrait projection could not be initialized.', error);
            });
    }, { rootMargin: '600px 0px' });
    preloadObserver.observe(stage);
}

function handleReducedMotionChange(event) {
    if (!event.matches) {
        initializePortrait();
        return;
    }
    preloadObserver?.disconnect();
    controller?.destroy();
    controller = null;
    const stage = aboutSection?.querySelector('[data-about-portrait]');
    if (stage) delete stage.dataset.aboutPortraitBound;
}

if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', initializePortrait, { once: true });
} else {
    initializePortrait();
}
document.addEventListener('portfolio:core-ready', initializePortrait);
reducedMotion.addEventListener('change', handleReducedMotionChange);
