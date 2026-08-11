const hero = document.getElementById('home');
const root = document.documentElement;
const reducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)');

let initializationQueued = false;
let initialized = false;

function markHeroCtaVisible() {
    if (!hero) {
        return;
    }

    const wasVisible = hero.classList.contains('hero-cta-visible');
    hero.classList.add('hero-cta-visible');
    if (!wasVisible) {
        document.dispatchEvent(new CustomEvent('portfolio:hero-cta-visible'));
    }
}

function showStaticFallback() {
    root.classList.remove('hero-wave-pending');
    root.classList.add('hero-wave-fallback');
    hero?.classList.remove('hero-intro-sweeping');
    hero?.style.removeProperty('--hero-reveal-opacity');
    hero?.classList.add('hero-intro-visible');
    markHeroCtaVisible();
}

function supportsWebGL() {
    return 'WebGLRenderingContext' in window || 'WebGL2RenderingContext' in window;
}

function queueInitialization() {
    if (!hero || initializationQueued || initialized) {
        return;
    }

    initializationQueued = true;
    const initialize = () => {
        initializationQueued = false;
        initializeHeroWave();
    };

    if (
        root.dataset.heroWaveBypassed === 'true'
        || window.matchMedia('(max-width: 767px)').matches
    ) {
        window.requestAnimationFrame(initialize);
    } else if ('requestIdleCallback' in window) {
        window.requestIdleCallback(initialize, { timeout: 800 });
    } else {
        window.setTimeout(initialize, 80);
    }
}

async function initializeHeroWave() {
    if (!hero || initialized) {
        return;
    }

    if (reducedMotion.matches || !supportsWebGL()) {
        showStaticFallback();
        return;
    }

    initialized = true;

    try {
        const THREE = await import('/vendor/three/build/three.module.js');
        createHeroWave(THREE);
    } catch (error) {
        initialized = false;
        showStaticFallback();
        console.warn('The animated hero background could not be initialized.', error);
    }
}

function createHeroWave(THREE) {
    const width = Math.max(hero.clientWidth, 1);
    const isCompact = width < 768
        || Boolean(navigator.connection?.saveData)
        || (navigator.hardwareConcurrency && navigator.hardwareConcurrency <= 4)
        || (navigator.deviceMemory && navigator.deviceMemory <= 4);
    const pixelRatioCap = isCompact ? 1.6 : 1.5;
    const waveSegments = isCompact ? [36, 18] : [56, 26];
    const wireSegments = isCompact ? [18, 9] : [28, 13];
    const driftingParticleCount = isCompact ? 150 : 340;
    const introCycleDuration = isCompact ? 4.8 : 5.6;
    const ambientCycleDuration = 10.5;
    const ambientPlaybackRate = 0.88;
    const ambientStartPhase = 0.44;
    const ctaRevealPhase = isCompact ? 0.42 : 0.44;
    const ctaRevealDelay = introCycleDuration * ctaRevealPhase;
    const burstPreviewPhase = ctaRevealPhase - (isCompact ? 0.08 : 0.075);
    const burstEndPhase = ctaRevealPhase + (isCompact ? 0.28 : 0.26);

    const scene = new THREE.Scene();
    const camera = new THREE.PerspectiveCamera(42, 1, 0.1, 60);
    const renderer = new THREE.WebGLRenderer({
        alpha: true,
        antialias: false,
        depth: false,
        powerPreference: 'low-power',
        premultipliedAlpha: false
    });

    renderer.setClearColor(0x101011, 0);
    renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, pixelRatioCap));
    renderer.domElement.className = 'hero-wave-canvas';
    renderer.domElement.setAttribute('aria-hidden', 'true');
    renderer.domElement.setAttribute('role', 'presentation');
    hero.prepend(renderer.domElement);

    const group = new THREE.Group();
    scene.add(group);

    const waveHeightShader = `
        float waveHeight(vec2 point, float time) {
            float primary = sin(point.x * 0.62 + time * 0.62) * 0.52;
            float crossing = sin(point.y * 0.96 - time * 0.46) * 0.28;
            float detail = sin((point.x + point.y) * 0.42 + time * 0.31) * 0.17;
            return primary + crossing + detail;
        }

        float firstCycleMask(float time) {
            return 1.0 - step(${introCycleDuration.toFixed(1)}, time);
        }

        float cyclePhase(float time) {
            float firstCycle = firstCycleMask(time);
            float introPhase = clamp(
                time / ${introCycleDuration.toFixed(1)},
                0.0,
                1.0
            );
            float ambientTime = max(
                time - ${introCycleDuration.toFixed(1)},
                0.0
            );
            float ambientPhase = fract(
                ambientTime / ${ambientCycleDuration.toFixed(1)}
            );
            return mix(ambientPhase, introPhase, firstCycle);
        }

        float cycleIndex(float time) {
            float firstCycle = firstCycleMask(time);
            float ambientCycle = 1.0 + floor(
                max(time - ${introCycleDuration.toFixed(1)}, 0.0)
                    / ${ambientCycleDuration.toFixed(1)}
            );
            return mix(ambientCycle, 0.0, firstCycle);
        }

        float clusterPulse(float time) {
            float phase = cyclePhase(time);
            float gather = smoothstep(0.18, 0.32, phase);
            float open = 1.0 - smoothstep(0.36, 0.52, phase);
            return gather * open;
        }

        float meshPulse(float time) {
            float phase = cyclePhase(time);
            float firstCycle = firstCycleMask(time);
            float assemble = smoothstep(0.38, 0.54, phase);
            float disperse = 1.0 - smoothstep(0.70, 0.92, phase);
            return assemble * disperse * (1.0 - firstCycle);
        }

        vec2 formationCenter(float time) {
            float cycle = cycleIndex(time);
            float shiftX = fract(sin((cycle + 1.0) * 17.13) * 43758.5453);
            float shiftZ = fract(sin((cycle + 1.0) * 31.71) * 24634.6345);
            return vec2(
                2.0 + (shiftX - 0.5) * 1.4,
                -0.35 + (shiftZ - 0.5) * 1.5
            );
        }

        float formationSelection(vec3 target, float time) {
            float distanceToCenter = distance(target.xz, formationCenter(time));
            return 1.0 - smoothstep(5.5, 6.8, distanceToCenter);
        }

        float formationAmount(vec3 target, float time) {
            float pulse = meshPulse(time);
            float radius = mix(0.55, 6.3, smoothstep(0.0, 1.0, pulse));
            float distanceToCenter = distance(target.xz, formationCenter(time));
            float patchMask = 1.0 - smoothstep(
                radius - 0.75,
                radius + 0.8,
                distanceToCenter
            );
            return pulse * patchMask;
        }

        float formationFrontier(vec3 target, float time) {
            float pulse = meshPulse(time);
            float radius = mix(0.55, 6.3, smoothstep(0.0, 1.0, pulse));
            float distanceToCenter = distance(target.xz, formationCenter(time));
            float ring = 1.0 - smoothstep(
                0.0,
                0.85,
                abs(distanceToCenter - radius)
            );
            return smoothstep(0.04, 0.28, pulse) * ring;
        }

        float surfaceBoundsFade(vec3 point) {
            float horizontal = smoothstep(-11.0, -9.2, point.x)
                * (1.0 - smoothstep(9.4, 11.0, point.x));
            float depth = smoothstep(-6.0, -4.9, point.z)
                * (1.0 - smoothstep(4.8, 6.0, point.z));
            return horizontal * depth;
        }

        float surfaceFade(vec3 point) {
            float contentClearance = mix(0.22, 1.0, smoothstep(-5.6, 0.6, point.x));
            return surfaceBoundsFade(point) * contentClearance;
        }
    `;

    const surfaceGeometry = new THREE.PlaneGeometry(
        22,
        12,
        waveSegments[0],
        waveSegments[1]
    );
    surfaceGeometry.rotateX(-Math.PI / 2);

    const surfaceParticleCount = surfaceGeometry.attributes.position.count;
    const surfaceSeeds = new Float32Array(surfaceParticleCount);
    const surfaceFlowOrigins = new Float32Array(surfaceParticleCount * 3);
    const surfaceSpeeds = new Float32Array(surfaceParticleCount);
    for (let index = 0; index < surfaceSeeds.length; index += 1) {
        const offset = index * 3;
        surfaceSeeds[index] = Math.random();
        surfaceFlowOrigins[offset] = Math.random() * 24 - 12;
        surfaceFlowOrigins[offset + 1] = Math.random() * 6.2 - 1.7;
        surfaceFlowOrigins[offset + 2] = Math.random() * 11 - 5.5;
        surfaceSpeeds[index] = Math.random();
    }

    surfaceGeometry.setAttribute('aSeed', new THREE.BufferAttribute(surfaceSeeds, 1));
    surfaceGeometry.setAttribute(
        'aFlowOrigin',
        new THREE.BufferAttribute(surfaceFlowOrigins, 3)
    );
    surfaceGeometry.setAttribute(
        'aSpeed',
        new THREE.BufferAttribute(surfaceSpeeds, 1)
    );
    const sharedUniforms = {
        uTime: { value: 0 },
        uPixelRatio: { value: Math.min(window.devicePixelRatio || 1, pixelRatioCap) },
        uCtaTarget: { value: new THREE.Vector3(-3, -2.4, 0) }
    };

    const particleMaterial = new THREE.ShaderMaterial({
        uniforms: sharedUniforms,
        transparent: true,
        depthTest: false,
        depthWrite: false,
        blending: THREE.AdditiveBlending,
        vertexShader: `
            uniform float uTime;
            uniform float uPixelRatio;
            uniform vec3 uCtaTarget;
            attribute float aSeed;
            attribute vec3 aFlowOrigin;
            attribute float aSpeed;
            varying float vAlpha;
            varying float vTint;

            ${waveHeightShader}

            void main() {
                vec3 target = position;
                target.y = waveHeight(target.xz, uTime);

                float firstCycle = firstCycleMask(uTime);
                float phase = cyclePhase(uTime);
                float pathNoise = fract(
                    sin(aSeed * 87.23 + aSpeed * 41.17) * 43758.5453
                );

                vec3 ambientFlowing = vec3(
                    aFlowOrigin.x,
                    0.06 + aSpeed * 0.34,
                    aFlowOrigin.z
                );
                float travel = uTime * (0.72 + aSpeed * 0.9);
                ambientFlowing.x = mod(
                    ambientFlowing.x + travel + 11.0,
                    22.0
                ) - 11.0;
                ambientFlowing.z += sin(
                    ambientFlowing.x * 0.48 + uTime * 0.42 + aSeed * 9.0
                ) * (0.13 + aSpeed * 0.2);
                ambientFlowing.z += cos(
                    uTime * 0.23 + aSeed * 15.0
                ) * 0.07;
                ambientFlowing.y += waveHeight(ambientFlowing.xz, uTime);
                ambientFlowing.y += sin(
                    uTime * 1.05 + aSeed * 17.0
                ) * 0.065;

                vec2 center = formationCenter(uTime);
                float regularSelection = formationSelection(position, uTime);
                float regularCluster = clusterPulse(uTime)
                    * regularSelection;
                float clusterAngle = aSeed * 6.283 + uTime * 0.22;
                float clusterRadius = 0.22 + aSpeed * 0.82;
                vec3 cloudTarget = vec3(
                    center.x + cos(clusterAngle) * clusterRadius,
                    waveHeight(center, uTime)
                        + 0.16
                        + sin(aSeed * 19.0 + uTime * 0.7) * 0.2,
                    center.y + sin(clusterAngle) * clusterRadius
                );
                float clusterEase = regularCluster
                    * regularCluster
                    * (3.0 - 2.0 * regularCluster);
                vec3 gathered = mix(
                    ambientFlowing,
                    cloudTarget,
                    clusterEase
                );

                float mesh = formationAmount(position, uTime);
                float meshEase = mesh * mesh * (3.0 - 2.0 * mesh);
                vec3 transformed = mix(gathered, target, meshEase);

                float previewStart = ${burstPreviewPhase.toFixed(3)};
                float burstStart = ${ctaRevealPhase.toFixed(3)};
                float burstEnd = ${burstEndPhase.toFixed(3)};
                float burstDelay = pathNoise * 0.055;
                float burstProgress = smoothstep(
                    burstStart + burstDelay,
                    burstEnd + burstDelay,
                    phase
                );
                float burstEase = burstProgress
                    * burstProgress
                    * (3.0 - 2.0 * burstProgress);
                float burstAngle = aSeed * 6.283
                    + pathNoise * 3.14159;
                vec3 burstDirection = normalize(vec3(
                    cos(burstAngle),
                    0.26 + sin(aSeed * 19.0) * 0.42,
                    sin(burstAngle) * 0.72
                ));
                vec3 burstOrigin = uCtaTarget + burstDirection
                    * (0.035 + aSpeed * 0.055);
                vec3 burstControl = uCtaTarget + burstDirection
                    * (1.75 + aSpeed * 2.0);
                burstControl.y += 0.72 + pathNoise * 0.95;
                vec3 burstStartPoint = mix(
                    burstOrigin,
                    burstControl,
                    burstEase
                );
                vec3 burstEndPoint = mix(
                    burstControl,
                    ambientFlowing,
                    burstEase
                );
                vec3 burstPosition = mix(
                    burstStartPoint,
                    burstEndPoint,
                    burstEase
                );
                float burstArc = sin(burstProgress * 3.14159);
                burstPosition += burstDirection
                    * burstArc
                    * (0.25 + aSpeed * 0.52);
                transformed = mix(transformed, burstPosition, firstCycle);

                float frontier = formationFrontier(position, uTime);
                float twinkle = 0.64 + 0.36 * sin(uTime * 1.15 + aSeed * 18.0);
                float visibility = surfaceFade(transformed);
                float volumeVisibility = surfaceBoundsFade(transformed);
                visibility = mix(
                    visibility,
                    volumeVisibility,
                    firstCycle
                );
                float flowingAlpha = 0.3 + 0.42 * twinkle;
                float formedAlpha = 0.56 + 0.2 * twinkle;
                vAlpha = visibility * mix(flowingAlpha, formedAlpha, meshEase);
                vAlpha += visibility * frontier * 0.28;
                float previewSelection = 1.0 - step(0.09, pathNoise);
                float previewAlpha = smoothstep(
                    previewStart,
                    burstStart,
                    phase
                ) * previewSelection * (0.28 + twinkle * 0.12);
                float burstVisibility = smoothstep(
                    burstStart,
                    burstStart + 0.085,
                    phase
                );
                float introVisibility = max(
                    previewAlpha,
                    burstVisibility * (0.68 + burstArc * 0.28)
                );
                float previewHold = previewAlpha * (
                    1.0 - smoothstep(
                        burstStart,
                        burstStart + 0.06,
                        phase
                    )
                );
                float introAlpha = max(
                    vAlpha * introVisibility,
                    previewHold
                );
                vAlpha = mix(vAlpha, introAlpha, firstCycle);
                vTint = clamp(
                    0.4
                        + transformed.y * 0.25
                        + aSeed * 0.28
                        + frontier * 0.24
                        + firstCycle * burstArc * 0.14,
                    0.0,
                    1.0
                );

                vec4 viewPosition = modelViewMatrix * vec4(transformed, 1.0);
                float perspective = clamp(9.0 / -viewPosition.z, 0.72, 1.55);
                gl_PointSize = (
                    1.35
                        + aSeed * 1.7
                        + meshEase * 0.35
                        + frontier * 0.8
                        + firstCycle * burstArc * 0.45
                )
                    * uPixelRatio
                    * perspective;
                gl_Position = projectionMatrix * viewPosition;
            }
        `,
        fragmentShader: `
            varying float vAlpha;
            varying float vTint;

            void main() {
                float distanceToCenter = distance(gl_PointCoord, vec2(0.5));
                float disc = 1.0 - smoothstep(0.18, 0.5, distanceToCenter);
                float core = 1.0 - smoothstep(0.0, 0.2, distanceToCenter);
                vec3 coolBlue = vec3(0.16, 0.64, 0.92);
                vec3 ice = vec3(0.78, 0.94, 1.0);
                vec3 color = mix(coolBlue, ice, vTint) + core * 0.16;
                gl_FragColor = vec4(color, vAlpha * disc);
            }
        `
    });

    const surfaceParticles = new THREE.Points(surfaceGeometry, particleMaterial);
    surfaceParticles.renderOrder = 2;
    group.add(surfaceParticles);

    const wireSurfaceGeometry = new THREE.PlaneGeometry(
        22,
        12,
        wireSegments[0],
        wireSegments[1]
    );
    wireSurfaceGeometry.rotateX(-Math.PI / 2);
    const wireGeometry = new THREE.WireframeGeometry(wireSurfaceGeometry);
    wireSurfaceGeometry.dispose();
    const wireMaterial = new THREE.ShaderMaterial({
        uniforms: sharedUniforms,
        transparent: true,
        depthTest: false,
        depthWrite: false,
        blending: THREE.NormalBlending,
        vertexShader: `
            uniform float uTime;
            varying float vAlpha;
            varying float vTint;

            ${waveHeightShader}

            void main() {
                vec3 transformed = position;
                transformed.y = waveHeight(transformed.xz, uTime);

                float mesh = formationAmount(position, uTime);
                float frontier = formationFrontier(position, uTime);
                float idleGrid = 0.045 + 0.018 * sin(
                    uTime * 0.48
                        + position.x * 0.22
                        + position.z * 0.18
                );
                vAlpha = surfaceFade(transformed)
                    * (
                        idleGrid
                            + pow(mesh, 1.15)
                                * (0.52 + frontier * 0.34)
                    );
                vTint = clamp(0.35 + transformed.y * 0.25 + frontier * 0.34, 0.0, 1.0);

                gl_Position = projectionMatrix
                    * modelViewMatrix
                    * vec4(transformed, 1.0);
            }
        `,
        fragmentShader: `
            varying float vAlpha;
            varying float vTint;

            void main() {
                vec3 color = mix(
                    vec3(0.12, 0.5, 0.8),
                    vec3(0.62, 0.91, 1.0),
                    vTint
                );
                gl_FragColor = vec4(color, vAlpha);
            }
        `
    });

    const surfaceWireframe = new THREE.LineSegments(wireGeometry, wireMaterial);
    surfaceWireframe.renderOrder = 1;
    group.add(surfaceWireframe);

    const driftingGeometry = new THREE.BufferGeometry();
    const driftingPositions = new Float32Array(driftingParticleCount * 3);
    const driftingSeeds = new Float32Array(driftingParticleCount);

    for (let index = 0; index < driftingParticleCount; index += 1) {
        const offset = index * 3;
        driftingPositions[offset] = Math.random() * 22 - 11;
        driftingPositions[offset + 1] = Math.random() * 1.8 + 0.15;
        driftingPositions[offset + 2] = Math.random() * 10.5 - 5.25;
        driftingSeeds[index] = Math.random();
    }

    driftingGeometry.setAttribute(
        'position',
        new THREE.BufferAttribute(driftingPositions, 3)
    );
    driftingGeometry.setAttribute(
        'aSeed',
        new THREE.BufferAttribute(driftingSeeds, 1)
    );

    const driftingMaterial = new THREE.ShaderMaterial({
        uniforms: sharedUniforms,
        transparent: true,
        depthTest: false,
        depthWrite: false,
        blending: THREE.AdditiveBlending,
        vertexShader: `
            uniform float uTime;
            uniform float uPixelRatio;
            uniform vec3 uCtaTarget;
            attribute float aSeed;
            varying float vAlpha;
            varying float vTint;

            ${waveHeightShader}

            void main() {
                vec3 transformed = position;
                float travel = uTime * (0.48 + aSeed * 0.62);
                transformed.x = mod(transformed.x + travel + 11.0, 22.0) - 11.0;
                transformed.z += sin(
                    transformed.x * 0.38 + uTime * 0.32 + aSeed * 9.0
                ) * 0.32;
                transformed.y += waveHeight(transformed.xz, uTime) * 0.68;
                transformed.y += sin(uTime * 0.72 + aSeed * 14.0) * 0.24;

                float firstCycle = firstCycleMask(uTime);
                float phase = cyclePhase(uTime);
                float previewStart = ${burstPreviewPhase.toFixed(3)};
                float burstStart = ${ctaRevealPhase.toFixed(3)};
                float burstEnd = ${burstEndPhase.toFixed(3)};
                float burstNoise = fract(
                    sin(aSeed * 97.41) * 43758.5453
                );
                float burstDelay = burstNoise * 0.07;
                float burstProgress = smoothstep(
                    burstStart + burstDelay,
                    burstEnd + burstDelay,
                    phase
                );
                float burstEase = burstProgress
                    * burstProgress
                    * (3.0 - 2.0 * burstProgress);
                float burstAngle = aSeed * 6.283
                    + burstNoise * 2.4;
                vec3 burstDirection = normalize(vec3(
                    cos(burstAngle),
                    0.32 + sin(aSeed * 17.0) * 0.38,
                    sin(burstAngle) * 0.74
                ));
                vec3 burstOrigin = uCtaTarget + burstDirection
                    * (0.04 + burstNoise * 0.06);
                vec3 burstControl = uCtaTarget + burstDirection
                    * (1.9 + burstNoise * 2.2);
                burstControl.y += 0.7 + burstNoise * 1.1;
                vec3 burstStartPoint = mix(
                    burstOrigin,
                    burstControl,
                    burstEase
                );
                vec3 burstEndPoint = mix(
                    burstControl,
                    transformed,
                    burstEase
                );
                vec3 burstPosition = mix(
                    burstStartPoint,
                    burstEndPoint,
                    burstEase
                );
                float burstArc = sin(burstProgress * 3.14159);
                burstPosition += burstDirection
                    * burstArc
                    * (0.3 + burstNoise * 0.62);
                transformed = mix(transformed, burstPosition, firstCycle);

                float edge = surfaceFade(vec3(transformed.x, 0.0, transformed.z));
                float formation = formationAmount(
                    vec3(transformed.x, 0.0, transformed.z),
                    uTime
                );
                float pulse = 0.52 + 0.48 * sin(uTime * 0.92 + aSeed * 21.0);
                vAlpha = edge
                    * (0.12 + pulse * 0.34)
                    * (0.35 + aSeed * 0.65)
                    * (1.0 - formation * 0.76);
                float previewSelection = 1.0 - step(0.09, burstNoise);
                float previewAlpha = smoothstep(
                    previewStart,
                    burstStart,
                    phase
                ) * previewSelection * (0.24 + pulse * 0.1);
                float burstVisibility = smoothstep(
                    burstStart,
                    burstStart + 0.085,
                    phase
                );
                float introVisibility = max(
                    previewAlpha,
                    burstVisibility * (0.7 + burstArc * 0.28)
                );
                float previewHold = previewAlpha * (
                    1.0 - smoothstep(
                        burstStart,
                        burstStart + 0.06,
                        phase
                    )
                );
                float introAlpha = max(
                    vAlpha * introVisibility,
                    previewHold
                );
                vAlpha = mix(vAlpha, introAlpha, firstCycle);
                vTint = clamp(
                    0.35 + aSeed * 0.65 + firstCycle * burstArc * 0.12,
                    0.0,
                    1.0
                );

                vec4 viewPosition = modelViewMatrix * vec4(transformed, 1.0);
                float perspective = clamp(8.0 / -viewPosition.z, 0.65, 1.45);
                gl_PointSize = (
                    1.1
                        + aSeed * 2.2
                        + firstCycle * burstArc * 0.48
                )
                    * uPixelRatio
                    * perspective;
                gl_Position = projectionMatrix * viewPosition;
            }
        `,
        fragmentShader: `
            varying float vAlpha;
            varying float vTint;

            void main() {
                float distanceToCenter = distance(gl_PointCoord, vec2(0.5));
                float disc = 1.0 - smoothstep(0.1, 0.5, distanceToCenter);
                vec3 color = mix(
                    vec3(0.18, 0.58, 0.88),
                    vec3(0.76, 0.93, 1.0),
                    vTint
                );
                gl_FragColor = vec4(color, vAlpha * disc);
            }
        `
    });

    const driftingParticles = new THREE.Points(driftingGeometry, driftingMaterial);
    driftingParticles.renderOrder = 3;
    group.add(driftingParticles);

    let animationFrame = 0;
    let animationTime = introCycleDuration
        + ambientCycleDuration * ambientStartPhase;
    let ctaElapsedTime = 0;
    let lastFrameTime = performance.now();
    let heroIsVisible = true;
    let pointerX = 0;
    let pointerY = 0;
    let pointerTargetX = 0;
    let pointerTargetY = 0;
    let disposed = false;
    let ctaTimelineStarted = false;
    let cameraParallaxX = 1.4;
    let cameraParallaxY = 0.68;
    let cameraParallaxZ = 0.62;
    let cameraDriftX = 0.34;
    let cameraDriftY = 0.12;
    let cameraDriftZ = 0.14;
    let groupParallaxX = 0.038;
    let groupParallaxY = 0.075;
    const cameraBasePosition = new THREE.Vector3();
    const cameraLookTarget = new THREE.Vector3();

    function startCtaTimeline() {
        if (disposed || ctaTimelineStarted) {
            return;
        }

        ctaTimelineStarted = true;
        ctaElapsedTime = 0;
    }

    function handlePreloaderHidden() {
        startCtaTimeline();
    }

    function queueCtaTimeline() {
        const preloader = document.getElementById('preloader');
        if (!preloader || preloader.dataset.revealComplete === 'true') {
            startCtaTimeline();
            return;
        }

        document.addEventListener(
            'portfolio:preloader-hidden',
            handlePreloaderHidden,
            { once: true }
        );
    }

    function setSceneLayout() {
        const rect = hero.getBoundingClientRect();
        const narrow = rect.width <= 768;
        const desktopProgress = Math.min(
            Math.max((rect.width - 768) / 432, 0),
            1
        );
        const desktopEase = desktopProgress
            * desktopProgress
            * (3 - 2 * desktopProgress);
        const pixelRatio = Math.min(window.devicePixelRatio || 1, pixelRatioCap);

        renderer.setPixelRatio(pixelRatio);
        renderer.setSize(Math.max(rect.width, 1), Math.max(rect.height, 1), false);
        sharedUniforms.uPixelRatio.value = pixelRatio;
        camera.aspect = Math.max(rect.width, 1) / Math.max(rect.height, 1);
        camera.fov = narrow ? 47 : 42;
        cameraBasePosition.set(0, narrow ? 4.8 : 4.25, narrow ? 10.8 : 10.2);
        cameraLookTarget.set(0, narrow ? -1.55 : -1.35, -1.8);
        cameraParallaxX = narrow ? 0.48 : 1.4;
        cameraParallaxY = narrow ? 0.28 : 0.68;
        cameraParallaxZ = narrow ? 0.22 : 0.62;
        cameraDriftX = narrow ? 0.24 : 0.34;
        cameraDriftY = narrow ? 0.08 : 0.12;
        cameraDriftZ = narrow ? 0.09 : 0.14;
        groupParallaxX = narrow ? 0.014 : 0.038;
        groupParallaxY = narrow ? 0.03 : 0.075;
        camera.position.copy(cameraBasePosition);
        camera.lookAt(cameraLookTarget);
        camera.updateProjectionMatrix();
        group.position.set(
            narrow ? -0.55 : 0.35 + desktopEase * 2.75,
            narrow ? -2.7 : -2.6 + desktopEase * 0.15,
            -1.65
        );
        camera.updateMatrixWorld(true);
        group.updateMatrixWorld(true);
        const ctaRaycaster = new THREE.Raycaster();
        const ctaPlane = new THREE.Plane().setFromNormalAndCoplanarPoint(
            new THREE.Vector3(0, 0, 1),
            new THREE.Vector3(0, 0, group.position.z)
        );
        const ctaWorldTarget = new THREE.Vector3();
        const ctaElement = hero.querySelector('.scroll-down-arrow');
        const ctaRect = ctaElement?.getBoundingClientRect();
        const ctaNdc = ctaRect
            ? new THREE.Vector2(
                (((ctaRect.left + ctaRect.width * 0.5) - rect.left) / rect.width) * 2 - 1,
                1 - (((ctaRect.top + ctaRect.height * 0.5) - rect.top) / rect.height) * 2
            )
            : new THREE.Vector2(0, -0.84);
        ctaRaycaster.setFromCamera(ctaNdc, camera);
        if (ctaRaycaster.ray.intersectPlane(ctaPlane, ctaWorldTarget)) {
            group.worldToLocal(ctaWorldTarget);
            sharedUniforms.uCtaTarget.value.copy(ctaWorldTarget);
        }
    }

    function render(frameTime) {
        animationFrame = 0;
        if (disposed || !heroIsVisible || document.hidden) {
            return;
        }

        const delta = Math.min((frameTime - lastFrameTime) / 1000, 0.05);
        lastFrameTime = frameTime;
        animationTime += Math.max(delta, 0) * ambientPlaybackRate;
        if (ctaTimelineStarted) {
            ctaElapsedTime += Math.max(delta, 0);
        }
        sharedUniforms.uTime.value = animationTime;

        if (
            !hero.classList.contains('hero-cta-visible')
            && ctaElapsedTime >= ctaRevealDelay
        ) {
            markHeroCtaVisible();
        }

        const pointerEase = 1 - Math.exp(-Math.max(delta, 0) * 7.2);
        const cameraEase = 1 - Math.exp(-Math.max(delta, 0) * 5.2);
        const groupEase = 1 - Math.exp(-Math.max(delta, 0) * 4.2);
        pointerX += (pointerTargetX - pointerX) * pointerEase;
        pointerY += (pointerTargetY - pointerY) * pointerEase;

        const driftX = Math.sin(animationTime * 0.24) * cameraDriftX;
        const driftY = Math.sin(animationTime * 0.17 + 1.2) * cameraDriftY;
        const driftZ = Math.cos(animationTime * 0.19 + 0.4) * cameraDriftZ;
        camera.position.x += (
            cameraBasePosition.x
                + driftX
                + pointerX * cameraParallaxX
                - camera.position.x
        ) * cameraEase;
        camera.position.y += (
            cameraBasePosition.y
                + driftY
                - pointerY * cameraParallaxY
                - camera.position.y
        ) * cameraEase;
        camera.position.z += (
            cameraBasePosition.z
                + driftZ
                + pointerY * cameraParallaxZ
                - camera.position.z
        ) * cameraEase;
        camera.lookAt(cameraLookTarget);

        const targetGroupRotationY = pointerX * groupParallaxY
            + Math.sin(animationTime * 0.16) * groupParallaxY * 0.28;
        const targetGroupRotationX = pointerY * groupParallaxX
            + Math.cos(animationTime * 0.13) * groupParallaxX * 0.32;
        group.rotation.y += (
            targetGroupRotationY - group.rotation.y
        ) * groupEase;
        group.rotation.x += (
            targetGroupRotationX - group.rotation.x
        ) * groupEase;

        renderer.render(scene, camera);
        animationFrame = window.requestAnimationFrame(render);
    }

    function startRendering() {
        if (disposed || animationFrame || !heroIsVisible || document.hidden) {
            return;
        }
        lastFrameTime = performance.now();
        animationFrame = window.requestAnimationFrame(render);
    }

    function stopRendering() {
        if (!animationFrame) {
            return;
        }
        window.cancelAnimationFrame(animationFrame);
        animationFrame = 0;
    }

    function shapePointerInput(value) {
        const clamped = Math.max(-1, Math.min(1, value));
        return Math.sign(clamped) * Math.pow(Math.abs(clamped), 0.82);
    }

    function handlePointerMove(event) {
        if (!heroIsVisible) {
            return;
        }

        const rect = hero.getBoundingClientRect();
        const normalizedX = ((event.clientX - rect.left) / Math.max(rect.width, 1) - 0.5) * 2;
        const normalizedY = ((event.clientY - rect.top) / Math.max(rect.height, 1) - 0.5) * 2;
        const influence = event.pointerType === 'touch' ? 0.55 : 1;
        pointerTargetX = shapePointerInput(normalizedX) * influence;
        pointerTargetY = shapePointerInput(normalizedY) * influence;
    }

    function resetPointerTarget(event) {
        if (!event || event.type === 'mouseleave' || event.pointerType === 'touch') {
            pointerTargetX = 0;
            pointerTargetY = 0;
        }
    }

    function handleVisibilityChange() {
        if (document.hidden) {
            stopRendering();
        } else {
            startRendering();
        }
    }

    function dispose() {
        if (disposed) {
            return;
        }
        disposed = true;
        stopRendering();
        resizeObserver?.disconnect();
        intersectionObserver?.disconnect();
        window.removeEventListener('pointermove', handlePointerMove);
        window.removeEventListener('pointerdown', handlePointerMove);
        window.removeEventListener('pointerup', resetPointerTarget);
        window.removeEventListener('pointercancel', resetPointerTarget);
        document.documentElement.removeEventListener('mouseleave', resetPointerTarget);
        document.removeEventListener('visibilitychange', handleVisibilityChange);
        document.removeEventListener(
            'portfolio:preloader-hidden',
            handlePreloaderHidden
        );
        surfaceGeometry.dispose();
        wireGeometry.dispose();
        driftingGeometry.dispose();
        particleMaterial.dispose();
        wireMaterial.dispose();
        driftingMaterial.dispose();
        renderer.dispose();
        renderer.domElement.remove();
    }

    const resizeObserver = 'ResizeObserver' in window
        ? new ResizeObserver(setSceneLayout)
        : null;
    const intersectionObserver = 'IntersectionObserver' in window
        ? new IntersectionObserver((entries) => {
            heroIsVisible = entries[0]?.isIntersecting ?? true;
            if (heroIsVisible) {
                startRendering();
            } else {
                pointerTargetX = 0;
                pointerTargetY = 0;
                stopRendering();
            }
        }, { threshold: 0.02 })
        : null;

    setSceneLayout();
    resizeObserver?.observe(hero);
    intersectionObserver?.observe(hero);
    window.addEventListener('pointermove', handlePointerMove, { passive: true });
    window.addEventListener('pointerdown', handlePointerMove, { passive: true });
    window.addEventListener('pointerup', resetPointerTarget, { passive: true });
    window.addEventListener('pointercancel', resetPointerTarget, { passive: true });
    document.documentElement.addEventListener('mouseleave', resetPointerTarget);
    document.addEventListener('visibilitychange', handleVisibilityChange);
    renderer.domElement.addEventListener('webglcontextlost', () => {
        dispose();
        showStaticFallback();
    }, { once: true });

    root.classList.remove('hero-wave-fallback');
    root.classList.remove('hero-wave-pending');
    root.classList.add('hero-wave-active');
    window.requestAnimationFrame(() => {
        hero.classList.add('hero-wave-ready');
    });

    hero.classList.remove('hero-intro-sweeping');
    hero.classList.add('hero-intro-visible');
    hero.style.removeProperty('--hero-reveal-opacity');
    if (root.dataset.heroWaveBypassed === 'true') {
        markHeroCtaVisible();
    } else {
        queueCtaTimeline();
    }
    startRendering();
}

if (!hero) {
    // The portfolio hero is not present on project or blog pages.
} else if (reducedMotion.matches) {
    showStaticFallback();
} else if (hero.querySelector('.hero-content')) {
    queueInitialization();
} else {
    hero.addEventListener('portfolio:section-ready', queueInitialization, { once: true });
    document.addEventListener('portfolio:core-ready', queueInitialization, { once: true });
}
