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

    if (window.matchMedia('(max-width: 767px)').matches) {
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
    const introCycleDuration = isCompact ? 6.4 : 8.2;
    const ambientCycleDuration = 10.5;
    const ambientPlaybackRate = 0.88;
    const introRevealStartPhase = 0.035;
    const introRevealEndPhase = 0.18;
    const ctaRevealPhase = isCompact ? 0.46 : 0.51;

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

        float ctaFlowPulse(float time) {
            float phase = cyclePhase(time);
            float travel = smoothstep(0.46, 0.56, phase);
            return travel * firstCycleMask(time);
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
    const surfaceHiTargets = new Float32Array(surfaceParticleCount * 3);

    for (let index = 0; index < surfaceSeeds.length; index += 1) {
        const offset = index * 3;
        surfaceSeeds[index] = Math.random();
        surfaceFlowOrigins[offset] = Math.random() * 24 - 12;
        surfaceFlowOrigins[offset + 1] = Math.random() * 6.2 - 1.7;
        surfaceFlowOrigins[offset + 2] = Math.random() * 11 - 5.5;
        surfaceSpeeds[index] = Math.random();

        const stroke = Math.random();
        let letterX = 0;
        let letterY = 0;

        if (stroke < 0.3) {
            letterX = -1.05;
            letterY = Math.random() * 2.1 - 1.05;
        } else if (stroke < 0.6) {
            letterX = -0.1;
            letterY = Math.random() * 2.1 - 1.05;
        } else if (stroke < 0.74) {
            letterX = Math.random() * 0.95 - 1.05;
            letterY = 0;
        } else if (stroke < 0.94) {
            letterX = 0.62;
            letterY = Math.random() * 1.48 - 1.05;
        } else {
            const dotAngle = Math.random() * Math.PI * 2;
            const dotRadius = Math.sqrt(Math.random()) * 0.13;
            letterX = 0.62 + Math.cos(dotAngle) * dotRadius;
            letterY = 0.91 + Math.sin(dotAngle) * dotRadius;
        }

        surfaceHiTargets[offset] = letterX + (Math.random() - 0.5) * 0.075;
        surfaceHiTargets[offset + 1] = letterY + (Math.random() - 0.5) * 0.075;
        surfaceHiTargets[offset + 2] = (Math.random() - 0.5) * 0.52;
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
    surfaceGeometry.setAttribute(
        'aHiTarget',
        new THREE.BufferAttribute(surfaceHiTargets, 3)
    );

    const sharedUniforms = {
        uTime: { value: 0 },
        uPixelRatio: { value: Math.min(window.devicePixelRatio || 1, pixelRatioCap) },
        uCtaTarget: { value: new THREE.Vector3(-3, -2.4, 0) },
        uHiOffset: { value: new THREE.Vector3() },
        uHiScale: { value: 1 },
        uHiDepth: { value: 1 },
        uHiLineAlpha: { value: 0.2 },
        uTrailBend: { value: new THREE.Vector3() },
        uTrailStrength: { value: 1 }
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
            uniform vec3 uHiOffset;
            uniform float uHiScale;
            uniform float uHiDepth;
            uniform vec3 uTrailBend;
            uniform float uTrailStrength;
            attribute float aSeed;
            attribute vec3 aFlowOrigin;
            attribute float aSpeed;
            attribute vec3 aHiTarget;
            varying float vAlpha;
            varying float vTint;

            ${waveHeightShader}

            void main() {
                vec3 target = position;
                target.y = waveHeight(target.xz, uTime);

                vec2 center = formationCenter(uTime);
                float firstCycle = firstCycleMask(uTime);
                float phase = cyclePhase(uTime);
                float pathNoise = fract(
                    sin(aSeed * 87.23 + aSpeed * 41.17) * 43758.5453
                );
                float ctaNoise = fract(
                    sin(aSeed * 131.71 + aSpeed * 73.19) * 24634.6345
                );
                float hiSelection = 1.0 - step(0.74, aSpeed);
                float ctaSelection = hiSelection
                    * (1.0 - step(0.52, ctaNoise));

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

                vec3 volumeFlowing = aFlowOrigin;
                volumeFlowing.x += sin(
                    uTime * (0.17 + aSpeed * 0.08) + aSeed * 19.0
                ) * (0.18 + aSpeed * 0.22);
                volumeFlowing.y += cos(
                    uTime * (0.2 + aSpeed * 0.07) + aSeed * 23.0
                ) * (0.24 + aSpeed * 0.28);
                volumeFlowing.z += sin(
                    uTime * (0.14 + aSpeed * 0.09) + aSeed * 29.0
                ) * (0.3 + aSpeed * 0.36);
                float ambientSettle = smoothstep(
                    0.80 + pathNoise * 0.025,
                    0.985,
                    phase
                );
                vec3 flowing = mix(
                    ambientFlowing,
                    volumeFlowing,
                    firstCycle * (1.0 - ambientSettle)
                );

                float regularSelection = formationSelection(position, uTime);
                float regularCluster = clusterPulse(uTime)
                    * regularSelection;
                float hiGather = 1.0;
                float releaseDelay = pathNoise * 0.045;
                float hiRelease = 1.0 - smoothstep(
                    0.52 + releaseDelay,
                    0.70 + releaseDelay,
                    phase
                );
                float introCluster = hiGather
                    * hiRelease
                    * hiSelection;
                float cluster = mix(
                    regularCluster,
                    introCluster,
                    firstCycle
                );
                float clusterAngle = aSeed * 6.283 + uTime * 0.22;
                float clusterRadius = 0.22 + aSpeed * 0.82;
                vec3 cloudTarget = vec3(
                    center.x + cos(clusterAngle) * clusterRadius,
                    waveHeight(center, uTime)
                        + 0.16
                        + sin(aSeed * 19.0 + uTime * 0.7) * 0.2,
                    center.y + sin(clusterAngle) * clusterRadius
                );
                float hiYaw = 0.31 + sin(uTime * 0.2 + aSeed * 4.0) * 0.03;
                float hiYawCos = cos(hiYaw);
                float hiYawSin = sin(hiYaw);
                vec3 hiLocal = vec3(
                    aHiTarget.x * uHiScale,
                    aHiTarget.y * uHiScale,
                    aHiTarget.z * uHiScale * uHiDepth
                );
                hiLocal.z += sin(
                    aHiTarget.x * 2.1 + aHiTarget.y * 1.6 + aSeed * 8.0
                ) * 0.12 * uHiDepth;
                vec3 hiTilted = vec3(
                    hiLocal.x * hiYawCos + hiLocal.z * hiYawSin,
                    hiLocal.y + hiLocal.z * 0.06,
                    -hiLocal.x * hiYawSin + hiLocal.z * hiYawCos
                );
                float hiDepthLight = smoothstep(-0.42, 0.42, hiTilted.z);
                vec3 hiTarget = vec3(
                    center.x + uHiOffset.x + hiTilted.x,
                    waveHeight(center, uTime)
                        + 0.18
                        + uHiOffset.y
                        + hiTilted.y,
                    center.y + uHiOffset.z + hiTilted.z
                );
                hiTarget += vec3(
                    sin(uTime * 0.34 + aSeed * 17.0) * 0.045,
                    cos(uTime * 0.29 + aSeed * 11.0) * 0.035,
                    sin(uTime * 0.25 + aSeed * 23.0) * 0.055
                ) * firstCycle * hiSelection * hiRelease;
                vec3 clusterTarget = mix(cloudTarget, hiTarget, firstCycle);
                float clusterEase = cluster * cluster * (3.0 - 2.0 * cluster);
                vec3 gatherCurl = vec3(
                    sin(aSeed * 13.0) * (0.56 + pathNoise * 0.22),
                    cos(aSeed * 17.0) * (0.42 + aSpeed * 0.2),
                    sin(aSeed * 23.0) * (0.78 + pathNoise * 0.32)
                ) * sin(hiGather * 3.14159);
                vec3 gathered = mix(flowing, clusterTarget, clusterEase);
                gathered += gatherCurl
                    * firstCycle
                    * hiSelection
                    * hiRelease;

                float mesh = formationAmount(position, uTime);
                float meshEase = mesh * mesh * (3.0 - 2.0 * mesh);
                vec3 transformed = mix(gathered, target, meshEase);

                float trailSelection = hiSelection * firstCycle;
                float trailRelease = 1.0 - smoothstep(
                    0.80 + releaseDelay,
                    0.94,
                    phase
                );
                float scatterProgress = smoothstep(
                    0.78 + releaseDelay,
                    0.94,
                    phase
                );
                float trailGuide = ctaFlowPulse(uTime)
                    * trailRelease
                    * trailSelection
                    * smoothstep(0.72, 1.0, hiGather);
                float easedTrailGuide = trailGuide * uTrailStrength;
                float trailDelay = pathNoise * 0.05;
                float trailProgress = smoothstep(
                    0.50 + trailDelay,
                    0.70 + trailDelay,
                    phase
                );
                float endpointAngle = aSeed * 6.283;
                float endpointRadius = mix(
                    1.4 + aSpeed * 2.2,
                    0.14 + aSpeed * 0.48,
                    ctaSelection
                );
                float endpointHeight = mix(
                    0.45 + aSpeed * 0.6,
                    0.11 + aSeed * 0.08,
                    ctaSelection
                );
                vec3 endpoint = uCtaTarget + vec3(
                    cos(endpointAngle) * endpointRadius,
                    sin(aSeed * 19.0) * endpointHeight,
                    sin(endpointAngle) * endpointRadius * 0.52
                );
                vec3 trailControl = mix(hiTarget, endpoint, 0.5)
                    + uTrailBend;
                vec3 trailStart = mix(
                    hiTarget,
                    trailControl,
                    trailProgress
                );
                vec3 trailEnd = mix(
                    trailControl,
                    endpoint,
                    trailProgress
                );
                vec3 trailPosition = mix(
                    trailStart,
                    trailEnd,
                    trailProgress
                );
                trailPosition.y += sin(trailProgress * 3.14159)
                    * (0.52 + aSpeed * 0.48);
                trailPosition.z += sin(trailProgress * 3.14159)
                    * (aSeed - 0.5)
                    * 0.8;
                transformed = mix(transformed, trailPosition, easedTrailGuide);
                float scatterArc = sin(scatterProgress * 3.14159)
                    * firstCycle
                    * trailSelection
                    * 0.58;
                transformed += vec3(
                    sin(aSeed * 29.0 + uTime * 0.3) * 0.5,
                    cos(aSeed * 19.0) * 0.34,
                    sin(aSeed * 37.0 - uTime * 0.24) * 0.72
                ) * scatterArc;

                float frontier = formationFrontier(position, uTime);
                float twinkle = 0.64 + 0.36 * sin(uTime * 1.15 + aSeed * 18.0);
                float visibility = surfaceFade(transformed);
                float volumeVisibility = surfaceBoundsFade(transformed);
                float contentClearance = smoothstep(0.12, 0.28, phase);
                visibility = mix(
                    visibility,
                    volumeVisibility,
                    firstCycle * (1.0 - contentClearance)
                );
                float flowingAlpha = 0.3 + 0.42 * twinkle;
                float formedAlpha = 0.56 + 0.2 * twinkle;
                vAlpha = visibility * mix(flowingAlpha, formedAlpha, meshEase);
                vAlpha += visibility * frontier * 0.28;
                vAlpha += visibility
                    * easedTrailGuide
                    * (0.12 + ctaSelection * 0.22);
                vAlpha += visibility
                    * firstCycle
                    * cluster
                    * (0.06 + hiDepthLight * 0.07);
                float hiReadGlow = mix(
                    0.42,
                    0.82,
                    smoothstep(0.36, 0.52, phase)
                );
                vAlpha *= mix(
                    1.0,
                    hiReadGlow,
                    firstCycle * hiSelection * cluster
                );
                float introBackground = mix(
                    0.58,
                    0.72,
                    smoothstep(0.0, 0.34, phase)
                );
                float backgroundFade = mix(
                    introBackground,
                    0.06,
                    smoothstep(0.46, 0.60, phase)
                );
                float introVisibility = mix(backgroundFade, 1.0, hiSelection);
                vAlpha *= mix(1.0, introVisibility, firstCycle);
                vTint = clamp(
                    0.4
                        + transformed.y * 0.25
                        + aSeed * 0.28
                        + frontier * 0.24
                        + easedTrailGuide * (0.08 + ctaSelection * 0.12)
                        + firstCycle * cluster * (0.08 + hiDepthLight * 0.2),
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
                        + easedTrailGuide * (0.2 + ctaSelection * 0.35)
                        + firstCycle * cluster * (0.18 + hiDepthLight * 0.36)
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
                vAlpha = surfaceFade(transformed)
                    * pow(mesh, 1.15)
                    * (0.52 + frontier * 0.34);
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

    const hiLineVertices = [];
    function addHiSegment(x1, y1, z1, x2, y2, z2) {
        hiLineVertices.push(x1, y1, z1, x2, y2, z2);
    }

    function addHiStrokeLayer(z) {
        addHiSegment(-1.05, -1.05, z, -1.05, 1.05, z);
        addHiSegment(-0.1, -1.05, z, -0.1, 1.05, z);
        addHiSegment(-1.05, 0, z, -0.1, 0, z);
        addHiSegment(0.62, -1.05, z, 0.62, 0.43, z);
        addHiSegment(0.48, 0.92, z, 0.76, 0.92, z);
        addHiSegment(0.62, 0.78, z, 0.62, 1.06, z);
    }

    const hiFrontDepth = 0.36;
    const hiBackDepth = -0.36;
    addHiStrokeLayer(hiFrontDepth);
    addHiStrokeLayer(hiBackDepth);
    [
        [-1.05, -1.05],
        [-1.05, 1.05],
        [-0.1, -1.05],
        [-0.1, 1.05],
        [-1.05, 0],
        [-0.1, 0],
        [0.62, -1.05],
        [0.62, 0.43],
        [0.48, 0.92],
        [0.76, 0.92],
        [0.62, 0.78],
        [0.62, 1.06]
    ].forEach(([x, y]) => {
        addHiSegment(x, y, hiBackDepth, x, y, hiFrontDepth);
    });

    const hiLineGeometry = new THREE.BufferGeometry();
    hiLineGeometry.setAttribute(
        'position',
        new THREE.Float32BufferAttribute(hiLineVertices, 3)
    );
    const hiLineMaterial = new THREE.ShaderMaterial({
        uniforms: sharedUniforms,
        transparent: true,
        depthTest: false,
        depthWrite: false,
        blending: THREE.AdditiveBlending,
        vertexShader: `
            uniform float uTime;
            uniform vec3 uHiOffset;
            uniform float uHiScale;
            uniform float uHiDepth;
            uniform float uHiLineAlpha;
            varying float vAlpha;
            varying float vTint;

            ${waveHeightShader}

            void main() {
                vec2 center = formationCenter(uTime);
                float phase = cyclePhase(uTime);
                float linePulse = firstCycleMask(uTime)
                    * smoothstep(0.0, 0.04, phase)
                    * (1.0 - smoothstep(0.48, 0.64, phase));
                float hiYaw = 0.31 + sin(uTime * 0.2) * 0.02;
                float hiYawCos = cos(hiYaw);
                float hiYawSin = sin(hiYaw);
                vec3 hiLocal = vec3(
                    position.x * uHiScale,
                    position.y * uHiScale,
                    position.z * uHiScale * uHiDepth
                );
                vec3 hiTilted = vec3(
                    hiLocal.x * hiYawCos + hiLocal.z * hiYawSin,
                    hiLocal.y + hiLocal.z * 0.06,
                    -hiLocal.x * hiYawSin + hiLocal.z * hiYawCos
                );
                float depthLight = smoothstep(-0.3, 0.3, hiTilted.z);
                vec3 transformed = vec3(
                    center.x + uHiOffset.x + hiTilted.x,
                    waveHeight(center, uTime)
                        + 0.18
                        + uHiOffset.y
                        + hiTilted.y,
                    center.y + uHiOffset.z + hiTilted.z
                );
                float lineReadGlow = mix(
                    0.48,
                    0.82,
                    smoothstep(0.36, 0.52, phase)
                );
                vAlpha = linePulse
                    * uHiLineAlpha
                    * (0.55 + depthLight * 0.45)
                    * lineReadGlow;
                vTint = depthLight;
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
                    vec3(0.18, 0.58, 0.78),
                    vec3(0.82, 0.96, 1.0),
                    vTint
                );
                gl_FragColor = vec4(color, vAlpha);
            }
        `
    });
    const hiLineFrame = new THREE.LineSegments(hiLineGeometry, hiLineMaterial);
    hiLineFrame.renderOrder = 4;
    group.add(hiLineFrame);

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
                float introBackgroundFade = mix(
                    1.0,
                    0.05,
                    smoothstep(
                        0.46,
                        0.60,
                        cyclePhase(uTime)
                    )
                );
                vAlpha *= mix(
                    1.0,
                    introBackgroundFade,
                    firstCycleMask(uTime)
                );
                vTint = 0.35 + aSeed * 0.65;

                vec4 viewPosition = modelViewMatrix * vec4(transformed, 1.0);
                float perspective = clamp(8.0 / -viewPosition.z, 0.65, 1.45);
                gl_PointSize = (1.1 + aSeed * 2.2)
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
    let animationTime = 0;
    let lastFrameTime = performance.now();
    let heroIsVisible = true;
    let pointerX = 0;
    let pointerY = 0;
    let disposed = false;
    let introHasStarted = false;
    let introStartQueued = false;
    let cameraParallaxX = 0.16;
    let cameraParallaxY = 0.1;
    const cameraBasePosition = new THREE.Vector3();
    const cameraLookTarget = new THREE.Vector3();

    function startIntroTimeline() {
        if (disposed || introHasStarted || introStartQueued) {
            return;
        }

        introStartQueued = true;
        window.requestAnimationFrame(() => {
            window.requestAnimationFrame(() => {
                if (disposed) {
                    return;
                }
                hero.classList.add('hero-intro-sweeping');
                hero.style.setProperty('--hero-reveal-opacity', '0');
                introHasStarted = true;
                lastFrameTime = performance.now();
            });
        });
    }

    function handlePreloaderHidden() {
        startIntroTimeline();
    }

    function queueIntroTimeline() {
        const preloader = document.getElementById('preloader');
        if (!preloader || preloader.dataset.revealComplete === 'true') {
            startIntroTimeline();
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
        const phone = rect.width <= 480;
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
        cameraParallaxX = narrow ? 0.08 : 0.16;
        cameraParallaxY = narrow ? 0.05 : 0.1;
        camera.position.copy(cameraBasePosition);
        camera.lookAt(cameraLookTarget);
        camera.updateProjectionMatrix();
        group.position.set(
            narrow ? -0.55 : 0.35 + desktopEase * 2.75,
            narrow ? -2.7 : -2.6 + desktopEase * 0.15,
            -1.65
        );
        sharedUniforms.uHiScale.value = narrow
            ? (phone ? 0.68 : 0.72)
            : 0.84 + desktopEase * 0.16;
        sharedUniforms.uHiDepth.value = phone ? 0.66 : (narrow ? 0.78 : 1.04);
        sharedUniforms.uHiLineAlpha.value = phone ? 0.26 : (narrow ? 0.38 : 0.64);
        sharedUniforms.uHiOffset.value.set(
            phone ? -1.25 : (narrow ? -0.45 : -0.8 - desktopEase * 1.75),
            phone ? 5.22 : (narrow ? 4.9 : 1.55 + desktopEase * 0.45),
            0
        );
        sharedUniforms.uTrailBend.value.set(
            phone ? -2.25 : (narrow ? 2.4 : 0.25),
            phone ? 0.75 : (narrow ? 0.2 : 0),
            0
        );
        sharedUniforms.uTrailStrength.value = phone ? 0.12 : (narrow ? 0.72 : 1);

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
        if (introHasStarted) {
            const playbackRate = animationTime < introCycleDuration
                ? 1
                : ambientPlaybackRate;
            animationTime += Math.max(delta, 0) * playbackRate;
        }
        sharedUniforms.uTime.value = animationTime;

        if (
            introHasStarted
            && !hero.classList.contains('hero-intro-visible')
        ) {
            const revealStart = introCycleDuration * introRevealStartPhase;
            const revealDuration = introCycleDuration
                * (introRevealEndPhase - introRevealStartPhase);
            const rawRevealProgress = Math.min(
                Math.max(
                    (animationTime - revealStart) / revealDuration,
                    0
                ),
                1
            );
            const revealProgress = rawRevealProgress
                * rawRevealProgress
                * (3 - 2 * rawRevealProgress);
            hero.style.setProperty(
                '--hero-reveal-opacity',
                revealProgress.toFixed(3)
            );

            if (rawRevealProgress >= 1) {
                hero.classList.remove('hero-intro-sweeping');
                hero.classList.add('hero-intro-visible');
                hero.style.removeProperty('--hero-reveal-opacity');
            }
        }

        if (
            !hero.classList.contains('hero-cta-visible')
            && animationTime >= introCycleDuration * ctaRevealPhase
        ) {
            markHeroCtaVisible();
        }

        camera.position.x += (
            cameraBasePosition.x + pointerX * cameraParallaxX - camera.position.x
        ) * 0.04;
        camera.position.y += (
            cameraBasePosition.y - pointerY * cameraParallaxY - camera.position.y
        ) * 0.04;
        camera.lookAt(cameraLookTarget);

        group.rotation.y += (pointerX * 0.012 - group.rotation.y) * 0.03;
        group.rotation.x += (pointerY * 0.007 - group.rotation.x) * 0.03;

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

    function handlePointerMove(event) {
        if (event.pointerType === 'touch') {
            return;
        }
        pointerX = (event.clientX / Math.max(window.innerWidth, 1) - 0.5) * 2;
        pointerY = (event.clientY / Math.max(window.innerHeight, 1) - 0.5) * 2;
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
        document.removeEventListener('visibilitychange', handleVisibilityChange);
        document.removeEventListener(
            'portfolio:preloader-hidden',
            handlePreloaderHidden
        );
        surfaceGeometry.dispose();
        wireGeometry.dispose();
        hiLineGeometry.dispose();
        driftingGeometry.dispose();
        particleMaterial.dispose();
        wireMaterial.dispose();
        hiLineMaterial.dispose();
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
                stopRendering();
            }
        }, { threshold: 0.02 })
        : null;

    setSceneLayout();
    resizeObserver?.observe(hero);
    intersectionObserver?.observe(hero);
    window.addEventListener('pointermove', handlePointerMove, { passive: true });
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

    if (root.dataset.heroWaveBypassed === 'true') {
        hero.classList.remove('hero-intro-sweeping');
        hero.classList.add('hero-intro-visible');
        markHeroCtaVisible();
        hero.style.removeProperty('--hero-reveal-opacity');
        animationTime = introCycleDuration;
        introHasStarted = true;
        lastFrameTime = performance.now();
    } else {
        queueIntroTimeline();
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
