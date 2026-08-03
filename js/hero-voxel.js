const hero = document.getElementById('home');
const root = document.documentElement;
const reducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)');
const query = new URLSearchParams(window.location.search);
const forcedHero = query.get('hero');
const standaloneLoop = root.dataset.heroVoxelStandalone === 'true';
const debugEnabled = query.get('heroVoxelDebug') === '1';
const cameraTunerEnabled = query.get('heroVoxelCameraTuner') === '1';
const debugTime = Number.parseFloat(query.get('heroVoxelTime'));

root.classList.toggle('hero-voxel-camera-tuner', cameraTunerEnabled);

const MANIFEST_VERSION = 2;
const RESOLUTIONS = [8, 16, 32, 64, 128];
const PLAYBACK_SLOWDOWN = 1.1;
const CYCLE_DURATION = 8.12;
const DENSE_REVEAL_END = 0.66;
const DENSE_HOLD_END = 0.95;
const PRUNE_END = 4.8;
const CASCADE_START = DENSE_HOLD_END;
const FINAL_HOLD_START = 6.25;
const FINAL_DISSOLVE_START = 6.6;
const REDUCED_MOTION_TIME = 6.41;
const HANDOFF_NDC_X = 0.38;
const HANDOFF_NDC_Y = 0.02;
const HANDOFF_MOBILE_NDC_X = 0.18;
const HANDOFF_MOBILE_NDC_Y = -0.02;
const CAMERA_AZIMUTH = 43 * Math.PI / 180;
const CAMERA_ELEVATION = 26.5 * Math.PI / 180;
const CAMERA_FOV = 41.5;
const CAMERA_DISTANCE = 9.8;
const CAMERA_SCREEN_OFFSET = 0;
const CAMERA_VERTICAL_CENTER = -0.45;
const STANDALONE_ORBIT_LIMIT = 10 * Math.PI / 180;
const STANDALONE_ELEVATION_LIMIT = 85 * Math.PI / 180;
const WORLD_SCALE = 4.05;
const TRANSITIONS = [
    { parent: 8, child: 16 },
    { parent: 16, child: 32 },
    { parent: 32, child: 64 },
    { parent: 64, child: 128 }
];

let initializationQueued = false;
let initialized = false;
let activeDisposer = null;

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
    root.classList.remove('hero-wave-pending', 'hero-wave-active', 'hero-voxel-active');
    root.classList.add('hero-wave-fallback');
    hero?.classList.remove('hero-intro-sweeping', 'hero-wave-ready', 'hero-voxel-ready');
    hero?.style.removeProperty('--hero-reveal-opacity');
    hero?.classList.add('hero-intro-visible');
    markHeroCtaVisible();
}

async function startWaveFallback(error) {
    activeDisposer?.();
    activeDisposer = null;
    root.classList.remove('hero-voxel-active');
    hero?.classList.remove('hero-voxel-ready');
    if (standaloneLoop) {
        showStaticFallback();
        console.warn('The standalone voxel visualization could not initialize.', error);
        return;
    }
    if (error) {
        console.warn('The voxel hierarchy could not be initialized; using the wave hero.', error);
    }
    try {
        await import('./hero-wave.js');
    } catch (waveError) {
        showStaticFallback();
        console.warn('The wave hero could not initialize; using the static fallback.', waveError);
    }
}

function hasConstrainedDevice() {
    return Boolean(navigator.connection?.saveData)
        || (navigator.hardwareConcurrency && navigator.hardwareConcurrency <= 4)
        || (navigator.deviceMemory && navigator.deviceMemory <= 4);
}

function shouldUseMobileAsset(width) {
    return width <= 768 || hasConstrainedDevice();
}

function queueInitialization() {
    if (!hero || initializationQueued || initialized) {
        return;
    }

    initializationQueued = true;
    const initialize = () => {
        initializationQueued = false;
        initializeHeroVoxel();
    };

    if (window.matchMedia('(max-width: 767px)').matches) {
        window.requestAnimationFrame(initialize);
    } else if ('requestIdleCallback' in window) {
        window.requestIdleCallback(initialize, { timeout: 800 });
    } else {
        window.setTimeout(initialize, 80);
    }
}

function assert(condition, message) {
    if (!condition) {
        throw new Error(message);
    }
}

function validateRange(buffer, descriptor, type, components, count) {
    const bytesPerElement = type === 'uint8' ? 1 : type === 'uint16' ? 2 : 4;
    assert(descriptor?.type === type, `Expected ${type} hierarchy data.`);
    assert(Number.isInteger(descriptor.offset) && descriptor.offset >= 16, 'Invalid hierarchy byte offset.');
    assert(descriptor.offset % 4 === 0, 'Hierarchy byte offsets must be 4-byte aligned.');
    assert(descriptor.byteLength === count * components * bytesPerElement, 'Hierarchy array byte length is inconsistent.');
    assert(descriptor.offset + descriptor.byteLength <= buffer.byteLength, 'Hierarchy array is outside the binary payload.');
}

function createView(buffer, descriptor, count, components = 1) {
    validateRange(buffer, descriptor, descriptor.type, components, count);
    if (descriptor.type === 'uint8') {
        return new Uint8Array(buffer, descriptor.offset, count * components);
    }
    if (descriptor.type === 'uint16') {
        return new Uint16Array(buffer, descriptor.offset, count * components);
    }
    if (descriptor.type === 'uint32') {
        return new Uint32Array(buffer, descriptor.offset, count * components);
    }
    throw new Error(`Unsupported hierarchy type: ${descriptor.type}`);
}

async function loadHierarchy(variant) {
    const manifestUrl = `/assets/hero/voxel-hierarchy/hero-voxels-${variant}.json`;
    const manifestResponse = await fetch(manifestUrl, {
        cache: 'no-cache',
        credentials: 'same-origin'
    });
    assert(manifestResponse.ok, `Hierarchy manifest request failed with ${manifestResponse.status}.`);
    const manifest = await manifestResponse.json();
    assert(manifest.schema === 'hero-voxel-hierarchy', 'Unknown hero hierarchy schema.');
    assert(manifest.version === MANIFEST_VERSION, `Unsupported hero hierarchy version ${manifest.version}.`);
    assert(manifest.variant === variant, 'The loaded hierarchy variant is inconsistent.');
    assert(manifest.axisOrder === 'xyz', 'The hero hierarchy must use xyz coordinate order.');
    assert(JSON.stringify(manifest.productionResolutions) === JSON.stringify(RESOLUTIONS), 'The hero hierarchy levels are incomplete.');
    assert(manifest.binary?.littleEndian === true, 'The hero hierarchy must be little-endian.');

    const binaryUrl = new URL(manifest.binary.url, manifestResponse.url);
    const binaryResponse = await fetch(binaryUrl, {
        cache: 'no-cache',
        credentials: 'same-origin'
    });
    assert(binaryResponse.ok, `Hierarchy binary request failed with ${binaryResponse.status}.`);
    const buffer = await binaryResponse.arrayBuffer();
    assert(buffer.byteLength === manifest.binary.byteLength, 'Hierarchy binary length is inconsistent.');
    assert(new Uint8Array(new Uint16Array([1]).buffer)[0] === 1, 'This browser is not little-endian.');

    const header = new DataView(buffer, 0, 16);
    assert(
        header.getUint8(0) === 72
            && header.getUint8(1) === 86
            && header.getUint8(2) === 79
            && header.getUint8(3) === 88,
        'Hierarchy binary magic is invalid.'
    );
    assert(header.getUint32(4, true) === MANIFEST_VERSION, 'Hierarchy binary version is invalid.');
    assert(header.getUint32(8, true) === RESOLUTIONS.length, 'Hierarchy binary level count is invalid.');
    assert(header.getUint32(12, true) === 16, 'Hierarchy binary header length is invalid.');

    const levels = new Map();
    for (const resolution of RESOLUTIONS) {
        const level = manifest.levels[String(resolution)];
        assert(level && Number.isInteger(level.count) && level.count > 0, `Hierarchy level ${resolution} is missing.`);
        validateRange(buffer, level.indices, 'uint16', 3, level.count);
        const parsed = {
            count: level.count,
            coordinates: createView(buffer, level.indices, level.count, 3),
            masks: null,
            demoParents: null,
            transition: level.transition
        };
        if (resolution !== RESOLUTIONS[RESOLUTIONS.length - 1]) {
            validateRange(buffer, level.childMasks, 'uint8', 1, level.count);
            validateRange(buffer, level.demoParents, 'uint32', 1, level.demoParents.count);
            parsed.masks = createView(buffer, level.childMasks, level.count);
            parsed.demoParents = createView(buffer, level.demoParents, level.demoParents.count);
        }
        levels.set(resolution, parsed);
    }

    return { buffer, levels, manifest, variant };
}

function createInstancedGeometry(THREE, baseGeometry, coordinates, activeStates = null) {
    const geometry = new THREE.InstancedBufferGeometry();
    geometry.setIndex(baseGeometry.index);
    geometry.setAttribute('position', baseGeometry.getAttribute('position'));
    geometry.setAttribute('normal', baseGeometry.getAttribute('normal'));
    geometry.setAttribute('aCoordinate', new THREE.InstancedBufferAttribute(coordinates, 3, false));
    if (activeStates) {
        geometry.setAttribute('aActive', new THREE.InstancedBufferAttribute(activeStates, 1, false));
    }
    geometry.instanceCount = coordinates.length / 3;
    return geometry;
}

function createVoxelMaterial(THREE, mode, color, opacity) {
    return new THREE.ShaderMaterial({
        transparent: true,
        depthTest: true,
        depthWrite: true,
        side: THREE.FrontSide,
        uniforms: {
            uParentResolution: { value: 8 },
            uChildResolution: { value: 16 },
            uColor: { value: new THREE.Color(color) },
            uFineColor: { value: new THREE.Color(0x8faab2) },
            uRejectedColor: { value: new THREE.Color(0x535c60) },
            uGlobalOpacity: { value: 1 },
            uFinalLevel: { value: 0 },
            uMode: { value: mode },
            uOpacity: { value: opacity },
            uTime: { value: 0 },
            uTransition: { value: 0 },
            uWorldScale: { value: WORLD_SCALE }
        },
        vertexShader: `
            attribute vec3 aCoordinate;
            attribute float aActive;
            uniform float uParentResolution;
            uniform float uChildResolution;
            uniform float uGlobalOpacity;
            uniform float uFinalLevel;
            uniform float uMode;
            uniform float uOpacity;
            uniform float uTime;
            uniform float uTransition;
            uniform float uWorldScale;
            varying float vAlpha;
            varying float vActive;
            varying float vGradient;
            varying vec3 vNormal;

            float ramp(float edge0, float edge1, float value) {
                return clamp((value - edge0) / max(edge1 - edge0, 0.0001), 0.0, 1.0);
            }

            float coordinateHash(vec3 coordinate) {
                return fract(sin(dot(coordinate, vec3(17.13, 91.71, 43.37))) * 43758.5453);
            }

            vec3 voxelCenter(vec3 coordinate, float resolution) {
                return ((coordinate + 0.5) / resolution - 0.5) * uWorldScale;
            }

            float voxelFill(float resolution) {
                float refinement = clamp(log2(resolution / 8.0) / 4.0, 0.0, 1.0);
                return mix(0.88, 0.72, refinement);
            }

            float cornerRank(vec3 coordinate, float resolution) {
                float maximumDistance = max((resolution - 1.0) * 3.0, 1.0);
                vec3 distanceFromUpperLeft = vec3(
                    coordinate.x,
                    resolution - 1.0 - coordinate.y,
                    resolution - 1.0 - coordinate.z
                );
                return clamp(dot(distanceFromUpperLeft, vec3(1.0)) / maximumDistance, 0.0, 1.0);
            }

            float octantRank(vec3 coordinate) {
                vec3 offset = mod(coordinate, 2.0);
                return (offset.x + (1.0 - offset.y) + (1.0 - offset.z)) / 3.0;
            }

            // Schedule refinement recursively. A coarse 8^3 BFS chooses the
            // region, then each octree offset supplies the local BFS order for
            // the next group of eight children.
            float transitionStart(vec3 coordinate, float resolution) {
                vec3 rootCoordinate = floor(coordinate / max(resolution / 8.0, 1.0));
                // This is the dense classifier's exact linear corner rank in
                // cascade-clock units, so a coarse cell never waits after reveal.
                float start = cornerRank(rootCoordinate, 8.0) * 0.648
                    + coordinateHash(rootCoordinate) * 0.012;
                if (resolution > 8.5) {
                    vec3 coordinate16 = floor(coordinate / max(resolution / 16.0, 1.0));
                    start += 0.035
                        + octantRank(coordinate16) * 0.012
                        + coordinateHash(coordinate16) * 0.002;
                }
                if (resolution > 16.5) {
                    vec3 coordinate32 = floor(coordinate / max(resolution / 32.0, 1.0));
                    start += 0.035
                        + octantRank(coordinate32) * 0.012
                        + coordinateHash(coordinate32) * 0.002;
                }
                if (resolution > 32.5) {
                    vec3 coordinate64 = floor(coordinate / max(resolution / 64.0, 1.0));
                    start += 0.035
                        + octantRank(coordinate64) * 0.012
                        + coordinateHash(coordinate64) * 0.002;
                }
                return start;
            }

            float transitionPhase(vec3 coordinate, float resolution) {
                float start = transitionStart(coordinate, resolution);
                return ramp(start, start + 0.16, uTransition);
            }

            void main() {
                bool isParent = uMode < 0.5;
                vec3 parentCoordinate = isParent ? aCoordinate : floor(aCoordinate * 0.5);
                float localTransition = transitionPhase(parentCoordinate, uParentResolution);
                float movement = ramp(0.02, 0.48, localTransition);
                float parentCell = uWorldScale / uParentResolution;
                float childCell = uWorldScale / uChildResolution;
                vec3 parentCenter = voxelCenter(parentCoordinate, uParentResolution);
                vec3 childCenter = voxelCenter(aCoordinate, uChildResolution);
                vec3 center = parentCenter;
                float size = parentCell * voxelFill(uParentResolution);
                float alpha = uOpacity;

                if (isParent) {
                    size *= 1.0 - ramp(0.06, 0.54, localTransition) * 0.84;
                    alpha *= 1.0 - ramp(0.08, 0.56, localTransition);
                } else {
                    center = mix(parentCenter, childCenter, movement);
                    size = mix(parentCell * 0.14, childCell * voxelFill(uChildResolution), movement);
                    float candidateReveal = ramp(0.0, 0.14, localTransition);
                    float rejection = (1.0 - aActive) * ramp(0.5, 0.94, localTransition);
                    size *= 1.0 - rejection;
                    alpha *= candidateReveal * mix(0.22, 1.0, aActive) * (1.0 - rejection);
                    // Once a finest child has settled, let a broad traveling
                    // ripple move it by roughly one cell instead of freezing the
                    // completed silhouette into a static sculpture.
                    float settled = uFinalLevel * aActive * ramp(0.78, 1.0, localTransition);
                    float wavePhase = aCoordinate.x * 0.085
                        + aCoordinate.z * 0.05
                        - uTime * 1.05;
                    float wave = sin(wavePhase) * 0.78
                        + sin(wavePhase * 0.52 + aCoordinate.y * 0.022) * 0.22;
                    center.y += wave * childCell * 1.05 * settled;
                    center.x += cos(wavePhase * 0.58) * childCell * 0.16 * settled;
                    if (uFinalLevel < 0.5 && aActive > 0.5) {
                        float childRefinement = transitionPhase(aCoordinate, uChildResolution);
                        size *= 1.0 - ramp(0.06, 0.54, childRefinement) * 0.84;
                        alpha *= 1.0 - ramp(0.08, 0.56, childRefinement);
                    }
                }

                vec4 worldPosition = modelMatrix * vec4(center + position * size, 1.0);
                vAlpha = alpha * uGlobalOpacity;
                vActive = isParent ? 1.0 : aActive;
                float displayResolution = isParent ? uParentResolution : uChildResolution;
                vec3 normalizedCoordinate = aCoordinate / max(displayResolution - 1.0, 1.0);
                vGradient = dot(normalizedCoordinate, vec3(0.25, 0.55, 0.20));
                vNormal = normalize(mat3(modelMatrix) * normal);
                gl_Position = projectionMatrix * viewMatrix * worldPosition;
            }
        `,
        fragmentShader: `
            uniform vec3 uColor;
            uniform vec3 uFineColor;
            uniform vec3 uRejectedColor;
            uniform float uParentResolution;
            uniform float uChildResolution;
            uniform float uMode;
            varying float vAlpha;
            varying float vActive;
            varying float vGradient;
            varying vec3 vNormal;

            void main() {
                if (vAlpha <= 0.002) {
                    discard;
                }
                float displayResolution = uMode < 0.5 ? uParentResolution : uChildResolution;
                float refinement = clamp(log2(displayResolution / 16.0) / 3.0, 0.0, 1.0);
                vec3 activeColor = mix(uColor, uFineColor, refinement * 0.42);
                vec3 baseColor = mix(uRejectedColor, activeColor, vActive);
                vec3 gradientShift = mix(
                    vec3(-0.018, 0.0, 0.018),
                    vec3(0.018, 0.008, -0.014),
                    vGradient
                );
                baseColor = clamp(baseColor + gradientShift * 0.65, 0.0, 1.0);
                float light = 0.68 + 0.32 * max(dot(normalize(vNormal), normalize(vec3(0.28, 0.62, 0.74))), 0.0);
                gl_FragColor = vec4(baseColor * light, vAlpha);
            }
        `
    });
}

function createDenseQuery(THREE, baseGeometry, activeCoordinates, resolution) {
    const count = resolution * resolution * resolution;
    const coordinates = new Uint16Array(count * 3);
    const active = new Uint8Array(count);
    let cursor = 0;
    for (let x = 0; x < resolution; x += 1) {
        for (let y = 0; y < resolution; y += 1) {
            for (let z = 0; z < resolution; z += 1) {
                coordinates[cursor * 3] = x;
                coordinates[cursor * 3 + 1] = y;
                coordinates[cursor * 3 + 2] = z;
                cursor += 1;
            }
        }
    }
    for (let index = 0; index < activeCoordinates.length; index += 3) {
        const code = activeCoordinates[index] * resolution * resolution
            + activeCoordinates[index + 1] * resolution
            + activeCoordinates[index + 2];
        active[code] = 1;
    }

    const geometry = createInstancedGeometry(THREE, baseGeometry, coordinates, active);
    const material = new THREE.ShaderMaterial({
        transparent: true,
        depthTest: true,
        depthWrite: true,
        uniforms: {
            uCascadeProgress: { value: 0 },
            uPruneProgress: { value: 0 },
            uResolution: { value: resolution },
            uReveal: { value: 0 },
            uWorldScale: { value: WORLD_SCALE }
        },
        vertexShader: `
            attribute vec3 aCoordinate;
            attribute float aActive;
            uniform float uCascadeProgress;
            uniform float uPruneProgress;
            uniform float uResolution;
            uniform float uReveal;
            uniform float uWorldScale;
            varying float vAlpha;
            varying float vConfirmed;
            varying float vGradient;
            varying vec3 vNormal;

            float eased(float edge0, float edge1, float value) {
                float t = clamp((value - edge0) / max(edge1 - edge0, 0.0001), 0.0, 1.0);
                return t * t * t * (t * (t * 6.0 - 15.0) + 10.0);
            }

            float ramp(float edge0, float edge1, float value) {
                return clamp((value - edge0) / max(edge1 - edge0, 0.0001), 0.0, 1.0);
            }

            float voxelHash(vec3 coordinate) {
                return fract(sin(dot(coordinate, vec3(17.13, 91.71, 43.37))) * 43758.5453);
            }

            void main() {
                float maximumDistance = max((uResolution - 1.0) * 3.0, 1.0);
                float bfsDistance = aCoordinate.x
                    + (uResolution - 1.0 - aCoordinate.y)
                    + (uResolution - 1.0 - aCoordinate.z);
                float rank = bfsDistance / maximumDistance + voxelHash(aCoordinate) * 0.018;
                float visited = 1.0 - eased(uPruneProgress - 0.055, uPruneProgress + 0.055, rank);
                visited *= eased(0.0, 0.03, uPruneProgress);
                float rejected = (1.0 - aActive) * visited;
                float cellSize = uWorldScale / uResolution;
                float size = cellSize * 0.92 * mix(1.0, 0.04, eased(0.0, 1.0, rejected));
                float cornerDistance = bfsDistance / maximumDistance;
                float cascadeStart = cornerDistance * 0.648 + voxelHash(aCoordinate) * 0.012;
                float localCascade = ramp(cascadeStart, cascadeStart + 0.16, uCascadeProgress);
                float parentScale = 1.0 - ramp(0.06, 0.54, localCascade) * 0.84;
                float parentAlpha = 1.0 - ramp(0.08, 0.56, localCascade);
                size *= mix(1.0, parentScale, aActive);
                vec3 center = ((aCoordinate + 0.5) / uResolution - 0.5) * uWorldScale;
                vec4 worldPosition = modelMatrix * vec4(center + position * size, 1.0);
                vConfirmed = aActive * visited;
                vAlpha = uReveal
                    * mix(0.42, 0.68, vConfirmed)
                    * (1.0 - eased(0.0, 1.0, rejected))
                    * mix(1.0, parentAlpha, aActive);
                vec3 normalizedCoordinate = aCoordinate / max(uResolution - 1.0, 1.0);
                vGradient = dot(normalizedCoordinate, vec3(0.25, 0.55, 0.20));
                vNormal = normalize(mat3(modelMatrix) * normal);
                gl_Position = projectionMatrix * viewMatrix * worldPosition;
            }
        `,
        fragmentShader: `
            varying float vAlpha;
            varying float vConfirmed;
            varying float vGradient;
            varying vec3 vNormal;

            void main() {
                if (vAlpha <= 0.01) {
                    discard;
                }
                vec3 neutral = vec3(0.285, 0.305, 0.32);
                vec3 activeColor = vec3(0.38, 0.47, 0.52);
                vec3 gradientShift = mix(
                    vec3(-0.018, 0.0, 0.018),
                    vec3(0.018, 0.008, -0.014),
                    vGradient
                );
                vec3 baseColor = mix(neutral, activeColor, vConfirmed);
                baseColor = clamp(baseColor + gradientShift * 0.65, 0.0, 1.0);
                float light = 0.68 + 0.32 * max(dot(normalize(vNormal), normalize(vec3(0.3, 0.72, 0.64))), 0.0);
                gl_FragColor = vec4(baseColor * light, vAlpha);
            }
        `
    });
    const mesh = new THREE.Mesh(geometry, material);
    mesh.frustumCulled = false;
    mesh.renderOrder = 1;
    return { geometry, material, mesh };
}

function deterministicUnit(index, salt) {
    let value = Math.imul(index + 1, 0x9e3779b1) ^ Math.imul(salt + 1, 0x85ebca6b);
    value ^= value >>> 16;
    value = Math.imul(value, 0x7feb352d);
    value ^= value >>> 15;
    return (value >>> 0) / 4294967295;
}

function createAmbientParticles(count) {
    const positions = new Float32Array(count * 2);
    const seeds = new Float32Array(count);
    for (let index = 0; index < count; index += 1) {
        positions[index * 2] = deterministicUnit(index, 11);
        positions[index * 2 + 1] = deterministicUnit(index, 29);
        seeds[index] = deterministicUnit(index, 47);
    }

    const canvas = document.createElement('canvas');
    canvas.className = 'hero-voxel-particle-canvas';
    canvas.setAttribute('aria-hidden', 'true');
    canvas.setAttribute('role', 'presentation');
    const context = canvas.getContext('2d', { alpha: true });
    assert(context, 'The ambient particle canvas could not be initialized.');
    let width = 1;
    let height = 1;
    let pixelRatio = 1;
    let wasVisible = false;

    function resize(cssWidth, cssHeight, nextPixelRatio) {
        width = Math.max(cssWidth, 1);
        height = Math.max(cssHeight, 1);
        pixelRatio = nextPixelRatio;
        canvas.width = Math.max(1, Math.round(width * pixelRatio));
        canvas.height = Math.max(1, Math.round(height * pixelRatio));
        context.setTransform(pixelRatio, 0, 0, pixelRatio, 0, 0);
    }

    function draw(time, opacity, handoffProgress, targetX, targetY) {
        if (opacity <= 0.001) {
            if (wasVisible) {
                context.clearRect(0, 0, width, height);
                wasVisible = false;
            }
            return;
        }
        context.clearRect(0, 0, width, height);
        wasVisible = true;
        const gather = smoothstep(0.04, 0.82, handoffProgress);
        context.fillStyle = '#6d8188';
        for (let index = 0; index < count; index += 1) {
            const seed = seeds[index];
            const angle = seed * Math.PI * 2 + time * (0.025 + seed * 0.018);
            const drift = 1.5 + seed * 2.2;
            const baseX = positions[index * 2] * width + Math.cos(angle) * drift;
            const baseY = positions[index * 2 + 1] * height + Math.sin(angle) * drift;
            const x = baseX + (targetX - baseX) * gather;
            const y = baseY + (targetY - baseY) * gather;
            const radius = (0.44 + seed * 0.58) * (1 - gather * 0.18);
            const pulse = 0.9 + Math.sin(time * 0.38 + seed * 17) * 0.1;
            context.globalAlpha = opacity * (0.26 + seed * 0.28) * pulse;
            context.beginPath();
            context.arc(x, y, radius, 0, Math.PI * 2);
            context.fill();
        }
        context.globalAlpha = 1;
    }

    return { canvas, draw, resize };
}

function createCandidateData(parentLevel, expectedActiveChildren) {
    const candidateCount = parentLevel.count * 8;
    const coordinates = new Uint16Array(candidateCount * 3);
    const active = new Uint8Array(candidateCount);
    let cursor = 0;
    let activeCount = 0;
    for (let parentIndex = 0; parentIndex < parentLevel.count; parentIndex += 1) {
        const px = parentLevel.coordinates[parentIndex * 3];
        const py = parentLevel.coordinates[parentIndex * 3 + 1];
        const pz = parentLevel.coordinates[parentIndex * 3 + 2];
        const mask = parentLevel.masks[parentIndex];
        for (let bit = 0; bit < 8; bit += 1) {
            coordinates[cursor * 3] = px * 2 + ((bit >> 2) & 1);
            coordinates[cursor * 3 + 1] = py * 2 + ((bit >> 1) & 1);
            coordinates[cursor * 3 + 2] = pz * 2 + (bit & 1);
            active[cursor] = (mask >> bit) & 1;
            activeCount += active[cursor];
            cursor += 1;
        }
    }
    assert(cursor === candidateCount, 'Candidate child count does not match the parent count.');
    assert(activeCount === expectedActiveChildren, 'Child masks do not reproduce the active child count.');
    return { coordinates, active };
}

function createDebugOverlay(variant, manifest) {
    if (!debugEnabled) {
        return null;
    }

    const overlay = document.createElement('aside');
    overlay.className = 'hero-voxel-debug';
    overlay.setAttribute('aria-label', 'Hero voxel debug controls');
    overlay.innerHTML = `
        <output class="hero-voxel-debug__stats"></output>
        <div class="hero-voxel-debug__controls">
            <button type="button">Pause</button>
            <input type="range" min="0" max="${CYCLE_DURATION}" step="0.01" value="0" aria-label="Scrub hero voxel timeline">
        </div>
        <small>${variant} · ${manifest.binary.byteLength.toLocaleString()} bytes</small>
    `;
    hero.append(overlay);
    return {
        overlay,
        output: overlay.querySelector('output'),
        button: overlay.querySelector('button'),
        range: overlay.querySelector('input'),
        manual: Number.isFinite(debugTime),
        paused: Number.isFinite(debugTime),
        time: Number.isFinite(debugTime) ? Math.max(0, Math.min(CYCLE_DURATION, debugTime)) : 0,
        lastUpdate: 0
    };
}

function createCameraTuner() {
    if (!cameraTunerEnabled) {
        return null;
    }

    const defaults = {
        azimuth: CAMERA_AZIMUTH * 180 / Math.PI,
        elevation: CAMERA_ELEVATION * 180 / Math.PI,
        fov: CAMERA_FOV,
        distance: CAMERA_DISTANCE,
        screenOffset: CAMERA_SCREEN_OFFSET,
        verticalCenter: CAMERA_VERTICAL_CENTER
    };
    const overlay = document.createElement('aside');
    overlay.setAttribute('aria-label', 'Voxel hero camera tuner');
    const overlayPositionStyles = standaloneLoop
        ? ['position:relative', 'align-self:center']
        : ['position:absolute', 'top:1rem', 'right:1rem'];
    overlay.style.cssText = [
        ...overlayPositionStyles,
        'z-index:22',
        'width:min(320px,calc(100% - 2rem))',
        'padding:14px',
        'border:1px solid rgba(255,255,255,.16)',
        'background:rgba(16,16,17,.9)',
        'color:rgba(240,241,243,.9)',
        'font:11px/1.35 IBM Plex Mono,monospace',
        'backdrop-filter:blur(12px)'
    ].join(';');
    overlay.innerHTML = `
        <form>
            <strong style="display:block;margin-bottom:10px;font-size:12px">VOXEL CAMERA TUNER</strong>
            ${Object.entries(defaults).map(([name, value]) => {
                const ranges = {
                    azimuth: standaloneLoop ? [-180, 180, 0.5] : [-90, 90, 0.5],
                    elevation: standaloneLoop ? [-20, 85, 0.5] : [5, 85, 0.5],
                    fov: standaloneLoop ? [15, 85, 0.5] : [20, 70, 0.5],
                    distance: standaloneLoop ? [6, 36, 0.1] : [8, 28, 0.1],
                    screenOffset: standaloneLoop ? [-7, 7, 0.05] : [-1, 7, 0.05],
                    verticalCenter: standaloneLoop ? [-6, 6, 0.05] : [-4, 4, 0.05]
                };
                const [minimum, maximum, step] = ranges[name];
                return `
                    <label style="display:grid;grid-template-columns:1fr auto;gap:4px;margin:8px 0">
                        <span>${name}</span>
                        <output data-output="${name}">${value}</output>
                        <input style="grid-column:1/-1;width:100%" name="${name}" type="range"
                            min="${minimum}" max="${maximum}" step="${step}" value="${value}">
                    </label>
                `;
            }).join('')}
            <div style="display:flex;gap:8px;margin-top:12px">
                <button type="submit" style="flex:1;padding:7px;border:1px solid rgba(255,255,255,.3);background:#858b8c;color:#101011">SUBMIT</button>
                <button type="reset" style="padding:7px;border:1px solid rgba(255,255,255,.2);background:transparent;color:inherit">RESET</button>
            </div>
            <p data-status style="min-height:2.7em;margin:9px 0 0;color:rgba(210,214,215,.72)">Adjust live, then submit.</p>
        </form>
    `;
    (standaloneLoop ? document.body : hero).append(overlay);
    const form = overlay.querySelector('form');
    const inputs = Object.fromEntries(
        Object.keys(defaults).map((name) => [name, form.elements.namedItem(name)])
    );
    const outputs = Object.fromEntries(
        Object.keys(defaults).map((name) => [name, overlay.querySelector(`[data-output="${name}"]`)])
    );
    const read = () => Object.fromEntries(
        Object.entries(inputs).map(([name, input]) => [name, Number.parseFloat(input.value)])
    );
    const updateOutputs = () => {
        const values = read();
        for (const [name, output] of Object.entries(outputs)) {
            output.value = values[name].toFixed(name === 'azimuth' || name === 'elevation' ? 1 : 2);
        }
        return values;
    };
    updateOutputs();
    return {
        defaults,
        form,
        inputs,
        overlay,
        read,
        reset: form.querySelector('[type="reset"]'),
        status: overlay.querySelector('[data-status]'),
        updateOutputs,
        values: { ...defaults }
    };
}

function smoothstep(minimum, maximum, value) {
    const normalized = Math.max(0, Math.min(1, (value - minimum) / Math.max(maximum - minimum, 0.0001)));
    return normalized * normalized * (3 - 2 * normalized);
}

function createHeroVoxel(THREE, hierarchy, compact) {
    const width = Math.max(hero.clientWidth, 1);
    const lowSpec = hasConstrainedDevice();
    const assetCamera = hierarchy.manifest.visibilityCulling?.camera;
    const pixelRatioCap = lowSpec ? 1 : compact ? 1.25 : 1.5;
    const scene = new THREE.Scene();
    const camera = new THREE.PerspectiveCamera(40, 1, 0.1, 50);
    const renderer = new THREE.WebGLRenderer({
        alpha: true,
        antialias: false,
        depth: true,
        powerPreference: lowSpec ? 'low-power' : 'high-performance',
        premultipliedAlpha: false
    });
    renderer.setClearColor(0x101011, 0);
    renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, pixelRatioCap));
    renderer.domElement.className = 'hero-voxel-canvas';
    renderer.domElement.setAttribute('aria-hidden', 'true');
    renderer.domElement.setAttribute('role', 'presentation');
    hero.prepend(renderer.domElement);

    const ambientParticles = createAmbientParticles(
        lowSpec ? 60 : compact ? 96 : 160
    );
    hero.prepend(ambientParticles.canvas);

    const group = new THREE.Group();
    scene.add(group);
    const baseGeometry = new THREE.BoxGeometry(1, 1, 1);
    const candidateMaterials = [];
    const candidateMeshes = [];
    const instancedGeometries = [];

    for (let transitionIndex = 0; transitionIndex < TRANSITIONS.length; transitionIndex += 1) {
        const transition = TRANSITIONS[transitionIndex];
        const parentLevel = hierarchy.levels.get(transition.parent);
        const childLevel = hierarchy.levels.get(transition.child);
        const candidates = createCandidateData(parentLevel, childLevel.count);
        const candidateGeometry = createInstancedGeometry(
            THREE,
            baseGeometry,
            candidates.coordinates,
            candidates.active
        );
        const candidateMaterial = createVoxelMaterial(THREE, 1, 0x647d88, 0.66);
        candidateMaterial.uniforms.uParentResolution.value = transition.parent;
        candidateMaterial.uniforms.uChildResolution.value = transition.child;
        candidateMaterial.uniforms.uFinalLevel.value = transitionIndex === TRANSITIONS.length - 1 ? 1 : 0;
        const candidateMesh = new THREE.Mesh(candidateGeometry, candidateMaterial);
        candidateMesh.frustumCulled = false;
        candidateMesh.renderOrder = 3 + transitionIndex;
        candidateMesh.visible = false;
        group.add(candidateMesh);
        candidateMaterials.push(candidateMaterial);
        candidateMeshes.push(candidateMesh);
        instancedGeometries.push(candidateGeometry);
    }
    const voxelMaterials = candidateMaterials;

    const dense = createDenseQuery(
        THREE,
        baseGeometry,
        hierarchy.levels.get(RESOLUTIONS[0]).coordinates,
        RESOLUTIONS[0]
    );
    group.add(dense.mesh);

    const boundsBoxGeometry = new THREE.BoxGeometry(WORLD_SCALE, WORLD_SCALE, WORLD_SCALE);
    const boundsGeometry = new THREE.EdgesGeometry(boundsBoxGeometry);
    const boundsMaterial = new THREE.LineBasicMaterial({
        color: 0x61747b,
        transparent: true,
        opacity: 0.055,
        depthTest: true,
        depthWrite: false
    });
    const bounds = new THREE.LineSegments(boundsGeometry, boundsMaterial);
    bounds.renderOrder = 0;
    group.add(bounds);

    const debug = createDebugOverlay(hierarchy.variant, hierarchy.manifest);
    if (debug) {
        debug.range.value = String(debug.time);
        debug.button.textContent = debug.paused ? 'Play' : 'Pause';
    }
    const cameraTuner = createCameraTuner();

    let animationFrame = 0;
    let elapsedTime = 0;
    let lastFrameTime = performance.now();
    let heroIsVisible = true;
    let parentFrameVisible = !standaloneLoop
        || root.dataset.heroVoxelParentVisible !== 'false';
    let disposed = false;
    let handoffStarted = false;
    let handoffCompleted = false;
    let handoffPromise = null;
    let handoffError = null;
    let pointerX = 0;
    let pointerY = 0;
    let pointerTargetX = 0;
    let pointerTargetY = 0;
    let cameraParallaxX = 0.28;
    let cameraParallaxY = 0.16;
    let fpsAverage = 60;
    const cameraBasePosition = new THREE.Vector3();
    const cameraBaseOffset = new THREE.Vector3();
    const cameraLookTarget = new THREE.Vector3();
    const groupBasePosition = new THREE.Vector3();
    const handoffTargetPosition = new THREE.Vector3();
    const handoffViewDirection = new THREE.Vector3();
    const handoffPlane = new THREE.Plane();
    const handoffRaycaster = new THREE.Raycaster();
    const handoffNdc = new THREE.Vector2();
    let handoffScreenX = 0;
    let handoffScreenY = 0;
    let handoffTargetNdcX = HANDOFF_NDC_X;
    let handoffTargetNdcY = HANDOFF_NDC_Y;
    let cameraOrbitRadius = 1;
    let cameraOrbitAzimuth = 0;
    let cameraOrbitElevation = 0;
    const standalonePlaybackButton = standaloneLoop
        ? document.querySelector('[data-voxel-playback]')
        : null;
    let standalonePaused = false;
    const timelineState = {
        activeIndex: -1,
        ambientOpacity: 0,
        globalOpacity: 1,
        handoffProgress: 0,
        transitionProgress: 0
    };

    function setUniforms(progress, opacity) {
        for (const material of voxelMaterials) {
            material.uniforms.uTransition.value = progress;
            material.uniforms.uGlobalOpacity.value = opacity;
        }
    }

    function updateTimeline(cycleTime) {
        const handoffProgress = cycleTime <= FINAL_DISSOLVE_START
            ? 0
            : Math.min(
                (cycleTime - FINAL_DISSOLVE_START)
                    / (CYCLE_DURATION - FINAL_DISSOLVE_START),
                1
            );
        const handoffMovement = standaloneLoop
            ? 0
            : smoothstep(0.02, 0.86, handoffProgress);
        group.position.lerpVectors(groupBasePosition, handoffTargetPosition, handoffMovement);
        group.scale.setScalar(1 - handoffMovement * 0.78);
        dense.mesh.visible = cycleTime < FINAL_HOLD_START;
        dense.material.uniforms.uReveal.value = smoothstep(0, DENSE_REVEAL_END, cycleTime);
        dense.material.uniforms.uPruneProgress.value = cycleTime <= DENSE_HOLD_END
            ? 0
            : Math.min((cycleTime - DENSE_HOLD_END) / (PRUNE_END - DENSE_HOLD_END) * 1.12, 1.12);
        const cascadeProgress = cycleTime <= CASCADE_START
            ? 0
            : Math.min((cycleTime - CASCADE_START) / (FINAL_HOLD_START - CASCADE_START), 1);
        dense.material.uniforms.uCascadeProgress.value = cascadeProgress;
        for (let index = 0; index < TRANSITIONS.length; index += 1) {
            candidateMeshes[index].visible = false;
        }

        let activeIndex = -1;
        let transitionProgress = 0;
        let globalOpacity = 1;
        if (cycleTime >= CASCADE_START && cycleTime < FINAL_HOLD_START) {
            transitionProgress = cascadeProgress;
            activeIndex = transitionProgress < 0.04
                ? 0
                : transitionProgress < 0.075
                    ? 1
                    : transitionProgress < 0.11
                        ? 2
                        : 3;
            for (const candidateMesh of candidateMeshes) {
                candidateMesh.visible = true;
            }
            setUniforms(transitionProgress, 1);
        } else if (cycleTime >= FINAL_HOLD_START) {
            activeIndex = TRANSITIONS.length - 1;
            transitionProgress = 1;
            globalOpacity = cycleTime <= FINAL_DISSOLVE_START
                ? 1
                : 1 - smoothstep(FINAL_DISSOLVE_START, CYCLE_DURATION, cycleTime);
            candidateMeshes[activeIndex].visible = true;
            setUniforms(1, globalOpacity);
        }

        boundsMaterial.opacity = cycleTime < DENSE_REVEAL_END
            ? smoothstep(0, DENSE_REVEAL_END, cycleTime) * 0.055
            : cycleTime > FINAL_DISSOLVE_START
                ? globalOpacity * 0.055
                : 0.055;
        const ambientOpacity = smoothstep(0, DENSE_REVEAL_END, cycleTime)
            * (1 - smoothstep(0.62, 1, handoffProgress));
        timelineState.activeIndex = activeIndex;
        timelineState.ambientOpacity = ambientOpacity;
        timelineState.transitionProgress = transitionProgress;
        timelineState.globalOpacity = globalOpacity;
        timelineState.handoffProgress = handoffProgress;
        return timelineState;
    }

    function updateDebug(frameTime, cycleTime, timeline, delta) {
        if (!debug || frameTime - debug.lastUpdate < 160) {
            return;
        }
        debug.lastUpdate = frameTime;
        fpsAverage += ((delta > 0 ? 1 / delta : 60) - fpsAverage) * 0.18;
        let phase = cycleTime < DENSE_HOLD_END
            ? 'occupied 8^3 volume'
            : cycleTime < CASCADE_START
                ? 'corner BFS prune 8^3'
                : cycleTime < PRUNE_END
                        ? 'prune-to-refinement handoff'
                    : cycleTime < FINAL_HOLD_START
                        ? 'recursive octree BFS cascade'
                        : cycleTime < FINAL_DISSOLVE_START
                            ? 'settled 128^3 visible-surface ripple'
                            : 'final voxel dissolve';
        let detail = '512 occupied query cells';
        if (timeline.activeIndex >= 0) {
            const transition = TRANSITIONS[timeline.activeIndex];
            const parentLevel = hierarchy.levels.get(transition.parent);
            const retainedCount = hierarchy.levels.get(transition.child).count;
            const candidateCount = parentLevel.count * 8;
            detail = `${Math.round(timeline.transitionProgress * 100)}% cascade | deepest front ${transition.parent}->${transition.child} | candidates ${candidateCount.toLocaleString()} | retained ${retainedCount.toLocaleString()}`;
        }
        debug.output.textContent = `${phase}\n${detail}\n${renderer.info.render.calls} calls | ${fpsAverage.toFixed(0)} fps`;
        if (!debug.manual) {
            debug.range.value = cycleTime.toFixed(2);
        }
    }

    function setSceneLayout() {
        const rect = hero.getBoundingClientRect();
        const narrow = rect.width <= 768;
        const progress = Math.min(Math.max((rect.width - 768) / 500, 0), 1);
        const desktopEase = progress * progress * (3 - 2 * progress);
        const pixelRatio = Math.min(window.devicePixelRatio || 1, pixelRatioCap);
        renderer.setPixelRatio(pixelRatio);
        renderer.setSize(Math.max(rect.width, 1), Math.max(rect.height, 1), false);
        ambientParticles.resize(rect.width, rect.height, Math.min(pixelRatio, 1));
        camera.aspect = Math.max(rect.width, 1) / Math.max(rect.height, 1);
        const tuning = cameraTuner?.values;
        const manifestFov = Number(assetCamera?.fovDegrees);
        const manifestAzimuth = Number(assetCamera?.azimuthDegrees);
        const manifestElevation = Number(assetCamera?.elevationDegrees);
        const manifestDistance = Number(assetCamera?.distance);
        const manifestScreenOffset = Number(assetCamera?.screenOffset);
        const manifestVerticalCenter = Number(assetCamera?.verticalCenter);
        camera.fov = tuning?.fov
            ?? (Number.isFinite(manifestFov) ? manifestFov : narrow ? 48 : CAMERA_FOV);
        const verticalCenter = tuning?.verticalCenter
            ?? (Number.isFinite(manifestVerticalCenter)
                ? manifestVerticalCenter
                : narrow ? -0.3 : CAMERA_VERTICAL_CENTER);
        const rightOffset = narrow ? 0.55 : 0.95 + desktopEase * 1.45;
        const wideScreenOffset = Number.isFinite(manifestScreenOffset)
            ? manifestScreenOffset
            : CAMERA_SCREEN_OFFSET;
        const responsiveScreenOffset = rightOffset * Math.SQRT2
            + desktopEase * (wideScreenOffset - 2.4 * Math.SQRT2);
        // The standalone Note owns the complete frame, so its submitted
        // view-plane offset must not pass through the old right-biased home
        // hero interpolation. This keeps tuner and iframe framing identical.
        const screenOffset = tuning?.screenOffset
            ?? (standaloneLoop
                ? (Number.isFinite(manifestScreenOffset) ? manifestScreenOffset : 0)
                : responsiveScreenOffset);
        const azimuth = (tuning?.azimuth
            ?? (Number.isFinite(manifestAzimuth)
                ? manifestAzimuth
                : CAMERA_AZIMUTH * 180 / Math.PI)) * Math.PI / 180;
        const elevation = (tuning?.elevation
            ?? (Number.isFinite(manifestElevation)
                ? manifestElevation
                : CAMERA_ELEVATION * 180 / Math.PI)) * Math.PI / 180;
        // Translate along the current camera's normalized view-plane right axis
        // so tuning the angle does not alter the model's screen-space height.
        group.position.set(
            Math.cos(azimuth) * screenOffset,
            verticalCenter,
            -Math.sin(azimuth) * screenOffset
        );
        group.rotation.set(0, 0, 0);
        const cameraDistance = tuning?.distance
            ?? (Number.isFinite(manifestDistance)
                ? manifestDistance
                : narrow ? 18 : CAMERA_DISTANCE);
        const horizontalDistance = cameraDistance * Math.cos(elevation);
        // Keep the camera independent from the model's view-plane offset.
        cameraBasePosition.set(
            horizontalDistance * Math.sin(azimuth),
            cameraDistance * Math.sin(elevation),
            horizontalDistance * Math.cos(azimuth)
        );
        cameraLookTarget.set(0, verticalCenter, 0);
        // Standalone pointer motion is a bounded orbit around the submitted
        // look target. Deriving the spherical basis from the authored camera
        // position preserves its exact center-frame composition.
        cameraBaseOffset.subVectors(cameraBasePosition, cameraLookTarget);
        cameraOrbitRadius = Math.max(cameraBaseOffset.length(), 0.001);
        cameraOrbitAzimuth = Math.atan2(cameraBaseOffset.x, cameraBaseOffset.z);
        cameraOrbitElevation = Math.asin(
            Math.max(-1, Math.min(1, cameraBaseOffset.y / cameraOrbitRadius))
        );
        cameraParallaxX = narrow ? 0.06 : 0.12;
        cameraParallaxY = narrow ? 0.04 : 0.07;
        camera.position.copy(cameraBasePosition);
        camera.lookAt(cameraLookTarget);
        camera.updateProjectionMatrix();
        camera.updateMatrixWorld(true);
        groupBasePosition.copy(group.position);
        const targetNdcX = narrow ? HANDOFF_MOBILE_NDC_X : HANDOFF_NDC_X;
        const targetNdcY = narrow ? HANDOFF_MOBILE_NDC_Y : HANDOFF_NDC_Y;
        handoffTargetNdcX = targetNdcX;
        handoffTargetNdcY = targetNdcY;
        handoffNdc.set(targetNdcX, targetNdcY);
        camera.getWorldDirection(handoffViewDirection);
        handoffPlane.setFromNormalAndCoplanarPoint(handoffViewDirection, groupBasePosition);
        handoffRaycaster.setFromCamera(handoffNdc, camera);
        if (!handoffRaycaster.ray.intersectPlane(handoffPlane, handoffTargetPosition)) {
            handoffTargetPosition.copy(groupBasePosition);
        }
        handoffScreenX = (targetNdcX * 0.5 + 0.5) * rect.width;
        handoffScreenY = (0.5 - targetNdcY * 0.5) * rect.height;
        if (reducedMotion.matches || cameraTuner) {
            renderer.render(scene, camera);
        }
    }

    function handleCameraTunerInput() {
        cameraTuner.values = cameraTuner.updateOutputs();
        setSceneLayout();
    }

    function handleCameraTunerReset() {
        window.requestAnimationFrame(() => {
            if (!disposed) {
                cameraTuner.values = cameraTuner.updateOutputs();
                setSceneLayout();
            }
        });
    }

    async function handleCameraTunerSubmit(event) {
        event.preventDefault();
        cameraTuner.values = cameraTuner.updateOutputs();
        const selection = {
            ...cameraTuner.values,
            viewport: {
                width: Math.round(hero.clientWidth),
                height: Math.round(hero.clientHeight)
            }
        };
        cameraTuner.status.textContent = 'Saving selection...';
        try {
            const response = await fetch('/__hero-camera-tuner/selection', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(selection)
            });
            if (!response.ok) {
                throw new Error(`Camera tuner server returned ${response.status}.`);
            }
            cameraTuner.status.textContent = 'Saved. Return to Codex and say submitted.';
        } catch (error) {
            await navigator.clipboard?.writeText(JSON.stringify(selection));
            cameraTuner.status.textContent = 'Local save failed; values copied to clipboard.';
            console.warn('Camera tuner selection could not be saved locally.', error);
        }
    }

    function beginWaveHandoff() {
        if (disposed || handoffStarted) {
            return;
        }
        handoffStarted = true;
        root.dataset.heroWaveBypassed = 'true';
        root.dataset.heroWaveHandoff = 'true';
        root.dataset.heroWaveHandoffX = handoffTargetNdcX.toFixed(4);
        root.dataset.heroWaveHandoffY = handoffTargetNdcY.toFixed(4);
        handoffPromise = import('./hero-wave.js').catch((error) => {
            handoffError = error;
        });
    }

    async function completeWaveHandoff() {
        if (disposed || handoffCompleted) {
            return;
        }
        handoffCompleted = true;
        beginWaveHandoff();
        await handoffPromise;
        root.classList.remove('hero-voxel-active');
        hero.classList.remove('hero-voxel-ready');
        dispose();
        if (handoffError) {
            showStaticFallback();
            console.warn('The ambient wave could not initialize after the voxel intro.', handoffError);
        }
    }

    function render(frameTime) {
        animationFrame = 0;
        if (disposed || !heroIsVisible || !parentFrameVisible || document.hidden) {
            return;
        }

        const delta = Math.min(Math.max((frameTime - lastFrameTime) / 1000, 0), 0.05);
        lastFrameTime = frameTime;
        if (debug?.manual) {
            elapsedTime = debug.time;
        } else if (!debug?.paused && !standalonePaused) {
            elapsedTime += delta;
        }
        const keepVoxelLoop = Boolean(debug || cameraTuner || standaloneLoop);
        const cycleTime = reducedMotion.matches
            ? REDUCED_MOTION_TIME
            : keepVoxelLoop
                ? elapsedTime % CYCLE_DURATION
                : Math.min(elapsedTime, CYCLE_DURATION);
        const timeline = updateTimeline(cycleTime);
        const animationTime = cycleTime / PLAYBACK_SLOWDOWN;
        ambientParticles.draw(
            animationTime,
            timeline.ambientOpacity,
            standaloneLoop ? 0 : timeline.handoffProgress,
            handoffScreenX,
            handoffScreenY
        );
        for (const material of voxelMaterials) {
            material.uniforms.uTime.value = animationTime;
        }
        if (
            !reducedMotion.matches
            && !keepVoxelLoop
            && cycleTime >= FINAL_DISSOLVE_START
        ) {
            beginWaveHandoff();
        }

        if (!reducedMotion.matches) {
            const pointerEase = 1 - Math.exp(-delta * 4.4);
            const cameraEase = 1 - Math.exp(-delta * 3.2);
            pointerX += (pointerTargetX - pointerX) * pointerEase;
            pointerY += (pointerTargetY - pointerY) * pointerEase;
            if (standaloneLoop) {
                const pointerMagnitude = Math.hypot(pointerX, pointerY);
                const pointerScale = pointerMagnitude > 1 ? 1 / pointerMagnitude : 1;
                const orbitX = pointerX * pointerScale;
                const orbitY = pointerY * pointerScale;
                const orbitAzimuth = cameraOrbitAzimuth + orbitX * STANDALONE_ORBIT_LIMIT;
                const orbitElevation = Math.max(
                    -STANDALONE_ELEVATION_LIMIT,
                    Math.min(
                        STANDALONE_ELEVATION_LIMIT,
                        cameraOrbitElevation - orbitY * STANDALONE_ORBIT_LIMIT
                    )
                );
                const orbitHorizontal = cameraOrbitRadius * Math.cos(orbitElevation);
                camera.position.set(
                    cameraLookTarget.x + orbitHorizontal * Math.sin(orbitAzimuth),
                    cameraLookTarget.y + cameraOrbitRadius * Math.sin(orbitElevation),
                    cameraLookTarget.z + orbitHorizontal * Math.cos(orbitAzimuth)
                );
            } else {
                camera.position.x += (
                    cameraBasePosition.x + pointerX * cameraParallaxX - camera.position.x
                ) * cameraEase;
                camera.position.y += (
                    cameraBasePosition.y - pointerY * cameraParallaxY - camera.position.y
                ) * cameraEase;
            }
            camera.lookAt(cameraLookTarget);
        }

        renderer.render(scene, camera);
        updateDebug(frameTime, cycleTime, timeline, delta);
        if (!reducedMotion.matches && !keepVoxelLoop && elapsedTime >= CYCLE_DURATION) {
            completeWaveHandoff();
            return;
        }
        if (!reducedMotion.matches && !standalonePaused) {
            animationFrame = window.requestAnimationFrame(render);
        }
    }

    function startRendering() {
        if (
            disposed
            || animationFrame
            || !heroIsVisible
            || !parentFrameVisible
            || document.hidden
            || standalonePaused
        ) {
            return;
        }
        lastFrameTime = performance.now();
        animationFrame = window.requestAnimationFrame(render);
    }

    function stopRendering() {
        if (animationFrame) {
            window.cancelAnimationFrame(animationFrame);
            animationFrame = 0;
        }
    }

    function handlePointerMove(event) {
        if (!heroIsVisible || !parentFrameVisible || reducedMotion.matches) {
            return;
        }
        const rect = hero.getBoundingClientRect();
        const normalizedX = ((event.clientX - rect.left) / Math.max(rect.width, 1) - 0.5) * 2;
        const normalizedY = ((event.clientY - rect.top) / Math.max(rect.height, 1) - 0.5) * 2;
        const influence = event.pointerType === 'touch' ? 0.5 : 1;
        pointerTargetX = Math.max(-1, Math.min(1, normalizedX)) * influence;
        pointerTargetY = Math.max(-1, Math.min(1, normalizedY)) * influence;
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

    function handleDebugButton() {
        debug.paused = !debug.paused;
        debug.manual = false;
        debug.button.textContent = debug.paused ? 'Play' : 'Pause';
        if (!debug.paused) {
            startRendering();
        }
    }

    function handleDebugInput() {
        debug.manual = true;
        debug.paused = true;
        debug.time = Number.parseFloat(debug.range.value);
        debug.button.textContent = 'Play';
        startRendering();
    }

    function handleStandalonePlayback() {
        standalonePaused = !standalonePaused;
        if (standalonePlaybackButton) {
            standalonePlaybackButton.textContent = standalonePaused ? 'Play' : 'Pause';
            standalonePlaybackButton.setAttribute('aria-pressed', String(standalonePaused));
        }
        if (standalonePaused) {
            stopRendering();
        } else {
            startRendering();
        }
    }

    function handleParentFrameVisibility(event) {
        parentFrameVisible = event.detail?.visible !== false;
        if (parentFrameVisible) {
            startRendering();
        } else {
            pointerTargetX = 0;
            pointerTargetY = 0;
            stopRendering();
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
        window.removeEventListener('pagehide', dispose);
        document.documentElement.removeEventListener('mouseleave', resetPointerTarget);
        document.removeEventListener('visibilitychange', handleVisibilityChange);
        debug?.button.removeEventListener('click', handleDebugButton);
        debug?.range.removeEventListener('input', handleDebugInput);
        debug?.overlay.remove();
        cameraTuner?.form.removeEventListener('input', handleCameraTunerInput);
        cameraTuner?.form.removeEventListener('submit', handleCameraTunerSubmit);
        cameraTuner?.reset.removeEventListener('click', handleCameraTunerReset);
        standalonePlaybackButton?.removeEventListener('click', handleStandalonePlayback);
        window.removeEventListener('hero-voxel-parent-visibility', handleParentFrameVisibility);
        cameraTuner?.overlay.remove();
        dense.geometry.dispose();
        dense.material.dispose();
        ambientParticles.canvas.remove();
        boundsBoxGeometry.dispose();
        boundsGeometry.dispose();
        boundsMaterial.dispose();
        for (const geometry of instancedGeometries) {
            geometry.dispose();
        }
        baseGeometry.dispose();
        for (const material of candidateMaterials) {
            material.dispose();
        }
        renderer.dispose();
        renderer.domElement.remove();
    }

    const resizeObserver = 'ResizeObserver' in window ? new ResizeObserver(setSceneLayout) : null;
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
    const initialTime = reducedMotion.matches ? REDUCED_MOTION_TIME : debug?.time ?? 0;
    const initialTimeline = updateTimeline(initialTime);
    ambientParticles.draw(
        initialTime,
        initialTimeline.ambientOpacity,
        initialTimeline.handoffProgress,
        handoffScreenX,
        handoffScreenY
    );
    renderer.render(scene, camera);
    resizeObserver?.observe(hero);
    intersectionObserver?.observe(hero);
    window.addEventListener('pointermove', handlePointerMove, { passive: true });
    window.addEventListener('pointerdown', handlePointerMove, { passive: true });
    window.addEventListener('pointerup', resetPointerTarget, { passive: true });
    window.addEventListener('pointercancel', resetPointerTarget, { passive: true });
    window.addEventListener('pagehide', dispose, { once: true });
    document.documentElement.addEventListener('mouseleave', resetPointerTarget);
    document.addEventListener('visibilitychange', handleVisibilityChange);
    debug?.button.addEventListener('click', handleDebugButton);
    debug?.range.addEventListener('input', handleDebugInput);
    cameraTuner?.form.addEventListener('input', handleCameraTunerInput);
    cameraTuner?.form.addEventListener('submit', handleCameraTunerSubmit);
    cameraTuner?.reset.addEventListener('click', handleCameraTunerReset);
    if (standalonePlaybackButton) {
        standalonePlaybackButton.disabled = reducedMotion.matches;
        standalonePlaybackButton.textContent = reducedMotion.matches ? 'Static' : 'Pause';
        standalonePlaybackButton.setAttribute('aria-pressed', 'false');
        standalonePlaybackButton.addEventListener('click', handleStandalonePlayback);
    }
    if (standaloneLoop) {
        window.addEventListener('hero-voxel-parent-visibility', handleParentFrameVisibility);
    }
    renderer.domElement.addEventListener('webglcontextlost', (event) => {
        event.preventDefault();
        dispose();
        showStaticFallback();
    }, { once: true });

    root.classList.remove('hero-wave-fallback', 'hero-wave-pending');
    root.classList.add('hero-wave-active', 'hero-voxel-active');
    hero.classList.remove('hero-intro-sweeping');
    hero.classList.add('hero-intro-visible');
    hero.style.removeProperty('--hero-reveal-opacity');
    markHeroCtaVisible();
    window.requestAnimationFrame(() => {
        if (!disposed) {
            hero.classList.add('hero-voxel-ready');
        }
    });
    startRendering();
    return dispose;
}

async function initializeHeroVoxel() {
    if (!hero || initialized) {
        return;
    }
    initialized = true;
    const compact = shouldUseMobileAsset(Math.max(hero.clientWidth, 1));
    let hierarchy;
    try {
        hierarchy = await loadHierarchy(compact ? 'mobile' : 'desktop');
    } catch (error) {
        initialized = false;
        await startWaveFallback(error);
        return;
    }

    try {
        const THREE = await import('/vendor/three/build/three.module.js');
        activeDisposer = createHeroVoxel(THREE, hierarchy, compact);
    } catch (error) {
        initialized = false;
        showStaticFallback();
        console.warn('WebGL could not initialize the voxel hero; using the static fallback.', error);
    }
}

if (!hero) {
    // The portfolio hero is not present on project or blog pages.
} else if (forcedHero === 'wave') {
    startWaveFallback();
} else if (hero.querySelector('.hero-content')) {
    queueInitialization();
} else {
    hero.addEventListener('portfolio:section-ready', queueInitialization, { once: true });
    document.addEventListener('portfolio:core-ready', queueInitialization, { once: true });
}
