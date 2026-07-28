import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';
import { OBJLoader } from 'three/addons/loaders/OBJLoader.js';
import { MTLLoader } from 'three/addons/loaders/MTLLoader.js';
import { FBXLoader } from 'three/addons/loaders/FBXLoader.js';
import { PLYLoader } from 'three/addons/loaders/PLYLoader.js';
import { STLLoader } from 'three/addons/loaders/STLLoader.js';
import { RGBELoader } from 'three/addons/loaders/RGBELoader.js';
import { EXRLoader } from 'three/addons/loaders/EXRLoader.js';
import { DRACOLoader } from 'three/addons/loaders/DRACOLoader.js';

const ENVIRONMENT_URLS = {
    env1: '/assets/viewer/spruit-sunrise.hdr',
    env2: '/assets/viewer/aircraft-workshop.hdr',
    env3: '/assets/viewer/lebombo.hdr'
};

const DEFAULT_ENVIRONMENT_PRESET = 'forest';
const ENVIRONMENT_PRESETS = {
    studio: {
        id: 'studio',
        label: 'Studio',
        url: '/assets/viewer/blender-studio-lights/studio.exr',
        environmentIntensity: 1,
        backgroundIntensity: 0.82,
        rotation: 0
    },
    interior: {
        id: 'interior',
        label: 'Interior',
        url: '/assets/viewer/blender-studio-lights/interior.exr',
        environmentIntensity: 1.08,
        backgroundIntensity: 0.84,
        rotation: 0.15
    },
    city: {
        id: 'city',
        label: 'City',
        url: '/assets/viewer/blender-studio-lights/city.exr',
        environmentIntensity: 0.92,
        backgroundIntensity: 0.78,
        rotation: -0.3
    },
    sunrise: {
        id: 'sunrise',
        label: 'Sunrise',
        url: '/assets/viewer/blender-studio-lights/sunrise.exr',
        environmentIntensity: 0.96,
        backgroundIntensity: 0.78,
        rotation: 0.05
    },
    forest: {
        id: 'forest',
        label: 'Forest',
        url: '/assets/viewer/blender-studio-lights/forest.exr',
        environmentIntensity: 1.04,
        backgroundIntensity: 0.82,
        rotation: -0.2
    },
    courtyard: {
        id: 'courtyard',
        label: 'Courtyard',
        url: '/assets/viewer/blender-studio-lights/courtyard.exr',
        environmentIntensity: 1,
        backgroundIntensity: 0.8,
        rotation: 0.22
    },
    env1: {
        id: 'env1',
        label: 'Sunrise HDR',
        url: ENVIRONMENT_URLS.env1,
        environmentIntensity: 1,
        backgroundIntensity: 0.9,
        rotation: 0
    },
    env2: {
        id: 'env2',
        label: 'Workshop HDR',
        url: ENVIRONMENT_URLS.env2,
        environmentIntensity: 0.95,
        backgroundIntensity: 0.88,
        rotation: 0
    },
    env3: {
        id: 'env3',
        label: 'Lebombo HDR',
        url: ENVIRONMENT_URLS.env3,
        environmentIntensity: 1,
        backgroundIntensity: 0.9,
        rotation: 0
    }
};

const SUPPORTED_MODEL_EXTENSIONS = new Set(['glb', 'gltf', 'obj', 'fbx', 'ply', 'stl']);
const COMPANION_EXTENSIONS = new Set([
    'mtl',
    'bin',
    'png',
    'jpg',
    'jpeg',
    'webp',
    'avif',
    'bmp',
    'gif',
    'hdr',
    'exr',
    'tga',
    'ktx2'
]);
const TEXTURE_PROPERTIES = [
    ['map', 'Base color'],
    ['normalMap', 'Normal'],
    ['roughnessMap', 'Roughness'],
    ['metalnessMap', 'Metalness'],
    ['aoMap', 'AO'],
    ['emissiveMap', 'Emissive'],
    ['alphaMap', 'Alpha'],
    ['bumpMap', 'Bump'],
    ['displacementMap', 'Displace']
];
const HISTORY_DB_NAME = 'simple-model-viewer-history';
const HISTORY_DB_VERSION = 1;
const HISTORY_STORE_NAME = 'records';
const HISTORY_LIMIT = 5;
const HISTORY_BYTE_LIMIT = 100 * 1024 * 1024;
const DEFAULT_BACKGROUND_COLOR = '#070707';
const DEFAULT_CAMERA_DISTANCE = 3.5;
const DEFAULT_WIREFRAME_COLOR = 0x111111;
const MAX_TEXTURE_ANISOTROPY_CAP = 8;
const PBR_DISPLAY_LOOK_SATURATION = 1.02;
const PBR_DISPLAY_LOOK_CONTRAST = 1.01;
const PBR_DISPLAY_LOOK_PIVOT = 0.5;
const QUAD_EDGE_NORMAL_DOT = Math.cos(THREE.MathUtils.degToRad(3));
const QUAD_EDGE_LENGTH_RATIO = 0.98;

function extensionFromPath(path) {
    const clean = String(path || '').split(/[?#]/)[0];
    const match = clean.match(/\.([a-z0-9]+)$/i);
    return (match?.[1] || '').toLowerCase();
}

function normalizePath(path) {
    const parts = String(path || '')
        .replaceAll('\\', '/')
        .split('/')
        .filter((part) => part && part !== '.');
    const normalized = [];
    parts.forEach((part) => {
        if (part === '..') {
            normalized.pop();
        } else {
            normalized.push(part);
        }
    });
    return normalized.join('/');
}

function dirname(path) {
    const normalized = normalizePath(path);
    const slash = normalized.lastIndexOf('/');
    return slash >= 0 ? normalized.slice(0, slash) : '';
}

function basename(path) {
    const normalized = normalizePath(path);
    const slash = normalized.lastIndexOf('/');
    return slash >= 0 ? normalized.slice(slash + 1) : normalized;
}

function joinPath(base, path) {
    return normalizePath(base ? `${base}/${path}` : path);
}

function pathWithoutSearch(path) {
    return String(path || '').split(/[?#]/)[0];
}

function escapeHtml(value) {
    return String(value).replace(/[&<>"']/g, (character) => ({
        '&': '&amp;',
        '<': '&lt;',
        '>': '&gt;',
        '"': '&quot;',
        "'": '&#39;'
    })[character]);
}

function sanitizeFilenameSegment(value, fallback = 'model') {
    const normalized = String(value || '')
        .trim()
        .replace(/\.[a-z0-9]+$/i, '')
        .replace(/[^a-z0-9-_]+/gi, '-')
        .replace(/^-+|-+$/g, '')
        .toLowerCase();
    return normalized || fallback;
}

function formatBytes(value) {
    const bytes = Number(value);
    if (!Number.isFinite(bytes) || bytes < 0) return '0 B';
    if (bytes < 1024) return `${bytes} B`;
    const units = ['KB', 'MB', 'GB', 'TB'];
    let size = bytes / 1024;
    let unit = units.shift();
    while (size >= 1024 && units.length) {
        size /= 1024;
        unit = units.shift();
    }
    return `${size >= 10 ? size.toFixed(0) : size.toFixed(1)} ${unit}`;
}

function formatCount(value) {
    const count = Number(value);
    if (!Number.isFinite(count) || count <= 0) return '0';
    if (count >= 1000000) return `${(count / 1000000).toFixed(1)}M`;
    if (count >= 1000) return `${(count / 1000).toFixed(1)}k`;
    return String(Math.round(count));
}

function environmentRotationShift(rotationRadians, width) {
    if (!width) return 0;
    const turns = ((rotationRadians / (Math.PI * 2)) % 1 + 1) % 1;
    return Math.round(turns * width) % width;
}

function environmentTextureChannelCount(texture) {
    if (texture?.format === THREE.RedFormat) return 1;
    if (texture?.format === THREE.RGFormat) return 2;
    if (texture?.format === THREE.RGBFormat) return 3;
    return 4;
}

function createRotatedEnvironmentTexture(texture, rotationRadians) {
    const image = texture?.image;
    const data = image?.data;
    const width = image?.width || 0;
    const height = image?.height || 0;
    const shift = environmentRotationShift(rotationRadians, width);
    if (!data || !width || !height || shift === 0) return texture;
    const channels = environmentTextureChannelCount(texture);
    const rotatedData = new data.constructor(data.length);
    const rowStride = width * channels;
    const shiftStride = shift * channels;
    for (let y = 0; y < height; y += 1) {
        const rowOffset = y * rowStride;
        for (let x = 0; x < width; x += 1) {
            const sourceOffset = rowOffset + ((x * channels + shiftStride) % rowStride);
            const targetOffset = rowOffset + x * channels;
            for (let channel = 0; channel < channels; channel += 1) {
                rotatedData[targetOffset + channel] = data[sourceOffset + channel];
            }
        }
    }
    const rotated = new THREE.DataTexture(rotatedData, width, height, texture.format, texture.type);
    rotated.mapping = THREE.EquirectangularReflectionMapping;
    rotated.colorSpace = texture.colorSpace;
    rotated.flipY = texture.flipY;
    rotated.generateMipmaps = texture.generateMipmaps;
    rotated.magFilter = texture.magFilter;
    rotated.minFilter = texture.minFilter;
    rotated.wrapS = texture.wrapS;
    rotated.wrapT = texture.wrapT;
    rotated.userData.smvRotatedEnvironment = true;
    rotated.needsUpdate = true;
    return rotated;
}

function previewColorFromEnvironment(texture, u, v) {
    const image = texture?.image;
    const data = image?.data;
    const width = image?.width || 0;
    const height = image?.height || 0;
    if (!data || !width || !height) return [0.05, 0.05, 0.05];
    const x = Math.max(0, Math.min(width - 1, Math.floor(((u % 1 + 1) % 1) * width)));
    const y = Math.max(0, Math.min(height - 1, Math.floor(THREE.MathUtils.clamp(v, 0, 1) * height)));
    const channels = environmentTextureChannelCount(texture);
    const offset = (y * width + x) * channels;
    const read = (channel) => {
        const value = data[offset + Math.min(channel, channels - 1)] || 0;
        if (texture.type === THREE.HalfFloatType || data instanceof Uint16Array) {
            return THREE.DataUtils.fromHalfFloat(value);
        }
        return value;
    };
    const r = read(0);
    const g = channels > 1 ? read(1) : r;
    const b = channels > 2 ? read(2) : r;
    return [r, g, b];
}

function previewDisplayValue(linearValue) {
    const mapped = 1 - Math.exp(-Math.max(0, linearValue) * 0.85);
    return Math.round(Math.pow(THREE.MathUtils.clamp(mapped, 0, 1), 1 / 2.2) * 255);
}

function drawEnvironmentPreviewCanvas(canvas, texture, preset, degrees = 0) {
    if (!canvas || !texture) return;
    const size = Math.max(24, Math.floor(canvas.width || canvas.clientWidth || 56));
    if (canvas.width !== size || canvas.height !== size) {
        canvas.width = size;
        canvas.height = size;
    }
    const context = canvas.getContext('2d');
    if (!context) return;
    const imageData = context.createImageData(size, size);
    const target = imageData.data;
    const center = (size - 1) / 2;
    const radius = size * 0.46;
    const rotation = THREE.MathUtils.degToRad(degrees) + Number(preset.rotation || 0);
    const cos = Math.cos(rotation);
    const sin = Math.sin(rotation);
    for (let py = 0; py < size; py += 1) {
        for (let px = 0; px < size; px += 1) {
            const dx = (px - center) / radius;
            const dy = (py - center) / radius;
            const distanceSq = dx * dx + dy * dy;
            const offset = (py * size + px) * 4;
            if (distanceSq > 1) {
                target[offset + 3] = 0;
                continue;
            }
            const z = Math.sqrt(Math.max(0, 1 - distanceSq));
            const dirX = dx * cos + z * sin;
            const dirY = -dy;
            const dirZ = z * cos - dx * sin;
            const u = Math.atan2(dirX, dirZ) / (Math.PI * 2) + 0.5;
            const v = Math.acos(THREE.MathUtils.clamp(dirY, -1, 1)) / Math.PI;
            const shade = 0.62 + 0.38 * Math.max(0, z);
            const [r, g, b] = previewColorFromEnvironment(texture, u, v);
            target[offset] = previewDisplayValue(r * shade);
            target[offset + 1] = previewDisplayValue(g * shade);
            target[offset + 2] = previewDisplayValue(b * shade);
            target[offset + 3] = 255;
        }
    }
    context.putImageData(imageData, 0, 0);
}

function materialArray(material) {
    if (!material) return [];
    return Array.isArray(material) ? material.filter(Boolean) : [material];
}

function isTextEntryElement(element) {
    if (!(element instanceof Element)) return false;
    return ['INPUT', 'TEXTAREA', 'SELECT'].includes(element.tagName) || element.isContentEditable;
}

function isExternalUrl(value) {
    return /^(?:https?:)?\/\//i.test(String(value || '')) || /^(?:blob|data):/i.test(String(value || ''));
}

function filePathFor(file) {
    return normalizePath(file.webkitRelativePath || file.relativePath || file.name || 'model');
}

function createHistoryId() {
    return `${Date.now().toString(36)}-${Math.random().toString(36).slice(2, 9)}`;
}

function requestToPromise(request) {
    return new Promise((resolve, reject) => {
        request.onsuccess = () => resolve(request.result);
        request.onerror = () => reject(request.error || new Error('IndexedDB request failed'));
    });
}

function openHistoryDatabase() {
    if (!('indexedDB' in window)) return Promise.resolve(null);
    return new Promise((resolve, reject) => {
        const request = indexedDB.open(HISTORY_DB_NAME, HISTORY_DB_VERSION);
        request.onupgradeneeded = () => {
            const database = request.result;
            if (!database.objectStoreNames.contains(HISTORY_STORE_NAME)) {
                database.createObjectStore(HISTORY_STORE_NAME, { keyPath: 'id' });
            }
        };
        request.onsuccess = () => resolve(request.result);
        request.onerror = () => reject(request.error || new Error('Could not open viewer history'));
    });
}

async function readHistoryRecords() {
    const database = await openHistoryDatabase();
    if (!database) return [];
    const transaction = database.transaction(HISTORY_STORE_NAME, 'readonly');
    const records = await requestToPromise(transaction.objectStore(HISTORY_STORE_NAME).getAll());
    return (records || []).sort((a, b) => (b.createdAt || 0) - (a.createdAt || 0));
}

async function writeHistoryRecords(records) {
    const database = await openHistoryDatabase();
    if (!database) return;
    await new Promise((resolve, reject) => {
        const transaction = database.transaction(HISTORY_STORE_NAME, 'readwrite');
        const store = transaction.objectStore(HISTORY_STORE_NAME);
        store.clear();
        records.forEach((record) => store.put(record));
        transaction.oncomplete = () => resolve();
        transaction.onerror = () => reject(transaction.error || new Error('Could not write viewer history'));
        transaction.onabort = () => reject(transaction.error || new Error('Viewer history write was aborted'));
    });
}

function getTextureSource(texture) {
    const image = texture?.source?.data || texture?.image || null;
    return image?.currentSrc || image?.src || texture?.userData?.source || '';
}

function isDrawableTextureImage(image) {
    return Boolean(image && (
        (typeof HTMLImageElement !== 'undefined' && image instanceof HTMLImageElement)
        || (typeof HTMLCanvasElement !== 'undefined' && image instanceof HTMLCanvasElement)
        || (typeof HTMLVideoElement !== 'undefined' && image instanceof HTMLVideoElement)
        || (typeof ImageBitmap !== 'undefined' && image instanceof ImageBitmap)
        || (typeof OffscreenCanvas !== 'undefined' && image instanceof OffscreenCanvas)
    ));
}

function drawFallbackTexture(canvas, entry, size = 96) {
    canvas.width = size;
    canvas.height = size;
    const context = canvas.getContext('2d');
    const tile = Math.max(8, Math.round(size / 8));
    for (let y = 0; y < size; y += tile) {
        for (let x = 0; x < size; x += tile) {
            context.fillStyle = ((x / tile + y / tile) % 2) ? '#171817' : '#2b2d2b';
            context.fillRect(x, y, tile, tile);
        }
    }
    context.fillStyle = 'rgba(250, 249, 245, 0.86)';
    context.font = `700 ${Math.max(10, Math.round(size / 10))}px Inter, sans-serif`;
    context.textAlign = 'center';
    context.textBaseline = 'middle';
    context.fillText(entry.compressed ? 'KTX2' : entry.label, size / 2, size / 2);
}

function drawTextureMap(canvas, entry, maxSize = 96, square = false) {
    const image = entry.image || entry.texture?.source?.data || entry.texture?.image || null;
    const width = Number(image?.width || image?.naturalWidth || entry.width) || maxSize;
    const height = Number(image?.height || image?.naturalHeight || entry.height) || maxSize;
    const scale = Math.min(1, maxSize / Math.max(width, height));
    const drawWidth = Math.max(1, Math.round(width * scale));
    const drawHeight = Math.max(1, Math.round(height * scale));
    const canvasWidth = square ? maxSize : drawWidth;
    const canvasHeight = square ? maxSize : drawHeight;
    if (!isDrawableTextureImage(image)) {
        drawFallbackTexture(canvas, entry, maxSize);
        return false;
    }
    canvas.width = canvasWidth;
    canvas.height = canvasHeight;
    const context = canvas.getContext('2d');
    context.fillStyle = '#0f0f0e';
    context.fillRect(0, 0, canvasWidth, canvasHeight);
    context.drawImage(
        image,
        Math.round((canvasWidth - drawWidth) / 2),
        Math.round((canvasHeight - drawHeight) / 2),
        drawWidth,
        drawHeight
    );
    return true;
}

function setTextureColorSpace(texture, color = false) {
    if (!texture) return;
    if ('colorSpace' in texture) {
        texture.colorSpace = color ? THREE.SRGBColorSpace : THREE.NoColorSpace;
    } else if ('encoding' in texture) {
        texture.encoding = color ? THREE.sRGBEncoding : THREE.LinearEncoding;
    }
    texture.needsUpdate = true;
}

function setTextureSampling(texture, maxAnisotropy) {
    if (!texture?.isTexture) return;
    let changed = false;
    if (Number.isFinite(maxAnisotropy) && maxAnisotropy > 1 && texture.anisotropy !== maxAnisotropy) {
        texture.anisotropy = maxAnisotropy;
        changed = true;
    }
    if (texture.magFilter !== THREE.LinearFilter) {
        texture.magFilter = THREE.LinearFilter;
        changed = true;
    }
    if (texture.minFilter !== THREE.LinearMipmapLinearFilter) {
        texture.minFilter = THREE.LinearMipmapLinearFilter;
        changed = true;
    }
    if (changed) texture.needsUpdate = true;
}

function applyPbrDisplayLook(material) {
    if (!material || material.userData?.smvPbrDisplayLook === true) return;
    if (!material.isMeshStandardMaterial && !material.isMeshPhysicalMaterial) return;
    const originalOnBeforeCompile = material.onBeforeCompile;
    const originalCustomProgramCacheKey = material.customProgramCacheKey;
    material.onBeforeCompile = function smvPbrDisplayLook(shader, renderer) {
        originalOnBeforeCompile.call(this, shader, renderer);
        shader.fragmentShader = shader.fragmentShader.replace(
            '#include <dithering_fragment>',
            [
                'vec3 smvLookLumaWeights = vec3(0.2126, 0.7152, 0.0722);',
                'float smvLookLuma = dot(gl_FragColor.rgb, smvLookLumaWeights);',
                `gl_FragColor.rgb = mix(vec3(smvLookLuma), gl_FragColor.rgb, ${PBR_DISPLAY_LOOK_SATURATION.toFixed(3)});`,
                `gl_FragColor.rgb = (gl_FragColor.rgb - vec3(${PBR_DISPLAY_LOOK_PIVOT.toFixed(3)})) * ${PBR_DISPLAY_LOOK_CONTRAST.toFixed(3)} + vec3(${PBR_DISPLAY_LOOK_PIVOT.toFixed(3)});`,
                'gl_FragColor.rgb = max(gl_FragColor.rgb, vec3(0.0));',
                '#include <dithering_fragment>'
            ].join('\n')
        );
    };
    material.customProgramCacheKey = function smvPbrDisplayLookCacheKey() {
        return `${originalCustomProgramCacheKey.call(this)}|smv-pbr-display-look`;
    };
    material.userData.smvPbrDisplayLook = true;
    material.needsUpdate = true;
}

function mapMaterialEntry(materialEntry, fn) {
    return Array.isArray(materialEntry)
        ? materialEntry.map((material) => fn(material))
        : fn(materialEntry);
}

function parseBooleanAttribute(value, fallback = false) {
    if (value === null || value === undefined) return fallback;
    const normalized = String(value).trim().toLowerCase();
    if (!normalized || normalized === 'true' || normalized === '1' || normalized === 'yes') return true;
    if (normalized === 'false' || normalized === '0' || normalized === 'no') return false;
    return fallback;
}

function parseNumber(value, fallback) {
    if (value === null || value === undefined || String(value).trim() === '') return fallback;
    const number = Number(value);
    return Number.isFinite(number) ? number : fallback;
}

function normalizeViewMode(mode) {
    const normalized = String(mode || '').trim().toLowerCase();
    if (['geometry', 'mesh', 'clay'].includes(normalized)) return 'geometry';
    if (['normal', 'normals'].includes(normalized)) return 'normal';
    if (['albedo', 'base', 'basecolor', 'base-color'].includes(normalized)) return 'albedo';
    if (['rough', 'roughness'].includes(normalized)) return 'roughness';
    if (['metal', 'metallic', 'metalness'].includes(normalized)) return 'metalness';
    if (['diffuse', 'texture', 'textured', 'pbr', 'default', ''].includes(normalized)) return 'pbr';
    return 'pbr';
}

const VIEW_MODE_LABELS = {
    pbr: 'PBR',
    albedo: 'Albedo',
    roughness: 'Rough',
    metalness: 'Metal',
    geometry: 'Geo',
    normal: 'Normal'
};

function viewModeLabel(mode) {
    return VIEW_MODE_LABELS[normalizeViewMode(mode)] || VIEW_MODE_LABELS.pbr;
}

function normalizeWireframeMode(mode) {
    const normalized = String(mode || '').trim().toLowerCase();
    return ['tri', 'triangle', 'triangles'].includes(normalized) ? 'tri' : 'quad';
}

function environmentPresetFor(id) {
    return ENVIRONMENT_PRESETS[id] || ENVIRONMENT_PRESETS[DEFAULT_ENVIRONMENT_PRESET];
}

function makeButtonIcon(pathData) {
    return `<svg viewBox="0 0 24 24" aria-hidden="true" focusable="false"><path d="${pathData}"></path></svg>`;
}

const ICONS = {
    open: makeButtonIcon('M12 16V4m0 0L7 9m5-5 5 5M5 14v5h14v-5'),
    fit: makeButtonIcon('M8 3H3v5M3 3l6 6M16 3h5v5m0-5-6 6M8 21H3v-5m0 5 6-6m7 6h5v-5m0 5-6-6'),
    reset: makeButtonIcon('M4 4v6h6M20 20v-6h-6M5.6 14A7 7 0 0 0 18 17.4M18.4 10A7 7 0 0 0 6 6.6'),
    rotate: makeButtonIcon('M17 2l4 4-4 4M3 11a7 7 0 0 1 14-5h4M7 22l-4-4 4-4M21 13a7 7 0 0 1-14 5H3'),
    grid: makeButtonIcon('M4 4h16v16H4zM4 9h16M4 15h16M9 4v16M15 4v16'),
    wire: makeButtonIcon('M12 3 20 7.5v9L12 21 4 16.5v-9L12 3ZM4 7.5l8 4.5 8-4.5M12 12v9'),
    clip: makeButtonIcon('M4 5h16M7 5v14m10-14v14M4 19h16'),
    shot: makeButtonIcon('M7 7h2l1.5-2h3L15 7h2a3 3 0 0 1 3 3v6a3 3 0 0 1-3 3H7a3 3 0 0 1-3-3v-6a3 3 0 0 1 3-3ZM12 16a3 3 0 1 0 0-6 3 3 0 0 0 0 6Z'),
    full: makeButtonIcon('M8 3H3v5M3 3l6 6M16 3h5v5m0-5-6 6M8 21H3v-5m0 5 6-6m7 6h5v-5m0 5-6-6'),
    close: makeButtonIcon('M6 6l12 12M18 6 6 18'),
    play: makeButtonIcon('M8 5v14l11-7z'),
    pause: makeButtonIcon('M7 5h4v14H7zM13 5h4v14h-4z')
};

class SimpleModelViewer extends HTMLElement {
    static get observedAttributes() {
        return [
            'src',
            'camera-orbit',
            'camera-target',
            'camera-up',
            'background-color',
            'view-mode',
            'wireframe-mode',
            'auto-rotate',
            'angle-per-second',
            'environment',
            'environment-url',
            'environment-intensity',
            'environment-background',
            'environment-rotation',
            'exposure',
            'selection-mode',
            'performance-mode',
            'animation',
            'animation-speed',
            'animation-loop'
        ];
    }

    constructor() {
        super();
        this.attachShadow({ mode: 'open' });
        this.shadowRoot.innerHTML = this.renderTemplate();

        this.rootEl = this.shadowRoot.querySelector('#viewerRoot');
        this.canvasContainer = this.shadowRoot.querySelector('#canvasContainer');
        this.fileInputContainer = this.shadowRoot.querySelector('#fileInputContainer');
        this.fileInput = this.shadowRoot.querySelector('#fileInput');
        this.folderInput = this.shadowRoot.querySelector('#folderInput');
        this.relinkInput = this.shadowRoot.querySelector('#relinkInput');
        this.statusEl = this.shadowRoot.querySelector('#status');
        this.entriesEl = this.shadowRoot.querySelector('#entries');
        this.preflightEl = this.shadowRoot.querySelector('#preflight');
        this.preflightFactsEl = this.shadowRoot.querySelector('#preflightFacts');
        this.preflightResourcesEl = this.shadowRoot.querySelector('#preflightResources');
        this.loadingEl = this.shadowRoot.querySelector('#loading');
        this.loadingLabelEl = this.shadowRoot.querySelector('#loadingLabel');
        this.loadingValueEl = this.shadowRoot.querySelector('#loadingValue');
        this.loadingBarEl = this.shadowRoot.querySelector('#loadingBar');
        this.statsEl = this.shadowRoot.querySelector('#stats');
        this.textureMapsEl = this.shadowRoot.querySelector('#textureMaps');
        this.textureStripEl = this.shadowRoot.querySelector('#textureStrip');
        this.textureCountEl = this.shadowRoot.querySelector('#textureCount');
        this.textureToggleBtn = this.shadowRoot.querySelector('#textureToggleBtn');
        this.textureDialog = this.shadowRoot.querySelector('#textureDialog');
        this.textureDialogCanvas = this.shadowRoot.querySelector('#textureDialogCanvas');
        this.textureDialogCaption = this.shadowRoot.querySelector('#textureDialogCaption');
        this.environmentControls = this.shadowRoot.querySelector('#environmentControls');
        this.environmentMenu = this.shadowRoot.querySelector('#environmentMenu');
        this.environmentToggle = this.shadowRoot.querySelector('#environmentToggle');
        this.environmentDial = this.shadowRoot.querySelector('#environmentDial');
        this.modeToggleBtn = this.shadowRoot.querySelector('#modeToggleBtn');
        this.modeMenu = this.shadowRoot.querySelector('#modeMenu');
        this.historySelect = this.shadowRoot.querySelector('#historySelect');
        this.sectionControls = this.shadowRoot.querySelector('#sectionControls');

        this.scene = new THREE.Scene();
        this.camera = new THREE.PerspectiveCamera(45, 1, 0.01, 10000);
        this.camera.position.set(0, 0, DEFAULT_CAMERA_DISTANCE);
        this.renderer = new THREE.WebGLRenderer({
            antialias: true,
            alpha: true,
            preserveDrawingBuffer: true
        });
        if ('outputColorSpace' in this.renderer) {
            this.renderer.outputColorSpace = THREE.SRGBColorSpace;
        } else {
            this.renderer.outputEncoding = THREE.sRGBEncoding;
        }
        this.renderer.toneMapping = THREE.ACESFilmicToneMapping;
        this.renderer.toneMappingExposure = 1;
        this.renderer.localClippingEnabled = false;
        this.renderer.setClearColor(0x000000, 0);
        this.renderer.domElement.setAttribute('aria-label', '3D model viewer canvas');
        this.canvasContainer.appendChild(this.renderer.domElement);

        this.controls = new OrbitControls(this.camera, this.renderer.domElement);
        this.controls.enableDamping = true;
        this.controls.dampingFactor = 0.08;
        this.controls.screenSpacePanning = true;

        this.pmremGenerator = new THREE.PMREMGenerator(this.renderer);
        this.rgbeLoader = new RGBELoader();
        this.exrLoader = new EXRLoader();
        this.textureLoader = new THREE.TextureLoader();
        this.raycaster = new THREE.Raycaster();
        this.pointer = new THREE.Vector2();
        this.pointerStart = new THREE.Vector2();

        this.ambientLight = new THREE.HemisphereLight(0xffffff, 0x59616d, 0.65);
        this.keyLight = new THREE.DirectionalLight(0xffffff, 1.55);
        this.fillLight = new THREE.DirectionalLight(0xd6e4ff, 0.42);
        this.rimLight = new THREE.DirectionalLight(0xffffff, 0.72);
        this.keyLight.position.set(4.5, 5.2, 4.0);
        this.fillLight.position.set(-3.5, 2.4, -3.0);
        this.rimLight.position.set(-2.4, 4.8, 3.4);
        this.scene.add(this.ambientLight, this.keyLight, this.fillLight, this.rimLight);

        this.grid = new THREE.GridHelper(2, 20, 0x9aa3af, 0xc5cad3);
        this.grid.material.transparent = true;
        this.grid.material.opacity = 0.38;
        this.grid.visible = false;
        this.scene.add(this.grid);

        this.placeholderMaterial = this.createGeometryMaterial();
        this.geometryMaterial = this.createGeometryMaterial();
        this.normalMaterial = new THREE.MeshNormalMaterial({ side: THREE.DoubleSide });
        this.wireframeMaterial = new THREE.LineBasicMaterial({
            color: DEFAULT_WIREFRAME_COLOR,
            transparent: true,
            opacity: 0.62,
            depthTest: true,
            depthWrite: false
        });
        this.selectionHelper = new THREE.BoxHelper(new THREE.Object3D(), 0x41c7ff);
        this.selectionHelper.visible = false;
        this.scene.add(this.selectionHelper);

        this.placeholder = this.createPlaceholder();
        this.scene.add(this.placeholder);

        this.state = {
            viewMode: normalizeViewMode(this.getAttribute('view-mode')),
            wireframe: false,
            wireframeMode: normalizeWireframeMode(this.getAttribute('wireframe-mode')),
            autoRotate: this.hasAttribute('auto-rotate'),
            anglePerSecond: parseNumber(this.getAttribute('angle-per-second'), 30),
            grid: false,
            section: {
                enabled: false,
                axis: 'x',
                value: 0,
                flipped: false
            },
            backgroundColor: this.getAttribute('background-color') || DEFAULT_BACKGROUND_COLOR,
            environment: this.getAttribute('environment') || DEFAULT_ENVIRONMENT_PRESET,
            environmentUrl: this.getAttribute('environment-url') || '',
            environmentIntensity: parseNumber(this.getAttribute('environment-intensity'), 1),
            environmentBackground: parseBooleanAttribute(this.getAttribute('environment-background'), false),
            environmentRotation: parseNumber(this.getAttribute('environment-rotation'), 0),
            selectionMode: this.getAttribute('selection-mode') || 'all',
            performanceMode: this.getAttribute('performance-mode') || 'default',
            animationSpeed: parseNumber(this.getAttribute('animation-speed'), 1),
            animationLoop: this.getAttribute('animation-loop') || 'repeat'
        };

        this.model = null;
        this.modelBounds = new THREE.Box3();
        this.modelCenter = new THREE.Vector3();
        this.modelSize = new THREE.Vector3(1, 1, 1);
        this.modelRadius = 1;
        this.meshParts = [];
        this.selectedMesh = null;
        this.animations = [];
        this.mixer = null;
        this.currentAction = null;
        this.isAnimationPlaying = true;
        this.historyRecords = [];
        this.texturePanelOpen = false;
        this.pendingBundle = null;
        this.pendingMainEntry = null;
        this.objectUrls = new Set();
        this.currentSource = null;
        this.currentFileName = '';
        this.loadToken = 0;
        this.frameRequest = null;
        this.lastFrameTime = performance.now();
        this.connected = false;
        this.initialCameraOrbit = null;
        this.cameraTransitionDefault = null;
        this.environmentSourceTexture = null;
        this.environmentTexture = null;
        this.environmentRenderTarget = null;
        this.environmentRotationKey = '';
        this.environmentPreviewTextures = new Map();
        this.environmentPreviewPromises = new Map();
        this.environmentUrl = '';
        this.draggingPointer = false;

        this.sectionPlane = new THREE.Plane(new THREE.Vector3(1, 0, 0), 0);
        this.resizeObserver = new ResizeObserver(() => this.resizeRenderer());

        this.boundHandleKeyDown = (event) => this.handleKeyDown(event);
        this.boundPointerDown = (event) => this.handlePointerDown(event);
        this.boundPointerUp = (event) => this.handlePointerUp(event);
        this.boundPointerMove = (event) => this.handlePointerMove(event);
    }

    renderTemplate() {
        return /*html*/`
            <style>
                :host {
                    display: block;
                    min-height: 360px;
                    color: #f1ede2;
                    font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
                    --viewer-panel: rgba(8, 8, 7, 0.62);
                    --viewer-panel-strong: rgba(8, 8, 7, 0.84);
                    --viewer-ink: #f1ede2;
                    --viewer-muted: rgba(241, 237, 226, 0.6);
                    --viewer-line: rgba(241, 237, 226, 0.14);
                    --viewer-accent: #5db8a6;
                    --viewer-accent-2: #f3b35b;
                }

                *, *::before, *::after {
                    box-sizing: border-box;
                }

                #viewerRoot {
                    position: relative;
                    width: 100%;
                    height: 100%;
                    min-height: inherit;
                    overflow: hidden;
                    background:
                        radial-gradient(ellipse at 50% 68%, rgba(86, 86, 82, 0.34) 0%, rgba(38, 38, 36, 0.18) 29%, rgba(15, 15, 14, 0) 58%),
                        radial-gradient(ellipse at 50% 52%, rgba(48, 48, 45, 0.2) 0%, rgba(15, 15, 14, 0) 54%),
                        linear-gradient(180deg, #111110 0%, #0d0d0c 48%, var(--viewer-bg-color, #070707) 100%);
                }

                #viewerRoot.upload-open::after {
                    position: absolute;
                    inset: 0;
                    z-index: 8;
                    background: rgba(2, 3, 3, 0.34);
                    content: "";
                    pointer-events: none;
                }

                #canvasContainer,
                #canvasContainer > canvas {
                    position: absolute;
                    inset: 0;
                    width: 100%;
                    height: 100%;
                }

                canvas {
                    display: block;
                }

                #canvasContainer > canvas {
                    outline: none;
                    touch-action: none;
                }

                button,
                select,
                input {
                    font: inherit;
                }

                button {
                    border: 1px solid var(--viewer-line);
                    background: rgba(255, 255, 255, 0.045);
                    color: var(--viewer-ink);
                    cursor: pointer;
                }

                button:disabled {
                    cursor: default;
                    opacity: 0.42;
                }

                button:hover:not(:disabled),
                button:focus-visible,
                select:focus-visible,
                input:focus-visible {
                    border-color: rgba(93, 184, 166, 0.56);
                    outline: 2px solid rgba(93, 184, 166, 0.2);
                    outline-offset: 1px;
                }

                svg {
                    width: 1rem;
                    height: 1rem;
                    fill: none;
                    stroke: currentColor;
                    stroke-linecap: round;
                    stroke-linejoin: round;
                    stroke-width: 1.8;
                }

                .icon-button {
                    display: inline-flex;
                    align-items: center;
                    justify-content: center;
                    width: 2.05rem;
                    height: 2.05rem;
                    border-radius: 7px;
                    padding: 0;
                }

                .text-button {
                    display: inline-flex;
                    align-items: center;
                    justify-content: center;
                    min-height: 2.05rem;
                    border-radius: 7px;
                    padding: 0 0.72rem;
                    font-size: 0.78rem;
                    font-weight: 700;
                    white-space: nowrap;
                }

                .primary {
                    border-color: rgba(93, 184, 166, 0.36);
                    background: rgba(93, 184, 166, 0.22);
                    color: #dffcf7;
                }

                .secondary {
                    background: rgba(255, 255, 255, 0.06);
                }

                .active,
                .icon-button[aria-pressed="true"],
                .text-button[aria-pressed="true"] {
                    border-color: rgba(93, 184, 166, 0.58);
                    background: rgba(93, 184, 166, 0.15);
                    color: #bcfff3;
                }

                #toolbar {
                    position: absolute;
                    z-index: 8;
                    top: 0.75rem;
                    left: 0.75rem;
                    right: 0.75rem;
                    display: flex;
                    align-items: center;
                    gap: 0.45rem;
                    pointer-events: none;
                }

                #toolbar > * {
                    pointer-events: auto;
                }

                .tool-group,
                .mode-segment {
                    display: inline-flex;
                    align-items: center;
                    gap: 0.25rem;
                    min-width: 0;
                    padding: 0.25rem;
                    border: 1px solid var(--viewer-line);
                    border-radius: 8px;
                    background: var(--viewer-panel);
                    box-shadow: 0 12px 34px rgba(17, 24, 39, 0.12);
                    backdrop-filter: blur(14px);
                }

                .mode-segment {
                    position: relative;
                    overflow: visible;
                }

                .toolbar-spacer {
                    flex: 1;
                }

                #modeToggleBtn,
                #modeMenu .mode-button {
                    display: inline-flex;
                    align-items: center;
                    justify-content: center;
                    min-height: 1.78rem;
                    border-radius: 6px;
                    padding: 0 0.48rem;
                    font-size: 0.68rem;
                    font-weight: 800;
                    letter-spacing: 0;
                    text-transform: uppercase;
                }

                #modeToggleBtn {
                    min-width: 4.25rem;
                    gap: 0.42rem;
                }

                #modeToggleBtn::after {
                    width: 0;
                    height: 0;
                    border-left: 0.22rem solid transparent;
                    border-right: 0.22rem solid transparent;
                    border-top: 0.3rem solid currentColor;
                    content: "";
                    opacity: 0.72;
                }

                #modeMenu {
                    position: absolute;
                    z-index: 14;
                    top: calc(100% + 0.35rem);
                    left: 0.25rem;
                    display: grid;
                    gap: 0.25rem;
                    min-width: 7.1rem;
                    padding: 0.35rem;
                    border: 1px solid var(--viewer-line);
                    border-radius: 8px;
                    background: var(--viewer-panel-strong);
                    box-shadow: 0 18px 42px rgba(0, 0, 0, 0.32);
                    backdrop-filter: blur(16px);
                }

                #modeMenu[hidden] {
                    display: none;
                }

                #modeMenu .mode-button {
                    justify-content: flex-start;
                    width: 100%;
                    min-width: 0;
                }

                #modeMenu .mode-button:not(.active) {
                    border-color: transparent;
                    background: transparent;
                    color: rgba(241, 237, 226, 0.62);
                }

                #wireModeSelect,
                #sectionAxis {
                    width: 4.3rem;
                    min-height: 2.05rem;
                    border: 1px solid var(--viewer-line);
                    border-radius: 7px;
                    background: rgba(255, 255, 255, 0.06);
                    color: var(--viewer-ink);
                    font-size: 0.74rem;
                    font-weight: 700;
                    padding: 0 0.45rem;
                }

                #stats {
                    position: absolute;
                    z-index: 7;
                    left: 0.75rem;
                    bottom: 0.75rem;
                    max-width: calc(100% - 1.5rem);
                    border: 1px solid var(--viewer-line);
                    border-radius: 8px;
                    background: rgba(8, 8, 7, 0.58);
                    box-shadow: 0 12px 30px rgba(0, 0, 0, 0.22);
                    padding: 0.46rem 0.62rem;
                    color: rgba(241, 237, 226, 0.78);
                    font-size: 0.74rem;
                    font-weight: 700;
                    backdrop-filter: blur(14px);
                }

                #textureMaps {
                    position: absolute;
                    z-index: 7;
                    right: 0.75rem;
                    bottom: 5.25rem;
                    width: min(380px, calc(100% - 1.5rem));
                    border: 1px solid var(--viewer-line);
                    border-radius: 8px;
                    background: rgba(8, 8, 7, 0.72);
                    box-shadow: 0 12px 30px rgba(0, 0, 0, 0.28);
                    padding: 0.48rem;
                    backdrop-filter: blur(14px);
                }

                #textureMaps.collapsed {
                    width: auto;
                    min-width: 0;
                    border-color: transparent;
                    background: transparent;
                    box-shadow: none;
                    padding: 0;
                    backdrop-filter: none;
                }

                #environmentControls {
                    position: absolute;
                    right: 0.75rem;
                    bottom: 0.75rem;
                    z-index: 8;
                    pointer-events: none;
                }

                .pbr-light-control {
                    position: relative;
                    display: inline-flex;
                    align-items: center;
                    gap: 0.5rem;
                    width: 190px;
                    min-height: 4rem;
                    padding: 0.38rem 0.5rem;
                    border: 1px solid var(--viewer-line);
                    border-radius: 8px;
                    background: rgba(15, 15, 14, 0.72);
                    box-shadow: 0 16px 34px rgba(0, 0, 0, 0.34);
                    backdrop-filter: blur(14px);
                    pointer-events: auto;
                }

                .pbr-env-picker {
                    position: relative;
                    flex: 0 0 auto;
                }

                .pbr-env-toggle {
                    display: grid;
                    align-content: center;
                    gap: 0.12rem;
                    min-width: 4.85rem;
                    min-height: 2.75rem;
                    border-radius: 8px;
                    padding: 0.32rem 0.5rem;
                    background: rgba(8, 8, 7, 0.52);
                    color: var(--viewer-ink);
                    font-size: 0.68rem;
                    font-weight: 900;
                    line-height: 1;
                    text-align: left;
                }

                .pbr-env-toggle:hover,
                .pbr-env-toggle:focus-visible,
                .pbr-env-toggle[aria-expanded="true"] {
                    border-color: rgba(243, 179, 91, 0.72);
                    background: rgba(243, 179, 91, 0.14);
                    color: var(--viewer-accent-2);
                }

                .pbr-env-active-label {
                    max-width: 4.25rem;
                    overflow: hidden;
                    color: var(--viewer-muted);
                    font-size: 0.56rem;
                    font-weight: 800;
                    text-overflow: ellipsis;
                    text-transform: uppercase;
                    white-space: nowrap;
                }

                .pbr-env-menu {
                    position: absolute;
                    left: 0;
                    bottom: calc(100% + 0.5rem);
                    z-index: 9;
                    display: grid;
                    gap: 0.32rem;
                    width: 5.9rem;
                    max-height: min(390px, calc(100vh - 160px));
                    overflow-y: auto;
                    padding: 0.44rem;
                    border: 1px solid var(--viewer-line);
                    border-radius: 8px;
                    background: rgba(10, 10, 9, 0.92);
                    box-shadow: 0 16px 34px rgba(0, 0, 0, 0.42);
                    backdrop-filter: blur(12px);
                }

                .pbr-env-menu button {
                    display: grid;
                    justify-items: center;
                    gap: 0.25rem;
                    min-width: 0;
                    min-height: 4.9rem;
                    border: 1px solid rgba(241, 237, 226, 0.16);
                    border-radius: 8px;
                    padding: 0.36rem 0.25rem;
                    background: rgba(8, 8, 7, 0.46);
                    color: var(--viewer-muted);
                    font-size: 0.62rem;
                    font-weight: 900;
                    line-height: 1;
                }

                .pbr-env-menu .pbr-env-background-toggle {
                    grid-template-columns: minmax(0, 1fr) auto;
                    align-items: center;
                    justify-items: stretch;
                    min-height: 1.75rem;
                    padding: 0.32rem 0.38rem;
                    text-align: left;
                }

                .pbr-env-background-toggle span:first-child {
                    overflow: hidden;
                    font-size: 0.5rem;
                    text-overflow: ellipsis;
                    text-transform: uppercase;
                    white-space: nowrap;
                }

                .pbr-env-background-switch {
                    position: relative;
                    width: 1.5rem;
                    height: 0.88rem;
                    border-radius: 999px;
                    background: rgba(241, 237, 226, 0.16);
                    box-shadow: inset 0 0 0 1px rgba(241, 237, 226, 0.18);
                }

                .pbr-env-background-switch::after {
                    position: absolute;
                    top: 0.19rem;
                    left: 0.19rem;
                    width: 0.5rem;
                    height: 0.5rem;
                    border-radius: 50%;
                    background: var(--viewer-muted);
                    content: "";
                    transition: transform 120ms ease, background 120ms ease;
                }

                .pbr-env-background-toggle.active .pbr-env-background-switch {
                    background: rgba(243, 179, 91, 0.32);
                    box-shadow: inset 0 0 0 1px rgba(243, 179, 91, 0.48);
                }

                .pbr-env-background-toggle.active .pbr-env-background-switch::after {
                    transform: translateX(0.62rem);
                    background: var(--viewer-accent-2);
                }

                .pbr-env-menu button:hover,
                .pbr-env-menu button:focus-visible,
                .pbr-env-menu button.active {
                    border-color: rgba(243, 179, 91, 0.72);
                    color: var(--viewer-accent-2);
                    background: rgba(243, 179, 91, 0.14);
                }

                .pbr-env-preview {
                    width: 3rem;
                    height: 3rem;
                    border-radius: 50%;
                    background: rgba(0, 0, 0, 0.28);
                    box-shadow:
                        inset 0 0 0 1px rgba(255, 255, 255, 0.16),
                        0 4px 10px rgba(0, 0, 0, 0.28);
                }

                .pbr-light-dial {
                    position: relative;
                    flex: 0 0 3.5rem;
                    width: 3.5rem;
                    height: 3.5rem;
                    overflow: hidden;
                    border: 1px solid rgba(243, 179, 91, 0.72);
                    border-radius: 50%;
                    padding: 0;
                    background: rgba(243, 179, 91, 0.08);
                    color: var(--viewer-accent-2);
                    cursor: grab;
                    touch-action: none;
                }

                .pbr-light-dial:active {
                    cursor: grabbing;
                }

                .pbr-light-env-preview {
                    position: absolute;
                    inset: 0;
                    width: 100%;
                    height: 100%;
                    border-radius: 50%;
                }

                .pbr-light-dial-ring {
                    position: absolute;
                    inset: 0.25rem;
                    border: 1px solid rgba(250, 249, 245, 0.42);
                    border-radius: 50%;
                    box-shadow: inset 0 0 10px rgba(0, 0, 0, 0.45);
                }

                .pbr-light-dial-hand {
                    position: absolute;
                    left: 50%;
                    top: 0.32rem;
                    width: 2px;
                    height: 1.45rem;
                    margin-left: -1px;
                    border-radius: 999px;
                    background: currentColor;
                    transform-origin: 50% 1.45rem;
                    box-shadow:
                        0 0 0 1px rgba(0, 0, 0, 0.3),
                        0 0 8px rgba(243, 179, 91, 0.48);
                }

                .pbr-light-copy {
                    display: grid;
                    min-width: 1.5rem;
                    justify-items: end;
                }

                .pbr-light-value {
                    color: var(--viewer-accent-2);
                    font-size: 0.68rem;
                    font-weight: 800;
                    line-height: 1.1;
                }

                .texture-head {
                    display: flex;
                    align-items: center;
                    justify-content: space-between;
                    gap: 0.65rem;
                    margin-bottom: 0.38rem;
                    color: rgba(241, 237, 226, 0.78);
                    font-size: 0.74rem;
                    font-weight: 800;
                }

                #textureMaps.collapsed .texture-head {
                    margin-bottom: 0;
                }

                #textureToggleBtn {
                    display: inline-flex;
                    align-items: center;
                    justify-content: center;
                    width: auto;
                    min-width: 4rem;
                    min-height: 1.9rem;
                    gap: 0.38rem;
                    border-color: rgba(241, 237, 226, 0.16);
                    border-radius: 999px;
                    background: rgba(8, 8, 7, 0.58);
                    box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.055);
                    color: rgba(241, 237, 226, 0.76);
                    font-size: 0.62rem;
                    font-weight: 900;
                    letter-spacing: 0;
                    line-height: 1;
                    padding: 0 0.62rem;
                    text-transform: uppercase;
                }

                #textureToggleBtn:hover,
                #textureToggleBtn:focus-visible,
                #textureToggleBtn[aria-expanded="true"] {
                    border-color: rgba(93, 184, 166, 0.58);
                    background: rgba(93, 184, 166, 0.13);
                    color: #bcfff3;
                }

                #textureCount {
                    display: inline-flex;
                    align-items: center;
                    justify-content: center;
                    min-width: 1.05rem;
                    height: 1.05rem;
                    border-radius: 999px;
                    background: rgba(241, 237, 226, 0.12);
                    color: rgba(241, 237, 226, 0.68);
                    font-size: 0.56rem;
                    font-weight: 900;
                    line-height: 1;
                    padding: 0 0.2rem;
                }

                #textureToggleBtn[aria-expanded="true"] #textureCount {
                    background: rgba(93, 184, 166, 0.22);
                    color: #d9fff8;
                }

                #textureStrip {
                    display: grid;
                    grid-auto-flow: column;
                    grid-auto-columns: 4.85rem;
                    gap: 0.48rem;
                    overflow-x: auto;
                    padding-bottom: 0.05rem;
                }

                #textureMaps.collapsed #textureStrip {
                    display: none;
                }

                .texture-card {
                    display: grid;
                    grid-template-rows: 4.15rem auto;
                    gap: 0.32rem;
                    border: 1px solid rgba(241, 237, 226, 0.12);
                    border-radius: 7px;
                    background: rgba(255, 255, 255, 0.055);
                    padding: 0.3rem;
                    color: rgba(241, 237, 226, 0.8);
                    text-align: left;
                    min-width: 0;
                }

                .texture-card span {
                    overflow: hidden;
                    text-overflow: ellipsis;
                    white-space: nowrap;
                    font-size: 0.64rem;
                    font-weight: 800;
                }

                .texture-preview {
                    display: grid;
                    place-items: center;
                    min-width: 0;
                    min-height: 0;
                    border-radius: 5px;
                    background:
                        linear-gradient(45deg, rgba(241, 237, 226, 0.16) 25%, transparent 25%),
                        linear-gradient(-45deg, rgba(241, 237, 226, 0.16) 25%, transparent 25%),
                        linear-gradient(45deg, transparent 75%, rgba(241, 237, 226, 0.16) 75%),
                        linear-gradient(-45deg, transparent 75%, rgba(241, 237, 226, 0.16) 75%),
                        rgba(0, 0, 0, 0.34);
                    background-size: 14px 14px;
                    background-position: 0 0, 0 7px, 7px -7px, -7px 0;
                    overflow: hidden;
                }

                .texture-preview img,
                .texture-preview canvas {
                    width: 100%;
                    height: 100%;
                    object-fit: contain;
                }

                #fileInputContainer {
                    position: absolute;
                    z-index: 9;
                    left: 50%;
                    top: 50%;
                    width: min(640px, calc(100% - 2.5rem));
                    max-height: min(78vh, 700px);
                    overflow: auto;
                    transform: translate(-50%, -50%);
                    border: 1px solid var(--viewer-line);
                    border-radius: 14px;
                    background: var(--viewer-panel-strong);
                    box-shadow: 0 24px 72px rgba(0, 0, 0, 0.54);
                    padding: 1.05rem;
                    backdrop-filter: blur(18px);
                }

                .panel-head {
                    display: flex;
                    align-items: start;
                    justify-content: space-between;
                    gap: 0.75rem;
                    margin-bottom: 0.65rem;
                }

                .panel-head p {
                    margin: 0;
                    color: var(--viewer-accent);
                    font-size: 0.7rem;
                    font-weight: 800;
                    text-transform: uppercase;
                }

                .panel-head h2 {
                    margin: 0.12rem 0 0;
                    color: var(--viewer-ink);
                    font-size: 1.05rem;
                    line-height: 1.1;
                    letter-spacing: 0;
                }

                #dropZone {
                    display: grid;
                    gap: 0.3rem;
                    place-items: center;
                    min-height: 11rem;
                    border: 1px dashed rgba(93, 184, 166, 0.46);
                    border-radius: 12px;
                    background:
                        radial-gradient(circle at 50% 12%, rgba(93, 184, 166, 0.12), transparent 52%),
                        rgba(93, 184, 166, 0.035);
                    padding: 0.85rem;
                    text-align: center;
                    color: rgba(241, 237, 226, 0.82);
                }

                #dropZone svg {
                    width: 1.7rem;
                    height: 1.7rem;
                    color: var(--viewer-accent);
                }

                #dropZone strong {
                    font-size: 0.88rem;
                    line-height: 1.2;
                }

                #dropZone span,
                #status,
                .storage-row,
                .preflight-resources {
                    color: var(--viewer-muted);
                    font-size: 0.72rem;
                    line-height: 1.35;
                }

                #viewerRoot.dragging #dropZone {
                    border-color: rgba(93, 184, 166, 0.94);
                    background:
                        radial-gradient(circle at 50% 12%, rgba(93, 184, 166, 0.22), transparent 56%),
                        rgba(93, 184, 166, 0.08);
                }

                .upload-actions,
                .history-row,
                .storage-row,
                .preflight-actions {
                    display: flex;
                    align-items: center;
                    gap: 0.45rem;
                    flex-wrap: wrap;
                    margin-top: 0.62rem;
                }

                .history-row select {
                    flex: 1;
                    min-width: 11rem;
                    min-height: 2.05rem;
                    border: 1px solid var(--viewer-line);
                    border-radius: 7px;
                    background: rgba(255, 255, 255, 0.06);
                    color: var(--viewer-ink);
                    padding: 0 0.5rem;
                    font-size: 0.76rem;
                    font-weight: 700;
                }

                #entries {
                    display: grid;
                    gap: 0.35rem;
                    max-height: 8.5rem;
                    overflow: auto;
                    margin-top: 0.6rem;
                }

                .entry-button {
                    display: grid;
                    grid-template-columns: 1fr auto;
                    gap: 0.5rem;
                    width: 100%;
                    min-height: 2.3rem;
                    border-radius: 7px;
                    padding: 0.45rem 0.55rem;
                    text-align: left;
                }

                .entry-button strong,
                .entry-button span {
                    overflow: hidden;
                    text-overflow: ellipsis;
                    white-space: nowrap;
                }

                .entry-button strong {
                    color: var(--viewer-ink);
                    font-size: 0.76rem;
                }

                .entry-button span {
                    color: rgba(241, 237, 226, 0.54);
                    font-size: 0.68rem;
                }

                #preflight {
                    margin-top: 0.7rem;
                    border-top: 1px solid var(--viewer-line);
                    padding-top: 0.65rem;
                }

                #preflightFacts {
                    display: grid;
                    grid-template-columns: repeat(2, minmax(0, 1fr));
                    gap: 0.35rem;
                    margin: 0;
                }

                #preflightFacts div {
                    min-width: 0;
                    border: 1px solid rgba(241, 237, 226, 0.1);
                    border-radius: 7px;
                    background: rgba(255, 255, 255, 0.055);
                    padding: 0.43rem 0.48rem;
                }

                #preflightFacts dt,
                #preflightFacts dd {
                    margin: 0;
                    overflow: hidden;
                    text-overflow: ellipsis;
                    white-space: nowrap;
                }

                #preflightFacts dt {
                    color: rgba(241, 237, 226, 0.54);
                    font-size: 0.62rem;
                    font-weight: 800;
                    text-transform: uppercase;
                }

                #preflightFacts dd {
                    color: var(--viewer-ink);
                    font-size: 0.76rem;
                    font-weight: 800;
                }

                #sectionControls {
                    position: absolute;
                    z-index: 8;
                    top: 4.15rem;
                    right: 0.75rem;
                    width: min(320px, calc(100% - 1.5rem));
                    border: 1px solid var(--viewer-line);
                    border-radius: 8px;
                    background: var(--viewer-panel);
                    box-shadow: 0 12px 34px rgba(17, 24, 39, 0.12);
                    padding: 0.55rem;
                    backdrop-filter: blur(14px);
                }

                .section-grid {
                    display: grid;
                    grid-template-columns: auto 1fr auto;
                    gap: 0.45rem;
                    align-items: center;
                }

                #sectionSlider {
                    width: 100%;
                    accent-color: var(--viewer-accent);
                }

                #sectionValue {
                    min-width: 3.4rem;
                    color: rgba(241, 237, 226, 0.8);
                    font-size: 0.72rem;
                    font-weight: 800;
                    text-align: right;
                }

                #loading {
                    position: absolute;
                    z-index: 10;
                    left: 50%;
                    bottom: 1rem;
                    width: min(340px, calc(100% - 2rem));
                    transform: translateX(-50%);
                    border: 1px solid var(--viewer-line);
                    border-radius: 8px;
                    background: rgba(8, 8, 7, 0.86);
                    box-shadow: 0 14px 42px rgba(0, 0, 0, 0.34);
                    padding: 0.65rem 0.75rem;
                    backdrop-filter: blur(14px);
                }

                .loading-head {
                    display: flex;
                    justify-content: space-between;
                    gap: 0.75rem;
                    color: rgba(241, 237, 226, 0.82);
                    font-size: 0.76rem;
                    font-weight: 800;
                    margin-bottom: 0.45rem;
                }

                .loading-track {
                    height: 0.42rem;
                    overflow: hidden;
                    border-radius: 999px;
                    background: rgba(93, 184, 166, 0.16);
                }

                .loading-track span {
                    display: block;
                    width: 0%;
                    height: 100%;
                    border-radius: inherit;
                    background: linear-gradient(90deg, var(--viewer-accent), var(--viewer-accent-2));
                    transition: width 160ms ease;
                }

                #textureDialog {
                    width: min(88vmin, 960px);
                    height: min(88vmin, 960px);
                    max-width: calc(100vw - 2rem);
                    max-height: calc(100vh - 2rem);
                    border: 1px solid rgba(241, 237, 226, 0.16);
                    border-radius: 10px;
                    background: rgba(8, 8, 7, 0.96);
                    padding: 0;
                    color: var(--viewer-ink);
                    overflow: hidden;
                }

                #textureDialog::backdrop {
                    background: rgba(0, 0, 0, 0.68);
                    backdrop-filter: blur(2px);
                }

                #textureDialogCanvas {
                    width: 100%;
                    height: 100%;
                    object-fit: contain;
                    background:
                        linear-gradient(45deg, rgba(241, 237, 226, 0.12) 25%, transparent 25%),
                        linear-gradient(-45deg, rgba(241, 237, 226, 0.12) 25%, transparent 25%),
                        linear-gradient(45deg, transparent 75%, rgba(241, 237, 226, 0.12) 75%),
                        linear-gradient(-45deg, transparent 75%, rgba(241, 237, 226, 0.12) 75%),
                        rgba(0, 0, 0, 0.42);
                    background-size: 24px 24px;
                    background-position: 0 0, 0 12px, 12px -12px, -12px 0;
                }

                #textureDialogCaption {
                    position: absolute;
                    left: 0.75rem;
                    right: 0.75rem;
                    bottom: 0.75rem;
                    margin: 0;
                    border: 1px solid rgba(241, 237, 226, 0.14);
                    border-radius: 7px;
                    background: rgba(0, 0, 0, 0.46);
                    padding: 0.45rem 0.55rem;
                    color: rgba(241, 237, 226, 0.82);
                    font-size: 0.78rem;
                    font-weight: 700;
                }

                #textureDialogClose {
                    position: absolute;
                    top: 0.75rem;
                    right: 0.75rem;
                    z-index: 1;
                    background: rgba(0, 0, 0, 0.52);
                }

                .visually-hidden {
                    position: absolute;
                    width: 1px;
                    height: 1px;
                    overflow: hidden;
                    clip: rect(0 0 0 0);
                    white-space: nowrap;
                    clip-path: inset(50%);
                }

                [hidden] {
                    display: none !important;
                }

                @media (max-width: 760px) {
                    #toolbar {
                        align-items: start;
                        flex-wrap: wrap;
                    }

                    .toolbar-spacer {
                        display: none;
                    }

                    #modeMenu {
                        left: 0.25rem;
                    }

                    #fileInputContainer {
                        width: calc(100% - 1.5rem);
                        max-height: calc(100% - 7.5rem);
                    }

                    #textureMaps {
                        left: 0.75rem;
                        right: auto;
                        bottom: 5.25rem;
                        width: min(320px, calc(100% - 1.5rem));
                    }

                    #textureMaps.collapsed {
                        width: auto;
                    }

                    #stats {
                        display: none;
                    }

                    #sectionControls {
                        top: auto;
                        right: 0.75rem;
                        bottom: 0.75rem;
                    }
                }
            </style>

            <div id="viewerRoot">
                <div id="canvasContainer"></div>

                <div id="toolbar" aria-label="Viewer controls">
                    <div class="tool-group">
                        <button id="openPanelBtn" class="icon-button" type="button" title="Open files" aria-label="Open files">${ICONS.open}</button>
                        <button id="fitBtn" class="icon-button" type="button" title="Fit" aria-label="Fit model">${ICONS.fit}</button>
                        <button id="resetBtn" class="icon-button" type="button" title="Reset view" aria-label="Reset view">${ICONS.reset}</button>
                    </div>
                    <div class="mode-segment" aria-label="View mode">
                        <button id="modeToggleBtn" type="button" aria-haspopup="menu" aria-expanded="false">PBR</button>
                        <div id="modeMenu" role="menu" hidden>
                            <button class="mode-button" role="menuitemradio" type="button" data-mode="pbr">PBR</button>
                            <button class="mode-button" role="menuitemradio" type="button" data-mode="albedo">Albedo</button>
                            <button class="mode-button" role="menuitemradio" type="button" data-mode="roughness">Rough</button>
                            <button class="mode-button" role="menuitemradio" type="button" data-mode="metalness">Metal</button>
                            <button class="mode-button" role="menuitemradio" type="button" data-mode="geometry">Geo</button>
                            <button class="mode-button" role="menuitemradio" type="button" data-mode="normal">Normal</button>
                        </div>
                    </div>
                    <div class="tool-group">
                        <button id="wireBtn" class="icon-button" type="button" title="Wireframe" aria-label="Wireframe" aria-pressed="false">${ICONS.wire}</button>
                        <select id="wireModeSelect" title="Wireframe mode" aria-label="Wireframe mode">
                            <option value="quad">Quad</option>
                            <option value="tri">Tri</option>
                        </select>
                        <button id="sectionBtn" class="icon-button" type="button" title="Section" aria-label="Section" aria-pressed="false">${ICONS.clip}</button>
                    </div>
                    <div class="toolbar-spacer"></div>
                    <div class="tool-group">
                        <button id="rotateBtn" class="icon-button" type="button" title="Auto rotate" aria-label="Auto rotate" aria-pressed="false">${ICONS.rotate}</button>
                        <button id="gridBtn" class="icon-button" type="button" title="Grid" aria-label="Grid" aria-pressed="false">${ICONS.grid}</button>
                        <button id="playBtn" class="icon-button" type="button" title="Play animation" aria-label="Play animation" disabled>${ICONS.play}</button>
                        <button id="snapshotBtn" class="icon-button" type="button" title="Snapshot" aria-label="Snapshot">${ICONS.shot}</button>
                        <button id="fullscreenBtn" class="icon-button" type="button" title="Fullscreen" aria-label="Fullscreen">${ICONS.full}</button>
                    </div>
                </div>

                <div id="environmentControls" aria-label="Environment controls">
                    <div class="pbr-light-control">
                        <div class="pbr-env-picker">
                            <button id="environmentToggle" class="pbr-env-toggle" type="button" aria-expanded="false">
                                <span>Env maps</span>
                                <span id="environmentActiveLabel" class="pbr-env-active-label">Forest</span>
                            </button>
                            <div id="environmentMenu" class="pbr-env-menu" hidden>
                                <button id="environmentBgBtn" class="pbr-env-background-toggle" type="button" aria-pressed="false" title="Show HDRI background">
                                    <span>Background</span>
                                    <span class="pbr-env-background-switch" aria-hidden="true"></span>
                                </button>
                                ${Object.values(ENVIRONMENT_PRESETS)
                                    .filter((preset) => extensionFromPath(preset.url) === 'exr')
                                    .map((preset) => `
                                        <button type="button" data-environment-preset="${preset.id}" aria-pressed="false">
                                            <canvas class="pbr-env-preview" width="56" height="56" data-environment-preview="${preset.id}" aria-hidden="true"></canvas>
                                            <span>${preset.label}</span>
                                        </button>
                                    `).join('')}
                            </div>
                        </div>
                        <button id="environmentDial" class="pbr-light-dial" type="button" role="slider" aria-label="Environment rotation" aria-valuemin="0" aria-valuemax="360" aria-valuenow="0">
                            <canvas id="environmentDialPreview" class="pbr-light-env-preview" width="56" height="56" aria-hidden="true"></canvas>
                            <span class="pbr-light-dial-ring"></span>
                            <span id="environmentDialHand" class="pbr-light-dial-hand"></span>
                        </button>
                        <span class="pbr-light-copy"><span id="environmentRotationValue" class="pbr-light-value">0deg</span></span>
                    </div>
                </div>

                <section id="sectionControls" aria-label="Section controls" hidden>
                    <div class="section-grid">
                        <select id="sectionAxis" aria-label="Section axis">
                            <option value="x">X</option>
                            <option value="y">Y</option>
                            <option value="z">Z</option>
                        </select>
                        <input id="sectionSlider" type="range" min="-100" max="100" step="1" value="0" aria-label="Section position">
                        <span id="sectionValue">0%</span>
                    </div>
                </section>

                <section id="fileInputContainer" aria-label="Open 3D model">
                    <div class="panel-head">
                        <div>
                            <p>Local mesh</p>
                            <h2>Open 3D model</h2>
                        </div>
                        <button id="closePanelBtn" class="icon-button" type="button" title="Close" aria-label="Close panel">${ICONS.close}</button>
                    </div>

                    <div id="dropZone" tabindex="0">
                        ${ICONS.open}
                        <strong>Drop mesh files</strong>
                        <span>GLB, GLTF, OBJ, FBX, PLY, STL with companion textures or MTL</span>
                    </div>

                    <div class="upload-actions">
                        <button id="chooseFilesBtn" class="text-button primary" type="button">Choose files</button>
                        <button id="chooseFolderBtn" class="text-button secondary" type="button">Choose folder</button>
                    </div>

                    <div class="history-row">
                        <select id="historySelect" aria-label="Recent local meshes">
                            <option value="">No saved meshes</option>
                        </select>
                        <button id="clearHistoryBtn" class="text-button secondary" type="button" disabled>Clear</button>
                    </div>
                    <div id="historyUsage" class="storage-row">Saved 0 B / 100 MB</div>

                    <p id="status">Files stay in this browser.</p>
                    <div id="entries" hidden></div>

                    <section id="preflight" hidden>
                        <dl id="preflightFacts"></dl>
                        <div id="preflightResources" class="preflight-resources"></div>
                        <div class="preflight-actions">
                            <button id="relinkBtn" class="text-button secondary" type="button">Add files</button>
                            <button id="loadBtn" class="text-button primary" type="button">Load mesh</button>
                        </div>
                    </section>
                </section>

                <div id="stats" hidden></div>

                <aside id="textureMaps" aria-label="Texture maps" hidden>
                    <div class="texture-head">
                        <button id="textureToggleBtn" type="button" title="Texture maps" aria-label="Toggle texture map previews" aria-expanded="false">
                            <span>TEX</span>
                            <span id="textureCount"></span>
                        </button>
                    </div>
                    <div id="textureStrip"></div>
                </aside>

                <div id="loading" role="status" aria-live="polite" hidden>
                    <div class="loading-head">
                        <strong id="loadingLabel">Loading mesh</strong>
                        <span id="loadingValue">0%</span>
                    </div>
                    <div class="loading-track" aria-hidden="true"><span id="loadingBar"></span></div>
                </div>

                <dialog id="textureDialog">
                    <button id="textureDialogClose" class="text-button secondary" type="button">Close</button>
                    <canvas id="textureDialogCanvas"></canvas>
                    <p id="textureDialogCaption"></p>
                </dialog>

                <input id="fileInput" class="visually-hidden" type="file" accept=".glb,.gltf,.obj,.fbx,.ply,.stl,.mtl,.bin,.png,.jpg,.jpeg,.webp,.avif,.bmp,.gif,.hdr,.exr,.tga,.ktx2" multiple>
                <input id="folderInput" class="visually-hidden" type="file" webkitdirectory directory multiple>
                <input id="relinkInput" class="visually-hidden" type="file" multiple>
            </div>
        `;
    }

    connectedCallback() {
        if (this.connected) return;
        this.connected = true;
        this.initEventListeners();
        this.resizeObserver.observe(this);
        document.addEventListener('keydown', this.boundHandleKeyDown);
        this.applyAttributes();
        this.resizeRenderer();
        this.animate();
        this.loadHistory().catch((error) => {
            this.showStatus(`History unavailable: ${error.message}`, 'warning');
        });
        const source = this.getAttribute('src');
        if (source) {
            this.setPanelOpen(false, { forceClose: true });
            void this.loadModelFromUrl(source, basename(pathWithoutSearch(source)));
        } else {
            this.setEmptyState(true);
        }
    }

    disconnectedCallback() {
        this.connected = false;
        cancelAnimationFrame(this.frameRequest);
        this.resizeObserver.disconnect();
        document.removeEventListener('keydown', this.boundHandleKeyDown);
        this.renderer.domElement.removeEventListener('pointerdown', this.boundPointerDown);
        this.renderer.domElement.removeEventListener('pointerup', this.boundPointerUp);
        this.renderer.domElement.removeEventListener('pointermove', this.boundPointerMove);
        this.controls.dispose();
        this.clearModel();
        this.disposeObject(this.placeholder);
        this.revokeObjectUrls();
        this.disposeEnvironmentResources();
        this.environmentPreviewTextures.forEach((texture) => {
            if (texture !== this.environmentSourceTexture) texture.dispose();
        });
        this.environmentPreviewTextures.clear();
        this.environmentPreviewPromises.clear();
        this.geometryMaterial.dispose();
        this.normalMaterial.dispose();
        this.wireframeMaterial.dispose();
        this.placeholderMaterial.dispose();
        this.grid.geometry.dispose();
        this.grid.material.dispose();
        this.pmremGenerator.dispose();
        this.renderer.dispose();
    }

    attributeChangedCallback(name, oldValue, newValue) {
        if (oldValue === newValue) return;
        if (!this.shadowRoot) return;
        if (name === 'src' && this.connected && newValue) {
            void this.loadModelFromUrl(newValue, basename(pathWithoutSearch(newValue)));
            return;
        }
        this.applyAttributes();
    }

    initEventListeners() {
        this.shadowRoot.querySelector('#openPanelBtn').addEventListener('click', () => this.setPanelOpen(true));
        this.shadowRoot.querySelector('#closePanelBtn').addEventListener('click', () => this.setPanelOpen(false, { forceClose: true }));
        this.shadowRoot.querySelector('#chooseFilesBtn').addEventListener('click', () => this.fileInput.click());
        this.shadowRoot.querySelector('#chooseFolderBtn').addEventListener('click', () => this.folderInput.click());
        this.shadowRoot.querySelector('#loadBtn').addEventListener('click', () => {
            if (this.pendingBundle && this.pendingMainEntry) {
                void this.loadBundle(this.pendingBundle, this.pendingMainEntry, { saveHistory: true });
            }
        });
        this.shadowRoot.querySelector('#relinkBtn').addEventListener('click', () => this.relinkInput.click());
        this.shadowRoot.querySelector('#clearHistoryBtn').addEventListener('click', () => {
            void this.clearHistory();
        });
        this.historySelect.addEventListener('change', () => {
            const record = this.historyRecords.find((entry) => entry.id === this.historySelect.value);
            if (record) void this.loadHistoryRecord(record);
        });
        this.fileInput.addEventListener('change', (event) => {
            void this.handleFileSelection(event.target.files);
            event.target.value = '';
        });
        this.folderInput.addEventListener('change', (event) => {
            void this.handleFileSelection(event.target.files);
            event.target.value = '';
        });
        this.relinkInput.addEventListener('change', (event) => {
            this.mergePendingFiles(event.target.files);
            event.target.value = '';
        });

        const dropZone = this.shadowRoot.querySelector('#dropZone');
        ['dragenter', 'dragover'].forEach((type) => {
            this.rootEl.addEventListener(type, (event) => {
                event.preventDefault();
                this.rootEl.classList.add('dragging');
            });
        });
        ['dragleave', 'drop'].forEach((type) => {
            this.rootEl.addEventListener(type, (event) => {
                event.preventDefault();
                if (type === 'drop' && event.dataTransfer?.files?.length) {
                    void this.handleFileSelection(event.dataTransfer.files);
                }
                this.rootEl.classList.remove('dragging');
            });
        });
        dropZone.addEventListener('keydown', (event) => {
            if (event.key === 'Enter' || event.key === ' ') {
                event.preventDefault();
                this.fileInput.click();
            }
        });

        this.modeToggleBtn.addEventListener('click', (event) => {
            event.stopPropagation();
            this.setModeMenuOpen(this.modeMenu.hidden);
        });
        this.modeMenu.addEventListener('click', (event) => {
            event.stopPropagation();
        });
        this.shadowRoot.querySelectorAll('#modeMenu .mode-button').forEach((button) => {
            button.addEventListener('click', () => {
                this.setViewMode(button.dataset.mode);
                this.setModeMenuOpen(false);
            });
        });
        this.rootEl.addEventListener('pointerdown', (event) => {
            if (this.modeMenu.hidden) return;
            const path = event.composedPath();
            if (!path.includes(this.modeMenu) && !path.includes(this.modeToggleBtn)) {
                this.setModeMenuOpen(false);
            }
        });
        this.textureToggleBtn.addEventListener('click', () => {
            this.setTexturePanelOpen(!this.texturePanelOpen);
        });
        this.shadowRoot.querySelector('#wireBtn').addEventListener('click', () => {
            this.setWireframeEnabled(!this.state.wireframe);
        });
        this.shadowRoot.querySelector('#wireModeSelect').addEventListener('change', (event) => {
            this.state.wireframeMode = normalizeWireframeMode(event.target.value);
            this.setAttribute('wireframe-mode', this.state.wireframeMode);
            if (this.state.wireframe) this.rebuildWireframeOverlay();
        });
        this.shadowRoot.querySelector('#sectionBtn').addEventListener('click', () => {
            this.state.section.enabled = !this.state.section.enabled;
            this.syncSectionUi();
            this.syncSectionClip();
        });
        this.environmentToggle.addEventListener('click', (event) => {
            event.preventDefault();
            this.setEnvironmentMenuOpen(this.environmentMenu.hidden);
        });
        this.shadowRoot.querySelectorAll('[data-environment-preset]').forEach((button) => {
            button.addEventListener('click', (event) => {
                event.preventDefault();
                this.setEnvironmentMenuOpen(false);
                void this.setEnvironment(button.dataset.environmentPreset);
            });
        });
        this.shadowRoot.querySelector('#environmentBgBtn').addEventListener('click', () => {
            this.setEnvironmentBackgroundVisible(!this.state.environmentBackground);
        });
        const degreesFromDialPointer = (event) => {
            const rect = this.environmentDial.getBoundingClientRect();
            const x = event.clientX - rect.left - rect.width / 2;
            const y = event.clientY - rect.top - rect.height / 2;
            return (THREE.MathUtils.radToDeg(Math.atan2(x, -y)) + 360) % 360;
        };
        const applyDialPointer = (event) => {
            event.preventDefault();
            this.setEnvironmentRotationDegrees(degreesFromDialPointer(event));
        };
        this.environmentDial.addEventListener('pointerdown', (event) => {
            this.environmentDial.setPointerCapture(event.pointerId);
            applyDialPointer(event);
        });
        this.environmentDial.addEventListener('pointermove', (event) => {
            if (!this.environmentDial.hasPointerCapture(event.pointerId)) return;
            applyDialPointer(event);
        });
        this.environmentDial.addEventListener('keydown', (event) => {
            const step = event.shiftKey ? 15 : 5;
            const current = THREE.MathUtils.radToDeg(this.state.environmentRotation || 0);
            if (event.key === 'ArrowRight' || event.key === 'ArrowUp') {
                event.preventDefault();
                this.setEnvironmentRotationDegrees(current + step);
            } else if (event.key === 'ArrowLeft' || event.key === 'ArrowDown') {
                event.preventDefault();
                this.setEnvironmentRotationDegrees(current - step);
            } else if (event.key === 'Home') {
                event.preventDefault();
                this.setEnvironmentRotationDegrees(0);
            } else if (event.key === 'End') {
                event.preventDefault();
                this.setEnvironmentRotationDegrees(359);
            }
        });
        this.shadowRoot.querySelector('#sectionAxis').addEventListener('change', (event) => {
            this.state.section.axis = event.target.value;
            this.syncSectionClip();
        });
        this.shadowRoot.querySelector('#sectionSlider').addEventListener('input', (event) => {
            this.state.section.value = Number(event.target.value) / 100;
            this.syncSectionUi();
            this.syncSectionClip();
        });
        this.shadowRoot.querySelector('#fitBtn').addEventListener('click', () => this.fitCameraToModel({ animate: false }));
        this.shadowRoot.querySelector('#resetBtn').addEventListener('click', () => this.resetView());
        this.shadowRoot.querySelector('#rotateBtn').addEventListener('click', () => {
            this.state.autoRotate = !this.state.autoRotate;
            this.reflectBooleanAttribute('auto-rotate', this.state.autoRotate);
            this.syncToolbar();
        });
        this.shadowRoot.querySelector('#gridBtn').addEventListener('click', () => {
            this.state.grid = !this.state.grid;
            this.grid.visible = this.state.grid;
            this.syncToolbar();
        });
        this.shadowRoot.querySelector('#playBtn').addEventListener('click', () => {
            this.setAnimationPlaying(!this.isAnimationPlaying);
        });
        this.shadowRoot.querySelector('#snapshotBtn').addEventListener('click', () => {
            void this.captureScreenshot({ download: true });
        });
        this.shadowRoot.querySelector('#fullscreenBtn').addEventListener('click', () => {
            void this.toggleFullscreen();
        });
        this.shadowRoot.querySelector('#textureDialogClose').addEventListener('click', () => {
            this.textureDialog.close();
        });

        this.renderer.domElement.addEventListener('pointerdown', this.boundPointerDown);
        this.renderer.domElement.addEventListener('pointerup', this.boundPointerUp);
        this.renderer.domElement.addEventListener('pointermove', this.boundPointerMove);
        this.controls.addEventListener('change', () => {
            this.emitEvent('viewer-camera-change', this.getCameraStateSnapshot());
        });
    }

    applyAttributes() {
        this.state.viewMode = normalizeViewMode(this.getAttribute('view-mode') || this.state.viewMode);
        this.state.wireframeMode = normalizeWireframeMode(this.getAttribute('wireframe-mode') || this.state.wireframeMode);
        this.state.autoRotate = this.hasAttribute('auto-rotate');
        this.state.anglePerSecond = parseNumber(this.getAttribute('angle-per-second'), this.state.anglePerSecond);
        this.state.backgroundColor = this.getAttribute('background-color') || this.state.backgroundColor || DEFAULT_BACKGROUND_COLOR;
        this.state.environment = this.getAttribute('environment') || this.state.environment || DEFAULT_ENVIRONMENT_PRESET;
        this.state.environmentUrl = this.getAttribute('environment-url') || '';
        const preset = environmentPresetFor(this.state.environment);
        this.state.environmentIntensity = parseNumber(this.getAttribute('environment-intensity'), this.state.environmentIntensity || preset.environmentIntensity);
        this.state.environmentBackground = parseBooleanAttribute(
            this.getAttribute('environment-background'),
            this.state.environmentBackground
        );
        this.state.environmentRotation = this.hasAttribute('environment-rotation')
            ? parseNumber(this.getAttribute('environment-rotation'), this.state.environmentRotation || 0)
            : (this.state.environmentRotation || 0);
        this.state.selectionMode = this.getAttribute('selection-mode') || this.state.selectionMode;
        this.state.performanceMode = this.getAttribute('performance-mode') || this.state.performanceMode;
        this.state.animationSpeed = parseNumber(this.getAttribute('animation-speed'), this.state.animationSpeed);
        this.state.animationLoop = this.getAttribute('animation-loop') || this.state.animationLoop;
        this.renderer.toneMappingExposure = parseNumber(this.getAttribute('exposure'), this.renderer.toneMappingExposure || 1);
        this.initialCameraOrbit = this.parseCameraOrbit(this.getAttribute('camera-orbit'));
        this.camera.up.copy(this.parseVector3Attribute('camera-up') || new THREE.Vector3(0, 1, 0));
        this.rootEl?.style.setProperty('--viewer-bg-color', this.state.backgroundColor || DEFAULT_BACKGROUND_COLOR);
        this.rebuildEnvironmentTexture();
        this.applyEnvironmentPresentation();
        this.applyViewMode();
        this.syncToolbar();
        this.syncSectionUi();
        void this.loadEnvironment();
    }

    parseVector3Attribute(name) {
        const value = this.getAttribute(name);
        if (!value) return null;
        const entries = value.trim().split(/\s+/).map((part) => Number(part));
        if (entries.length < 3 || entries.some((entry) => !Number.isFinite(entry))) return null;
        return new THREE.Vector3(entries[0], entries[1], entries[2]);
    }

    parseCameraOrbit(value) {
        if (!value) return null;
        const tokens = value.trim().split(/\s+/);
        if (tokens.length < 3) return null;
        const hasUnits = /deg|rad|m/i.test(value);
        const numeric = tokens.slice(0, 3).map((token) => Number(String(token).replace(/(?:deg|rad|m)$/gi, '')));
        if (numeric.some((entry) => !Number.isFinite(entry))) return null;
        if (!hasUnits) return new THREE.Vector3(numeric[0], numeric[1], numeric[2]);
        const theta = /rad/i.test(tokens[0]) ? numeric[0] : THREE.MathUtils.degToRad(numeric[0]);
        const phi = /rad/i.test(tokens[1]) ? numeric[1] : THREE.MathUtils.degToRad(numeric[1]);
        const radius = Math.max(0.01, numeric[2]);
        return new THREE.Vector3(
            radius * Math.sin(phi) * Math.sin(theta),
            radius * Math.cos(phi),
            radius * Math.sin(phi) * Math.cos(theta)
        );
    }

    reflectBooleanAttribute(name, enabled) {
        if (enabled) {
            if (!this.hasAttribute(name)) this.setAttribute(name, '');
        } else if (this.hasAttribute(name)) {
            this.removeAttribute(name);
        }
    }

    createGeometryMaterial() {
        return new THREE.ShaderMaterial({
            uniforms: {
                baseColor: { value: new THREE.Color(0xd8d0c7) },
                rimColor: { value: new THREE.Color(0x5db8a6) }
            },
            vertexShader: /*glsl*/`
                varying vec3 vViewNormal;
                varying vec3 vViewPosition;

                void main() {
                    vec4 viewPosition = modelViewMatrix * vec4(position, 1.0);
                    vViewPosition = -viewPosition.xyz;
                    vViewNormal = normalize(normalMatrix * normal);
                    gl_Position = projectionMatrix * viewPosition;
                }
            `,
            fragmentShader: /*glsl*/`
                uniform vec3 baseColor;
                uniform vec3 rimColor;
                varying vec3 vViewNormal;
                varying vec3 vViewPosition;

                void main() {
                    vec3 normal = normalize(vViewNormal);
                    vec3 viewDir = normalize(vViewPosition);
                    vec3 keyDir = normalize(vec3(-0.38, 0.58, 0.72));
                    vec3 fillDir = normalize(vec3(0.72, 0.18, 0.42));
                    float key = max(dot(normal, keyDir), 0.0);
                    float fill = max(dot(normal, fillDir), 0.0);
                    float fresnel = pow(1.0 - max(dot(normal, viewDir), 0.0), 2.35);
                    float shade = 0.28 + key * 0.62 + fill * 0.22;
                    vec3 color = baseColor * shade + rimColor * fresnel * 0.28;
                    gl_FragColor = vec4(color, 1.0);
                }
            `,
            side: THREE.DoubleSide,
            toneMapped: false
        });
    }

    createAlbedoMaterialEntry(materialEntry) {
        return mapMaterialEntry(materialEntry, (material) => {
            const color = material?.color?.clone ? material.color.clone() : new THREE.Color(0xffffff);
            const preview = new THREE.MeshBasicMaterial({
                color,
                map: material?.map || null,
                alphaMap: material?.alphaMap || null,
                vertexColors: material?.vertexColors === true,
                transparent: material?.transparent === true || Number(material?.opacity) < 1 || Boolean(material?.alphaMap),
                opacity: Number.isFinite(material?.opacity) ? material.opacity : 1,
                side: material?.side ?? THREE.FrontSide,
                toneMapped: false
            });
            preview.name = material?.name ? `${material.name} Albedo` : 'Albedo Preview';
            return preview;
        });
    }

    createScalarPreviewMaterialEntry(materialEntry, property, scalarProperty, channelIndex, label) {
        return mapMaterialEntry(materialEntry, (material) => {
            const texture = material?.[property] || null;
            const scalar = Number.isFinite(material?.[scalarProperty]) ? material[scalarProperty] : 1;
            const preview = new THREE.ShaderMaterial({
                uniforms: {
                    mapTexture: { value: texture },
                    scalarValue: { value: scalar },
                    channelIndex: { value: channelIndex },
                    hasMap: { value: texture?.isTexture === true ? 1 : 0 }
                },
                vertexShader: /*glsl*/`
                    varying vec2 vUv;
                    void main() {
                        vUv = uv;
                        gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
                    }
                `,
                fragmentShader: /*glsl*/`
                    uniform sampler2D mapTexture;
                    uniform float scalarValue;
                    uniform int channelIndex;
                    uniform int hasMap;
                    varying vec2 vUv;

                    void main() {
                        float value = scalarValue;
                        if (hasMap == 1) {
                            vec4 texel = texture2D(mapTexture, vUv);
                            if (channelIndex == 0) value = texel.r;
                            else if (channelIndex == 1) value = texel.g;
                            else if (channelIndex == 2) value = texel.b;
                            else value = texel.a;
                        }
                        gl_FragColor = vec4(vec3(value), 1.0);
                    }
                `,
                side: material?.side ?? THREE.FrontSide,
                toneMapped: false
            });
            preview.name = material?.name ? `${material.name} ${label}` : `${label} Preview`;
            return preview;
        });
    }

    createPlaceholder() {
        const group = new THREE.Group();
        const sphere = new THREE.Mesh(
            new THREE.IcosahedronGeometry(0.72, 2),
            this.placeholderMaterial
        );
        const wire = new THREE.LineSegments(
            new THREE.WireframeGeometry(sphere.geometry),
            new THREE.LineBasicMaterial({
                color: 0x64748b,
                transparent: true,
                opacity: 0.3
            })
        );
        wire.renderOrder = 2;
        sphere.add(wire);
        group.add(sphere);
        return group;
    }

    async loadEnvironment() {
        const preset = environmentPresetFor(this.state.environment);
        const url = this.state.environmentUrl || preset.url || ENVIRONMENT_URLS[this.state.environment] || '';
        if (!url) {
            this.disposeEnvironmentResources();
            this.environmentUrl = '';
            this.applyEnvironmentPresentation();
            return;
        }
        if (this.environmentUrl === url && this.environmentSourceTexture) {
            this.rebuildEnvironmentTexture();
            this.applyEnvironmentPresentation();
            return;
        }
        try {
            const loader = extensionFromPath(url) === 'exr' ? this.exrLoader : this.rgbeLoader;
            const texture = await new Promise((resolve, reject) => {
                loader.load(url, resolve, undefined, reject);
            });
            texture.mapping = THREE.EquirectangularReflectionMapping;
            texture.userData.url = url;
            this.disposeEnvironmentResources();
            this.environmentSourceTexture = texture;
            this.environmentUrl = url;
            this.rebuildEnvironmentTexture();
            this.applyEnvironmentPresentation();
        } catch (error) {
            this.scene.environment = null;
            this.scene.background = null;
            console.warn('Environment load failed:', error);
        }
    }

    disposeEnvironmentResources() {
        if (this.environmentRenderTarget) {
            this.environmentRenderTarget.dispose();
            this.environmentRenderTarget = null;
        }
        if (
            this.environmentTexture
            && this.environmentTexture !== this.environmentSourceTexture
            && this.environmentTexture.userData?.smvRotatedEnvironment
        ) {
            this.environmentTexture.dispose();
        }
        this.environmentTexture = null;
        if (this.environmentSourceTexture) {
            this.environmentSourceTexture.dispose();
            this.environmentSourceTexture = null;
        }
        this.environmentRotationKey = '';
    }

    environmentRotationRadians() {
        const preset = environmentPresetFor(this.state.environment);
        return Number(this.state.environmentRotation || 0) + Number(preset.rotation || 0);
    }

    rebuildEnvironmentTexture() {
        if (!this.environmentSourceTexture) return;
        const rotation = this.environmentRotationRadians();
        const width = this.environmentSourceTexture.image?.width || 0;
        const rotationKey = `${this.environmentUrl}:${environmentRotationShift(rotation, width)}`;
        if (rotationKey === this.environmentRotationKey && this.environmentTexture && this.environmentRenderTarget) return;
        if (this.environmentRenderTarget) {
            this.environmentRenderTarget.dispose();
            this.environmentRenderTarget = null;
        }
        if (
            this.environmentTexture
            && this.environmentTexture !== this.environmentSourceTexture
            && this.environmentTexture.userData?.smvRotatedEnvironment
        ) {
            this.environmentTexture.dispose();
        }
        this.environmentTexture = createRotatedEnvironmentTexture(this.environmentSourceTexture, rotation);
        this.environmentRenderTarget = this.pmremGenerator.fromEquirectangular(this.environmentTexture);
        this.environmentRotationKey = rotationKey;
    }

    applyEnvironmentPresentation() {
        if (!this.scene) return;
        const preset = environmentPresetFor(this.state?.environment);
        const texture = this.environmentRenderTarget?.texture || null;
        this.scene.environment = texture;
        this.scene.background = this.state.environmentBackground && this.environmentTexture ? this.environmentTexture : null;
        if (this.scene.environmentIntensity !== undefined) {
            this.scene.environmentIntensity = this.state.environmentIntensity || preset.environmentIntensity || 1;
        }
        if (this.scene.backgroundIntensity !== undefined) {
            this.scene.backgroundIntensity = preset.backgroundIntensity || 1;
        }
        if (this.scene.backgroundBlurriness !== undefined) {
            this.scene.backgroundBlurriness = preset.backgroundBlurriness || 0;
        }
        this.applyEnvironmentIntensity();
        this.syncToolbar();
    }

    async loadEnvironmentPreviewTexture(presetId) {
        const preset = environmentPresetFor(presetId);
        if (this.environmentUrl === preset.url && this.environmentSourceTexture) {
            return this.environmentSourceTexture;
        }
        if (this.environmentPreviewTextures.has(preset.id)) {
            return this.environmentPreviewTextures.get(preset.id);
        }
        if (!this.environmentPreviewPromises.has(preset.id)) {
            const loader = extensionFromPath(preset.url) === 'exr' ? this.exrLoader : this.rgbeLoader;
            this.environmentPreviewPromises.set(
                preset.id,
                new Promise((resolve, reject) => {
                    loader.load(preset.url, resolve, undefined, reject);
                }).then((texture) => {
                    texture.mapping = THREE.EquirectangularReflectionMapping;
                    this.environmentPreviewTextures.set(preset.id, texture);
                    return texture;
                })
            );
        }
        return this.environmentPreviewPromises.get(preset.id);
    }

    renderEnvironmentPreview(canvas, presetId, degrees = this.environmentRotationDegrees()) {
        if (!canvas) return;
        const preset = environmentPresetFor(presetId);
        const token = `${preset.id}:${Math.round(degrees)}`;
        if (canvas.dataset.envPreviewToken === token) return;
        canvas.dataset.envPreviewToken = token;
        this.loadEnvironmentPreviewTexture(preset.id)
            .then((texture) => {
                if (canvas.dataset.envPreviewToken !== token) return;
                drawEnvironmentPreviewCanvas(canvas, texture, preset, degrees);
            })
            .catch((error) => {
                console.warn('HDR environment preview failed:', error);
            });
    }

    applyEnvironmentIntensity() {
        if (!this.model) return;
        this.model.traverse((node) => {
            if (!node.isMesh) return;
            const original = node.userData.smvOriginalMaterial || node.material;
            materialArray(original).forEach((material) => {
                if ('envMapIntensity' in material) {
                    material.envMapIntensity = this.state.environmentIntensity;
                    material.needsUpdate = true;
                }
            });
        });
    }

    async handleFileSelection(fileList) {
        const files = Array.from(fileList || []);
        if (!files.length) return;
        const bundle = this.createFileBundle(files);
        if (!bundle.mainEntries.length) {
            this.pendingBundle = null;
            this.pendingMainEntry = null;
            this.entriesEl.hidden = true;
            this.preflightEl.hidden = true;
            this.showStatus('No supported mesh file found.', 'error');
            return;
        }
        this.pendingBundle = bundle;
        this.pendingMainEntry = bundle.mainEntries[0];
        this.renderEntries(bundle);
        await this.renderPreflight();
        this.setPanelOpen(true);
        this.showStatus(`${bundle.mainEntries.length} mesh ${bundle.mainEntries.length === 1 ? 'entry' : 'entries'} ready.`, 'ready');
    }

    mergePendingFiles(fileList) {
        const files = Array.from(fileList || []);
        if (!files.length) return;
        const nextEntries = this.pendingBundle ? [...this.pendingBundle.entries] : [];
        const byPath = new Map(nextEntries.map((entry) => [entry.path.toLowerCase(), entry]));
        files.forEach((file) => {
            const entry = this.createBundleEntry(file);
            byPath.set(entry.path.toLowerCase(), entry);
        });
        const bundle = this.createFileBundle([...byPath.values()].map((entry) => entry.file));
        bundle.entries = [...byPath.values()].sort((a, b) => a.path.localeCompare(b.path));
        bundle.mainEntries = bundle.entries.filter((entry) => SUPPORTED_MODEL_EXTENSIONS.has(entry.extension));
        bundle.bytes = bundle.entries.reduce((sum, entry) => sum + entry.file.size, 0);
        this.pendingBundle = bundle;
        if (!this.pendingMainEntry || !bundle.entries.includes(this.pendingMainEntry)) {
            this.pendingMainEntry = bundle.mainEntries[0] || null;
        }
        this.renderEntries(bundle);
        void this.renderPreflight();
    }

    createBundleEntry(file) {
        const path = filePathFor(file);
        return {
            file,
            path,
            name: basename(path),
            extension: extensionFromPath(path),
            size: Number(file.size) || 0
        };
    }

    createFileBundle(files) {
        const entries = files.map((file) => this.createBundleEntry(file))
            .filter((entry) => SUPPORTED_MODEL_EXTENSIONS.has(entry.extension) || COMPANION_EXTENSIONS.has(entry.extension))
            .sort((a, b) => a.path.localeCompare(b.path));
        return {
            entries,
            mainEntries: entries.filter((entry) => SUPPORTED_MODEL_EXTENSIONS.has(entry.extension)),
            bytes: entries.reduce((sum, entry) => sum + entry.size, 0)
        };
    }

    renderEntries(bundle) {
        const entries = bundle.mainEntries || [];
        this.entriesEl.hidden = entries.length <= 1;
        this.entriesEl.innerHTML = entries.map((entry) => {
            const active = entry === this.pendingMainEntry ? ' active' : '';
            return `
                <button class="entry-button${active}" type="button" data-path="${escapeHtml(entry.path)}">
                    <strong>${escapeHtml(entry.name)}</strong>
                    <span>${escapeHtml(entry.extension.toUpperCase())} ${formatBytes(entry.size)}</span>
                </button>
            `;
        }).join('');
        this.entriesEl.querySelectorAll('.entry-button').forEach((button) => {
            button.addEventListener('click', () => {
                const path = button.dataset.path;
                this.pendingMainEntry = entries.find((entry) => entry.path === path) || entries[0];
                this.renderEntries(bundle);
                void this.renderPreflight();
            });
        });
    }

    async renderPreflight() {
        if (!this.pendingBundle || !this.pendingMainEntry) {
            this.preflightEl.hidden = true;
            return;
        }
        const entry = this.pendingMainEntry;
        const references = await this.inspectReferences(this.pendingBundle, entry);
        const presentCount = references.filter((reference) => reference.present).length;
        const missingCount = references.filter((reference) => !reference.present).length;
        this.preflightFactsEl.innerHTML = `
            <div><dt>Format</dt><dd>${escapeHtml(entry.extension.toUpperCase())}</dd></div>
            <div><dt>Mesh</dt><dd>${escapeHtml(entry.name)}</dd></div>
            <div><dt>Bundle</dt><dd>${formatBytes(this.pendingBundle.bytes)}</dd></div>
            <div><dt>Files</dt><dd>${this.pendingBundle.entries.length}</dd></div>
        `;
        if (references.length) {
            const state = missingCount ? `${missingCount} missing, ${presentCount} linked` : `${presentCount} linked`;
            this.preflightResourcesEl.textContent = `References: ${state}`;
            this.preflightResourcesEl.dataset.state = missingCount ? 'warning' : 'ready';
        } else {
            this.preflightResourcesEl.textContent = 'References: none detected';
            this.preflightResourcesEl.dataset.state = 'ready';
        }
        this.preflightEl.hidden = false;
    }

    async inspectReferences(bundle, mainEntry) {
        const lowerPaths = new Set(bundle.entries.map((entry) => entry.path.toLowerCase()));
        const lowerNames = new Set(bundle.entries.map((entry) => entry.name.toLowerCase()));
        const resolvePresent = (reference) => {
            const clean = normalizePath(reference);
            return lowerPaths.has(joinPath(dirname(mainEntry.path), clean).toLowerCase())
                || lowerPaths.has(clean.toLowerCase())
                || lowerNames.has(basename(clean).toLowerCase());
        };
        try {
            if (mainEntry.extension === 'gltf') {
                const json = JSON.parse(await mainEntry.file.text());
                const refs = [];
                (json.buffers || []).forEach((buffer) => {
                    if (buffer.uri && !isExternalUrl(buffer.uri)) refs.push(buffer.uri);
                });
                (json.images || []).forEach((image) => {
                    if (image.uri && !isExternalUrl(image.uri)) refs.push(image.uri);
                });
                return refs.map((reference) => ({ reference, present: resolvePresent(reference) }));
            }
            if (mainEntry.extension === 'obj') {
                const text = await mainEntry.file.text();
                const refs = [];
                text.split(/\r?\n/).forEach((line) => {
                    const trimmed = line.trim();
                    if (/^mtllib\s+/i.test(trimmed)) {
                        refs.push(trimmed.replace(/^mtllib\s+/i, '').trim());
                    }
                });
                const mtlEntries = refs
                    .map((reference) => this.findEntryForReference(bundle, dirname(mainEntry.path), reference))
                    .filter(Boolean);
                for (const mtlEntry of mtlEntries) {
                    const mtlText = await mtlEntry.file.text();
                    mtlText.split(/\r?\n/).forEach((line) => {
                        const trimmed = line.trim();
                        if (/^map_/i.test(trimmed) || /^bump\s+/i.test(trimmed) || /^disp\s+/i.test(trimmed) || /^decal\s+/i.test(trimmed)) {
                            const parts = trimmed.split(/\s+/);
                            const reference = parts[parts.length - 1];
                            if (reference && !reference.startsWith('-')) refs.push(joinPath(dirname(mtlEntry.path), reference));
                        }
                    });
                }
                return refs.map((reference) => ({ reference, present: resolvePresent(reference) }));
            }
        } catch (error) {
            console.warn('Preflight inspection failed:', error);
        }
        return [];
    }

    findEntryForReference(bundle, baseDir, reference) {
        const candidates = [
            normalizePath(reference).toLowerCase(),
            joinPath(baseDir, reference).toLowerCase(),
            basename(reference).toLowerCase()
        ];
        return bundle.entries.find((entry) => {
            const path = entry.path.toLowerCase();
            const name = entry.name.toLowerCase();
            return candidates.includes(path) || candidates.includes(name);
        }) || null;
    }

    async loadBundle(bundle, mainEntry, options = {}) {
        if (!bundle || !mainEntry) return;
        const token = ++this.loadToken;
        this.setLoading(true, 0, `Loading ${mainEntry.name}`);
        try {
            const resolver = this.createBundleResolver(bundle, mainEntry);
            const scene = await this.parseSceneFromEntry(mainEntry, resolver);
            if (token !== this.loadToken) {
                this.disposeObject(scene);
                return;
            }
            await this.setModel(scene, {
                source: resolver.mainUrl,
                fileName: mainEntry.name,
                format: mainEntry.extension,
                animations: scene.userData.smvAnimations || [],
                keepObjectUrls: true
            });
            this.currentSource = null;
            if (options.saveHistory !== false) {
                void this.saveBundleToHistory(bundle, mainEntry);
            }
            this.setPanelOpen(false);
            this.showStatus(`Loaded ${mainEntry.name}`, 'ready');
        } catch (error) {
            this.emitViewerError('load-file-bundle', error, { fileName: mainEntry.name });
            this.showStatus(error.message || 'Mesh load failed.', 'error');
        } finally {
            if (token === this.loadToken) this.setLoading(false);
        }
    }

    createBundleResolver(bundle, mainEntry) {
        this.revokeObjectUrls();
        const urlByPath = new Map();
        const urlByName = new Map();
        bundle.entries.forEach((entry) => {
            const url = URL.createObjectURL(entry.file);
            this.objectUrls.add(url);
            urlByPath.set(entry.path.toLowerCase(), url);
            if (!urlByName.has(entry.name.toLowerCase())) {
                urlByName.set(entry.name.toLowerCase(), url);
            }
        });
        const baseDir = dirname(mainEntry.path);
        const manager = new THREE.LoadingManager();
        manager.setURLModifier((url) => {
            if (isExternalUrl(url)) return url;
            const clean = normalizePath(decodeURIComponent(pathWithoutSearch(url)));
            const candidates = [
                clean.toLowerCase(),
                joinPath(baseDir, clean).toLowerCase(),
                basename(clean).toLowerCase()
            ];
            for (const candidate of candidates) {
                if (urlByPath.has(candidate)) return urlByPath.get(candidate);
                if (urlByName.has(candidate)) return urlByName.get(candidate);
            }
            return url;
        });
        return {
            bundle,
            manager,
            baseDir,
            baseUrl: baseDir ? `${baseDir}/` : '',
            mainUrl: urlByPath.get(mainEntry.path.toLowerCase())
        };
    }

    async parseSceneFromEntry(entry, resolver) {
        const format = entry.extension;
        const buffer = await entry.file.arrayBuffer();
        if (format === 'glb' || format === 'gltf') {
            const loader = this.createGltfLoader(resolver.manager);
            const gltf = await new Promise((resolve, reject) => {
                loader.parse(buffer, resolver.baseUrl, resolve, reject);
            });
            gltf.scene.userData.smvAnimations = gltf.animations || [];
            return gltf.scene;
        }
        if (format === 'obj') {
            const text = new TextDecoder('utf-8').decode(buffer);
            const loader = new OBJLoader(resolver.manager);
            const mtlReference = this.extractMtlReference(text);
            const mtlEntry = mtlReference
                ? this.findEntryForReference(resolver.bundle, resolver.baseDir, mtlReference)
                : null;
            if (mtlEntry) {
                const materials = new MTLLoader(resolver.manager).parse(await mtlEntry.file.text(), dirname(mtlEntry.path) ? `${dirname(mtlEntry.path)}/` : '');
                materials.preload();
                loader.setMaterials(materials);
            }
            return loader.parse(text);
        }
        if (format === 'fbx') {
            const object = new FBXLoader(resolver.manager).parse(buffer, resolver.baseUrl);
            object.userData.smvAnimations = object.animations || [];
            return object;
        }
        if (format === 'ply') {
            return this.geometryAsScene(new PLYLoader(resolver.manager).parse(buffer), entry.name);
        }
        if (format === 'stl') {
            return this.geometryAsScene(new STLLoader(resolver.manager).parse(buffer), entry.name);
        }
        throw new Error(`Unsupported mesh format: ${format}`);
    }

    extractMtlReference(objText) {
        const line = objText.split(/\r?\n/).find((entry) => /^mtllib\s+/i.test(entry.trim()));
        return line ? line.trim().replace(/^mtllib\s+/i, '').trim() : '';
    }

    geometryAsScene(geometry, name = 'mesh') {
        if (!geometry.attributes.normal) geometry.computeVertexNormals();
        const material = new THREE.MeshStandardMaterial({
            color: 0xd8d0c7,
            roughness: 0.76,
            metalness: 0.02,
            vertexColors: geometry.hasAttribute('color')
        });
        const mesh = new THREE.Mesh(geometry, material);
        mesh.name = name;
        const group = new THREE.Group();
        group.name = sanitizeFilenameSegment(name, 'mesh');
        group.add(mesh);
        return group;
    }

    async loadModelFromFile(file) {
        const bundle = this.createFileBundle([file]);
        const entry = bundle.mainEntries[0];
        if (!entry) throw new Error('Unsupported model file');
        await this.loadBundle(bundle, entry, { saveHistory: true });
    }

    async loadModelFromUrl(url, fileName = basename(pathWithoutSearch(url))) {
        if (!url) return;
        const token = ++this.loadToken;
        this.setPanelOpen(false, { forceClose: true });
        this.setLoading(true, 0, `Loading ${fileName || 'mesh'}`);
        try {
            this.revokeObjectUrls();
            const format = extensionFromPath(fileName || url) || 'glb';
            const scene = await this.loadSceneFromUrl(url, format);
            if (token !== this.loadToken) {
                this.disposeObject(scene);
                return;
            }
            await this.setModel(scene, {
                source: url,
                fileName: fileName || basename(pathWithoutSearch(url)),
                format,
                animations: scene.userData.smvAnimations || [],
                keepObjectUrls: false
            });
            this.currentSource = url;
            this.setPanelOpen(false, { forceClose: true });
            this.showStatus(`Loaded ${fileName || 'model'}`, 'ready');
        } catch (error) {
            this.emitViewerError('load-url', error, { source: url });
            this.showStatus(error.message || 'Mesh load failed.', 'error');
            if (!this.model) this.setEmptyState(true);
        } finally {
            if (token === this.loadToken) this.setLoading(false);
        }
    }

    async loadSceneFromUrl(url, format) {
        if (format === 'glb' || format === 'gltf') {
            const loader = this.createGltfLoader();
            const gltf = await this.loaderLoad(loader, url);
            gltf.scene.userData.smvAnimations = gltf.animations || [];
            return gltf.scene;
        }
        if (format === 'obj') {
            return this.loadRemoteObj(url);
        }
        if (format === 'fbx') {
            const object = await this.loaderLoad(new FBXLoader(), url);
            object.userData.smvAnimations = object.animations || [];
            return object;
        }
        if (format === 'ply') {
            const geometry = await this.loaderLoad(new PLYLoader(), url);
            return this.geometryAsScene(geometry, basename(pathWithoutSearch(url)));
        }
        if (format === 'stl') {
            const geometry = await this.loaderLoad(new STLLoader(), url);
            return this.geometryAsScene(geometry, basename(pathWithoutSearch(url)));
        }
        throw new Error(`Unsupported mesh format: ${format}`);
    }

    createGltfLoader(manager = undefined) {
        const loader = new GLTFLoader(manager);
        const dracoLoader = new DRACOLoader(manager);
        dracoLoader.setDecoderPath('/vendor/three/examples/jsm/libs/draco/gltf/');
        loader.setDRACOLoader(dracoLoader);
        return loader;
    }

    loaderLoad(loader, url) {
        return new Promise((resolve, reject) => {
            loader.load(
                url,
                resolve,
                (event) => {
                    if (event.lengthComputable && event.total) {
                        this.setLoading(true, event.loaded / event.total, 'Loading mesh');
                    }
                },
                reject
            );
        });
    }

    async loadRemoteObj(url) {
        const objText = await this.fetchText(url);
        const loader = new OBJLoader();
        const mtlReference = this.extractMtlReference(objText);
        if (mtlReference) {
            try {
                const mtlUrl = new URL(mtlReference, url).href;
                const mtlText = await this.fetchText(mtlUrl);
                const materials = new MTLLoader().parse(mtlText, new URL('.', mtlUrl).href);
                materials.preload();
                loader.setMaterials(materials);
            } catch (error) {
                console.warn('MTL load failed:', error);
            }
        }
        return loader.parse(objText);
    }

    async fetchText(url) {
        const response = await fetch(url);
        if (!response.ok) throw new Error(`Request failed: ${response.status}`);
        return response.text();
    }

    async setModel(scene, options = {}) {
        this.clearModel({ keepObjectUrls: options.keepObjectUrls === true });
        this.model = scene;
        this.currentFileName = options.fileName || 'model';
        this.animations = options.animations || [];
        this.prepareModel(scene);
        this.scene.add(scene);
        this.placeholder.visible = false;
        this.setEmptyState(false);
        this.applyEnvironmentIntensity();
        this.applyViewMode();
        this.setWireframeEnabled(this.state.wireframe, { rebuild: true });
        this.syncSectionClip();
        this.fitCameraToModel({ animate: false, useInitialOrbit: true });
        this.setupAnimations();
        this.updateStats();
        this.texturePanelOpen = false;
        this.updateTextureMaps();
        this.emitEvent('viewer-load', {
            source: options.source || null,
            fileName: this.currentFileName,
            format: options.format || extensionFromPath(this.currentFileName),
            meshCount: this.meshParts.length,
            ...this.getStats()
        });
    }

    prepareModel(model) {
        this.meshParts = [];
        model.traverse((node) => {
            if (!node.isMesh || !node.geometry) return;
            if (!node.geometry.attributes.normal) node.geometry.computeVertexNormals();
            if (node.geometry.attributes.uv && !node.geometry.attributes.uv2) {
                node.geometry.setAttribute('uv2', node.geometry.attributes.uv);
            }
            node.castShadow = true;
            node.receiveShadow = true;
            node.userData.smvOriginalMaterial = node.material;
            materialArray(node.material).forEach((material) => this.prepareMaterial(material));
            node.userData.smvAlbedoMaterial = this.createAlbedoMaterialEntry(node.material);
            node.userData.smvRoughnessMaterial = this.createScalarPreviewMaterialEntry(node.material, 'roughnessMap', 'roughness', 1, 'Roughness');
            node.userData.smvMetalnessMaterial = this.createScalarPreviewMaterialEntry(node.material, 'metalnessMap', 'metalness', 2, 'Metalness');
            this.meshParts.push(node);
        });
    }

    prepareMaterial(material) {
        if (!material) return;
        const maxAnisotropy = Math.min(
            this.renderer?.capabilities?.getMaxAnisotropy?.() || 1,
            MAX_TEXTURE_ANISOTROPY_CAP
        );
        setTextureColorSpace(material.map, true);
        setTextureColorSpace(material.emissiveMap, true);
        ['roughnessMap', 'metalnessMap', 'normalMap', 'aoMap', 'alphaMap', 'bumpMap', 'displacementMap'].forEach((key) => {
            setTextureColorSpace(material[key], false);
            setTextureSampling(material[key], maxAnisotropy);
        });
        setTextureSampling(material.map, maxAnisotropy);
        setTextureSampling(material.emissiveMap, maxAnisotropy);
        if ('envMapIntensity' in material) material.envMapIntensity = this.state.environmentIntensity;
        applyPbrDisplayLook(material);
        material.needsUpdate = true;
    }

    setupAnimations() {
        if (this.mixer) {
            this.mixer.stopAllAction();
            this.mixer = null;
            this.currentAction = null;
        }
        const playButton = this.shadowRoot.querySelector('#playBtn');
        playButton.disabled = !this.animations.length;
        if (!this.model || !this.animations.length) {
            playButton.innerHTML = ICONS.play;
            return;
        }
        this.mixer = new THREE.AnimationMixer(this.model);
        const requested = this.getAttribute('animation');
        const clip = this.resolveAnimationClip(requested) || this.animations[0];
        this.currentAction = this.mixer.clipAction(clip);
        this.configureAnimationAction(this.currentAction);
        if (this.isAnimationPlaying) this.currentAction.play();
        playButton.innerHTML = this.isAnimationPlaying ? ICONS.pause : ICONS.play;
    }

    resolveAnimationClip(selection) {
        if (!selection || !this.animations.length) return null;
        if (/^\d+$/.test(selection)) return this.animations[Number(selection)] || null;
        const normalized = selection.toLowerCase();
        return this.animations.find((clip) => clip.name === selection || clip.name.toLowerCase() === normalized) || null;
    }

    configureAnimationAction(action) {
        if (!action) return;
        action.setEffectiveTimeScale(this.state.animationSpeed);
        if (this.state.animationLoop === 'once') {
            action.setLoop(THREE.LoopOnce, 1);
            action.clampWhenFinished = true;
        } else if (this.state.animationLoop === 'ping-pong') {
            action.setLoop(THREE.LoopPingPong, Infinity);
        } else {
            action.setLoop(THREE.LoopRepeat, Infinity);
        }
    }

    setAnimationPlaying(enabled) {
        this.isAnimationPlaying = enabled === true;
        if (this.currentAction) {
            this.currentAction.paused = !this.isAnimationPlaying;
            if (this.isAnimationPlaying && !this.currentAction.isRunning()) this.currentAction.play();
        }
        this.shadowRoot.querySelector('#playBtn').innerHTML = this.isAnimationPlaying ? ICONS.pause : ICONS.play;
        this.emitEvent('viewer-animation-change', {
            playing: this.isAnimationPlaying,
            animation: this.currentAction?._clip?.name || null
        });
    }

    clearModel(options = {}) {
        this.selectionHelper.visible = false;
        this.selectedMesh = null;
        if (this.mixer) {
            this.mixer.stopAllAction();
            this.mixer = null;
            this.currentAction = null;
        }
        if (this.model) {
            this.clearWireframeOverlay();
            this.scene.remove(this.model);
            this.disposeObject(this.model);
            this.model = null;
        }
        this.meshParts = [];
        this.animations = [];
        this.updateStats();
        this.updateTextureMaps();
        if (!options.keepObjectUrls) this.revokeObjectUrls();
        this.placeholder.visible = true;
        this.setEmptyState(true);
    }

    discardModel() {
        this.loadToken += 1;
        this.clearModel();
        this.currentSource = null;
        this.currentFileName = '';
        this.showStatus('Model cleared.', 'info');
    }

    disposeObject(object) {
        const geometries = new Set();
        const materials = new Set();
        const textures = new Set();
        object.traverse((node) => {
            if (node.geometry && !node.userData.smvWireframeOverlay) geometries.add(node.geometry);
            [
                node.material,
                node.userData?.smvOriginalMaterial,
                node.userData?.smvAlbedoMaterial,
                node.userData?.smvRoughnessMaterial,
                node.userData?.smvMetalnessMaterial
            ].forEach((entry) => {
                materialArray(entry).forEach((material) => {
                    if (
                        material
                        && material !== this.geometryMaterial
                        && material !== this.normalMaterial
                        && material !== this.wireframeMaterial
                    ) {
                        materials.add(material);
                    }
                });
            });
        });
        geometries.forEach((geometry) => geometry.dispose());
        materials.forEach((material) => {
            Object.keys(material).forEach((key) => {
                const value = material[key];
                if (value?.isTexture) textures.add(value);
            });
            material.dispose();
        });
        textures.forEach((texture) => texture.dispose());
    }

    revokeObjectUrls() {
        this.objectUrls.forEach((url) => URL.revokeObjectURL(url));
        this.objectUrls.clear();
    }

    setModeMenuOpen(open) {
        this.modeMenu.hidden = !open;
        this.modeToggleBtn.setAttribute('aria-expanded', String(open));
    }

    setViewMode(mode) {
        this.state.viewMode = normalizeViewMode(mode);
        this.setAttribute('view-mode', this.state.viewMode);
        this.applyViewMode();
        this.syncToolbar();
    }

    setFinalViewMode(mode) {
        this.setViewMode(mode);
    }

    renderMode() {
        this.applyViewMode();
    }

    applyViewMode() {
        if (!this.model) {
            this.syncToolbar();
            return;
        }
        this.model.traverse((node) => {
            if (!node.isMesh || node.userData.smvWireframeOverlay) return;
            if (this.state.viewMode === 'normal') {
                node.material = this.normalMaterial;
            } else if (this.state.viewMode === 'geometry') {
                node.material = this.geometryMaterial;
            } else if (this.state.viewMode === 'albedo') {
                node.material = node.userData.smvAlbedoMaterial || node.userData.smvOriginalMaterial || node.material;
            } else if (this.state.viewMode === 'roughness') {
                node.material = node.userData.smvRoughnessMaterial || node.userData.smvOriginalMaterial || node.material;
            } else if (this.state.viewMode === 'metalness') {
                node.material = node.userData.smvMetalnessMaterial || node.userData.smvOriginalMaterial || node.material;
            } else {
                node.material = node.userData.smvOriginalMaterial || node.material;
            }
        });
        this.syncSectionClip();
        this.syncToolbar();
    }

    setWireframeEnabled(enabled, options = {}) {
        this.state.wireframe = enabled === true;
        if (!this.model) {
            this.syncToolbar();
            return;
        }
        if (this.state.wireframe) {
            if (options.rebuild) this.clearWireframeOverlay();
            this.rebuildWireframeOverlay();
        } else {
            this.setWireframeOverlayVisible(false);
        }
        this.syncToolbar();
    }

    showWireframe() {
        this.setWireframeEnabled(!this.state.wireframe);
    }

    rebuildWireframeOverlay() {
        if (!this.model) return;
        this.clearWireframeOverlay();
        this.meshParts.forEach((mesh) => {
            const geometry = this.state.wireframeMode === 'tri'
                ? this.createTriangleWireframeGeometry(mesh.geometry)
                : this.createQuadAwareWireframeGeometry(mesh.geometry);
            if (!geometry.attributes.position || geometry.attributes.position.count === 0) {
                geometry.dispose();
                return;
            }
            const wire = new THREE.LineSegments(geometry, this.wireframeMaterial);
            wire.renderOrder = 20;
            wire.frustumCulled = false;
            wire.userData.smvWireframeOverlay = true;
            mesh.add(wire);
        });
        this.syncSectionClip();
    }

    clearWireframeOverlay() {
        if (!this.model) return;
        const overlays = [];
        this.model.traverse((node) => {
            if (node.userData.smvWireframeOverlay) overlays.push(node);
        });
        overlays.forEach((overlay) => {
            overlay.parent?.remove(overlay);
            overlay.geometry?.dispose();
        });
    }

    setWireframeOverlayVisible(visible) {
        if (!this.model) return 0;
        let count = 0;
        this.model.traverse((node) => {
            if (!node.userData.smvWireframeOverlay) return;
            node.visible = visible;
            count += 1;
        });
        return count;
    }

    createTriangleWireframeGeometry(sourceGeometry) {
        const position = sourceGeometry.attributes.position;
        if (!position || position.count < 3) return new THREE.BufferGeometry();
        const index = sourceGeometry.index?.array || null;
        const indexCount = sourceGeometry.index ? sourceGeometry.index.count : position.count;
        const linePositions = [];
        const pushEdge = (a, b) => {
            linePositions.push(
                position.getX(a), position.getY(a), position.getZ(a),
                position.getX(b), position.getY(b), position.getZ(b)
            );
        };
        for (let cursor = 0; cursor + 2 < indexCount; cursor += 3) {
            const a = index ? index[cursor] : cursor;
            const b = index ? index[cursor + 1] : cursor + 1;
            const c = index ? index[cursor + 2] : cursor + 2;
            pushEdge(a, b);
            pushEdge(b, c);
            pushEdge(c, a);
        }
        const geometry = new THREE.BufferGeometry();
        geometry.setAttribute('position', new THREE.Float32BufferAttribute(linePositions, 3));
        return geometry;
    }

    createQuadAwareWireframeGeometry(sourceGeometry) {
        const position = sourceGeometry.attributes.position;
        if (!position || position.count < 3) return new THREE.BufferGeometry();
        const index = sourceGeometry.index?.array || null;
        const indexCount = sourceGeometry.index ? sourceGeometry.index.count : position.count;
        const edgeMap = new Map();
        const va = new THREE.Vector3();
        const vb = new THREE.Vector3();
        const vc = new THREE.Vector3();
        const normal = new THREE.Vector3();

        const vertexKey = (vertexIndex) => [
            Math.round(position.getX(vertexIndex) * 1000000),
            Math.round(position.getY(vertexIndex) * 1000000),
            Math.round(position.getZ(vertexIndex) * 1000000)
        ].join(',');
        const addEdge = (a, b, opposite, faceNormal) => {
            const keyA = vertexKey(a);
            const keyB = vertexKey(b);
            const key = keyA < keyB ? `${keyA}|${keyB}` : `${keyB}|${keyA}`;
            if (!edgeMap.has(key)) edgeMap.set(key, []);
            edgeMap.get(key).push({ a, b, opposite, normal: faceNormal.clone() });
        };

        for (let cursor = 0; cursor + 2 < indexCount; cursor += 3) {
            const a = index ? index[cursor] : cursor;
            const b = index ? index[cursor + 1] : cursor + 1;
            const c = index ? index[cursor + 2] : cursor + 2;
            va.fromBufferAttribute(position, a);
            vb.fromBufferAttribute(position, b);
            vc.fromBufferAttribute(position, c);
            normal.subVectors(vb, va).cross(vc.clone().sub(va));
            if (normal.lengthSq() < 1e-20) continue;
            normal.normalize();
            addEdge(a, b, c, normal);
            addEdge(b, c, a, normal);
            addEdge(c, a, b, normal);
        }

        const linePositions = [];
        const pushEdge = (a, b) => {
            linePositions.push(
                position.getX(a), position.getY(a), position.getZ(a),
                position.getX(b), position.getY(b), position.getZ(b)
            );
        };
        edgeMap.forEach((entries) => {
            if (entries.length !== 2 || !this.shouldSuppressQuadDiagonal(position, entries[0], entries[1])) {
                pushEdge(entries[0].a, entries[0].b);
            }
        });

        const geometry = new THREE.BufferGeometry();
        geometry.setAttribute('position', new THREE.Float32BufferAttribute(linePositions, 3));
        return geometry;
    }

    shouldSuppressQuadDiagonal(position, first, second) {
        if (first.normal.dot(second.normal) < QUAD_EDGE_NORMAL_DOT) return false;
        const a = new THREE.Vector3().fromBufferAttribute(position, first.a);
        const b = new THREE.Vector3().fromBufferAttribute(position, first.b);
        const c = new THREE.Vector3().fromBufferAttribute(position, first.opposite);
        const d = new THREE.Vector3().fromBufferAttribute(position, second.opposite);
        const ab = b.clone().sub(a);
        const ac = c.clone().sub(a);
        const ad = d.clone().sub(a);
        const sideC = ab.clone().cross(ac).dot(first.normal);
        const sideD = ab.clone().cross(ad).dot(first.normal);
        if (Math.abs(sideC) < 1e-10 || Math.abs(sideD) < 1e-10 || sideC * sideD >= 0) return false;

        const sharedLength = a.distanceTo(b);
        const perimeterLengths = [
            a.distanceTo(c),
            c.distanceTo(b),
            b.distanceTo(d),
            d.distanceTo(a)
        ];
        const longestPerimeter = Math.max(...perimeterLengths);
        return sharedLength >= longestPerimeter * QUAD_EDGE_LENGTH_RATIO;
    }

    syncSectionUi() {
        const sectionButton = this.shadowRoot.querySelector('#sectionBtn');
        const sectionSlider = this.shadowRoot.querySelector('#sectionSlider');
        const sectionAxis = this.shadowRoot.querySelector('#sectionAxis');
        const sectionValue = this.shadowRoot.querySelector('#sectionValue');
        sectionButton.setAttribute('aria-pressed', String(this.state.section.enabled));
        this.sectionControls.hidden = !this.state.section.enabled;
        sectionSlider.value = String(Math.round(this.state.section.value * 100));
        sectionAxis.value = this.state.section.axis;
        sectionValue.textContent = `${Math.round(this.state.section.value * 100)}%`;
    }

    syncSectionClip() {
        const enabled = this.state.section.enabled && Boolean(this.model);
        this.renderer.localClippingEnabled = enabled;
        if (!this.model) return;
        const axis = this.state.section.axis;
        const normal = {
            x: new THREE.Vector3(1, 0, 0),
            y: new THREE.Vector3(0, 1, 0),
            z: new THREE.Vector3(0, 0, 1)
        }[axis] || new THREE.Vector3(1, 0, 0);
        if (this.state.section.flipped) normal.multiplyScalar(-1);
        const min = this.modelBounds.min[axis];
        const max = this.modelBounds.max[axis];
        const coordinate = THREE.MathUtils.lerp(min, max, (this.state.section.value + 1) / 2);
        this.sectionPlane.normal.copy(normal);
        this.sectionPlane.constant = this.state.section.flipped ? coordinate : -coordinate;
        this.model.traverse((node) => {
            if (!node.isMesh && !node.isLine) return;
            materialArray(node.material).forEach((material) => {
                material.clippingPlanes = enabled ? [this.sectionPlane] : null;
                material.needsUpdate = true;
            });
        });
    }

    updateStats() {
        if (!this.model) {
            this.statsEl.hidden = true;
            return;
        }
        const stats = this.getStats();
        this.statsEl.textContent = `${formatCount(stats.vertices)} verts / ${formatCount(stats.triangles)} tris / ${stats.materials} mats`;
        this.statsEl.hidden = false;
    }

    getStats() {
        const materials = new Set();
        let vertices = 0;
        let triangles = 0;
        this.meshParts.forEach((mesh) => {
            const position = mesh.geometry?.attributes?.position;
            if (position) vertices += position.count;
            if (mesh.geometry?.index) triangles += Math.floor(mesh.geometry.index.count / 3);
            else if (position) triangles += Math.floor(position.count / 3);
            materialArray(mesh.userData.smvOriginalMaterial || mesh.material).forEach((material) => materials.add(material));
        });
        return { vertices, triangles, materials: materials.size };
    }

    getTextureMaps() {
        const maps = new Map();
        this.meshParts.forEach((mesh) => {
            materialArray(mesh.userData.smvOriginalMaterial || mesh.material).forEach((material) => {
                TEXTURE_PROPERTIES.forEach(([property, label]) => {
                    const texture = material?.[property];
                    if (!texture?.isTexture) return;
                    const id = texture.uuid || `${property}:${maps.size}`;
                    if (!maps.has(id)) {
                        const image = texture.source?.data || texture.image || null;
                        maps.set(id, {
                            id,
                            texture,
                            image,
                            labels: new Set([label]),
                            materialNames: new Set(material.name ? [material.name] : []),
                            name: texture.name || material.name || label,
                            width: Number(image?.width || texture.image?.width) || 0,
                            height: Number(image?.height || texture.image?.height) || 0,
                            compressed: texture.isCompressedTexture === true
                        });
                    } else {
                        const entry = maps.get(id);
                        entry.labels.add(label);
                        if (material.name) entry.materialNames.add(material.name);
                    }
                });
            });
        });
        return [...maps.values()].map((entry) => {
            const labels = [...entry.labels];
            const isOrm = ['AO', 'Roughness', 'Metalness'].every((label) => entry.labels.has(label));
            return {
                ...entry,
                label: isOrm ? 'ORM' : labels.join(' + '),
                slots: labels,
                name: entry.texture.name || [...entry.materialNames][0] || labels.join(' + ')
            };
        });
    }

    updateTextureMaps() {
        const maps = this.getTextureMaps();
        if (!maps.length) this.texturePanelOpen = false;
        this.textureMapsEl.hidden = maps.length === 0;
        this.textureCountEl.textContent = maps.length ? `${maps.length}` : '';
        this.textureMapsEl.classList.toggle('collapsed', !this.texturePanelOpen);
        this.textureToggleBtn.setAttribute('aria-expanded', String(this.texturePanelOpen));
        this.textureStripEl.innerHTML = '';
        maps.forEach((entry, index) => {
            const button = document.createElement('button');
            button.type = 'button';
            button.className = 'texture-card';
            button.title = `${entry.label}: ${entry.name || ''}`;
            const preview = document.createElement('div');
            preview.className = 'texture-preview';
            const label = document.createElement('span');
            label.textContent = entry.label;
            this.renderTexturePreview(entry, preview);
            button.append(preview, label);
            button.addEventListener('click', () => this.openTextureDialog(entry, index));
            this.textureStripEl.appendChild(button);
        });
    }

    setTexturePanelOpen(open) {
        this.texturePanelOpen = open === true;
        this.textureMapsEl.classList.toggle('collapsed', !this.texturePanelOpen);
        this.textureToggleBtn.setAttribute('aria-expanded', String(this.texturePanelOpen));
    }

    renderTexturePreview(entry, container) {
        const canvas = document.createElement('canvas');
        drawTextureMap(canvas, entry, 128, true);
        container.appendChild(canvas);
    }

    openTextureDialog(entry) {
        const canvas = this.textureDialogCanvas;
        drawTextureMap(canvas, entry, 1024, false);
        const dimensions = entry.width && entry.height ? ` - ${entry.width}x${entry.height}` : '';
        this.textureDialogCaption.textContent = `${entry.label}${dimensions}${entry.name ? ` - ${entry.name}` : ''}`;
        this.textureDialog.showModal();
    }

    fitCameraToModel(options = {}) {
        const targetObject = this.model || this.placeholder;
        const bounds = new THREE.Box3().setFromObject(targetObject);
        if (bounds.isEmpty()) return;
        bounds.getCenter(this.modelCenter);
        bounds.getSize(this.modelSize);
        this.modelBounds.copy(bounds);
        this.modelRadius = Math.max(this.modelSize.length() * 0.5, 0.01);
        this.controls.target.copy(this.parseVector3Attribute('camera-target') || this.modelCenter);
        const distance = this.modelRadius / Math.sin(THREE.MathUtils.degToRad(this.camera.fov * 0.5));
        const offset = options.useInitialOrbit && this.initialCameraOrbit
            ? this.initialCameraOrbit.clone()
            : new THREE.Vector3(0.55, 0.32, 1).normalize().multiplyScalar(distance * 1.35);
        if (offset.length() < this.modelRadius * 0.5) {
            offset.normalize().multiplyScalar(distance * 1.35);
        }
        this.camera.position.copy(this.controls.target).add(offset);
        this.camera.near = Math.max(0.001, this.modelRadius / 100);
        this.camera.far = Math.max(1000, this.modelRadius * 100);
        this.camera.updateProjectionMatrix();
        this.controls.update();
        this.updateGridScale();
        this.cameraTransitionDefault = this.getCameraStateSnapshot();
    }

    resetView() {
        if (this.cameraTransitionDefault) {
            this.setCameraStateSnapshot(this.cameraTransitionDefault);
        } else {
            this.fitCameraToModel({ animate: false, useInitialOrbit: true });
        }
    }

    getCameraStateSnapshot() {
        return {
            position: this.camera.position.toArray(),
            target: this.controls.target.toArray(),
            up: this.camera.up.toArray(),
            fov: this.camera.fov
        };
    }

    setCameraStateSnapshot(snapshot) {
        if (!snapshot) return;
        if (Array.isArray(snapshot.position)) this.camera.position.fromArray(snapshot.position);
        if (Array.isArray(snapshot.target)) this.controls.target.fromArray(snapshot.target);
        if (Array.isArray(snapshot.up)) this.camera.up.fromArray(snapshot.up);
        if (Number.isFinite(snapshot.fov)) {
            this.camera.fov = snapshot.fov;
            this.camera.updateProjectionMatrix();
        }
        this.controls.update();
    }

    updateGridScale() {
        const size = Math.max(this.modelSize.x, this.modelSize.y, this.modelSize.z, 1);
        this.grid.scale.setScalar(size);
        this.grid.position.copy(this.modelCenter);
        this.grid.position.y = this.modelBounds.min.y;
    }

    handlePointerDown(event) {
        this.pointerStart.set(event.clientX, event.clientY);
        this.draggingPointer = false;
    }

    handlePointerMove(event) {
        if (this.pointerStart.distanceTo(new THREE.Vector2(event.clientX, event.clientY)) > 4) {
            this.draggingPointer = true;
        }
    }

    handlePointerUp(event) {
        if (this.draggingPointer || this.state.selectionMode === 'none' || !this.model) return;
        const rect = this.renderer.domElement.getBoundingClientRect();
        this.pointer.x = ((event.clientX - rect.left) / rect.width) * 2 - 1;
        this.pointer.y = -((event.clientY - rect.top) / rect.height) * 2 + 1;
        this.raycaster.setFromCamera(this.pointer, this.camera);
        const hits = this.raycaster.intersectObjects(this.meshParts, false);
        if (!hits.length) {
            this.clearSelection();
            return;
        }
        this.selectMeshPart(hits[0].object);
    }

    selectMeshPart(mesh) {
        this.selectedMesh = mesh || null;
        if (this.selectedMesh) {
            this.selectionHelper.setFromObject(this.selectedMesh);
            this.selectionHelper.visible = true;
        } else {
            this.selectionHelper.visible = false;
        }
        this.emitEvent('viewer-selection-change', {
            name: this.selectedMesh?.name || null,
            index: this.selectedMesh ? this.meshParts.indexOf(this.selectedMesh) : -1
        });
    }

    selectMeshByName(name) {
        const normalized = String(name || '').toLowerCase();
        const mesh = this.meshParts.find((entry) => (entry.name || '').toLowerCase() === normalized)
            || this.meshParts.find((entry) => (entry.name || '').toLowerCase().includes(normalized));
        this.selectMeshPart(mesh || null);
        return Boolean(mesh);
    }

    selectMeshByIndex(index) {
        const mesh = this.meshParts[Number(index)] || null;
        this.selectMeshPart(mesh);
        return Boolean(mesh);
    }

    clearSelection() {
        this.selectMeshPart(null);
    }

    handleKeyDown(event) {
        if (isTextEntryElement(event.target)) return;
        if (event.key === 'Escape') {
            this.setModeMenuOpen(false);
            this.clearSelection();
            this.setPanelOpen(false);
        } else if (event.key.toLowerCase() === 'f') {
            this.fitCameraToModel({ animate: false });
        } else if (event.key.toLowerCase() === 'r') {
            this.resetView();
        } else if (event.key === ' ') {
            if (this.animations.length) {
                event.preventDefault();
                this.setAnimationPlaying(!this.isAnimationPlaying);
            }
        }
    }

    async captureScreenshot(options = {}) {
        this.renderer.render(this.scene, this.camera);
        const blob = await new Promise((resolve, reject) => {
            this.renderer.domElement.toBlob((result) => {
                if (result) resolve(result);
                else reject(new Error('Screenshot capture failed'));
            }, 'image/png');
        });
        if (options.download) {
            const url = URL.createObjectURL(blob);
            const anchor = document.createElement('a');
            anchor.href = url;
            anchor.download = `${sanitizeFilenameSegment(this.currentFileName, 'viewer')}-snapshot.png`;
            document.body.appendChild(anchor);
            anchor.click();
            anchor.remove();
            window.setTimeout(() => URL.revokeObjectURL(url), 2000);
        }
        return blob;
    }

    async toggleFullscreen() {
        if (document.fullscreenElement === this) {
            await document.exitFullscreen();
        } else {
            await this.requestFullscreen();
        }
        window.setTimeout(() => this.resizeRenderer(), 60);
    }

    setEnvironmentMenuOpen(open) {
        const nextOpen = open === true;
        this.environmentMenu.hidden = !nextOpen;
        this.environmentToggle.setAttribute('aria-expanded', String(nextOpen));
        if (nextOpen) this.syncEnvironmentControls();
    }

    setEnvironment(environment) {
        if (!environment) return;
        if (ENVIRONMENT_PRESETS[environment]) {
            this.state.environment = environment;
            this.state.environmentIntensity = parseNumber(this.getAttribute('environment-intensity'), ENVIRONMENT_PRESETS[environment].environmentIntensity);
            this.state.environmentUrl = '';
            this.setAttribute('environment', environment);
            this.removeAttribute('environment-url');
        } else {
            this.state.environmentUrl = environment;
            this.setAttribute('environment-url', environment);
        }
        return this.loadEnvironment();
    }

    environmentRotationDegrees() {
        return ((THREE.MathUtils.radToDeg(this.state.environmentRotation || 0) % 360) + 360) % 360;
    }

    setEnvironmentRotationDegrees(degrees) {
        const normalized = ((Number(degrees || 0) % 360) + 360) % 360;
        this.state.environmentRotation = THREE.MathUtils.degToRad(normalized);
        this.rebuildEnvironmentTexture();
        this.applyEnvironmentPresentation();
        this.emitEvent('viewer-environment-change', {
            environment: this.state.environment,
            background: this.state.environmentBackground,
            rotationDegrees: normalized,
            source: this.environmentUrl || this.state.environmentUrl
        });
    }

    setEnvironmentBackgroundVisible(visible) {
        this.state.environmentBackground = visible === true;
        this.setAttribute('environment-background', String(this.state.environmentBackground));
        this.applyEnvironmentPresentation();
        this.emitEvent('viewer-environment-change', {
            environment: this.state.environment,
            background: this.state.environmentBackground,
            source: this.environmentUrl || this.state.environmentUrl
        });
    }

    exportState() {
        const state = {
            schema: 'simple-model-viewer-state/v2',
            src: this.currentSource,
            fileName: this.currentFileName,
            viewMode: this.state.viewMode,
            wireframe: this.state.wireframe,
            wireframeMode: this.state.wireframeMode,
            autoRotate: this.state.autoRotate,
            anglePerSecond: this.state.anglePerSecond,
            environment: this.state.environment,
            environmentUrl: this.state.environmentUrl,
            environmentBackground: this.state.environmentBackground,
            environmentRotation: this.state.environmentRotation,
            camera: this.getCameraStateSnapshot()
        };
        this.emitEvent('viewer-state-export', state);
        return state;
    }

    async importState(state) {
        if (!state) return;
        if (state.src) await this.loadModelFromUrl(state.src, state.fileName || basename(pathWithoutSearch(state.src)));
        this.state.viewMode = normalizeViewMode(state.viewMode);
        this.state.wireframe = state.wireframe === true;
        this.state.wireframeMode = normalizeWireframeMode(state.wireframeMode);
        this.state.autoRotate = state.autoRotate === true;
        this.state.anglePerSecond = parseNumber(state.anglePerSecond, this.state.anglePerSecond);
        this.state.environment = state.environment || this.state.environment;
        this.state.environmentUrl = state.environmentUrl || '';
        this.state.environmentBackground = state.environmentBackground === true;
        this.state.environmentRotation = parseNumber(state.environmentRotation, this.state.environmentRotation);
        await this.loadEnvironment();
        this.applyViewMode();
        this.setWireframeEnabled(this.state.wireframe, { rebuild: true });
        if (state.camera) this.setCameraStateSnapshot(state.camera);
        this.syncToolbar();
    }

    async saveBundleToHistory(bundle, mainEntry) {
        if (!bundle?.entries?.length || bundle.bytes > HISTORY_BYTE_LIMIT) return;
        try {
            const files = [];
            for (const entry of bundle.entries) {
                files.push({
                    path: entry.path,
                    name: entry.name,
                    type: entry.file.type || '',
                    size: entry.size,
                    lastModified: entry.file.lastModified || Date.now(),
                    data: await entry.file.arrayBuffer()
                });
            }
            const signature = files.map((file) => `${file.path}:${file.size}`).join('|');
            const record = {
                id: createHistoryId(),
                name: mainEntry.name,
                mainPath: mainEntry.path,
                bytes: bundle.bytes,
                createdAt: Date.now(),
                signature,
                files
            };
            const records = [record, ...this.historyRecords.filter((entry) => entry.signature !== signature)];
            let total = 0;
            const pruned = [];
            for (const entry of records) {
                if (pruned.length >= HISTORY_LIMIT) continue;
                if (total + entry.bytes > HISTORY_BYTE_LIMIT) continue;
                total += entry.bytes;
                pruned.push(entry);
            }
            await writeHistoryRecords(pruned);
            this.historyRecords = pruned;
            this.renderHistory();
        } catch (error) {
            console.warn('History save failed:', error);
        }
    }

    async loadHistory() {
        this.historyRecords = await readHistoryRecords();
        this.renderHistory();
    }

    renderHistory() {
        const total = this.historyRecords.reduce((sum, record) => sum + (record.bytes || 0), 0);
        this.shadowRoot.querySelector('#historyUsage').textContent = `Saved ${formatBytes(total)} / 100 MB`;
        this.shadowRoot.querySelector('#clearHistoryBtn').disabled = this.historyRecords.length === 0;
        this.historySelect.innerHTML = this.historyRecords.length
            ? '<option value="">Recent meshes</option>'
            : '<option value="">No saved meshes</option>';
        this.historyRecords.forEach((record) => {
            const option = document.createElement('option');
            option.value = record.id;
            option.textContent = `${record.name} - ${formatBytes(record.bytes)}`;
            this.historySelect.appendChild(option);
        });
    }

    async clearHistory() {
        this.historyRecords = [];
        await writeHistoryRecords([]);
        this.renderHistory();
        this.showStatus('History cleared.', 'info');
    }

    async loadHistoryRecord(record) {
        const entries = record.files.map((entry) => {
            const path = normalizePath(entry.path);
            const file = new File([entry.data], basename(path), {
                type: entry.type || '',
                lastModified: entry.lastModified || Date.now()
            });
            return {
                file,
                path,
                name: basename(path),
                extension: extensionFromPath(path),
                size: Number(entry.size || file.size) || 0
            };
        }).filter((entry) => SUPPORTED_MODEL_EXTENSIONS.has(entry.extension) || COMPANION_EXTENSIONS.has(entry.extension));
        const bundle = {
            entries,
            mainEntries: entries.filter((entry) => SUPPORTED_MODEL_EXTENSIONS.has(entry.extension)),
            bytes: entries.reduce((sum, entry) => sum + entry.size, 0)
        };
        bundle.mainEntries = bundle.entries.filter((entry) => SUPPORTED_MODEL_EXTENSIONS.has(entry.extension));
        const mainEntry = bundle.entries.find((entry) => entry.path === record.mainPath) || bundle.mainEntries[0];
        await this.loadBundle(bundle, mainEntry, { saveHistory: false });
        this.historySelect.value = '';
    }

    setPanelOpen(open, options = {}) {
        const shouldOpen = open || (!this.model && !options.forceClose);
        this.fileInputContainer.hidden = !shouldOpen;
        this.rootEl.classList.toggle('upload-open', shouldOpen);
    }

    setEmptyState(empty) {
        this.rootEl.classList.toggle('has-model', !empty);
        this.placeholder.visible = empty;
        if (empty) {
            this.fileInputContainer.hidden = false;
            this.rootEl.classList.add('upload-open');
        } else {
            this.fileInputContainer.hidden = true;
            this.rootEl.classList.remove('upload-open');
        }
    }

    setLoading(visible, progress = 0, label = 'Loading mesh') {
        this.loadingEl.hidden = !visible;
        if (!visible) return;
        const percent = Number.isFinite(progress) && progress > 0
            ? Math.round(THREE.MathUtils.clamp(progress, 0, 1) * 100)
            : 0;
        this.loadingLabelEl.textContent = label;
        this.loadingValueEl.textContent = percent ? `${percent}%` : '';
        this.loadingBarEl.style.width = `${percent || 8}%`;
    }

    showStatus(message, type = 'info') {
        this.statusEl.textContent = message;
        this.statusEl.dataset.state = type;
    }

    syncToolbar() {
        const modeLabel = viewModeLabel(this.state.viewMode);
        this.modeToggleBtn.textContent = modeLabel;
        this.modeToggleBtn.setAttribute('aria-label', `View mode: ${modeLabel}`);
        this.shadowRoot.querySelectorAll('#modeMenu .mode-button').forEach((button) => {
            const active = button.dataset.mode === this.state.viewMode;
            button.classList.toggle('active', active);
            button.setAttribute('aria-checked', String(active));
        });
        this.shadowRoot.querySelector('#wireBtn').setAttribute('aria-pressed', String(this.state.wireframe));
        this.shadowRoot.querySelector('#wireModeSelect').value = this.state.wireframeMode;
        this.shadowRoot.querySelector('#rotateBtn').setAttribute('aria-pressed', String(this.state.autoRotate));
        this.shadowRoot.querySelector('#gridBtn').setAttribute('aria-pressed', String(this.state.grid));
        this.syncEnvironmentControls();
    }

    syncEnvironmentControls() {
        const preset = environmentPresetFor(this.state.environment);
        const degrees = this.environmentRotationDegrees();
        this.shadowRoot.querySelector('#environmentActiveLabel').textContent = preset.label;
        this.environmentToggle.setAttribute('title', `Environment: ${preset.label}`);
        this.shadowRoot.querySelectorAll('[data-environment-preset]').forEach((button) => {
            const active = button.dataset.environmentPreset === preset.id;
            button.classList.toggle('active', active);
            button.setAttribute('aria-pressed', String(active));
        });
        const backgroundButton = this.shadowRoot.querySelector('#environmentBgBtn');
        backgroundButton?.setAttribute('aria-pressed', String(this.state.environmentBackground));
        backgroundButton?.classList.toggle('active', this.state.environmentBackground);
        const dialHand = this.shadowRoot.querySelector('#environmentDialHand');
        if (dialHand) dialHand.style.transform = `rotate(${degrees}deg)`;
        this.environmentDial.setAttribute('aria-valuenow', String(Math.round(degrees)));
        this.environmentDial.setAttribute('aria-valuetext', `${Math.round(degrees)} degrees`);
        this.shadowRoot.querySelector('#environmentRotationValue').textContent = `${Math.round(degrees)}deg`;
        this.renderEnvironmentPreview(this.shadowRoot.querySelector('#environmentDialPreview'), preset.id, degrees);
        this.shadowRoot.querySelectorAll('[data-environment-preview]').forEach((canvas) => {
            this.renderEnvironmentPreview(canvas, canvas.dataset.environmentPreview, degrees);
        });
    }

    emitEvent(name, detail = {}) {
        this.dispatchEvent(new CustomEvent(name, {
            bubbles: true,
            composed: true,
            detail
        }));
    }

    emitViewerError(action, error, extra = {}) {
        console.error(`Viewer ${action} failed:`, error);
        this.emitEvent('viewer-error', {
            action,
            message: error?.message || String(error),
            error,
            ...extra
        });
    }

    resizeRenderer() {
        const rect = this.getBoundingClientRect();
        const width = Math.max(1, Math.floor(rect.width || this.clientWidth || 1));
        const height = Math.max(1, Math.floor(rect.height || this.clientHeight || 1));
        const pixelRatioLimit = this.state.performanceMode === 'quality' ? 2 : this.state.performanceMode === 'performance' ? 1.25 : 1.6;
        this.renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, pixelRatioLimit));
        this.renderer.setSize(width, height, false);
        this.camera.aspect = width / height;
        this.camera.updateProjectionMatrix();
    }

    animate(time = performance.now()) {
        this.frameRequest = requestAnimationFrame((nextTime) => this.animate(nextTime));
        const delta = Math.min(0.1, Math.max(0, (time - this.lastFrameTime) / 1000));
        this.lastFrameTime = time;
        if (this.mixer && this.isAnimationPlaying) {
            this.mixer.update(delta * this.state.animationSpeed);
        }
        if (this.model && this.state.autoRotate) {
            this.model.rotation.y += THREE.MathUtils.degToRad(this.state.anglePerSecond) * delta;
            if (this.selectedMesh) this.selectionHelper.setFromObject(this.selectedMesh);
        }
        this.controls.update();
        this.renderer.render(this.scene, this.camera);
    }
}

if (!customElements.get('simple-model-viewer')) {
    customElements.define('simple-model-viewer', SimpleModelViewer);
}

function initSimpleModelViewer() {
    return SimpleModelViewer;
}

function initGaussianViewer() {
    return initSimpleModelViewer();
}

export { SimpleModelViewer, initSimpleModelViewer, initGaussianViewer };
