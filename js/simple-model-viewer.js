import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';
import { OBJLoader } from 'three/addons/loaders/OBJLoader.js';
import { FBXLoader } from 'three/addons/loaders/FBXLoader.js';
import { PLYLoader } from 'three/addons/loaders/PLYLoader.js';
import { RGBELoader } from 'three/addons/loaders/RGBELoader.js';
import { DRACOLoader } from 'three/addons/loaders/DRACOLoader.js';
import { TransformControls } from 'three/addons/controls/TransformControls.js';

const ENVIRONMENT_URLS = {
    env1: '/assets/viewer/spruit-sunrise.hdr',
    env2: '/assets/viewer/aircraft-workshop.hdr',
    env3: '/assets/viewer/lebombo.hdr',
};

const TEXTURE_INPUT_IDS = {
    map: 'diffuseMapInput',
    roughnessMap: 'roughnessMapInput',
    metalnessMap: 'metalnessMapInput',
    normalMap: 'normalMapInput',
    aoMap: 'aoMapInput',
    emissiveMap: 'emissiveMapInput',
};

const PREVIEW_TEXT_STYLE = {
    lineHeight: '1.4',
    textAlign: 'center',
};

const DEFAULT_BACKGROUND_COLOR = '#eeeeee';
const DEFAULT_EXPOSURE = 1;
const DEFAULT_ENVIRONMENT_INTENSITY = 1;
const DEFAULT_ENVIRONMENT_ROTATION = 0;
const DEFAULT_ENVIRONMENT_BACKGROUND_VISIBLE = true;
const DEFAULT_SELECTION_MODE = 'scene-graph';
const DEFAULT_PERFORMANCE_MODE = 'default';
const DEFAULT_CAMERA_DISTANCE = 10;
const DEFAULT_ANIMATION_SPEED = 1;
const DEFAULT_ANIMATION_LOOP = 'repeat';
const DEFAULT_RECORDING_DURATION = 5;
const CUSTOM_ENVIRONMENT_ID = 'custom';
const POINTER_DRAG_THRESHOLD = 6;
const STATE_SCHEMA_VERSION = 'simple-model-viewer-state/v1';
const VALID_SELECTION_MODES = new Set(['scene-graph', 'canvas', 'all', 'none']);
const VALID_PERFORMANCE_MODES = new Set(['default', 'performance', 'quality']);
const VALID_ANIMATION_LOOP_MODES = new Set(['repeat', 'once', 'ping-pong']);
const ANIMATION_CROSS_FADE_DURATION = 0.2;
const SUPPORTED_MODEL_EXTENSIONS = new Set(['glb', 'gltf', 'obj', 'fbx', 'ply']);

const DISPOSABLE_TEXTURE_KEYS = [
    'alphaMap',
    'aoMap',
    'bumpMap',
    'displacementMap',
    'emissiveMap',
    'envMap',
    'lightMap',
    'map',
    'metalnessMap',
    'normalMap',
    'roughnessMap',
    'specularMap',
];

function getEditableMaterial(mesh) {
    if (!mesh || !mesh.material) {
        return null;
    }

    return Array.isArray(mesh.material) ? mesh.material[0] : mesh.material;
}

function getMaterialCount(mesh) {
    if (!mesh?.material) {
        return 0;
    }

    return Array.isArray(mesh.material) ? mesh.material.length : 1;
}

function cloneTexture(texture) {
    return texture && texture.clone ? texture.clone() : texture || null;
}

function getMaterialArray(material) {
    if (!material) {
        return [];
    }

    return Array.isArray(material) ? material : [material];
}

function getMaterialEntryAt(materialEntry, index = 0) {
    if (Array.isArray(materialEntry)) {
        return materialEntry[index] || materialEntry[0] || null;
    }

    return materialEntry || null;
}

function cloneMaterialEntry(materialEntry) {
    if (Array.isArray(materialEntry)) {
        return materialEntry.map((material) => material?.clone ? material.clone() : material || null);
    }

    return materialEntry?.clone ? materialEntry.clone() : materialEntry || null;
}

function getTextureSource(texture) {
    if (!texture?.image) {
        return null;
    }

    return texture.image.currentSrc
        || texture.image.src
        || texture.source?.data?.currentSrc
        || texture.source?.data?.src
        || null;
}

function createStandardMaterialFromMaterial(material) {
    if (material instanceof THREE.MeshStandardMaterial) {
        return material;
    }

    const standardMaterial = new THREE.MeshStandardMaterial({
        color: material?.color?.clone ? material.color.clone() : new THREE.Color(0xffffff),
        map: material?.map || null,
        alphaMap: material?.alphaMap || null,
        aoMap: material?.aoMap || null,
        bumpMap: material?.bumpMap || null,
        displacementMap: material?.displacementMap || null,
        emissive: material?.emissive?.clone ? material.emissive.clone() : new THREE.Color(0x000000),
        emissiveIntensity: material?.emissiveIntensity ?? 1,
        emissiveMap: material?.emissiveMap || null,
        envMapIntensity: material?.envMapIntensity ?? 1,
        lightMap: material?.lightMap || null,
        metalness: material?.metalness ?? 0.5,
        metalnessMap: material?.metalnessMap || null,
        normalMap: material?.normalMap || null,
        opacity: material?.opacity ?? 1,
        roughness: material?.roughness ?? 0.5,
        roughnessMap: material?.roughnessMap || null,
        side: material?.side ?? THREE.FrontSide,
        transparent: material?.transparent ?? false,
        vertexColors: !!material?.vertexColors,
    });

    if (material?.name) {
        standardMaterial.name = material.name;
    }

    if (material?.normalScale?.clone) {
        standardMaterial.normalScale.copy(material.normalScale);
    }

    if (material?.userData) {
        standardMaterial.userData = {
            ...material.userData,
        };
    }

    return standardMaterial;
}

function parseVector3String(value) {
    if (value instanceof THREE.Vector3) {
        return value.clone();
    }

    if (typeof value !== 'string') {
        return null;
    }

    const [x, y, z] = value.trim().split(/\s+/).map(parseFloat);
    if ([x, y, z].some((entry) => Number.isNaN(entry))) {
        return null;
    }

    return new THREE.Vector3(x, y, z);
}

function formatVector3String(vector, precision = 4) {
    return [vector.x, vector.y, vector.z]
        .map((value) => Number(value.toFixed(precision)))
        .join(' ');
}

function isTextEntryElement(element) {
    if (!(element instanceof Element)) {
        return false;
    }

    const tagName = element.tagName;
    return tagName === 'INPUT'
        || tagName === 'TEXTAREA'
        || tagName === 'SELECT'
        || element.isContentEditable;
}

function sanitizeFilenameSegment(value, fallback = 'capture') {
    const normalized = `${value || ''}`
        .trim()
        .replace(/\.[a-z0-9]+$/i, '')
        .replace(/[^a-z0-9-_]+/gi, '-')
        .replace(/^-+|-+$/g, '')
        .toLowerCase();

    return normalized || fallback;
}

function formatTimestampForFilename(date = new Date()) {
    const pad = (value) => `${value}`.padStart(2, '0');
    return [
        date.getFullYear(),
        pad(date.getMonth() + 1),
        pad(date.getDate()),
    ].join('') + '-' + [
        pad(date.getHours()),
        pad(date.getMinutes()),
        pad(date.getSeconds()),
    ].join('');
}

function getFileExtension(name = '') {
    return `${name}`.split('.').pop().toLowerCase();
}

function isSupportedModelFileName(name = '') {
    return SUPPORTED_MODEL_EXTENSIONS.has(getFileExtension(name));
}

class SimpleModelViewer extends HTMLElement {
    constructor() {
        super();
        this.attachShadow({ mode: 'open' });
        this.shadowRoot.innerHTML = /*html*/`
            <style>
                :host {
                    --viewer-bg: linear-gradient(145deg, rgba(247, 248, 251, 0.95), rgba(222, 228, 236, 0.82));
                    --panel-bg: rgba(255, 255, 255, 0.72);
                    --panel-border: rgba(58, 72, 89, 0.12);
                    --panel-shadow: 0 20px 45px rgba(43, 58, 79, 0.16);
                    --button-bg: rgba(46, 57, 72, 0.88);
                    --button-hover: rgba(31, 112, 93, 0.92);
                    --button-active: rgba(18, 145, 116, 0.95);
                    --text-main: #1f2937;
                    --text-muted: #5f6b7a;
                    display: block;
                    border-radius: 18px;
                    min-height: 300px;
                    background: var(--viewer-bg);
                    font-family: "Avenir Next", "Segoe UI", Verdana, Geneva, Arial, sans-serif;
                    position: relative; /* Required for absolute positioning of panels */
                    overflow: hidden;
                }

                *, *::before, *::after {
                    box-sizing: border-box;
                }

                #loadingProgressBar {
                    position: absolute;
                    top: 0;
                    left: 0;
                    width: 0%;
                    height: 5px;
                    background-color: #4CAF50;
                    z-index: 1;
                    display: none;
                }

                #canvas-container {
                    width: 100%;
                    height: auto;
                    position: relative;
                }

                label {
                    font-size: 0.7rem;
                }

                canvas {
                    width: 100%;
                    height: 100%;
                }

                input {
                    font-size: 0.7rem;
                }

                .controls {
                    margin: 0;
                    position: absolute; /* Make controls container positioned relative to :host */
                    top: 1rem;
                    right: 1rem;
                    z-index: 1000; /* Ensure it's above canvas */
                }

                button {
                    background: linear-gradient(180deg, rgba(60, 72, 88, 0.96), rgba(36, 45, 57, 0.94));
                    border: 1px solid rgba(255, 255, 255, 0.14);
                    color: white;
                    padding: 6px 10px;
                    border-radius: 10px;
                    cursor: pointer;
                    text-align: center;
                    text-decoration: none;
                    display: inline-block;
                    font-size: 0.8rem;
                    font-weight: 600;
                    letter-spacing: 0.01em;
                    z-index: 1001;
                    margin-right: 0px;
                    margin-top: 0.1rem;
                    margin-bottom: 0.1rem;
                    min-width: 4rem;
                    width: 32.5%;
                    box-shadow: 0 10px 24px rgba(31, 41, 55, 0.18);
                    transition: transform 0.16s ease, background-color 0.16s ease, box-shadow 0.16s ease;
                }

                button:hover {
                    background: linear-gradient(180deg, rgba(34, 126, 104, 0.96), rgba(24, 103, 84, 0.94));
                    transform: translateY(-1px);
                }

                button:disabled {
                    opacity: 0.45;
                    cursor: not-allowed;
                    transform: none;
                    box-shadow: none;
                }

                button:disabled:hover {
                    background: linear-gradient(180deg, rgba(60, 72, 88, 0.96), rgba(36, 45, 57, 0.94));
                }

                button.toggled-off {
                    background: linear-gradient(180deg, rgba(20, 157, 125, 0.98), rgba(15, 121, 97, 0.96));
                    box-shadow: 0 12px 28px rgba(16, 124, 98, 0.28);
                }

                #fileInputContainer {
                    position: absolute;
                    top: 50%;
                    left: 50%;
                    transform: translate(-50%, -50%);
                    text-align: center;
                }

                #fileInput {
                    font-size: 1rem;
                    padding: 10px;
                }

                .transform-buttons {
                    display: flex;
                    margin-top: 0.5rem;
                    gap: 5px; /* 버튼 사이 간격 */
                }

                .transform-button {
                    background: linear-gradient(180deg, rgba(60, 72, 88, 0.96), rgba(36, 45, 57, 0.94));
                    border: 1px solid rgba(255, 255, 255, 0.14);
                    color: white;
                    padding: 5px 10px;
                    border-radius: 10px;
                    cursor: pointer;
                    text-align: center;
                    text-decoration: none;
                    display: inline-block;
                    font-size: 0.8rem;
                    z-index: 1001;
                    min-width: 4rem;
                    width: 49%;
                }

                .transform-button.active {
                    background: linear-gradient(180deg, rgba(20, 157, 125, 0.98), rgba(15, 121, 97, 0.96));
                }

                .transform-button:hover {
                    background: linear-gradient(180deg, rgba(34, 126, 104, 0.96), rgba(24, 103, 84, 0.94));
                }

                .right-ui-panel { /* Renamed and unified panel */
                    position: absolute;
                    top: 0;
                    right: 0; /* Positioned to the right */
                    font-size: 0.7rem;
                    color: var(--text-main);
                    background: var(--panel-bg);
                    backdrop-filter: blur(16px);
                    -webkit-backdrop-filter: blur(16px);
                    border: 1px solid var(--panel-border);
                    padding: 0.6rem;
                    border-radius: 18px;
                    display: flex;
                    flex-direction: column;
                    gap: 0.35rem;
                    z-index: 1000;
                    width: min(25rem, calc(100vw - 2rem));
                    max-width: 25rem;
                    max-height: calc(100vh - 2rem);
                    box-shadow: var(--panel-shadow);
                    overflow: hidden;
                }

                #panelContent {
                    max-height: calc(100vh - 7rem);
                    overflow-y: auto;
                    overflow-x: hidden;
                    padding-right: 0.15rem;
                }

                #panelContent::-webkit-scrollbar {
                    width: 8px;
                }

                #panelContent::-webkit-scrollbar-thumb {
                    border-radius: 999px;
                    background: rgba(95, 107, 122, 0.3);
                }

                .right-ui-panel label {
                    display: flex;
                    justify-content: space-between;
                    align-items: center;
                    color: var(--text-muted);
                }

                .right-ui-panel input {
                    width: 3rem;
                }

                select,
                input[type="number"],
                input[type="text"],
                input[type="color"] {
                    border-radius: 8px;
                    border: 1px solid rgba(95, 107, 122, 0.25);
                    background: rgba(255, 255, 255, 0.9);
                    color: var(--text-main);
                    padding: 0.2rem 0.4rem;
                }

                .material-toggle {
                    margin-top: 5px;
                }

                .material-toggle label {
                    display: flex;
                    align-items: center;
                    gap: 5px;
                }

                input[type="range"] {
                    -webkit-appearance: none; /*  (Chrome, Safari) */
                    -moz-appearance: none;    /*  (Firefox) */
                    appearance: none;
                    background-color: transparent; /*  */
                    height: 8px; /*  */
                    cursor: pointer;
                }

                input[type="range"]::-webkit-slider-runnable-track {
                    background-color: rgba(63, 78, 94, 0.82);
                    height: 5px;
                    border-radius: 4px;
                }

                input[type="range"]::-moz-range-track {
                    background-color: rgba(63, 78, 94, 0.82);
                    height: 5px;
                    border-radius: 4px;
                }

                input[type="range"]::-webkit-slider-thumb {
                    -webkit-appearance: none;
                    appearance: none;
                    background-color: rgba(20, 157, 125, 0.95);
                    border: none;
                    height: 16px;
                    width: 16px;
                    border-radius: 50%;
                    margin-top: -5.5px;
                }

                input[type="range"]::-moz-range-thumb {
                    -moz-appearance: none;
                    appearance: none;
                    background-color: rgba(20, 157, 125, 0.95);
                    border: none;
                    height: 16px;
                    width: 16px;
                    border-radius: 50%;
                }

                input[type="range"]:focus {
                    outline: none;
                }

                input[type="range"]:focus::-webkit-slider-runnable-track {
                    background-color: #666666;
                }

                input[type="range"]:focus::-moz-range-track {
                    background-color: #666666;
                }

                input[type="range"]::-webkit-slider-thumb:active {
                    background-color: #666666;
                }

                input[type="range"]::-moz-range-thumb:active {
                    background-color: #666666;
                }

                input[type="range"]:disabled {
                    cursor: not-allowed;
                    opacity: 0.7;
                }

                input[type="range"]:disabled::-webkit-slider-runnable-track {
                    background-color: #aaaaaa;
                }

                input[type="range"]:disabled::-moz-range-track {
                    background-color: #aaaaaa;
                }

                input[type="range"]:disabled::-webkit-slider-thumb {
                    background-color: #aaaaaa;
                }

                input[type="range"]:disabled::-moz-range-thumb {
                    background-color: #aaaaaa;
                }

                input[type="checkbox"] {
                    -webkit-appearance: none;
                    -moz-appearance: none;
                    appearance: none;
                    width: 16px;
                    height: 16px;
                    border: 2px solid #444444;
                    border-radius: 3px;
                    background-color: transparent;
                    cursor: pointer;
                    top: 0;
                    position: relative;
                }

                input[type="checkbox"]:checked {
                    background-color: transparent;
                }


                input[type="checkbox"]:checked::before {
                    content: '';
                    position: absolute;
                    top: 50%;
                    left: 50%;
                    transform: translate(-50%, -50%);
                    width: 10px;
                    height: 10px;
                    background-color: #444444;
                    border-radius: 2px;
                }

                input[type="checkbox"]:focus {
                    outline: 1px solid #444444;
                }

                .texture-map-controls {
                    display: grid;
                    grid-template-columns: auto auto; /* Label and Controls */
                    gap: 5px;
                    align-items: center;
                    margin-bottom: 5px;
                }

                .texture-preview {
                    width: min(100%, 18rem);
                    aspect-ratio: 1;
                    min-height: 12rem;
                    border: 1px solid #ccc;
                    border-radius: 12px;
                    background-color: #eee;
                    display: flex;
                    align-items: center;
                    justify-content: center;
                    overflow: hidden;
                    transform-origin: 100% 0%;
                }

                .texture-preview:hover{
                    transform: scale(1.08);
                    border: 1px solid #666666;
                    z-index: 10;
                    cursor: cell;
                }

                .texture-preview img,
                .texture-preview canvas {
                    max-width: 100%;
                    max-height: 100%;
                    width: auto;
                    height: auto;
                    display: block;
                    object-fit: contain;
                }

                #videoModal {
                    /* Styles already defined inline, but you can move them here */
                    /* display: none; */ /* Controlled by JS */
                    /* position: fixed; */
                    /* ... etc ... */
                }

                #videoModal > div {
                    /* background-color: white; */
                    /* padding: 20px; */
                    /* border-radius: 5px; */
                    box-shadow: 0 4px 8px rgba(0,0,0,0.2);
                }

                #videoPreview {
                    border: 1px solid #ccc;
                }

                #recordBtn {
                    background-color: #d9534f; /* Red */
                    color: white;
                    border-color: #d43f3a;
                }
                #recordBtn:hover {
                    background-color: #c9302c;
                }

                #stopBtn {
                    background-color: #5bc0de; /* Blue */
                    color: white;
                    border-color: #46b8da;
                }
                #stopBtn:hover {
                    background-color: #31b0d5;
                }

                #downloadBtn {
                    background-color: #5cb85c; /* Green */
                    color: white;
                    border-color: #4cae4c;
                }
                #downloadBtn:hover {
                    background-color: #449d44;
                }

                ul {
                    left: 0;
                    padding-inline-start: 0.75rem;
                    font-size: 0.65rem;
                }

                .texture-button-group {
                    display: grid;
                    grid-template-columns: repeat(2, minmax(0, 1fr));
                    gap: 5px;
                    align-items: stretch;
                    flex: 1 1 12rem;
                }

                .texture-button-group button {
                    padding: 3px 6px;
                    font-size: 0.7rem;
                    margin: 0;
                }

                .texture-section {
                    margin-top: 10px;
                    padding-top: 10px;
                    border-top: 1px solid #ddd;
                }

                .material-editor-grid {
                    display: grid;
                    grid-template-columns: repeat(2, minmax(0, 1fr));
                    gap: 0.35rem 0.5rem;
                    margin-bottom: 0.45rem;
                }

                .material-editor-grid label {
                    gap: 0.5rem;
                }

                .material-editor-grid input[type="number"],
                .material-editor-grid input[type="color"],
                .material-editor-grid select {
                    width: 6rem;
                }

                .material-editor-grid input[type="checkbox"] {
                    margin-left: auto;
                }

                .material-meta {
                    font-size: 0.7rem;
                    line-height: 1.4;
                    color: var(--text-muted);
                    min-height: 2.6rem;
                    margin: 0.35rem 0;
                    white-space: pre-line;
                }

                .texture-toolbar {
                    display: flex;
                    flex-wrap: wrap;
                    align-items: center;
                    gap: 0.5rem;
                    margin-bottom: 0.35rem;
                }

                .texture-preview-row {
                    display: flex;
                    flex-wrap: wrap;
                    align-items: flex-start;
                    gap: 0.75rem;
                }

                .hidden {
                    display: none !important;
                }

                .scene-graph-tree ul {
                    list-style: none;
                    padding-left: 1px;
                    margin: 0;
                }

                .scene-graph-tree li {
                    margin-bottom: 2px;
                }

                .scene-graph-tree label {
                    display: flex;
                    align-items: center;
                    gap: 5px;
                    cursor: pointer;
                    padding: 2px 5px;
                    border-radius: 3px;
                }

                .scene-graph-tree label:hover,
                .scene-graph-tree label.selected {
                    background-color: rgba(0, 120, 215, 0.2);
                }

                .scene-graph-tree label.selected {
                    font-weight: bold; /* Bold font for selected item */
                }

                /* Tab Styles */
                .tab-buttons {
                    display: flex;
                    margin-bottom: 0.5rem;
                    gap: 0.35rem;
                }

                .tab-button {
                    background: rgba(214, 220, 228, 0.86);
                    border: 1px solid rgba(95, 107, 122, 0.14);
                    padding: 8px 16px;
                    cursor: pointer;
                    border-radius: 12px;
                    font-size: 0.8rem;
                    margin-right: 0;
                    color: var(--text-main);
                    flex: 1 1 0;
                    min-width: 0;
                    width: auto;
                    box-shadow: none;
                }

                .tab-button.active {
                    background: rgba(255, 255, 255, 0.96);
                    border-color: rgba(20, 157, 125, 0.2);
                    color: #0f5c4c;
                }

                .tab-button:hover {
                    background: rgba(255, 255, 255, 0.94);
                }

                .tab-content {
                    padding: 0.5rem;
                    border-radius: 0 0 5px 5px;
                    /* background-color: rgba(200, 200, 200, 0.5); Already set in .right-ui-panel */
                }

                fieldset {
                    max-width: 23rem;
                    border: 1px solid rgba(95, 107, 122, 0.16);
                    border-radius: 14px;
                    background: rgba(255, 255, 255, 0.48);
                    box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.4);
                }

                .stack-row {
                    display: flex;
                    gap: 0.35rem;
                    align-items: center;
                    flex-wrap: wrap;
                }

                .stack-row > * {
                    flex: 1 1 auto;
                }

                .compact-input {
                    width: 100%;
                    box-sizing: border-box;
                }

                .section-toggle {
                    width: 100%;
                    margin: 0 0 0.45rem;
                    display: flex;
                    align-items: center;
                    justify-content: space-between;
                    gap: 0.5rem;
                }

                .section-toggle .section-toggle-icon {
                    font-size: 0.95rem;
                    line-height: 1;
                }

                .collapsible-content[hidden] {
                    display: none !important;
                }

                .utility-grid {
                    display: grid;
                    grid-template-columns: repeat(3, minmax(0, 1fr));
                    gap: 0.35rem;
                    margin-bottom: 0.35rem;
                }

                .utility-grid > button,
                .utility-grid > select,
                .utility-grid > div {
                    width: 100%;
                    min-width: 0;
                    margin: 0;
                }

                .utility-record-stack {
                    display: grid;
                }

                .utility-record-stack > button {
                    width: 100%;
                }

                .utility-input-group {
                    display: flex;
                    align-items: center;
                    gap: 0.35rem;
                    border-radius: 10px;
                    border: 1px solid rgba(95, 107, 122, 0.18);
                    background: rgba(255, 255, 255, 0.68);
                    padding: 0.25rem 0.4rem;
                    min-height: 2.3rem;
                    box-sizing: border-box;
                }

                .utility-input-group input {
                    width: 100%;
                    min-width: 0;
                }

                .small-text {
                    font-size: 0.7rem;
                    color: var(--text-muted);
                }

                #viewerStatus {
                    position: absolute;
                    left: 1rem;
                    bottom: 1rem;
                    max-width: min(28rem, calc(100% - 2rem));
                    padding: 0.55rem 0.75rem;
                    border-radius: 12px;
                    font-size: 0.78rem;
                    line-height: 1.45;
                    box-shadow: 0 10px 24px rgba(31, 41, 55, 0.14);
                    background: rgba(255, 255, 255, 0.94);
                    color: var(--text-main);
                    border: 1px solid rgba(95, 107, 122, 0.18);
                    z-index: 5;
                    display: none;
                    white-space: pre-line;
                }

                #viewerStatus[data-type="error"] {
                    background: rgba(254, 242, 242, 0.96);
                    color: #991b1b;
                    border-color: rgba(220, 38, 38, 0.22);
                }

                #viewerStatus[data-type="success"] {
                    background: rgba(240, 253, 244, 0.96);
                    color: #166534;
                    border-color: rgba(22, 163, 74, 0.22);
                }

                #dropHint {
                    position: absolute;
                    inset: 1rem;
                    display: none;
                    align-items: center;
                    justify-content: center;
                    border: 2px dashed rgba(20, 157, 125, 0.6);
                    border-radius: 18px;
                    background: rgba(255, 255, 255, 0.62);
                    color: #0f5c4c;
                    font-size: 0.95rem;
                    font-weight: 600;
                    letter-spacing: 0.01em;
                    z-index: 4;
                    pointer-events: none;
                }

                #dropHint.active {
                    display: flex;
                }

                @media (max-width: 480px) {
                    .controls {
                        top: 0.5rem;
                        right: 0.5rem;
                        left: 0.5rem;
                    }

                    .right-ui-panel {
                        max-width: none;
                        max-height: calc(100vh - 1rem);
                    }

                    #panelContent {
                        max-height: calc(100vh - 6rem);
                    }

                    .tab-button {
                        padding-inline: 0.35rem;
                    }
                }
            </style>
            <div class="controls">
                <div class="right-ui-panel">
                    <button id="togglePanelBtn" type="button" aria-label="Expand controls" style="width:100%">&gt;</button>
                    <div id="panelContent" style="display: none;">
                        <div class="tab-buttons">
                            <button class="tab-button active" data-tab="render"><strong>Render</strong></button>
                            <button class="tab-button" data-tab="control"><strong>Control</strong></button>
                            <button class="tab-button" data-tab="edit"><strong>Edit</strong></button>
                        </div>

                        <div id="render-tab-content" class="tab-content" style="display: block;">
                            <div id='meta'>
                                <div id="modelInfo" style='padding-left: 0.1rem; font-size:0.8rem; margin-bottom: 0.5rem;'><strong>[Model Info]</strong> loading...</div>
                                <hr/>
                                <fieldset style="margin-top: 0.5rem;">
                                    <legend style="font-size: 0.8rem;"><strong>Scene</strong></legend>
                                    <label for="bgColorPicker">Background: <input type="color" id="bgColorPicker" value="#eeeeee"></label>
                                    <label> Toggle Grid Helper: <button type="button" id="toggleGridBtn">Show Grid</button></label>
                                </fieldset>


                                <fieldset style="margin-top: 0.5rem;">
                                    <legend style="font-size: 0.8rem;"><strong>Rendering</strong></legend>
                                    <button id="textureBtn" style=" width: 49%">Diffuse</button>
                                    <button id="meshBtn" style=" width: 49%">Geometry</button>
                                    <button id="normalBtn" style=" width: 49%">Normal</button>
                                    <button id="wireframeBtn" style=" width: 49%">Wireframe</button>
                                    <button id="toonShadingBtn" style="display: none;">Toon Shading</button>
                                </fieldset>

                                <fieldset style="margin-top: 0.5rem;">
                                    <legend style="font-size: 0.8rem;"><strong>Environment</strong></legend>
                                    <button id="toggleEnvironmentSectionBtn" class="section-toggle" type="button" aria-expanded="false" data-open-label="Hide controls" data-closed-label="Show controls">
                                        <span class="section-toggle-label">Show controls</span>
                                        <span class="section-toggle-icon">+</span>
                                    </button>
                                    <div id="environmentControlsBody" class="collapsible-content" hidden>
                                        <div class="stack-row" style="margin-bottom: 0.35rem;">
                                            <button id="setBgBtn1" type="button">Env1</button>
                                            <button id="setBgBtn2" type="button">Env2</button>
                                            <button id="setBgBtn3" type="button">Env3</button>
                                            <button id="clearEnvBtn" type="button">Clear</button>
                                        </div>
                                        <label style="display: block; margin-bottom: 0.35rem;">
                                            HDR URL
                                            <div class="stack-row">
                                                <input type="text" id="environmentUrlInput" class="compact-input" placeholder="https://.../studio.hdr">
                                                <button id="loadEnvironmentUrlBtn" type="button">Load HDR</button>
                                            </div>
                                        </label>
                                        <div class="stack-row" style="margin-bottom: 0.35rem;">
                                            <button id="uploadEnvironmentBtn" type="button">Upload HDR</button>
                                            <label style="display: flex; align-items: center; gap: 0.35rem; margin: 0;">
                                                <input type="checkbox" id="environmentBackgroundToggle" checked>
                                                Show HDR background
                                            </label>
                                        </div>
                                        <label style="display: block; margin-bottom: 0.2rem;">
                                            Environment Intensity
                                            <input type="range" id="environmentIntensityInput" style="width: 100%;" min="0" max="4" step="0.1" value="1">
                                        </label>
                                        <div id="environmentIntensityValue" class="small-text" style="margin-bottom: 0.35rem;">1.0x</div>
                                        <label style="display: block; margin-bottom: 0.2rem;">
                                            Exposure
                                            <input type="range" id="exposureInput" style="width: 100%;" min="0.1" max="3" step="0.1" value="1">
                                        </label>
                                        <div id="exposureValue" class="small-text" style="margin-bottom: 0.35rem;">1.0</div>
                                        <label style="display: block; margin-bottom: 0.2rem;">
                                            Rotation
                                            <input type="range" id="environmentRotationInput" style="width: 100%;" min="0" max="360" step="1" value="0">
                                        </label>
                                        <div id="environmentRotationValue" class="small-text">0deg</div>
                                    </div>
                                </fieldset>

                                <fieldset style="margin-top: 0.5rem;">
                                    <legend style="font-size: 0.8rem;"><strong>Util</strong></legend>
                                    <div class="utility-grid">
                                        <button id="autoRotateBtn" type="button">
                                        <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" fill="currentColor" class="bi bi-arrow-counterclockwise" viewBox="0 0 16 16">
                                            <path fill-rule="evenodd" d="M8 3a5 5 0 1 1-4.546 2.914.5.5 0 0 0-.908-.417A6 6 0 1 0 8 2z"/>
                                            <path d="M8 4.466V.534a.25.25 0 0 0-.41-.192L5.23 2.308a.25.25 0 0 0 0 .384l2.36 1.966A.25.25 0 0 0 8 4.466"/>
                                        </svg>
                                        </button>
                                        <button id="screenshotBtn" type="button">
                                        <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" fill="currentColor" class="bi bi-camera-fill" viewBox="0 0 16 16">
                                            <path d="M10.5 8.5a2.5 2.5 0 1 1-5 0 2.5 2.5 0 0 1 5 0"/>
                                            <path d="M2 4a2 2 0 0 0-2 2v6a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V6a2 2 0 0 0-2-2h-1.172a2 2 0 0 1-1.414-.586l-.828-.828A2 2 0 0 0 9.172 2H6.828a2 2 0 0 0-1.414.586l-.828.828A2 2 0 0 1 3.172 4zm.5 2a.5.5 0 1 1 0-1 .5.5 0 0 1 0 1m9 2.5a3.5 3.5 0 1 1-7 0 3.5 3.5 0 0 1 7 0"/>
                                        </svg>
                                        </button>
                                        <button id="downloadScreenshotBtn" type="button">PNG</button>
                                    </div>
                                    <div class="utility-grid">
                                        <div class="utility-record-stack">
                                            <button id="recordBtn" type="button">
                                            <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" fill="currentColor" class="bi bi-record-circle" viewBox="0 0 16 16">
                                                <path d="M8 15A7 7 0 1 1 8 1a7 7 0 0 1 0 14m0 1A8 8 0 1 0 8 0a8 8 0 0 0 0 16"/>
                                                <path d="M11 8a3 3 0 1 1-6 0 3 3 0 0 1 6 0"/>
                                            </svg>
                                            </button>
                                            <button id="stopBtn" type="button" style="display: none;">
                                            <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" fill="currentColor" class="bi bi-stop-circle" viewBox="0 0 16 16">
                                                <path d="M8 15A7 7 0 1 1 8 1a7 7 0 0 1 0 14m0 1A8 8 0 1 0 8 0a8 8 0 0 0 0 16"/>
                                                <path d="M5 6.5A1.5 1.5 0 0 1 6.5 5h3A1.5 1.5 0 0 1 11 6.5v3A1.5 1.5 0 0 1 9.5 11h-3A1.5 1.5 0 0 1 5 9.5z"/>
                                            </svg>
                                            </button>
                                        </div>
                                        <div class="utility-input-group">
                                            <input type="number" id="recordDurationInput" min="1" max="30" step="1" value="5" aria-label="Record duration in seconds">
                                            <span class="small-text">sec</span>
                                        </div>
                                        <button id="quickRecordBtn" type="button">Turntable</button>
                                    </div>
                                    <div id="recordingStatus" class="small-text" style="margin-bottom: 0.35rem;">Idle</div>
                                    <div class="utility-grid">
                                        <button id="copyStateBtn" type="button">Copy Config</button>
                                        <button id="applyStateBtn" type="button">Apply Config</button>
                                        <select id="performanceModeSelect" style="font-size: 0.8rem;">
                                            <option value="default">Default</option>
                                            <option value="performance">Performance</option>
                                            <option value="quality">Quality</option>
                                        </select>
                                    </div>
                                    <textarea id="stateConfigInput" rows="3" class="compact-input" style="font-size: 0.72rem; resize: vertical;" placeholder="State JSON from exportState()"></textarea>
                                    <div class="small-text" style="margin: 0.35rem 0;">Shortcuts: <code>F</code> fit, <code>R</code> reset, <code>Esc</code> clear selection, <code>Space</code> play/pause.</div>
                                    <button id="runAnimationBtn" type="button" style="display: none; background-color: #149ddd">
                                        <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" fill="currentColor" class="bi bi-play-fill" viewBox="0 0 16 16">
                                            <path d="m11.596 8.697-6.363 3.692c-.54.313-1.233-.066-1.233-.697V4.308c0-.63.692-1.01 1.233-.696l6.363 3.692a.802.802 0 0 1 0 1.393"/>
                                        </svg>
                                    </button>
                                    <button id="pauseAnimationBtn" type="button" style="display: none; background-color: #777777">
                                        <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" fill="currentColor" class="bi bi-pause-fill" viewBox="0 0 16 16">
                                            <path d="M5.5 3.5A1.5 1.5 0 0 1 7 5v6a1.5 1.5 0 0 1-3 0V5a1.5 1.5 0 0 1 1.5-1.5m5 0A1.5 1.5 0 0 1 12 5v6a1.5 1.5 0 0 1-3 0V5a1.5 1.5 0 0 1 1.5-1.5"/>
                                        </svg>
                                    </button>
                                    <div id="anim_description" style="display: none; margin-bottom: 0.5rem;">
                                        <strong>Actions:</strong>
                                    </div>
                                    <div id="animationControls" style="display: none; margin-top: 0.5rem;">
                                        <label style="display: block; margin-bottom: 0.35rem;">
                                            Clip:
                                            <select id="animationSelector" style="width: 100%; font-size: 0.8rem;">
                                                <option value="none">None</option>
                                            </select>
                                        </label>
                                        <label style="display: block; margin-bottom: 0.35rem;">
                                            Speed:
                                            <input type="range" id="animationSpeed" style="width: 100%;" min="0.1" max="3" step="0.05" value="1">
                                        </label>
                                        <div id="animationSpeedValue" style="font-size: 0.75rem; color: var(--text-muted); margin-bottom: 0.35rem;">1.00x</div>
                                        <label style="display: block; margin-bottom: 0.35rem;">
                                            Loop:
                                            <select id="animationLoopMode" style="width: 100%; font-size: 0.8rem;">
                                                <option value="repeat">Repeat</option>
                                                <option value="once">Once</option>
                                                <option value="ping-pong">Ping-Pong</option>
                                            </select>
                                        </label>
                                        <label style="display: block; margin-bottom: 0.2rem;">
                                            Timeline:
                                            <input type="range" id="animationTimeline" style="width: 100%;" min="0" max="0" step="0.001" value="0">
                                        </label>
                                        <div id="animationTimeDisplay" style="font-size: 0.75rem; color: var(--text-muted);">0:00 / 0:00</div>
                                    </div>
                                </fieldset>
                                <button id="discardModelBtn" style="background-color: red; width: 100%">Discard Model</button>
                            </div>
                        </div>

                        <div id="control-tab-content" class="tab-content" style="display: none;">
                            <div id="transformControls">
                                <div id="lightControls">
                                    <button type="button" id="toggleLightsBtn" style=" width: 49%">Lights Off</button>
                                    <button type="button" id="toggleLightHelpersBtn" style=" width: 49%;">Hide Light Helpers</button>

                                    <fieldset style="margin-top: 0.5rem;">
                                        <legend style="font-size: 0.8rem;"><strong>Ambient Light</strong></legend>
                                        <label>Color: <input type="color" id="ambientColorPicker" value="#404040"></label>
                                        <label>Intensity: <input type="number" id="ambientIntensity" step="0.5" value="3"></label>
                                    </fieldset>

                                    <fieldset style="margin-top: 0.5rem;">
                                        <legend style="font-size: 0.8rem;"><strong>Directional Light</strong></legend>
                                        <div style="margin-bottom: 0.3rem;">
                                            <select id="directionalLightList" style="width: 100%; font-size: 0.8rem;"></select>
                                        </div>
                                        <label>Color: <input type="color" id="directColorPicker" value="#ffffff"></label>
                                        <label>Position X: <input type="number" id="directPosX" step="0.1" value="5"></label>
                                        <label>Position Y: <input type="number" id="directPosY" step="0.1" value="7.5"></label>
                                        <label>Position Z: <input type="number" id="directPosZ" step="0.1" value="7.5"></label>
                                        <label>Intensity: <input type="number" id="directIntensity" step="0.1" value="3"></label>

                                        <button type="button" id="addLightBtn" style="margin-top: 0.5rem; width: 49%;">Add Light</button>
                                        <button type="button" id="removeLightBtn" style="margin-top: 0.5rem; width: 49%; background-color: red">Remove Light</button>
                                    </fieldset>
                                </div>

                                <fieldset style="margin-top: 0.5rem;">
                                    <legend style="font-size: 0.8rem;"><strong>Camera Setting</strong></legend>
                                    <button id="toggleCameraSectionBtn" class="section-toggle" type="button" aria-expanded="false" data-open-label="Hide controls" data-closed-label="Show controls">
                                        <span class="section-toggle-label">Show controls</span>
                                        <span class="section-toggle-icon">+</span>
                                    </button>
                                    <div id="cameraControlsBody" class="collapsible-content" hidden>
                                        <label>FOV: <input type="number" id="cameraFov" step="1" value="50"></label>
                                        <label>Near: <input type="number" id="cameraNear" step="0.1" value="0.1"></label>
                                        <label>Far: <input type="number" id="cameraFar" step="100" value="1000"></label>
                                        <div class="transform-buttons">
                                            <button class="transform-button" id="resetViewBtn" type="button">Reset View</button>
                                            <button class="transform-button" id="fitModelBtn" type="button">Fit Model</button>
                                        </div>
                                        <button id="frameSelectedBtn" type="button" style="width: 100%;">Frame Selected</button>
                                    </div>
                                </fieldset>

                                <fieldset style="margin-top: 0.5rem; ">
                                    <legend style="font-size: 0.8rem;"><strong>Model Transform</strong></legend>
                                    <div class="transform-buttons">
                                        <button class="transform-button" id="translateBtn">
                                            <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" fill="currentColor" class="bi bi-arrows-move" viewBox="0 0 16 16">
                                                <path fill-rule="evenodd" d="M7.646.146a.5.5 0 0 1 .708 0l2 2a.5.5 0 0 1-.708.708L8.5 1.707V5.5a.5.5 0 0 1-1 0V1.707L6.354 2.854a.5.5 0 1 1-.708-.708zM8 10a.5.5 0 0 1 .5.5v3.793l1.146-1.147a.5.5 0 0 1 .708.708l-2 2a.5.5 0 0 1-.708 0l-2-2a.5.5 0 0 1 .708-.708L7.5 14.293V10.5A.5.5 0 0 1 8 10M.146 8.354a.5.5 0 0 1 0-.708l2-2a.5.5 0 1 1 .708.708L1.707 7.5H5.5a.5.5 0 0 1 0 1H1.707l1.147 1.146a.5.5 0 0 1-.708.708zM10 8a.5.5 0 0 1 .5-.5h3.793l-1.147-1.146a.5.5 0 0 1 .708-.708l2 2a.5.5 0 0 1 0 .708l-2 2a.5.5 0 0 1-.708-.708L14.293 8.5H10.5A.5.5 0 0 1 10 8"/>
                                            </svg>
                                        </button>
                                        <button class="transform-button" id="rotateBtn">
                                            <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" fill="currentColor" class="bi bi-arrow-repeat" viewBox="0 0 16 16">
                                                <path d="M11.534 7h3.932a.25.25 0 0 1 .192.41l-1.966 2.36a.25.25 0 0 1-.384 0l-1.966-2.36a.25.25 0 0 1 .192-.41m-11 2h3.932a.25.25 0 0 0 .192-.41L2.692 6.23a.25.25 0 0 0-.384 0L.342 8.59A.25.25 0 0 0 .534 9"/>
                                                <path fill-rule="evenodd" d="M8 3c-1.552 0-2.94.707-3.857 1.818a.5.5 0 1 1-.771-.636A6.002 6.002 0 0 1 13.917 7H12.9A5 5 0 0 0 8 3M3.1 9a5.002 5.002 0 0 0 8.757 2.182.5.5 0 1 1 .771.636A6.002 6.002 0 0 1 2.083 9z"/>
                                            </svg>
                                        </button>
                                    </div>
                                    <label style="display:none;">Position X: <input type="number" id="posX" step="0.1" value="0"></label>
                                    <label style="display:none;">Position Y: <input type="number" id="posY" step="0.1" value="0"></label>
                                    <label style="display:none;">Position Z: <input type="number" id="posZ" step="0.1" value="0"></label>
                                    <label style="display:none;">Rotation X (deg): <input type="number" id="rotX" step="1" value="0"></label>
                                    <label style="display:none;">Rotation Y (deg): <input type="number" id="rotY" step="1" value="0"></label>
                                    <label style="display:none;">Rotation Z (deg): <input type="number" id="rotZ" step="1" value="0"></label>
                                    <div style="display: none;">Scale: <input type="range" id="scale" style="width: 17rem; background-color: #d3d9de; color: #d3d9de;" min="0.1" max="20" step="0.1" value="1"></div>
                                </fieldset>
                            </div>
                        </div>

                        <div id="edit-tab-content" class="tab-content" style="display: none;">
                            <div id="sceneGraphControls">
                                <fieldset>
                                    <legend style="font-size: 0.8rem;"><strong>Scene Graph</strong></legend>
                                    <div id="sceneGraphTree" style="max-height: 200px; overflow-y: auto;">
                                        <div class="material-toggle" id="materialToggles"></div>
                                    </div>
                                </fieldset>

                                <fieldset style="margin-top: 0.5rem; ">
                                    <legend style="font-size: 0.8rem;"><strong>Material</strong></legend>
                                    <div class="material-editor-grid">
                                        <label>Part
                                            <select id="texturePartSelector" style="font-size: 0.8rem;">
                                                <!-- Part options will be populated here -->
                                            </select>
                                        </label>
                                        <label>Slot
                                            <select id="materialSlotSelector" style="font-size: 0.8rem;"></select>
                                        </label>
                                        <label>Base Color
                                            <input type="color" id="baseColorInput" value="#ffffff">
                                        </label>
                                        <label>Emissive
                                            <input type="color" id="emissiveColorInput" value="#000000">
                                        </label>
                                        <label>Emissive Int.
                                            <input type="number" id="emissiveIntensityInput" min="0" step="0.1" value="1">
                                        </label>
                                        <label>Opacity
                                            <input type="number" id="opacityInput" min="0" max="1" step="0.05" value="1">
                                        </label>
                                        <label>Transparent
                                            <input type="checkbox" id="transparentToggle">
                                        </label>
                                        <label>Double Sided
                                            <input type="checkbox" id="doubleSidedToggle">
                                        </label>
                                        <label>Roughness
                                            <input type="range" id="roughness" min="0" max="1" step="0.01" value="0.5">
                                        </label>
                                        <label>Metalness
                                            <input type="range" id="metalness" min="0" max="1" step="0.01" value="0.5">
                                        </label>
                                        <label>Normal X
                                            <input type="number" id="normalScaleXInput" step="0.1" value="1">
                                        </label>
                                        <label>Normal Y
                                            <input type="number" id="normalScaleYInput" step="0.1" value="1">
                                        </label>
                                        <label>Env Int.
                                            <input type="number" id="envMapIntensityInput" min="0" step="0.1" value="1">
                                        </label>
                                        <label>UV Rot.
                                            <input type="number" id="uvRotationInput" step="0.01" value="0">
                                        </label>
                                        <label>Repeat X
                                            <input type="number" id="uvRepeatXInput" step="0.1" value="1">
                                        </label>
                                        <label>Repeat Y
                                            <input type="number" id="uvRepeatYInput" step="0.1" value="1">
                                        </label>
                                        <label>Offset X
                                            <input type="number" id="uvOffsetXInput" step="0.01" value="0">
                                        </label>
                                        <label>Offset Y
                                            <input type="number" id="uvOffsetYInput" step="0.01" value="0">
                                        </label>
                                    </div>
                                    <hr/>
                                    <div class="texture-toolbar">
                                        <label for="textureTypeSelector" style="font-size: 0.8rem;">Texture</label>
                                        <select id="textureTypeSelector" style="font-size: 0.8rem; max-width: 9rem;">
                                            <option value="map">Diffuse</option>
                                            <option value="roughnessMap">Roughness</option>
                                            <option value="metalnessMap">Metalness</option>
                                            <option value="normalMap">Normal</option>
                                            <option value="aoMap">AO</option>
                                            <option value="emissiveMap">Emissive</option>
                                        </select>
                                        <label for="textureHistorySelector" style="font-size: 0.8rem;">History</label>
                                        <select id="textureHistorySelector" style="font-size: 0.8rem; max-width: 9rem;">
                                            <option value="-1">Current</option>
                                        </select>
                                    </div>
                                    <div class="texture-preview-row">
                                        <div id="texturePreview" class="texture-preview"></div>
                                        <div class="texture-button-group">
                                            <button id="replaceTextureBtn" type="button">Replace</button>
                                            <button id="removeTextureBtn" type="button">Remove</button>
                                            <button id="resetTextureBtn" type="button">Reset</button>
                                            <button id="copyTextureSourceBtn" type="button">Copy Source</button>
                                        </div>
                                    </div>
                                    <div id="textureMetaInfo" class="material-meta">No texture selected</div>
                                </fieldset>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
            <div id="canvas-container" style='text-align: center'>
                <div id="loadingProgressBar"></div>
                <div id="dropHint">Drop a .glb, .gltf, .obj, .fbx, or .ply file here</div>
                <div id="viewerStatus" role="status" aria-live="polite"></div>
                <div id="fileInputContainer" style="display: none;">
                    <input type="file" id="fileInput" accept=".glb,.gltf, .obj,.fbx,.ply">
                    <p style="font-size: 0.8rem; margin-top: 5px;"><strong>Select a GLB/OBJ/FBX/PLY file</strong></p>
                    <hr/>
                    <p style="font-size: 0.8rem; margin-top: 5px;"><strong> or </strong></p>
                    <input type="text" id="urlInput" style="width: 12rem; height: 1.1rem; font-size: 0.8rem;" placeholder="Enter model URL">
                    <button id="loadUrlButton">Load URL</button>
                    <p style="font-size: 0.8rem; margin-top: 5px;"><em> https://huggingface.co/spaces/hhhwan/custom_gs/resolve/main/glbs/fox_quad.glb </em></p>
                </div>
            </div>
            <div id="videoModal" style="display: none; position: fixed; left: 0; top: 0; width: 100%; height: 100%; background-color: rgba(0,0,0,0.7); z-index: 1000; align-items: center; justify-content: center;">
                <div style="background-color: white; padding: 20px; border-radius: 5px; text-align: center;">
                    <h4>Video Preview</h4>
                    <video id="videoPreview" controls style="max-width: 80vw; max-height: 60vh; display: block; margin: 10px auto;"></video>
                    <button id="downloadBtn">Download Video</button>
                    <button id="closeModalBtn" style="margin-left: 10px;">Close</button>
                </div>
            </div>

            <!-- Hidden file inputs for texture replacement -->
            <input type="file" id="diffuseMapInput" style="display: none;" accept="image/*">
            <input type="file" id="roughnessMapInput" style="display: none;" accept="image/*">
            <input type="file" id="metalnessMapInput" style="display: none;" accept="image/*">
            <input type="file" id="normalMapInput" style="display: none;" accept="image/*">
            <input type="file" id="aoMapInput" style="display: none;" accept="image/*">
            <input type="file" id="emissiveMapInput" style="display: none;" accept="image/*">
            <input type="file" id="environmentFileInput" style="display: none;" accept=".hdr">
        `;

        this.scene = new THREE.Scene();
        this.camera = new THREE.PerspectiveCamera(50, 1, 0.1, 1000);
        this.camera.position.set(0, 0, DEFAULT_CAMERA_DISTANCE);
        this.renderer = new THREE.WebGLRenderer({
            antialias: true,
            alpha: true,
            preserveDrawingBuffer: true // screenshot
        });
        this.renderer.setPixelRatio(window.devicePixelRatio);
        this.renderer.outputEncoding = THREE.sRGBEncoding;
        this.renderer.setClearColor(DEFAULT_BACKGROUND_COLOR, 1); // (light gray)
        this.renderer.shadowMap.enabled = true;
        this.renderer.toneMapping = THREE.ACESFilmicToneMapping;
        this.renderer.toneMappingExposure = DEFAULT_EXPOSURE;

        this.animationGeometry = null;
        this.animationMesh = null;
        this.tweenGroup = null;
        this.isIdleAnimationRunning = false;

        this.gridHelper = new THREE.GridHelper(10, 10, 0x888888, 0x444444);
        this.gridHelper.visible = false;
        this.scene.add(this.gridHelper);

        this.shadowRoot.querySelector('#canvas-container').appendChild(this.renderer.domElement);
        this.renderer.domElement.tabIndex = 0;
        this.renderer.domElement.setAttribute('aria-label', '3D model viewer canvas');
        this.renderer.domElement.style.outline = 'none';

        this.state = {
            lightsOn: true,
            viewMode: 'default', // 'default', 'diffuse', 'geometry', 'normal'
            wireframeInitialized: false,
            isWireframeOn: false,
            environment: null, // null, 'env1', 'env2', 'env3'
            environmentUrl: null,
            environmentIntensity: DEFAULT_ENVIRONMENT_INTENSITY,
            environmentRotation: DEFAULT_ENVIRONMENT_ROTATION,
            environmentBackgroundVisible: DEFAULT_ENVIRONMENT_BACKGROUND_VISIBLE,
            isAnimationPlaying: false,
            animationSelection: 'none',
            animationSpeed: DEFAULT_ANIMATION_SPEED,
            animationLoopMode: DEFAULT_ANIMATION_LOOP,
            transformMode: 'none',
            backgroundColor: DEFAULT_BACKGROUND_COLOR,
            exposure: DEFAULT_EXPOSURE,
            selectionMode: DEFAULT_SELECTION_MODE,
            performanceMode: DEFAULT_PERFORMANCE_MODE,
        };

        this.mixer = null;
        this.animationActions = [];
        this.currentAction = null;
        this.isScrubbingAnimationTimeline = false;

        const ambientLightAttr = this.getAttribute('ambient-light');
        if (ambientLightAttr) {
            this.setAmbientLight(ambientLightAttr);
        } else {
            this.ambientLight = new THREE.AmbientLight(0x404040, 3);
            this.scene.add(this.ambientLight);
        }

        const directLightAttr = this.getAttribute('direct-light');
        if (directLightAttr) {
            this.setDirectLight(directLightAttr);
        } else {
            this.directionalLights = []; // Directional Lights array init
            this.directionalLightHelpers = []; // DirectionalLightHelper array init
            this.addDirectionalLight(); // Basic Directional Light
            this.selectedDirectionalLightIndex = 0; // first light selected
        }

        this.controls = new OrbitControls(this.camera, this.renderer.domElement);
        this.controls.enableDamping = true;
        this.controls.dampingFactor = 0.05;
        this.controls.target.set(0, 0, 0);
        this.controls.addEventListener('change', () => {
            this.updateControlPanel();
            if (!this.isApplyingCameraState) {
                this.emitCameraChange('controls');
            }
        });

        this.textureLoader = new THREE.TextureLoader();
        this.whiteTexture = this.textureLoader.load('/assets/viewer/white.jpg');
        this.whiteTexture.mapping = THREE.EquirectangularReflectionMapping;

        this.gradTexture = this.textureLoader.load('/assets/viewer/gradient.jpg');
        this.gradTexture.mapping = THREE.EquirectangularReflectionMapping;

        this.dracoLoader = new DRACOLoader();
        this.objLoader = new OBJLoader();
        this.dracoLoader.setDecoderPath('/vendor/three/examples/jsm/libs/draco/gltf/');
        this.gltfLoader = new GLTFLoader();
        this.fbxLoader = new FBXLoader(); 
        this.plyLoader = new PLYLoader();
        this.gltfLoader.setDRACOLoader(this.dracoLoader);
        this.model = null;
        this.originalMaterials = {};
        this.initialMaterials = {};
        this.wireframeMeshes = [];
        this.modelSize = 8;
        this.autoRotate = false;
        this.anglePerSecond = 30;
        this.lastTime = 0;
        this.toonEnabled = false;
        this.noPBR = false;
        this.ambientLight.visible = this.state.lightsOn;
        this.directionalLights.forEach(light => {
            light.visible = this.state.lightsOn;
        });

        this.toonMaterial = null;
        this.toonMaterialBackups = new Map();
        this.standardMaterials = [];

        this.showLightHelpers = false; // Light Helper visiblity - default true, changed to true initially for better UX
        this.canAdjustRoughnessMetalness = false;
        this.meshParts = [];
        this.meshPartTextureInfo = [];
        this.textureHistory = new Map();
        this.currentEnvironmentTexture = null;
        this.environmentLoadToken = 0;
        this.cameraTransitionFrame = null;
        this.currentModelSource = null;
        this.currentModelFileName = null;
        this.isReflectingAttributes = false;
        this.isApplyingCameraState = false;
        this.modelDefaultCameraState = null;
        this.raycaster = new THREE.Raycaster();
        this.pointerNdc = new THREE.Vector2();

        this.sceneGraphLabelByMeshUuid = new Map();
        this.selectedSceneGraphLabel = null;
        this.selectedMeshPart = null;
        this.selectedMeshPartIndex = -1;
        this.selectedMaterialIndex = 0;
        this.hoveredMeshPart = null;
        this.canvasPointerDown = null;
        this.selectionOutline = new THREE.Box3Helper(new THREE.Box3(), 0x18a957);
        this.selectionOutline.visible = false;
        this.scene.add(this.selectionOutline);
        this.hoverOutline = new THREE.Box3Helper(new THREE.Box3(), 0x2563eb);
        this.hoverOutline.visible = false;
        this.scene.add(this.hoverOutline);

        // --- State Variables for Recording ---
        this.mediaRecorder = null;
        this.recordedChunks = [];
        this.videoBlob = null; // To store the final blob
        this.stream = null;    // To store the canvas stream
        this.recordingProgressTimer = null;
        this.quickRecordingTimeout = null;
        this.quickRecordingStartedAt = 0;
        this.quickRecordingDuration = DEFAULT_RECORDING_DURATION;
        this.quickRecordingPreviousAutoRotate = null;

        // --- State Variables for Explode Effect ---
        this.modelCenter = null;
        this.modelMaxDim = 0;
        this.resizeObserver = null;
        this.isConnectedToDom = false;
        this.statusTimeout = null;
        this.dropHoverDepth = 0;
        this.handleAnimationMixerFinished = (event) => {
            if (event?.action !== this.currentAction) {
                return;
            }

            this.state.isAnimationPlaying = false;
            this.refreshUiFromState({ syncTextureUi: false });
            this.emitAnimationChange('mixer', 'finished');
        };

        this.startRecording = this.startRecording.bind(this);
        this.stopRecording = this.stopRecording.bind(this);
        this.downloadVideo = this.downloadVideo.bind(this);
        this.closeModal = this.closeModal.bind(this);
        this.handleResize = this.resizeRenderer.bind(this);
        this.handleDragEnter = this.handleDragEnter.bind(this);
        this.handleDragOver = this.handleDragOver.bind(this);
        this.handleDragLeave = this.handleDragLeave.bind(this);
        this.handleDrop = this.handleDrop.bind(this);

        // TransformControls instance generation
        this.transformControls = new TransformControls(this.camera, this.renderer.domElement);
        this.transformControls.addEventListener('change', () => {
            this.refreshUiFromState({ syncTextureUi: false });
            this.emitCameraChange('transform-controls');
            this.requestRender();
        });
        this.transformControls.visible = false; // init invisible
        this.scene.add(this.transformControls);

        this.initEventListeners();
        this.initLightUIValues(); // Light UI init
        this.initCameraUIValues(); // Camera UI init
        this.updateDirectionalLightHelpersVisibility(); // Initial helper visibility setup
        this.initDiscardButton();
        this.initTextureMapUI();
        this.initTabSwitching(); // Initialize tab switching functionality
        this.defaultCameraState = this.getCameraStateSnapshot();
        this.updateCameraActionButtons();
        this.updateRecordingStatus('Idle');
        this.refreshUiFromState();
    }

    initIdleAnimation() {
        if (typeof TWEEN === 'undefined') {
            console.error('TWEEN is not defined. Ensure Tween.js is loaded before simple-model-viewer.js.');
            return;
        }

        const vertexCount = 20; // max vertices num
        this.animationGeometry = new THREE.BufferGeometry();
        const material = new THREE.PointsMaterial({ color: 0x777777, size: 0.8 });
        this.pointColor = new THREE.Color();

        // const material = new THREE.PointsMaterial({
        //     color: this.pointColor, 
        //     size: 0.6, 
        //     blending: THREE.AdditiveBlending, 
        //     transparent: true, 
        //     opacity: 0.8, 
        // });

        this.animationMesh = new THREE.Points(this.animationGeometry, material);
        this.scene.add(this.animationMesh);

        this.tweenGroup = new TWEEN.Group();

        const initialPositions = new Float32Array(vertexCount * 3).fill(0);
        this.animationGeometry.setAttribute('position', new THREE.BufferAttribute(initialPositions, 3));
    
        this.setupAnimationSteps(vertexCount);
    
        this.isIdleAnimationRunning = true;
    }
    
    setupAnimationSteps(vertexCount) {
        const positions = this.animationGeometry.attributes.position.array;
    
        const hexagonPositions = [];
        const hexagonVertices = this.getHexagonVertices(6); 
        for (let i = 0; i < vertexCount; i++) {
            hexagonPositions.push(...hexagonVertices[i % hexagonVertices.length]); 
        }
    
        const tweenToHexagon = new TWEEN.Tween(positions, this.tweenGroup)
            .to(hexagonPositions, 1000)
            .easing(TWEEN.Easing.Quadratic.InOut)
            .onUpdate(() => {
                this.animationGeometry.attributes.position.needsUpdate = true;
                // const hue = (performance.now() / 5000) % 1; 
                // this.pointColor.setHSL(hue, 1, 0.5); 
                // this.animationMesh.material.color = this.pointColor;
            });
    
        const dodecahedronPositions = [];
        const dodecahedronVertices = this.getDodecahedronVertices(4); 
        for (let i = 0; i < vertexCount; i++) {
            dodecahedronPositions.push(...dodecahedronVertices[i % dodecahedronVertices.length]); 
        }
    
        const tweenToDodecahedron = new TWEEN.Tween(positions, this.tweenGroup)
            .to(dodecahedronPositions, 2000)
            .easing(TWEEN.Easing.Quadratic.InOut)
            .onUpdate(() => {
                this.animationGeometry.attributes.position.needsUpdate = true;
                // const hue = (performance.now() / 5000) % 1; 
                // this.pointColor.setHSL(hue, 1, 0.5); 
                // this.animationMesh.material.color = this.pointColor;
            });
    
        const icosahedronPositions = [];
        const icosahedronVertices = this.getIcosahedronVertices(3); 
        for (let i = 0; i < vertexCount; i++) {
            icosahedronPositions.push(...icosahedronVertices[i % icosahedronVertices.length]); 
        }
    
        const tweenToIcosahedron = new TWEEN.Tween(positions, this.tweenGroup)
            .to(icosahedronPositions, 2000)
            .easing(TWEEN.Easing.Quadratic.InOut)
            .onUpdate(() => {
                this.animationGeometry.attributes.position.needsUpdate = true;
                // const hue = (performance.now() / 5000) % 1; 
                // this.pointColor.setHSL(hue, 1, 0.5); 
                // this.animationMesh.material.color = this.pointColor;
            });
    
        // all vertices to origin
        const backToPointPositions = new Array(vertexCount * 3).fill(0);
    
        const backToPoint = new TWEEN.Tween(positions, this.tweenGroup)
            .to(backToPointPositions, 2000)
            .easing(TWEEN.Easing.Quadratic.InOut)
            .onUpdate(() => {
                this.animationGeometry.attributes.position.needsUpdate = true;
                // const hue = (performance.now() / 5000) % 1; 
                // this.pointColor.setHSL(hue, 1, 0.5); 
                // this.animationMesh.material.color = this.pointColor;
            })
            .onComplete(() => {
                tweenToHexagon.start(); // loop
            });

        tweenToHexagon.chain(tweenToIcosahedron);
        tweenToIcosahedron.chain(tweenToDodecahedron);
        tweenToDodecahedron.chain(backToPoint);
    
        tweenToHexagon.start(); // animation
    }
    
    // Hexagon index generator
    getHexagonVertices(radius) {
        const vertices = [];
        for (let i = 0; i < 6; i++) {
            const angle = (Math.PI / 3) * i;
            const x = radius * Math.cos(angle);
            const y = radius * Math.sin(angle);
            vertices.push([x, y, 0]);
        }
        return vertices;
    }
    
    // Dodecahedron index generator
    getDodecahedronVertices(radius) {
        const phi = (1 + Math.sqrt(5)) / 2; // golden ratio
        const vertices = [
            [-1, -1, -1], [-1, -1, 1], [-1, 1, -1], [-1, 1, 1],
            [1, -1, -1], [1, -1, 1], [1, 1, -1], [1, 1, 1],
            [0, -1 / phi, -phi], [0, -1 / phi, phi], [0, 1 / phi, -phi], [0, 1 / phi, phi],
            [-1 / phi, -phi, 0], [-1 / phi, phi, 0], [1 / phi, -phi, 0], [1 / phi, phi, 0],
            [-phi, 0, -1 / phi], [-phi, 0, 1 / phi], [phi, 0, -1 / phi], [phi, 0, 1 / phi]
        ].map(v => v.map(coord => coord * radius)); // 
        return vertices;
    }
    
    // Icosahedron index generator
    getIcosahedronVertices(radius) {
        const phi = (1 + Math.sqrt(5)) / 2; // golden ratio
        const vertices = [
            [0, 1, phi], [0, 1, -phi], [0, -1, phi], [0, -1, -phi],
            [1, phi, 0], [1, -phi, 0], [-1, phi, 0], [-1, -phi, 0],
            [phi, 0, 1], [phi, 0, -1], [-phi, 0, 1], [-phi, 0, -1]
        ].map(v => v.map(coord => coord * radius)); // 
        return vertices;
    }

    initTabSwitching() {
        const tabButtons = this.shadowRoot.querySelectorAll('.tab-button');
        const tabContents = this.shadowRoot.querySelectorAll('.tab-content');

        tabButtons.forEach(button => {
            button.addEventListener('click', () => {
                const tabName = button.dataset.tab;

                // Deactivate all tabs and hide all content
                tabButtons.forEach(btn => btn.classList.remove('active'));
                tabContents.forEach(content => content.style.display = 'none');

                // Activate the clicked tab and show its content
                button.classList.add('active');
                this.shadowRoot.querySelector(`#${tabName}-tab-content`).style.display = 'block';
            });
        });
    }


    initCameraUIValues() {
        // Camera Settings UI init
        this.shadowRoot.querySelector('#cameraFov').value = this.camera.fov;
        this.shadowRoot.querySelector('#cameraNear').value = this.camera.near;
        this.shadowRoot.querySelector('#cameraFar').value = this.camera.far;
    }

    formatAnimationTime(seconds) {
        const safeSeconds = Number.isFinite(seconds) ? Math.max(0, seconds) : 0;
        const minutes = Math.floor(safeSeconds / 60);
        const remainder = safeSeconds - (minutes * 60);
        const wholeSeconds = Math.floor(remainder);
        const tenths = Math.floor((remainder - wholeSeconds) * 10);
        const paddedSeconds = `${wholeSeconds}`.padStart(2, '0');

        if (safeSeconds >= 10) {
            return `${minutes}:${paddedSeconds}`;
        }

        return `${minutes}:${paddedSeconds}.${tenths}`;
    }

    getAnimationDuration(action = this.currentAction) {
        return action?.getClip?.()?.duration || 0;
    }

    getAnimationStateSnapshot() {
        const clip = this.currentAction?.getClip?.() || null;
        const clipIndex = clip
            ? this.animationActions.findIndex((animationAction) => animationAction.getClip() === clip)
            : -1;
        const duration = this.getAnimationDuration();

        return {
            index: clipIndex,
            name: clip?.name || null,
            selected: this.state.animationSelection,
            isPlaying: this.state.isAnimationPlaying,
            speed: this.state.animationSpeed,
            loopMode: this.state.animationLoopMode,
            time: this.currentAction ? Math.min(this.currentAction.time, duration || this.currentAction.time) : 0,
            duration,
            clipCount: this.animationActions.length,
        };
    }

    getAnimationLoopSettings(loopMode = this.state.animationLoopMode) {
        switch (loopMode) {
            case 'once':
                return {
                    loop: THREE.LoopOnce,
                    repetitions: 0,
                    clampWhenFinished: true,
                };
            case 'ping-pong':
                return {
                    loop: THREE.LoopPingPong,
                    repetitions: Infinity,
                    clampWhenFinished: false,
                };
            case 'repeat':
            default:
                return {
                    loop: THREE.LoopRepeat,
                    repetitions: Infinity,
                    clampWhenFinished: false,
                };
        }
    }

    configureAnimationAction(action) {
        if (!action) {
            return;
        }

        const { loop, repetitions, clampWhenFinished } = this.getAnimationLoopSettings();
        action.setLoop(loop, repetitions);
        action.clampWhenFinished = clampWhenFinished;
        action.setEffectiveTimeScale(this.state.animationSpeed);
        action.enabled = true;
    }

    populateAnimationSelector() {
        const animationSelector = this.shadowRoot.querySelector('#animationSelector');
        if (!animationSelector) {
            return;
        }

        animationSelector.innerHTML = '';

        const noneOption = document.createElement('option');
        noneOption.value = 'none';
        noneOption.textContent = 'None';
        animationSelector.appendChild(noneOption);

        this.animationActions.forEach((action, index) => {
            const option = document.createElement('option');
            option.value = `${index}`;
            option.textContent = action.getClip()?.name || `Animation ${index + 1}`;
            animationSelector.appendChild(option);
        });
    }

    updateAnimationUi() {
        const hasAnimations = this.animationActions.length > 0;
        const animationControls = this.shadowRoot.querySelector('#animationControls');
        const animationSelector = this.shadowRoot.querySelector('#animationSelector');
        const animationSpeed = this.shadowRoot.querySelector('#animationSpeed');
        const animationSpeedValue = this.shadowRoot.querySelector('#animationSpeedValue');
        const animationLoopMode = this.shadowRoot.querySelector('#animationLoopMode');
        const animationTimeline = this.shadowRoot.querySelector('#animationTimeline');
        const animationTimeDisplay = this.shadowRoot.querySelector('#animationTimeDisplay');
        const runButton = this.shadowRoot.querySelector('#runAnimationBtn');
        const pauseButton = this.shadowRoot.querySelector('#pauseAnimationBtn');
        const description = this.shadowRoot.querySelector('#anim_description');
        const animationState = this.getAnimationStateSnapshot();
        const duration = animationState.duration;
        const hasCurrentAction = !!this.currentAction;

        if (description) {
            description.style.display = hasAnimations ? 'block' : 'none';
        }

        if (animationControls) {
            animationControls.style.display = hasAnimations ? 'block' : 'none';
        }

        if (animationSelector) {
            const selectedValue = animationState.index >= 0 ? `${animationState.index}` : 'none';
            if (animationSelector.value !== selectedValue) {
                animationSelector.value = selectedValue;
            }
            animationSelector.disabled = !hasAnimations;
        }

        if (animationSpeed) {
            animationSpeed.value = `${this.state.animationSpeed}`;
            animationSpeed.disabled = !hasAnimations;
        }

        if (animationSpeedValue) {
            animationSpeedValue.textContent = `${this.state.animationSpeed.toFixed(2)}x`;
        }

        if (animationLoopMode) {
            animationLoopMode.value = this.state.animationLoopMode;
            animationLoopMode.disabled = !hasAnimations;
        }

        if (animationTimeline) {
            animationTimeline.disabled = !hasCurrentAction;
            animationTimeline.max = `${duration || 0}`;
            if (!this.isScrubbingAnimationTimeline) {
                animationTimeline.value = `${hasCurrentAction ? Math.min(this.currentAction.time, duration || this.currentAction.time) : 0}`;
            }
        }

        if (animationTimeDisplay) {
            animationTimeDisplay.textContent = `${this.formatAnimationTime(animationState.time)} / ${this.formatAnimationTime(duration)}`;
        }

        if (hasCurrentAction) {
            runButton.style.display = this.state.isAnimationPlaying ? 'none' : 'inline-block';
            pauseButton.style.display = this.state.isAnimationPlaying ? 'inline-block' : 'none';
        } else {
            runButton.style.display = 'none';
            pauseButton.style.display = 'none';
        }
    }


    initLightUIValues() {
        // Ambient Light UI init
        this.shadowRoot.querySelector('#ambientColorPicker').value = `#${this.ambientLight.color.getHexString()}`;
        this.shadowRoot.querySelector('#ambientIntensity').value = this.ambientLight.intensity;

        // Directional Light UI init
        if (this.directionalLights.length > 0) {
            this.populateDirectionalLightList(); // Directional Light List UI
            this.updateDirectionalLightUIValues(); // Directional Light UI init
        }
    }


    connectedCallback() {
        this.isConnectedToDom = true;
        this.resizeRenderer();
        if (!this.resizeObserver) {
            this.resizeObserver = new ResizeObserver(() => this.resizeRenderer());
        }
        this.resizeObserver.observe(this);
        this.renderer.setAnimationLoop((time) => this.animate(time));
        if (!this.getAttribute('src')) {
            const fileInputContainer = this.shadowRoot.querySelector('#fileInputContainer');
            fileInputContainer.style.display = 'block';
        }
    }

    disconnectedCallback() {
        this.isConnectedToDom = false;
        if (this.resizeObserver) {
            this.resizeObserver.disconnect();
        }
        this.stopCameraTransition();
        this.renderer.setAnimationLoop(null);
        this.stopRecording();
        this.clearQuickRecordingTimers();
        if (this.statusTimeout) {
            clearTimeout(this.statusTimeout);
            this.statusTimeout = null;
        }
    }

    requestRender() {
        if (!this.renderer) {
            return;
        }

        this.renderer.render(this.scene, this.camera);
    }

    emitEvent(name, detail = {}) {
        this.dispatchEvent(new CustomEvent(name, {
            detail,
            bubbles: true,
            composed: true,
        }));
    }

    emitViewerError(action, error, extra = {}) {
        const message = error instanceof Error ? error.message : String(error);
        this.showStatus(message, 'error', 5000);
        this.emitEvent('viewer-error', {
            action,
            message,
            error,
            ...extra,
        });
    }

    showStatus(message, type = 'info', timeout = 0) {
        const statusElement = this.shadowRoot.querySelector('#viewerStatus');
        if (!statusElement) {
            return;
        }

        if (this.statusTimeout) {
            clearTimeout(this.statusTimeout);
            this.statusTimeout = null;
        }

        if (!message) {
            statusElement.style.display = 'none';
            statusElement.textContent = '';
            statusElement.dataset.type = '';
            return;
        }

        statusElement.textContent = message;
        statusElement.dataset.type = type;
        statusElement.style.display = 'block';

        if (timeout > 0) {
            this.statusTimeout = window.setTimeout(() => {
                this.showStatus('');
            }, timeout);
        }
    }

    setDropHintVisible(visible) {
        const hint = this.shadowRoot.querySelector('#dropHint');
        if (!hint) {
            return;
        }

        hint.classList.toggle('active', !!visible);
    }

    getSuggestedBaseFileName(fallback = 'model') {
        return sanitizeFilenameSegment(
            this.currentModelFileName
            || this.getAttribute('src')
            || fallback,
            fallback
        );
    }

    createExportFileName(kind, extension) {
        return `${this.getSuggestedBaseFileName(kind)}-${kind}-${formatTimestampForFilename()}.${extension}`;
    }

    downloadBlob(blob, filename) {
        const url = URL.createObjectURL(blob);
        const anchor = document.createElement('a');
        anchor.style.display = 'none';
        anchor.href = url;
        anchor.download = filename;
        document.body.appendChild(anchor);
        anchor.click();
        document.body.removeChild(anchor);
        URL.revokeObjectURL(url);
    }

    loadModelFromFile(file) {
        if (!file) {
            return Promise.reject(new Error('A model file is required.'));
        }

        if (!isSupportedModelFileName(file.name)) {
            return Promise.reject(new Error(`Unsupported file format: .${getFileExtension(file.name)}`));
        }

        return this.loadModel(URL.createObjectURL(file), file.name);
    }

    reflectAttribute(name, value) {
        this.isReflectingAttributes = true;
        try {
            if (value === null || value === undefined || value === '') {
                this.removeAttribute(name);
                return;
            }

            this.setAttribute(name, `${value}`);
        } finally {
            this.isReflectingAttributes = false;
        }
    }

    reflectBooleanAttribute(name, enabled) {
        this.isReflectingAttributes = true;
        try {
            if (enabled) {
                this.setAttribute(name, '');
            } else {
                this.removeAttribute(name);
            }
        } finally {
            this.isReflectingAttributes = false;
        }
    }

    getCameraStateSnapshot() {
        return {
            position: {
                x: this.camera.position.x,
                y: this.camera.position.y,
                z: this.camera.position.z,
            },
            target: {
                x: this.controls.target.x,
                y: this.controls.target.y,
                z: this.controls.target.z,
            },
            up: {
                x: this.camera.up.x,
                y: this.camera.up.y,
                z: this.camera.up.z,
            },
            fov: this.camera.fov,
            near: this.camera.near,
            far: this.camera.far,
        };
    }

    cloneCameraState(snapshot) {
        if (!snapshot) {
            return null;
        }

        return {
            position: { ...snapshot.position },
            target: { ...snapshot.target },
            up: { ...snapshot.up },
            fov: snapshot.fov,
            near: snapshot.near,
            far: snapshot.far,
        };
    }

    captureCurrentCameraStateAsDefault(options = {}) {
        const { scope = this.model ? 'model' : 'viewer' } = options;
        const snapshot = this.cloneCameraState(this.getCameraStateSnapshot());

        if (scope === 'model' && this.model) {
            this.modelDefaultCameraState = snapshot;
        } else {
            this.defaultCameraState = snapshot;
        }

        return snapshot;
    }

    syncCameraAttributes() {
        this.reflectAttribute('camera-orbit', formatVector3String(this.camera.position));
        this.reflectAttribute('camera-target', formatVector3String(this.controls.target));
        this.reflectAttribute('camera-up', formatVector3String(this.camera.up));
    }

    stopCameraTransition() {
        if (this.cameraTransitionFrame !== null) {
            cancelAnimationFrame(this.cameraTransitionFrame);
            this.cameraTransitionFrame = null;
        }
    }

    setCameraStateSnapshot(snapshot) {
        if (!snapshot) {
            return false;
        }

        this.isApplyingCameraState = true;
        try {
            this.camera.position.set(snapshot.position.x, snapshot.position.y, snapshot.position.z);
            this.controls.target.set(snapshot.target.x, snapshot.target.y, snapshot.target.z);
            this.camera.up.set(snapshot.up.x, snapshot.up.y, snapshot.up.z);
            this.camera.fov = snapshot.fov;
            this.camera.near = snapshot.near;
            this.camera.far = snapshot.far;
            this.camera.lookAt(this.controls.target);
            this.camera.updateProjectionMatrix();
            this.controls.update();
            this.initCameraUIValues();
            this.requestRender();
        } finally {
            this.isApplyingCameraState = false;
        }
        return true;
    }

    interpolateCameraState(start, end, t) {
        const lerp = THREE.MathUtils.lerp;
        return {
            position: {
                x: lerp(start.position.x, end.position.x, t),
                y: lerp(start.position.y, end.position.y, t),
                z: lerp(start.position.z, end.position.z, t),
            },
            target: {
                x: lerp(start.target.x, end.target.x, t),
                y: lerp(start.target.y, end.target.y, t),
                z: lerp(start.target.z, end.target.z, t),
            },
            up: {
                x: lerp(start.up.x, end.up.x, t),
                y: lerp(start.up.y, end.up.y, t),
                z: lerp(start.up.z, end.up.z, t),
            },
            fov: lerp(start.fov, end.fov, t),
            near: lerp(start.near, end.near, t),
            far: lerp(start.far, end.far, t),
        };
    }

    applyCameraStateSnapshot(snapshot, options = {}) {
        const {
            source = 'api',
            emitEvent = true,
            syncAttribute = false,
            saveAsDefault = false,
            transitionDuration = 0,
        } = options;
        const targetSnapshot = this.cloneCameraState(snapshot);

        if (!targetSnapshot) {
            return false;
        }

        const finalize = () => {
            this.setCameraStateSnapshot(targetSnapshot);

            if (syncAttribute) {
                this.syncCameraAttributes();
            }

            if (saveAsDefault) {
                this.captureCurrentCameraStateAsDefault();
            }

            if (emitEvent) {
                this.emitCameraChange(source);
            }
        };

        this.stopCameraTransition();

        if (transitionDuration > 0) {
            const startSnapshot = this.cloneCameraState(this.getCameraStateSnapshot());
            const startTime = performance.now();
            const step = (timestamp) => {
                const progress = Math.min(1, (timestamp - startTime) / transitionDuration);
                const eased = 1 - ((1 - progress) * (1 - progress));
                this.setCameraStateSnapshot(this.interpolateCameraState(startSnapshot, targetSnapshot, eased));

                if (progress < 1) {
                    this.cameraTransitionFrame = requestAnimationFrame(step);
                    return;
                }

                this.cameraTransitionFrame = null;
                finalize();
            };

            this.cameraTransitionFrame = requestAnimationFrame(step);
            return true;
        }

        finalize();
        return true;
    }

    emitCameraChange(source = 'api') {
        this.emitEvent('viewer-camera-change', {
            source,
            camera: this.getCameraStateSnapshot(),
        });
    }

    emitSelectionChange(source = 'api') {
        const selection = this.selectedMeshPart ? {
            index: this.selectedMeshPartIndex,
            name: this.selectedMeshPart.name || null,
            uuid: this.selectedMeshPart.uuid,
            materialIndex: this.selectedMaterialIndex,
            materialCount: getMaterialCount(this.selectedMeshPart),
            isMultiMaterial: getMaterialCount(this.selectedMeshPart) > 1,
            visible: this.isObjectEffectivelyVisible(this.selectedMeshPart),
        } : null;

        this.emitEvent('viewer-selection-change', {
            source,
            selectionMode: this.state.selectionMode,
            selection,
        });
    }

    emitAnimationChange(source = 'api', action = 'update') {
        this.emitEvent('viewer-animation-change', {
            source,
            action,
            animation: this.getAnimationStateSnapshot(),
        });
    }

    emitEnvironmentChange(source = 'api', action = 'update') {
        this.emitEvent('viewer-environment-change', {
            source,
            action,
            environment: {
                preset: this.state.environment === CUSTOM_ENVIRONMENT_ID ? null : this.state.environment,
                url: this.state.environmentUrl,
                intensity: this.state.environmentIntensity,
                rotation: this.state.environmentRotation,
                backgroundVisible: this.state.environmentBackgroundVisible,
                backgroundColor: this.state.backgroundColor,
                exposure: this.state.exposure,
            },
        });
    }

    getEffectiveEnvironmentIntensity(materialIntensity = 1) {
        return (materialIntensity ?? 1) * this.state.environmentIntensity;
    }

    applyEnvironmentPresentation() {
        const texture = this.currentEnvironmentTexture || null;
        this.scene.environment = texture;
        this.scene.background = texture && this.state.environmentBackgroundVisible ? texture : null;

        if ('environmentIntensity' in this.scene) {
            this.scene.environmentIntensity = this.state.environmentIntensity;
        }

        if ('backgroundIntensity' in this.scene) {
            this.scene.backgroundIntensity = this.state.environmentBackgroundVisible ? 1 : 0;
        }

        const rotationRadians = THREE.MathUtils.degToRad(this.state.environmentRotation);
        if (this.scene.environmentRotation?.set) {
            this.scene.environmentRotation.set(0, rotationRadians, 0);
        }
        if (this.scene.backgroundRotation?.set) {
            this.scene.backgroundRotation.set(0, rotationRadians, 0);
        }
    }

    applyEnvironmentMaterialSettings() {
        if (!this.model) {
            return;
        }

        this.model.traverse((child) => {
            if (!child.isMesh) {
                return;
            }

            const storedMaterialEntry = this.getMaterialStoreEntry(child, this.originalMaterials);
            const liveMaterials = getMaterialArray(child.material);
            const storedMaterials = getMaterialArray(storedMaterialEntry);

            liveMaterials.forEach((material, index) => {
                const storedMaterial = storedMaterials[index] || storedMaterials[0] || null;
                if (!material || !storedMaterial || !('envMapIntensity' in material)) {
                    return;
                }

                material.envMapIntensity = this.getEffectiveEnvironmentIntensity(storedMaterial.envMapIntensity ?? 1);
                material.needsUpdate = true;
            });
        });
    }

    updateEnvironmentControls() {
        const environmentUrlInput = this.shadowRoot.querySelector('#environmentUrlInput');
        const environmentIntensityInput = this.shadowRoot.querySelector('#environmentIntensityInput');
        const environmentIntensityValue = this.shadowRoot.querySelector('#environmentIntensityValue');
        const exposureInput = this.shadowRoot.querySelector('#exposureInput');
        const exposureValue = this.shadowRoot.querySelector('#exposureValue');
        const environmentRotationInput = this.shadowRoot.querySelector('#environmentRotationInput');
        const environmentRotationValue = this.shadowRoot.querySelector('#environmentRotationValue');
        const environmentBackgroundToggle = this.shadowRoot.querySelector('#environmentBackgroundToggle');
        const performanceModeSelect = this.shadowRoot.querySelector('#performanceModeSelect');
        const clearEnvBtn = this.shadowRoot.querySelector('#clearEnvBtn');

        if (environmentUrlInput) {
            environmentUrlInput.value = this.state.environmentUrl || '';
        }
        if (environmentIntensityInput) {
            environmentIntensityInput.value = `${this.state.environmentIntensity}`;
        }
        if (environmentIntensityValue) {
            environmentIntensityValue.textContent = `${this.state.environmentIntensity.toFixed(1)}x`;
        }
        if (exposureInput) {
            exposureInput.value = `${this.state.exposure}`;
        }
        if (exposureValue) {
            exposureValue.textContent = `${this.state.exposure.toFixed(1)}`;
        }
        if (environmentRotationInput) {
            environmentRotationInput.value = `${this.state.environmentRotation}`;
        }
        if (environmentRotationValue) {
            environmentRotationValue.textContent = `${Math.round(this.state.environmentRotation)}deg`;
        }
        if (environmentBackgroundToggle) {
            environmentBackgroundToggle.checked = !!this.state.environmentBackgroundVisible;
        }
        if (performanceModeSelect) {
            performanceModeSelect.value = this.state.performanceMode;
        }
        if (clearEnvBtn) {
            clearEnvBtn.disabled = !this.currentEnvironmentTexture;
        }
    }

    updateDiscardButtonVisibility() {
        const discardButton = this.shadowRoot.querySelector('#discardModelBtn');
        if (discardButton) {
            discardButton.style.display = this.model ? 'inline-block' : 'none';
        }
    }

    getMaterialStoreEntry(mesh, store = this.originalMaterials) {
        return mesh ? store[mesh.uuid] || null : null;
    }

    getMaterialStoreMaterial(mesh, materialIndex = 0, store = this.originalMaterials) {
        return getMaterialEntryAt(this.getMaterialStoreEntry(mesh, store), materialIndex);
    }

    getMaterialSlotCountForMesh(mesh) {
        return getMaterialCount(mesh);
    }

    getTextureHistoryKey(mesh, materialIndex = 0) {
        return mesh ? `${mesh.uuid}:${materialIndex}` : '';
    }

    getSceneGraphLabelForMesh(mesh) {
        return mesh ? this.sceneGraphLabelByMeshUuid.get(mesh.uuid) || null : null;
    }

    setSelectedSceneGraphLabel(labelElement) {
        if (this.selectedSceneGraphLabel === labelElement) {
            return;
        }

        if (this.selectedSceneGraphLabel) {
            this.selectedSceneGraphLabel.classList.remove('selected');
        }

        this.selectedSceneGraphLabel = labelElement || null;

        if (this.selectedSceneGraphLabel) {
            this.selectedSceneGraphLabel.classList.add('selected');
        }
    }

    isSelectionChannelEnabled(channel) {
        const mode = this.state.selectionMode;
        if (mode === 'none') {
            return false;
        }

        if (mode === 'scene-graph' || mode === 'all') {
            return channel === 'scene-graph' || channel === 'canvas';
        }

        return mode === channel;
    }

    isObjectEffectivelyVisible(object) {
        let current = object;

        while (current) {
            if (current.visible === false) {
                return false;
            }
            current = current.parent;
        }

        return true;
    }

    isSelectableMesh(mesh) {
        return !!(
            mesh
            && mesh.isMesh
            && getEditableMaterial(mesh)
            && this.meshParts.includes(mesh)
            && this.isObjectEffectivelyVisible(mesh)
        );
    }

    updateOutlineHelper(helper, object) {
        if (!helper) {
            return;
        }

        if (!object || !this.isObjectEffectivelyVisible(object)) {
            helper.visible = false;
            return;
        }

        helper.box.setFromObject(object);
        helper.visible = !helper.box.isEmpty();
    }

    updateSelectionHelpers() {
        this.updateOutlineHelper(this.selectionOutline, this.selectedMeshPart);
        this.updateOutlineHelper(
            this.hoverOutline,
            this.hoveredMeshPart && this.hoveredMeshPart !== this.selectedMeshPart
                ? this.hoveredMeshPart
                : null
        );
    }

    setHoveredMeshPart(mesh) {
        const nextMesh = this.isSelectableMesh(mesh) ? mesh : null;
        if (this.hoveredMeshPart === nextMesh) {
            return;
        }

        this.hoveredMeshPart = nextMesh;
        this.renderer.domElement.style.cursor = nextMesh ? 'pointer' : 'default';
        this.updateSelectionHelpers();
        this.requestRender();
    }

    clearHoverState() {
        this.setHoveredMeshPart(null);
    }

    updateTransformButtons() {
        const translateBtn = this.shadowRoot.querySelector('#translateBtn');
        const rotateBtn = this.shadowRoot.querySelector('#rotateBtn');

        if (!translateBtn || !rotateBtn) {
            return;
        }

        translateBtn.classList.toggle('active', this.state.transformMode === 'translate');
        rotateBtn.classList.toggle('active', this.state.transformMode === 'rotate');
    }

    updateCameraActionButtons() {
        const fitModelBtn = this.shadowRoot.querySelector('#fitModelBtn');
        const frameSelectedBtn = this.shadowRoot.querySelector('#frameSelectedBtn');

        if (fitModelBtn) {
            fitModelBtn.disabled = !this.model;
        }

        if (frameSelectedBtn) {
            frameSelectedBtn.disabled = !this.selectedMeshPart;
        }
    }

    refreshUiFromState(options = {}) {
        const { syncTextureUi = true } = options;

        this.updateControlPanel();
        this.updateLightsButtonUI();
        this.updateViewModeButtons();
        this.updateEnvButtons();
        this.updateEnvironmentControls();
        this.updateAnimationUi();
        this.updateDirectionalLightHelpersVisibility();
        this.updateTransformButtons();
        this.updateCameraActionButtons();
        this.syncWireframeButton();
        this.updateDiscardButtonVisibility();

        if (syncTextureUi) {
            this.syncMaterialEditorControls();
        }
    }

    disposeTextureResource(texture, disposedTextures = new Set()) {
        if (!texture || typeof texture.dispose !== 'function' || disposedTextures.has(texture)) {
            return;
        }

        if (
            texture === this.whiteTexture
            || texture === this.gradTexture
            || texture === this.currentEnvironmentTexture
            || texture === this.scene.environment
            || texture === this.scene.background
        ) {
            return;
        }

        disposedTextures.add(texture);
        texture.dispose();
    }

    disposeMaterialResources(material, disposedTextures = new Set()) {
        if (!material) {
            return;
        }

        if (Array.isArray(material)) {
            material.forEach((entry) => this.disposeMaterialResources(entry, disposedTextures));
            return;
        }

        DISPOSABLE_TEXTURE_KEYS.forEach((key) => {
            this.disposeTextureResource(material[key], disposedTextures);
        });

        if (typeof material.dispose === 'function') {
            material.dispose();
        }
    }

    disposeMaterialSnapshotStore(materialStore) {
        const disposedTextures = new Set();
        Object.values(materialStore).forEach((material) => {
            this.disposeMaterialResources(material, disposedTextures);
        });
    }

    disposeTextureHistory() {
        const disposedTextures = new Set();

        this.textureHistory.forEach((typeHistory) => {
            typeHistory.forEach((historyEntries) => {
                historyEntries.forEach((texture) => {
                    this.disposeTextureResource(texture, disposedTextures);
                });
            });
        });
    }

    disposeEnvironmentTexture() {
        if (!this.currentEnvironmentTexture) {
            return;
        }

        this.currentEnvironmentTexture.dispose();
        this.currentEnvironmentTexture = null;
    }

    resetTransformState() {
        this.state.transformMode = 'none';
        this.transformControls.detach();
        this.transformControls.visible = false;
        this.controls.enabled = true;
        this.updateTransformButtons();
    }

    clearModelResources() {
        this.stopCameraTransition();
        this.disposeCurrentModel();
        this.disposeMaterialSnapshotStore(this.originalMaterials);
        this.disposeMaterialSnapshotStore(this.initialMaterials);
        this.disposeTextureHistory();

        this.originalMaterials = {};
        this.initialMaterials = {};
        this.standardMaterials = [];
        this.meshParts = [];
        this.meshPartTextureInfo = [];
        this.textureHistory = new Map();
        this.modelCenter = null;
        this.modelMaxDim = 0;
        this.modelSize = 1;
        this.canAdjustRoughnessMetalness = false;
        this.currentModelSource = null;
        this.currentModelFileName = null;
        this.modelDefaultCameraState = null;
        this.selectedMaterialIndex = 0;
        this.toonMaterialBackups.clear();
    }

    resetModelSession({ showLoading = true } = {}) {
        this.resetAnimationState();
        this.clearSelectionState();
        this.resetWireframeState();
        this.resetTransformState();
        this.clearModelResources();
        this.resetModelUiState(showLoading);
        this.refreshUiFromState();
    }

    updateLightsButtonUI() {
        const lightsButton = this.shadowRoot.querySelector('#toggleLightsBtn');
        if (!lightsButton) return;

        lightsButton.textContent = this.state.lightsOn ? 'Lights Off' : 'Lights On';
        if (this.state.lightsOn) {
            lightsButton.classList.add('toggled-off');
        } else {
            lightsButton.classList.remove('toggled-off');
        }
    }

    initCollapsibleSection(toggleSelector, contentSelector) {
        const toggle = this.shadowRoot.querySelector(toggleSelector);
        const content = this.shadowRoot.querySelector(contentSelector);
        if (!toggle || !content) {
            return;
        }

        const label = toggle.querySelector('.section-toggle-label');
        const icon = toggle.querySelector('.section-toggle-icon');
        const openLabel = toggle.dataset.openLabel || 'Hide controls';
        const closedLabel = toggle.dataset.closedLabel || 'Show controls';

        const syncState = () => {
            const isExpanded = !content.hidden;
            toggle.setAttribute('aria-expanded', isExpanded ? 'true' : 'false');
            if (label) {
                label.textContent = isExpanded ? openLabel : closedLabel;
            }
            if (icon) {
                icon.textContent = isExpanded ? '-' : '+';
            }
        };

        toggle.addEventListener('click', () => {
            content.hidden = !content.hidden;
            syncState();
        });

        syncState();
    }

    static get observedAttributes() {
        return [
            'src', // source for mesh file
            'auto-rotate', // auto-rotate option
            'angle-per-second', // animation angle per sec
            'camera-orbit',  // init camera orbit
            'hide-control-ui', // hide ui
            'ui', // show ui
            'light-off', // turn off basic light
            'no-pbr', // turn off light, default as diffuse mode
            'view-mode', // 'default', 'diffuse', 'geometry', 'normal'
            'ambient-light', // 0x color, intensity
            'direct-light', // x,y,z,intensity
            'environment',
            'environment-url',
            'environment-intensity',
            'environment-rotation',
            'environment-background',
            'background-color',
            'camera-target',
            'camera-up',
            'exposure',
            'animation',
            'animation-speed',
            'animation-loop',
            'autoplay',
            'selection-mode',
            'performance-mode',
        ];
    }

    attributeChangedCallback(name, oldValue, newValue) {
        if (oldValue === newValue || this.isReflectingAttributes) {
            return;
        }

        if (name === 'src' && newValue) {
            const fileName = newValue.split('/').pop()?.split('?')[0] || newValue;
            void this.loadModel(newValue, fileName).catch(() => {});
            const fileInputContainer = this.shadowRoot.querySelector('#fileInputContainer');
            if (fileInputContainer) {
                fileInputContainer.style.display = 'none';
            }
        } else if (name === 'src' && !newValue) {
            this.discardModel();
        } else if (name === 'auto-rotate') {
            this.autoRotate = newValue !== null;
        } else if (name === 'angle-per-second') {
            this.anglePerSecond = parseFloat(newValue) || 30;
        } else if (name === 'camera-orbit') {
            this.setCameraOrbit(newValue, {
                source: 'attribute',
                syncAttribute: false,
                saveAsDefault: true,
            });
        } else if (name === 'hide-control-ui') {
            const controlsDiv = this.shadowRoot.querySelector('.controls');
            if (newValue !== null) {
                controlsDiv.style.display = 'none';
            } else {
                controlsDiv.style.display = 'block';
            }
        } else if (name === 'ui'){
            const content = this.shadowRoot.querySelector('#panelContent');
            const button = this.shadowRoot.querySelector('#togglePanelBtn');
            content.style.display = 'block';
            button.textContent = '<';
            button.setAttribute('aria-label', 'Collapse controls');
        } else if (name === 'light-off') {
            this.state.lightsOn = !(newValue !== null);
            this.ambientLight.visible = this.state.lightsOn;
            this.directionalLights.forEach(light => {
                light.visible = this.state.lightsOn;
            });
            this.updateDirectionalLightHelpersVisibility(); // Update helper visibility when lights are toggled
        } else if (name === 'no-pbr') {
            this.state.lightsOn = !(newValue !== null);
            this.ambientLight.visible = this.state.lightsOn;
            this.directionalLights.forEach(light => {
                light.visible = this.state.lightsOn;
            });
            const light_controls = this.shadowRoot.querySelector('#lightControls')
            light_controls.style.display = 'none';
            this.updateDirectionalLightHelpersVisibility(); // Update helper visibility when lights are toggled
            this.noPBR = true;
            const light_btn = this.shadowRoot.querySelector('#toggleLightsBtn')
            light_btn.style.display = 'none';
            const diffuse_btn = this.shadowRoot.querySelector('#textureBtn')
            diffuse_btn.style.display = 'none';
            this.state.viewMode = 'diffuse';
            this.renderMode();
        } else if (name === 'view-mode') {
            this.state.viewMode = newValue;
            this.renderMode();
        } else if (name === 'environment' || name === 'environment-url') {
            this.applyEnvironmentAttributes({ source: 'attribute' });
        } else if (name === 'environment-intensity') {
            this.setEnvironmentIntensity(parseFloat(newValue), {
                source: 'attribute',
                syncAttribute: false,
            });
        } else if (name === 'environment-rotation') {
            this.setEnvironmentRotation(parseFloat(newValue), {
                source: 'attribute',
                syncAttribute: false,
            });
        } else if (name === 'environment-background') {
            this.setEnvironmentBackgroundVisible(newValue !== 'false', {
                source: 'attribute',
                syncAttribute: false,
            });
        } else if (name === 'background-color') {
            this.setBackgroundColor(newValue || DEFAULT_BACKGROUND_COLOR, {
                source: 'attribute',
                syncAttribute: false,
            });
        } else if (name === 'camera-target') {
            this.setCameraTarget(newValue || '0 0 0', {
                source: 'attribute',
                syncAttribute: false,
                saveAsDefault: true,
            });
        } else if (name === 'camera-up') {
            this.setCameraUp(newValue || '0 1 0', {
                source: 'attribute',
                syncAttribute: false,
                saveAsDefault: true,
            });
        } else if (name === 'exposure') {
            this.setExposure(parseFloat(newValue), {
                source: 'attribute',
                syncAttribute: false,
            });
        } else if (name === 'animation') {
            this.applyAnimationSelection(newValue || 'none', {
                autoplay: this.hasAttribute('autoplay'),
                source: 'attribute',
                syncAttribute: false,
            });
        } else if (name === 'animation-speed') {
            this.setAnimationSpeed(parseFloat(newValue), {
                source: 'attribute',
                syncAttribute: false,
            });
        } else if (name === 'animation-loop') {
            this.setAnimationLoopMode(newValue || DEFAULT_ANIMATION_LOOP, {
                source: 'attribute',
                syncAttribute: false,
            });
        } else if (name === 'autoplay') {
            this.applyAutoplayState(newValue !== null, {
                source: 'attribute',
            });
        } else if (name === 'selection-mode') {
            this.applySelectionMode(newValue || DEFAULT_SELECTION_MODE, {
                source: 'attribute',
                syncAttribute: false,
            });
        } else if (name === 'performance-mode') {
            this.applyPerformanceMode(newValue || DEFAULT_PERFORMANCE_MODE, {
                source: 'attribute',
                syncAttribute: false,
            });
        }
    }

    renderMode() {
        if (this.state.viewMode === 'diffuse') {
            this.showTexture();
            this.ambientLight.visible = false;
            this.directionalLights.forEach(light => {
                light.visible = false;
            });
            this.updateDirectionalLightHelpersVisibility(); // Hide helpers in diffuse mode
        } else if (this.state.viewMode === 'geometry') {
            this.showMesh();
            this.ambientLight.visible = false;
            this.directionalLights.forEach(light => {
                light.visible = false;
            });
            this.updateDirectionalLightHelpersVisibility(); // Hide helpers in geometry mode
        } else if (this.state.viewMode === 'normal') {
            this.showNormal();
            this.ambientLight.visible = false;
            this.directionalLights.forEach(light => {
                light.visible = false;
            });
            this.updateDirectionalLightHelpersVisibility(); // Hide helpers in normal mode
        } else { // default view mode
            this.updateDirectionalLightHelpersVisibility(); // Ensure helpers visibility based on toggle and lights on/off state
        }
        this.refreshUiFromState({ syncTextureUi: false });
        this.requestRender();
    }


    setAmbientLight(value) {
        const [color, intensity] = value.split(' ');
        const colorValue = parseInt(color, 16);
        const intensityValue = parseFloat(intensity);
        if (!isNaN(colorValue) && !isNaN(intensityValue)) {
            if (this.ambientLight) {
                this.scene.remove(this.ambientLight);
            }
            this.ambientLight = new THREE.AmbientLight(colorValue, intensityValue);
            this.scene.add(this.ambientLight);
        }
    }

    setDirectLight(value) {
        const [x, y, z, intensity] = value.split(' ').map(parseFloat);
        if (!isNaN(x) && !isNaN(y) && !isNaN(z) && !isNaN(intensity)) {
            // Remove old directional lights if any from attribute change.
            this.directionalLights.forEach(light => this.scene.remove(light));
            this.directionalLightHelpers.forEach(helper => this.scene.remove(helper));
            this.directionalLights = [];
            this.directionalLightHelpers = [];

            let newLight = new THREE.DirectionalLight(0xffffff, intensity);
            newLight.position.set(x, y, z);
            this.directionalLights.push(newLight);
            this.scene.add(newLight);

            let helper = new THREE.DirectionalLightHelper(newLight, 1, 0xaaaaaa);
            helper.visible = this.showLightHelpers && this.state.lightsOn; // Helpers visible by default and lights are on
            this.directionalLightHelpers.push(helper);
            this.scene.add(helper);

            this.selectedDirectionalLightIndex = 0;
            this.populateDirectionalLightList();
            this.updateDirectionalLightUIValues();
            this.updateDirectionalLightHelpersVisibility(); // Ensure helper visibility is updated
        }
    }

    setLight(temp_light_state){
        if (!temp_light_state){
            this.ambientLight.visible = temp_light_state;
            this.directionalLights.forEach(light => {
                light.visible = temp_light_state;
            });
        } else{
            this.ambientLight.visible = this.state.lightsOn;
            this.directionalLights.forEach(light => {
                light.visible = this.state.lightsOn;
            });
        }
        this.updateDirectionalLightHelpersVisibility();
    }

    initEventListeners() {
        this.shadowRoot.querySelector('#translateBtn').addEventListener('click', () => {
            this.setTransformMode('translate');
        });

        this.shadowRoot.querySelector('#rotateBtn').addEventListener('click', () => {
            this.setTransformMode('rotate');
        });

        this.shadowRoot.querySelector('#runAnimationBtn').addEventListener('click', () => this.runAnimation());
        this.shadowRoot.querySelector('#pauseAnimationBtn').addEventListener('click', () => this.pauseAnimation());
        this.shadowRoot.querySelector('#animationSelector').addEventListener('change', (event) => {
            this.setAnimation(event.target.value, {
                autoplay: event.target.value !== 'none',
                source: 'ui',
                syncAttribute: true,
            });
        });
        this.shadowRoot.querySelector('#animationSpeed').addEventListener('input', (event) => {
            this.setAnimationSpeed(parseFloat(event.target.value), {
                source: 'ui',
                syncAttribute: true,
            });
        });
        this.shadowRoot.querySelector('#animationLoopMode').addEventListener('change', (event) => {
            this.setAnimationLoopMode(event.target.value, {
                source: 'ui',
                syncAttribute: true,
            });
        });
        this.shadowRoot.querySelector('#animationTimeline').addEventListener('pointerdown', () => {
            this.isScrubbingAnimationTimeline = true;
        });
        this.shadowRoot.querySelector('#animationTimeline').addEventListener('pointerup', () => {
            this.isScrubbingAnimationTimeline = false;
        });
        this.shadowRoot.querySelector('#animationTimeline').addEventListener('pointercancel', () => {
            this.isScrubbingAnimationTimeline = false;
        });
        this.shadowRoot.querySelector('#animationTimeline').addEventListener('change', () => {
            this.isScrubbingAnimationTimeline = false;
        });
        this.shadowRoot.querySelector('#animationTimeline').addEventListener('input', (event) => {
            this.setAnimationTime(parseFloat(event.target.value), {
                source: 'ui',
            });
        });

        this.shadowRoot.querySelector('#toggleLightsBtn').addEventListener('click', () => {
            this.state.lightsOn = !this.state.lightsOn;
            this.ambientLight.visible = this.state.lightsOn;
            this.directionalLights.forEach(light => {
                light.visible = this.state.lightsOn;
            });
            this.updateDirectionalLightHelpersVisibility(); // Update helper visibility when lights are toggled

            this.updateLightsButtonUI();
        });

        this.shadowRoot.querySelector('#textureBtn').addEventListener('click', () => {
            if (this.state.viewMode !== 'diffuse') {
                this.showTexture();
                this.state.viewMode = 'diffuse';
            } else {
                this.setDefaultMat();
                this.state.viewMode = 'default';
                this.setLight(this.state.lightsOn);
            }
            this.updateViewModeButtons();
        });

        // Geometry
        this.shadowRoot.querySelector('#meshBtn').addEventListener('click', () => {
            if (this.state.viewMode !== 'geometry') {
                this.showMesh();
                this.state.viewMode = 'geometry';
                this.setLight(false);
            } else {
                this.setDefaultMat();
                this.state.viewMode = 'default';
                this.setLight(this.state.lightsOn);
            }
            this.updateViewModeButtons();
        });

        // Normal
        this.shadowRoot.querySelector('#normalBtn').addEventListener('click', () => {
            if (this.state.viewMode !== 'normal') {
                this.showNormal();
                this.state.viewMode = 'normal';
                this.setLight(false);
            } else {
                this.setDefaultMat();
                this.state.viewMode = 'default';
                this.setLight(this.state.lightsOn);
            }
            this.updateViewModeButtons();
        });

        this.shadowRoot.querySelector('#wireframeBtn').addEventListener('click', () => this.showWireframe());

        this.shadowRoot.querySelector('#setBgBtn1').addEventListener('click', () => {
            if (this.state.environment !== 'env1') {
                this.setEnvironment('env1', {
                    source: 'ui',
                    syncAttribute: true,
                });
            } else {
                this.clearEnvironment({
                    source: 'ui',
                    syncAttribute: true,
                });
            }
        });

        this.shadowRoot.querySelector('#setBgBtn2').addEventListener('click', () => {
            if (this.state.environment !== 'env2') {
                this.setEnvironment('env2', {
                    source: 'ui',
                    syncAttribute: true,
                });
            } else {
                this.clearEnvironment({
                    source: 'ui',
                    syncAttribute: true,
                });
            }
        });

        this.shadowRoot.querySelector('#setBgBtn3').addEventListener('click', () => {
            if (this.state.environment !== 'env3') {
                this.setEnvironment('env3', {
                    source: 'ui',
                    syncAttribute: true,
                });
            } else {
                this.clearEnvironment({
                    source: 'ui',
                    syncAttribute: true,
                });
            }
        });
        this.shadowRoot.querySelector('#clearEnvBtn').addEventListener('click', () => {
            this.clearEnvironment({
                source: 'ui',
                syncAttribute: true,
            });
        });
        this.shadowRoot.querySelector('#loadEnvironmentUrlBtn').addEventListener('click', () => {
            const url = this.shadowRoot.querySelector('#environmentUrlInput').value.trim();
            if (!url) {
                return;
            }

            void this.setEnvironment(url, {
                source: 'ui',
                syncAttribute: true,
            }).then(() => {
                this.showStatus('Environment HDR loaded.', 'success', 2500);
            }).catch(() => {});
        });
        this.shadowRoot.querySelector('#uploadEnvironmentBtn').addEventListener('click', () => {
            const environmentFileInput = this.shadowRoot.querySelector('#environmentFileInput');
            environmentFileInput.value = '';
            environmentFileInput.click();
        });
        this.shadowRoot.querySelector('#environmentFileInput').addEventListener('change', (event) => {
            const file = event.target.files?.[0];
            if (!file) {
                return;
            }

            const objectUrl = URL.createObjectURL(file);
            void this.loadEnvironmentTexture(objectUrl, {
                environmentId: CUSTOM_ENVIRONMENT_ID,
                source: 'ui',
                syncAttribute: true,
                revokeObjectUrl: true,
            }).then(() => {
                this.showStatus(`Environment loaded from ${file.name}.`, 'success', 2500);
            }).catch(() => {});
        });
        this.shadowRoot.querySelector('#environmentIntensityInput').addEventListener('input', (event) => {
            this.setEnvironmentIntensity(parseFloat(event.target.value), {
                source: 'ui',
                syncAttribute: true,
            });
        });
        this.shadowRoot.querySelector('#environmentRotationInput').addEventListener('input', (event) => {
            this.setEnvironmentRotation(parseFloat(event.target.value), {
                source: 'ui',
                syncAttribute: true,
            });
        });
        this.shadowRoot.querySelector('#environmentBackgroundToggle').addEventListener('change', (event) => {
            this.setEnvironmentBackgroundVisible(event.target.checked, {
                source: 'ui',
                syncAttribute: true,
            });
        });
        this.shadowRoot.querySelector('#exposureInput').addEventListener('input', (event) => {
            this.setExposure(parseFloat(event.target.value), {
                source: 'ui',
                syncAttribute: true,
            });
        });

        this.shadowRoot.querySelector('#posX').addEventListener('input', () => this.updateModelTransform());
        this.shadowRoot.querySelector('#posY').addEventListener('input', () => this.updateModelTransform());
        this.shadowRoot.querySelector('#posZ').addEventListener('input', () => this.updateModelTransform());
        this.shadowRoot.querySelector('#rotX').addEventListener('input', () => this.updateModelTransform());
        this.shadowRoot.querySelector('#rotY').addEventListener('input', () => this.updateModelTransform());
        this.shadowRoot.querySelector('#rotZ').addEventListener('input', () => this.updateModelTransform());

        this.shadowRoot.querySelector('#roughness').disabled = true;
        this.shadowRoot.querySelector('#metalness').disabled = true;

        this.shadowRoot.querySelector('#scale').addEventListener('input', (e) => {
            this.modelSize = parseFloat(e.target.value);
            if (this.model) this.model.scale.set(this.modelSize, this.modelSize, this.modelSize);
        });

        this.shadowRoot.querySelector('#autoRotateBtn').addEventListener('click', () => {
            this.autoRotate = !this.autoRotate;

            const rotateButton = this.shadowRoot.querySelector('#autoRotateBtn');
            // rotateButton.textContent = this.autoRotate ? 'Auto-Rotate Off' : 'Auto-Rotate';
            if (this.autoRotate) {
                rotateButton.classList.add('toggled-off');
            } else {
                rotateButton.classList.remove('toggled-off');
            }
        });

        this.shadowRoot.querySelector('#togglePanelBtn').addEventListener('click', () => {
            const controls = this.shadowRoot.querySelector('.right-ui-panel');
            const content = this.shadowRoot.querySelector('#panelContent');
            const button = this.shadowRoot.querySelector('#togglePanelBtn');
            if (content.style.display === 'none') {
                controls.style.width = 'min(25rem, calc(100vw - 2rem))';
                content.style.display = `block`;
                button.textContent = '<';
                button.setAttribute('aria-label', 'Collapse controls');
            } else {
                button.textContent = '>';
                button.setAttribute('aria-label', 'Expand controls');
                controls.style.width = '4rem';
                content.style.display = `none`;
            }
        });

        this.initCollapsibleSection('#toggleEnvironmentSectionBtn', '#environmentControlsBody');
        this.initCollapsibleSection('#toggleCameraSectionBtn', '#cameraControlsBody');


        this.shadowRoot.querySelector('#toonShadingBtn').addEventListener('click', () => {
            this.toonEnabled = !this.toonEnabled;
            if (this.toonEnabled) {
                this.enableToonShading();
                this.shadowRoot.querySelector('#toonShadingBtn').textContent = 'Toon Shading Off';
            } else {
                this.disableToonShading();
                this.shadowRoot.querySelector('#toonShadingBtn').textContent = 'Toon Shading On';
            }
        });

        const fileInput = this.shadowRoot.querySelector('#fileInput');
        const urlInput = this.shadowRoot.querySelector('#urlInput');
        const loadUrlButton = this.shadowRoot.querySelector('#loadUrlButton');
        const fileInputContainer = this.shadowRoot.querySelector('#fileInputContainer');

        fileInput.addEventListener('change', (e) => {
            const file = e.target.files[0];
            if (file) {
                void this.loadModelFromFile(file)
                    .then(() => {
                        this.showStatus(`Loaded ${file.name}.`, 'success', 2500);
                    })
                    .catch((error) => this.emitViewerError('load-model', error, { fileName: file.name }));
                fileInputContainer.style.display = 'none';
            }
        });

        loadUrlButton.addEventListener('click', () => {
            const url = urlInput.value.trim();
            if (url) {
                void this.loadModelFromUrl(url)
                    .then(() => {
                        this.showStatus('Model URL loaded.', 'success', 2500);
                    })
                    .catch(() => {});
                fileInputContainer.style.display = 'none';
                urlInput.value = ''; 
            }
        });
        
        // with enter button
        urlInput.addEventListener('keypress', (e) => {
            if (e.key === 'Enter') {
                loadUrlButton.click();
            }
        });


        if (this.canAdjustRoughnessMetalness) {
            this.shadowRoot.querySelector('#roughness').addEventListener('input', () => this.updateMaterialProperties());
            this.shadowRoot.querySelector('#metalness').addEventListener('input', () => this.updateMaterialProperties());
        }

        // Light Helpers
        this.shadowRoot.querySelector('#toggleLightHelpersBtn').addEventListener('click', () => {
            this.showLightHelpers = !this.showLightHelpers;
            this.updateDirectionalLightHelpersVisibility();
            this.shadowRoot.querySelector('#toggleLightHelpersBtn').textContent = this.showLightHelpers ? 'Hide Light Helpers' : 'Show Light Helpers';
        });

        // Add Light Button
        this.shadowRoot.querySelector('#addLightBtn').addEventListener('click', () => {
            this.addDirectionalLight();
            this.populateDirectionalLightList(); // Light List UI update
        });

        // Remove Light Button
        this.shadowRoot.querySelector('#removeLightBtn').addEventListener('click', () => {
            this.removeDirectionalLight();
            this.populateDirectionalLightList();
        });

        // Directional Light List
        this.shadowRoot.querySelector('#directionalLightList').addEventListener('change', (event) => {
            this.selectedDirectionalLightIndex = parseInt(event.target.value);
            this.updateDirectionalLightUIValues();
        });


        this.shadowRoot.querySelector('#bgColorPicker').addEventListener('input', (event) => {
            this.setBackgroundColor(event.target.value, {
                source: 'ui',
                syncAttribute: true,
            });
        });

        this.shadowRoot.querySelector('#screenshotBtn').addEventListener('click', () => {
            this.takeScreenshotToClipboard();
        });
        this.shadowRoot.querySelector('#downloadScreenshotBtn').addEventListener('click', () => {
            this.captureScreenshot({
                download: true,
                filename: this.createExportFileName('screenshot', 'png'),
            }).then(() => {
                this.showStatus('Screenshot download started.', 'success', 2500);
            }).catch((error) => {
                this.emitViewerError('download-screenshot', error);
            });
        });

        this.shadowRoot.querySelector('#discardModelBtn').addEventListener('click', () => {
            this.discardModel();
        });

        this.shadowRoot.querySelector('#toggleGridBtn').addEventListener('click', () => {
            this.toggleGrid();
        });

        this.shadowRoot.querySelector('#ambientColorPicker').addEventListener('input', (event) => this.updateAmbientLightColor(event.target.value));
        this.shadowRoot.querySelector('#ambientIntensity').addEventListener('input', (event) => this.updateAmbientLightIntensity(parseFloat(event.target.value)));


        this.shadowRoot.querySelector('#directColorPicker').addEventListener('input', (event) => {
            if (this.directionalLights.length > 0 && this.directionalLights[this.selectedDirectionalLightIndex]) {
                this.updateDirectLightColor(event.target.value, this.selectedDirectionalLightIndex);
            }
        });
        this.shadowRoot.querySelector('#directPosX').addEventListener('input', (event) => {
            if (this.directionalLights.length > 0 && this.directionalLights[this.selectedDirectionalLightIndex]) {
                this.updateDirectLightPosition(parseFloat(event.target.value), null, null, this.selectedDirectionalLightIndex);
            }
        });
        this.shadowRoot.querySelector('#directPosY').addEventListener('input', (event) => {
            if (this.directionalLights.length > 0 && this.directionalLights[this.selectedDirectionalLightIndex]) {
                this.updateDirectLightPosition(null, parseFloat(event.target.value), null, this.selectedDirectionalLightIndex);
            }
        });
        this.shadowRoot.querySelector('#directPosZ').addEventListener('input', (event) => {
            if (this.directionalLights.length > 0 && this.directionalLights[this.selectedDirectionalLightIndex]) {
                this.updateDirectLightPosition(null, null, parseFloat(event.target.value), this.selectedDirectionalLightIndex);
            }
        });
        this.shadowRoot.querySelector('#directIntensity').addEventListener('input', (event) => {
            if (this.directionalLights.length > 0 && this.directionalLights[this.selectedDirectionalLightIndex]) {
                this.updateDirectLightIntensity(parseFloat(event.target.value), this.selectedDirectionalLightIndex);
            }
        });

        this.shadowRoot.querySelector('#cameraFov').addEventListener('input', (event) => this.updateCameraFov(parseFloat(event.target.value)));
        this.shadowRoot.querySelector('#cameraNear').addEventListener('input', (event) => this.updateCameraNear(parseFloat(event.target.value)));
        this.shadowRoot.querySelector('#cameraFar').addEventListener('input', (event) => this.updateCameraFar(parseFloat(event.target.value)));
        this.shadowRoot.querySelector('#resetViewBtn').addEventListener('click', () => {
            this.resetView({
                source: 'ui',
                syncAttribute: true,
            });
        });
        this.shadowRoot.querySelector('#fitModelBtn').addEventListener('click', () => {
            this.fitCameraToModel({
                source: 'ui',
                syncAttribute: true,
            });
        });
        this.shadowRoot.querySelector('#frameSelectedBtn').addEventListener('click', () => {
            this.frameSelected({
                source: 'ui',
                syncAttribute: true,
            });
        });
        this.renderer.domElement.addEventListener('pointerdown', (event) => this.handleCanvasPointerDown(event));
        this.renderer.domElement.addEventListener('pointermove', (event) => this.handleCanvasPointerMove(event));
        this.renderer.domElement.addEventListener('pointerleave', () => this.handleCanvasPointerLeave());
        this.renderer.domElement.addEventListener('click', (event) => this.handleCanvasClick(event));
        this.renderer.domElement.addEventListener('dblclick', (event) => this.handleCanvasDoubleClick(event));
        this.shadowRoot.addEventListener('keydown', (event) => this.handleComponentKeyDown(event));
        this.shadowRoot.querySelector('#copyStateBtn').addEventListener('click', async () => {
            const serialized = JSON.stringify(this.exportState(), null, 2);
            this.shadowRoot.querySelector('#stateConfigInput').value = serialized;
            try {
                if (navigator.clipboard?.writeText) {
                    await navigator.clipboard.writeText(serialized);
                    this.showStatus('Viewer state copied to clipboard.', 'success', 2500);
                } else {
                    this.showStatus('Clipboard unavailable. State JSON is in the text box.', 'info', 3000);
                }
            } catch (error) {
                this.showStatus('Clipboard copy failed. State JSON is in the text box.', 'info', 3000);
            }
        });
        this.shadowRoot.querySelector('#applyStateBtn').addEventListener('click', async () => {
            const rawState = this.shadowRoot.querySelector('#stateConfigInput').value.trim();
            if (!rawState) {
                this.showStatus('Paste a state JSON payload first.', 'error', 3000);
                return;
            }

            try {
                const parsedState = JSON.parse(rawState);
                await this.importState(parsedState);
                this.showStatus('Viewer state imported.', 'success', 2500);
            } catch (error) {
                this.emitViewerError('import-state', error);
            }
        });
        this.shadowRoot.querySelector('#performanceModeSelect').addEventListener('change', (event) => {
            this.applyPerformanceMode(event.target.value, {
                source: 'ui',
                syncAttribute: true,
            });
        });

        this.recordBtn = this.shadowRoot.querySelector('#recordBtn');
        this.stopBtn = this.shadowRoot.querySelector('#stopBtn');
        this.videoModal = this.shadowRoot.querySelector('#videoModal');
        this.videoPreview = this.shadowRoot.querySelector('#videoPreview');
        this.downloadBtn = this.shadowRoot.querySelector('#downloadBtn');
        this.closeModalBtn = this.shadowRoot.querySelector('#closeModalBtn');
        this.quickRecordBtn = this.shadowRoot.querySelector('#quickRecordBtn');
        this.closeModal(); // Ensure modal is hidden initially

        if (this.recordBtn) this.recordBtn.addEventListener('click', this.startRecording); // No ()
        if (this.stopBtn) this.stopBtn.addEventListener('click', this.stopRecording);     // No ()
        if (this.downloadBtn) this.downloadBtn.addEventListener('click', this.downloadVideo); // No ()
        if (this.closeModalBtn) this.closeModalBtn.addEventListener('click', this.closeModal);   // No ()
        if (this.quickRecordBtn) this.quickRecordBtn.addEventListener('click', () => this.startQuickTurntableRecording());

        if (this.videoModal) {
            this.videoModal.addEventListener('click', (event) => {
                if (event.target === this.videoModal) {
                    this.closeModal();
                }
            });
        }

        const canvasContainer = this.shadowRoot.querySelector('#canvas-container');
        canvasContainer.addEventListener('dragenter', this.handleDragEnter);
        canvasContainer.addEventListener('dragover', this.handleDragOver);
        canvasContainer.addEventListener('dragleave', this.handleDragLeave);
        canvasContainer.addEventListener('drop', this.handleDrop);
    }

    // --- Video Recording Functions ---

    updateRecordingStatus(message) {
        const statusElement = this.shadowRoot.querySelector('#recordingStatus');
        if (statusElement) {
            statusElement.textContent = message;
        }
    }

    clearQuickRecordingTimers() {
        if (this.recordingProgressTimer) {
            clearInterval(this.recordingProgressTimer);
            this.recordingProgressTimer = null;
        }
        if (this.quickRecordingTimeout) {
            clearTimeout(this.quickRecordingTimeout);
            this.quickRecordingTimeout = null;
        }
        this.quickRecordingStartedAt = 0;
        this.quickRecordingPreviousAutoRotate = null;
    }

    startQuickTurntableRecording() {
        if (!this.model) {
            this.showStatus('Load a model before starting a turntable recording.', 'error', 3000);
            return;
        }

        if (this.mediaRecorder?.state === 'recording') {
            this.showStatus('A recording is already in progress.', 'info', 2500);
            return;
        }

        const durationInput = this.shadowRoot.querySelector('#recordDurationInput');
        const duration = THREE.MathUtils.clamp(parseInt(durationInput?.value || DEFAULT_RECORDING_DURATION, 10) || DEFAULT_RECORDING_DURATION, 1, 30);
        this.quickRecordingDuration = duration;
        const previousAutoRotate = this.autoRotate;
        this.quickRecordingPreviousAutoRotate = previousAutoRotate;
        this.autoRotate = true;
        this.shadowRoot.querySelector('#autoRotateBtn').classList.add('toggled-off');

        this.startRecording();
        if (!this.mediaRecorder || this.mediaRecorder.state !== 'recording') {
            this.autoRotate = previousAutoRotate;
            return;
        }

        this.quickRecordingStartedAt = performance.now();
        this.updateRecordingStatus(`Turntable recording... ${duration.toFixed(0)}s`);
        this.recordingProgressTimer = window.setInterval(() => {
            const elapsedSeconds = (performance.now() - this.quickRecordingStartedAt) / 1000;
            const remainingSeconds = Math.max(0, duration - elapsedSeconds);
            this.updateRecordingStatus(`Turntable recording... ${remainingSeconds.toFixed(1)}s left`);
        }, 100);
        this.quickRecordingTimeout = window.setTimeout(() => {
            this.stopRecording();
            this.updateRecordingStatus('Turntable recording complete.');
        }, duration * 1000);
    }

    startRecording() {
        if (!this.renderer) {
            this.showStatus('Renderer not ready.', 'error', 3000);
            return;
        }

        if (!this.model){
            this.showStatus('Load a model before recording.', 'error', 3000);
            return;
        }

        if (!window.MediaRecorder) {
            this.showStatus('MediaRecorder API is not supported in this browser.', 'error', 3500);
            return;
        }

        const canvas = this.renderer.domElement;
        if (!canvas.captureStream) {
             this.showStatus('Canvas captureStream is not supported in this browser.', 'error', 3500);
             return;
        }

        console.log("Starting recording...");
        this.recordedChunks = []; // Reset chunks
        this.videoBlob = null; // Reset final blob

        // Get stream from canvas (e.g., at 30fps)
        this.stream = canvas.captureStream(30); // Adjust frame rate as needed

        // --- Choose a MIME type ---
        // Prefer webm with vp9 or vp8, fallback to default
        const options = { mimeType: 'video/webm;codecs=vp9' };
        if (!MediaRecorder.isTypeSupported(options.mimeType)) {
            console.warn(`${options.mimeType} not supported, trying vp8`);
            options.mimeType = 'video/webm;codecs=vp8';
            if (!MediaRecorder.isTypeSupported(options.mimeType)) {
                 console.warn(`${options.mimeType} not supported, trying default`);
                 options.mimeType = 'video/webm'; // Or even '' to let browser decide
                 if (!MediaRecorder.isTypeSupported(options.mimeType)) {
                     console.error("No suitable video/webm MIME type supported");
                      // Clean up stream if necessary
                      this.stream.getTracks().forEach(track => track.stop());
                      this.stream = null;
                     this.showStatus('Could not find a supported video format for recording.', 'error', 4000);
                     return;
                 }
            }
        }
        console.log("Using MIME type:", options.mimeType);

        try {
            this.mediaRecorder = new MediaRecorder(this.stream, options);

            this.mediaRecorder.ondataavailable = (event) => {
                if (event.data.size > 0) {
                    this.recordedChunks.push(event.data);
                    // console.log("Received data chunk:", event.data.size);
                }
            };

            this.mediaRecorder.onstop = () => {
                console.log("Recording stopped. Processing chunks...");
                if (this.recordedChunks.length === 0) {
                    console.warn("No data recorded.");
                     this.showStatus('Recording failed: no video data captured.', 'error', 3500);
                     // No need to show modal if nothing was recorded
                    this.closeModal(); // Ensure it's hidden
                    return;
                }
                // Combine chunks into a single Blob
                this.videoBlob = new Blob(this.recordedChunks, {
                    type: options.mimeType // Use the determined MIME type
                });
                console.log("Video blob created:", this.videoBlob);

                // Create object URL for preview
                const videoUrl = URL.createObjectURL(this.videoBlob);

                // Set preview source and show modal
                this.videoPreview.src = videoUrl;
                // this.videoPreview.load(); // Usually not needed with createObjectURL
                this.videoModal.style.display = 'flex'; // Show modal
                this.showStatus('Recording ready for preview/download.', 'success', 3000);
            };

             this.mediaRecorder.onerror = (event) => {
                 console.error("MediaRecorder error:", event.error);
                 this.showStatus(`Recording failed: ${event.error.name} - ${event.error.message}`, 'error', 5000);
                 this.stopRecording(); // Attempt cleanup
             };

            // Start recording
            this.mediaRecorder.start();

            // Update UI
            this.recordBtn.style.display = 'none';
            this.stopBtn.style.display = 'inline-block'; // Or 'block'
            this.updateRecordingStatus('Recording in progress...');

        } catch (err) {
            console.error("Failed to create MediaRecorder:", err);
            this.showStatus('Failed to initialize video recorder.', 'error', 3500);
             // Clean up stream if necessary
             if (this.stream) {
                 this.stream.getTracks().forEach(track => track.stop());
                 this.stream = null;
             }
        }
    }

    stopRecording() {
        console.log("Stopping recording...");
        const previousAutoRotate = this.quickRecordingPreviousAutoRotate;
        this.clearQuickRecordingTimers();
        if (this.mediaRecorder && this.mediaRecorder.state === "recording") {
            this.mediaRecorder.stop(); // This triggers the 'onstop' event handler
            // Stop the stream tracks *after* recorder has fully stopped (in onstop is safer, but here is common)
            if (this.stream) {
                this.stream.getTracks().forEach(track => track.stop());
                this.stream = null; // Release the stream
            }
        }

        // Update UI immediately
        if (this.stopBtn) {
            this.stopBtn.style.display = 'none';
        }
        if (this.recordBtn) {
            this.recordBtn.style.display = 'inline-block'; // Or 'block'
        }
        if (previousAutoRotate !== null) {
            this.autoRotate = previousAutoRotate;
            this.shadowRoot.querySelector('#autoRotateBtn').classList.toggle('toggled-off', this.autoRotate);
        }
        this.updateRecordingStatus('Idle');
    }

    downloadVideo() {
        if (!this.videoBlob) {
            console.error("No video blob available to download.");
            alert("No recording available to download.");
            return;
        }

        // Create a temporary URL for the blob
        const url = URL.createObjectURL(this.videoBlob);

        // Create a temporary anchor element
        const a = document.createElement('a');
        a.style.display = 'none';
        a.href = url;
        // Suggest a filename (e.g., recording.webm)
        const extension = this.videoBlob.type.split('/')[1].split(';')[0]; // Get 'webm' etc.
        a.download = this.createExportFileName('recording', extension);

        // Append to body, click, and remove
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);

        // Revoke the object URL to free up memory
        URL.revokeObjectURL(url);

        console.log("Download initiated.");
        this.showStatus('Recording download started.', 'success', 2500);
        // Optional: Close modal after download starts
        // this.closeModal();
    }

    closeModal() {
        console.log("Closing modal.");
        this.videoModal.style.display = 'none';
        // Clean up video preview source to release blob memory sooner
        if (this.videoPreview.src) {
             URL.revokeObjectURL(this.videoPreview.src); // Revoke if it's an object URL
             this.videoPreview.src = ''; // Clear src
             this.videoPreview.removeAttribute('src'); // Remove attribute
             this.videoPreview.load(); // Ask video element to release file
        }
        // Reset state if needed (optional, depends on desired flow)
        // this.videoBlob = null;
        // this.recordedChunks = [];
    }

    setTransformMode(mode) {
        if (this.model){
            if (this.state.transformMode === mode) {
                this.resetTransformState();
            } else {
                this.state.transformMode = mode;
                this.transformControls.setMode(mode);
                this.transformControls.attach(this.model);
                this.transformControls.visible = true;
                this.controls.enabled = false;
            }

            this.updateTransformButtons();
            this.requestRender();
        }
    }

    initGridButton() {
        this.shadowRoot.querySelector('#toggleGridBtn').textContent = this.gridHelper.visible ? 'Hide Grid' : 'Show Grid';
    }

    toggleGrid() {
        this.gridHelper.visible = !this.gridHelper.visible;
        this.shadowRoot.querySelector('#toggleGridBtn').textContent = this.gridHelper.visible ? 'Hide Grid' : 'Show Grid';
        if (this.gridHelper.visible) {
            this.shadowRoot.querySelector('#toggleGridBtn').classList.add('toggled-off');
        } else {
            this.shadowRoot.querySelector('#toggleGridBtn').classList.remove('toggled-off');
        }
        this.requestRender();
    }

    initDiscardButton() {
        const discardButton = this.shadowRoot.querySelector('#discardModelBtn');
        discardButton.style.display = 'none';
    }


    updateCameraFov(fov, options = {}) {
        const { source = 'ui', emitEvent = true } = options;
        this.camera.fov = fov;
        this.camera.updateProjectionMatrix();
        this.shadowRoot.querySelector('#cameraFov').value = this.camera.fov;
        this.requestRender();
        if (emitEvent) {
            this.emitCameraChange(source);
        }
    }

    updateCameraNear(near, options = {}) {
        const { source = 'ui', emitEvent = true } = options;
        this.camera.near = near;
        this.camera.updateProjectionMatrix();
        this.shadowRoot.querySelector('#cameraNear').value = this.camera.near;
        this.requestRender();
        if (emitEvent) {
            this.emitCameraChange(source);
        }
    }

    updateCameraFar(far, options = {}) {
        const { source = 'ui', emitEvent = true } = options;
        this.camera.far = far;
        this.camera.updateProjectionMatrix();
        this.shadowRoot.querySelector('#cameraFar').value = this.camera.far;
        this.requestRender();
        if (emitEvent) {
            this.emitCameraChange(source);
        }
    }

    updateDirectionalLightHelpersVisibility() {
        const toggleHelpersBtn = this.shadowRoot.querySelector('#toggleLightHelpersBtn');
        if (!this.state.lightsOn) {
            this.showLightHelpers = false; // force hide if lights are off
            // if (toggleHelpersBtn) toggleHelpersBtn.style.display = 'none'; // Hide the toggle button if lights are off
        } else {
            // if (toggleHelpersBtn) toggleHelpersBtn.style.display = 'inline-block'; // Show toggle button if lights are on
        }

        this.directionalLightHelpers.forEach(helper => {
            helper.visible = this.showLightHelpers && this.state.lightsOn; // Consider both toggle and lights on/off state
        });
        if (toggleHelpersBtn) {
            toggleHelpersBtn.textContent = this.showLightHelpers ? 'Hide Light Helpers' : 'Show Light Helpers';
        }
    }

    addDirectionalLight() {
        const newLight = new THREE.DirectionalLight(0xffffff, 3);
        newLight.position.set(5, 5, 5);
        this.directionalLights.push(newLight);
        this.scene.add(newLight);

        const helper = new THREE.DirectionalLightHelper(newLight, 1, 0xff0f00);
        helper.visible = this.showLightHelpers && this.state.lightsOn; // Helpers visible by default and lights are on
        this.scene.add(helper);
        this.directionalLightHelpers.push(helper);

        this.selectedDirectionalLightIndex = this.directionalLights.length - 1;
        this.updateDirectionalLightUIValues();
        this.updateDirectionalLightHelpersVisibility(); // Update helper visibility when a new light is added
    }

    removeDirectionalLight() {
        if (this.directionalLights.length === 0) {
            return;
        }

        const lightToRemove = this.directionalLights[this.selectedDirectionalLightIndex];
        const helperToRemove = this.directionalLightHelpers[this.selectedDirectionalLightIndex];

        this.scene.remove(lightToRemove);
        this.scene.remove(helperToRemove);

        this.directionalLights.splice(this.selectedDirectionalLightIndex, 1);
        this.directionalLightHelpers.splice(this.selectedDirectionalLightIndex, 1);

        this.selectedDirectionalLightIndex = Math.max(0, this.selectedDirectionalLightIndex - 1);
        this.updateDirectionalLightUIValues();
        this.populateDirectionalLightList(); // Update the list after removal
        this.updateDirectionalLightHelpersVisibility();
    }

    populateDirectionalLightList() {
        const lightList = this.shadowRoot.querySelector('#directionalLightList');
        lightList.innerHTML = '';

        if (this.directionalLights.length === 0) {
            return;
        }

        this.directionalLights.forEach((light, index) => {
            const option = document.createElement('option');
            option.value = index;
            option.textContent = `Light ${index + 1}`;
            lightList.appendChild(option);
        });

        lightList.value = this.selectedDirectionalLightIndex;
    }

    updateDirectionalLightUIValues() {
        if (this.directionalLights.length === 0) return;
        const currentLight = this.directionalLights[this.selectedDirectionalLightIndex];

        this.shadowRoot.querySelector('#directColorPicker').value = `#${currentLight.color.getHexString()}`;
        this.shadowRoot.querySelector('#directPosX').value = currentLight.position.x;
        this.shadowRoot.querySelector('#directPosY').value = currentLight.position.y;
        this.shadowRoot.querySelector('#directPosZ').value = currentLight.position.z;
        this.shadowRoot.querySelector('#directIntensity').value = currentLight.intensity;
    }


    updateAmbientLightColor(color) {
        this.ambientLight.color.set(color);
        this.shadowRoot.querySelector('#ambientColorPicker').value = color;
    }

    updateAmbientLightIntensity(intensity) {
        this.ambientLight.intensity = intensity;
        this.shadowRoot.querySelector('#ambientIntensity').value = this.ambientLight.intensity;
    }

    updateDirectLightColor(color, lightIndex) {
        this.directionalLights[lightIndex].color.set(color);
        this.directionalLightHelpers[lightIndex].update(); // Helper update
        this.shadowRoot.querySelector('#directColorPicker').value = color;
    }

    updateDirectLightPosition(x = null, y = null, z = null, lightIndex) {
        const currentLight = this.directionalLights[lightIndex];
        if (!currentLight) return;

        if (x !== null) currentLight.position.x = x;
        if (y !== null) currentLight.position.y = y;
        if (z !== null) currentLight.position.z = z;

        this.shadowRoot.querySelector('#directPosX').value = currentLight.position.x;
        this.shadowRoot.querySelector('#directPosY').value = currentLight.position.y;
        this.shadowRoot.querySelector('#directPosZ').value = currentLight.position.z;

        // Helper update
        if (this.directionalLightHelpers[lightIndex]) {
            this.directionalLightHelpers[lightIndex].update();
        }
    }


    updateDirectLightIntensity(intensity, lightIndex) {
        this.directionalLights[lightIndex].intensity = intensity;
        this.directionalLightHelpers[lightIndex].update(); // Helper update
        this.shadowRoot.querySelector('#directIntensity').value = intensity;
    }


    setBackgroundColor(color, options = {}) {
        const {
            source = 'api',
            syncAttribute = false,
            emitEvent = true,
        } = options;

        this.state.backgroundColor = color || DEFAULT_BACKGROUND_COLOR;
        this.renderer.setClearColor(this.state.backgroundColor, 1);
        this.shadowRoot.querySelector('#bgColorPicker').value = this.state.backgroundColor;
        this.requestRender();

        if (syncAttribute) {
            this.reflectAttribute('background-color', this.state.backgroundColor);
        }

        if (emitEvent) {
            this.emitEnvironmentChange(source, 'background-color');
        }
    }

    takeScreenshotToClipboard() {
        this.captureScreenshot({
            toClipboard: true,
            downloadFallback: true,
            filename: this.createExportFileName('screenshot', 'png'),
        })
            .then((result) => {
                this.showStatus(
                    result?.downloaded
                        ? 'Clipboard unavailable. Saved a PNG download instead.'
                        : 'Screenshot copied to clipboard.',
                    'success',
                    3200
                );
            })
            .catch((err) => {
                console.error('Failed to copy to clipboard:', err);
                this.emitViewerError('capture-screenshot', err);
            });
    }

    captureScreenshot(options = {}) {
        const {
            type = 'image/png',
            quality = 1,
            toClipboard = false,
            download = false,
            downloadFallback = false,
            transparentBackground = false,
            filename = this.createExportFileName('screenshot', 'png'),
        } = options;

        return new Promise((resolve, reject) => {
            const canvas = this.renderer?.domElement;
            if (!canvas) {
                reject(new Error('Renderer not ready.'));
                return;
            }

            const previousBackground = this.scene.background;
            const previousAlpha = this.renderer.getClearAlpha ? this.renderer.getClearAlpha() : 1;
            if (transparentBackground) {
                this.scene.background = null;
                if (this.renderer.setClearAlpha) {
                    this.renderer.setClearAlpha(0);
                }
                this.requestRender();
            }

            canvas.toBlob(async (blob) => {
                if (transparentBackground) {
                    this.scene.background = previousBackground;
                    if (this.renderer.setClearAlpha) {
                        this.renderer.setClearAlpha(previousAlpha);
                    }
                    this.requestRender();
                }

                if (!blob) {
                    reject(new Error('Failed to create blob from canvas.'));
                    return;
                }

                try {
                    const shouldDownload = download || (toClipboard && downloadFallback && (!navigator.clipboard || typeof ClipboardItem === 'undefined'));
                    if (toClipboard) {
                        if (!navigator.clipboard || typeof ClipboardItem === 'undefined') {
                            if (!downloadFallback) {
                                reject(new Error('Clipboard API not supported.'));
                                return;
                            }
                        } else {
                            await navigator.clipboard.write([
                                new ClipboardItem({
                                    [blob.type]: blob
                                })
                            ]);
                        }
                    }

                    if (shouldDownload) {
                        this.downloadBlob(blob, filename);
                    }
                } catch (error) {
                    if (downloadFallback) {
                        this.downloadBlob(blob, filename);
                        resolve({
                            blob,
                            downloaded: true,
                        });
                        return;
                    }
                    reject(error);
                    return;
                }

                resolve({
                    blob,
                    downloaded: download || (toClipboard && downloadFallback && (!navigator.clipboard || typeof ClipboardItem === 'undefined')),
                });
            }, type, quality);
        });
    }

    setExposure(exposure, options = {}) {
        const {
            source = 'api',
            syncAttribute = false,
            emitEvent = true,
        } = options;
        const normalizedExposure = Number.isFinite(exposure) && exposure > 0 ? exposure : DEFAULT_EXPOSURE;

        this.renderer.toneMappingExposure = normalizedExposure;
        this.state.exposure = normalizedExposure;
        this.requestRender();

        if (syncAttribute) {
            this.reflectAttribute('exposure', normalizedExposure);
        }

        if (emitEvent) {
            this.emitEnvironmentChange(source, 'exposure');
        }
    }

    setEnvironmentIntensity(intensity, options = {}) {
        const {
            source = 'api',
            syncAttribute = false,
            emitEvent = true,
        } = options;
        const normalizedIntensity = Number.isFinite(intensity)
            ? THREE.MathUtils.clamp(intensity, 0, 4)
            : DEFAULT_ENVIRONMENT_INTENSITY;

        this.state.environmentIntensity = normalizedIntensity;
        this.applyEnvironmentPresentation();
        this.applyEnvironmentMaterialSettings();
        this.refreshUiFromState({ syncTextureUi: false });
        this.requestRender();

        if (syncAttribute) {
            this.reflectAttribute('environment-intensity', normalizedIntensity);
        }

        if (emitEvent) {
            this.emitEnvironmentChange(source, 'intensity');
        }

        return true;
    }

    setEnvironmentRotation(rotation, options = {}) {
        const {
            source = 'api',
            syncAttribute = false,
            emitEvent = true,
        } = options;
        const normalizedRotation = Number.isFinite(rotation)
            ? ((rotation % 360) + 360) % 360
            : DEFAULT_ENVIRONMENT_ROTATION;

        this.state.environmentRotation = normalizedRotation;
        this.applyEnvironmentPresentation();
        this.refreshUiFromState({ syncTextureUi: false });
        this.requestRender();

        if (syncAttribute) {
            this.reflectAttribute('environment-rotation', normalizedRotation);
        }

        if (emitEvent) {
            this.emitEnvironmentChange(source, 'rotation');
        }

        return true;
    }

    setEnvironmentBackgroundVisible(visible, options = {}) {
        const {
            source = 'api',
            syncAttribute = false,
            emitEvent = true,
        } = options;

        this.state.environmentBackgroundVisible = visible !== false;
        this.applyEnvironmentPresentation();
        this.refreshUiFromState({ syncTextureUi: false });
        this.requestRender();

        if (syncAttribute) {
            this.reflectAttribute('environment-background', this.state.environmentBackgroundVisible ? 'true' : 'false');
        }

        if (emitEvent) {
            this.emitEnvironmentChange(source, 'background-visible');
        }

        return true;
    }

    applySelectionMode(mode, options = {}) {
        const {
            source = 'api',
            syncAttribute = false,
        } = options;
        const normalizedMode = VALID_SELECTION_MODES.has(mode) ? mode : DEFAULT_SELECTION_MODE;
        this.state.selectionMode = normalizedMode;

        if (normalizedMode === 'none') {
            this.clearSelectionState({
                source,
                emitEvent: true,
            });
            this.clearHoverState();
        }

        this.requestRender();

        if (syncAttribute) {
            this.reflectAttribute('selection-mode', normalizedMode);
        }
    }

    applyPerformanceMode(mode, options = {}) {
        const {
            syncAttribute = false,
        } = options;
        const normalizedMode = VALID_PERFORMANCE_MODES.has(mode) ? mode : DEFAULT_PERFORMANCE_MODE;
        const devicePixelRatio = window.devicePixelRatio || 1;
        const pixelRatio = normalizedMode === 'performance'
            ? 1
            : normalizedMode === 'quality'
                ? devicePixelRatio
                : Math.min(devicePixelRatio, 2);

        this.state.performanceMode = normalizedMode;
        this.renderer.setPixelRatio(pixelRatio);
        this.renderer.shadowMap.enabled = normalizedMode !== 'performance';
        this.resizeRenderer();
        this.requestRender();

        if (syncAttribute) {
            this.reflectAttribute('performance-mode', normalizedMode);
        }
    }


    updateViewModeButtons() {
        // this.shadowRoot.querySelector('#textureBtn').textContent = this.state.viewMode === 'diffuse' ? 'Diffuse Off' : 'Diffuse';
        // this.shadowRoot.querySelector('#meshBtn').textContent = this.state.viewMode === 'geometry' ? 'Geometry Off' : 'Geometry';
        // this.shadowRoot.querySelector('#normalBtn').textContent = this.state.viewMode === 'normal' ? 'Normal Off' : 'Normal';

        if (this.state.viewMode === 'diffuse') {
            this.shadowRoot.querySelector('#textureBtn').classList.add('toggled-off');
        } else {
            this.shadowRoot.querySelector('#textureBtn').classList.remove('toggled-off');
        }

        if (this.state.viewMode === 'geometry') {
            this.shadowRoot.querySelector('#meshBtn').classList.add('toggled-off');
        } else {
            this.shadowRoot.querySelector('#meshBtn').classList.remove('toggled-off');
        }

        if (this.state.viewMode === 'normal') {
            this.shadowRoot.querySelector('#normalBtn').classList.add('toggled-off');
        } else {
            this.shadowRoot.querySelector('#normalBtn').classList.remove('toggled-off');
        }
    }

    updateEnvButtons() {
        // this.shadowRoot.querySelector('#setBgBtn1').textContent = this.state.environment === 'env1' ? 'Env1 Off' : 'Env1';
        // this.shadowRoot.querySelector('#setBgBtn2').textContent = this.state.environment === 'env2' ? 'Env2 Off' : 'Env2';
        // this.shadowRoot.querySelector('#setBgBtn3').textContent = this.state.environment === 'env3' ? 'Env3 Off' : 'Env3';

        if (this.state.environment === 'env1') {
            this.shadowRoot.querySelector('#setBgBtn1').classList.add('toggled-off');
        } else {
            this.shadowRoot.querySelector('#setBgBtn1').classList.remove('toggled-off');
        }

        if (this.state.environment === 'env2') {
            this.shadowRoot.querySelector('#setBgBtn2').classList.add('toggled-off');
        } else {
            this.shadowRoot.querySelector('#setBgBtn2').classList.remove('toggled-off');
        }

        if (this.state.environment === 'env3') {
            this.shadowRoot.querySelector('#setBgBtn3').classList.add('toggled-off');
        } else {
            this.shadowRoot.querySelector('#setBgBtn3').classList.remove('toggled-off');
        }

    }

    updateControlPanel() {
        if (this.model) {
            this.shadowRoot.querySelector('#posX').value = this.model.position.x.toFixed(1);
            this.shadowRoot.querySelector('#posY').value = this.model.position.y.toFixed(1);
            this.shadowRoot.querySelector('#posZ').value = this.model.position.z.toFixed(1);

            this.shadowRoot.querySelector('#rotX').value = THREE.MathUtils.radToDeg(this.model.rotation.x).toFixed(0);
            this.shadowRoot.querySelector('#rotY').value = THREE.MathUtils.radToDeg(this.model.rotation.y).toFixed(0);
            this.shadowRoot.querySelector('#rotZ').value = THREE.MathUtils.radToDeg(this.model.rotation.z).toFixed(0);
        }
    }

    updateModelTransform() {
        if (this.model) {
            const posX = parseFloat(this.shadowRoot.querySelector('#posX').value);
            const posY = parseFloat(this.shadowRoot.querySelector('#posY').value);
            const posZ = parseFloat(this.shadowRoot.querySelector('#posZ').value);
            this.model.position.set(posX, posY, posZ);

            const rotX = THREE.MathUtils.degToRad(parseFloat(this.shadowRoot.querySelector('#rotX').value));
            const rotY = THREE.MathUtils.degToRad(parseFloat(this.shadowRoot.querySelector('#rotY').value));
            const rotZ = THREE.MathUtils.degToRad(parseFloat(this.shadowRoot.querySelector('#rotZ').value));
            this.model.rotation.set(rotX, rotY, rotZ);
            this.requestRender();
        }
    }

    syncWireframeButton() {
        const wireframeBtn = this.shadowRoot.querySelector('#wireframeBtn');
        wireframeBtn.classList.toggle('toggled-off', this.state.isWireframeOn);
    }

    disableWireframe() {
        this.state.isWireframeOn = false;

        if (this.model) {
            this.model.traverse((child) => {
                if (child.isMesh) {
                    getMaterialArray(child.material).forEach((material) => {
                        if (material?.userData?.shader?.uniforms?.uWireframe) {
                            material.userData.shader.uniforms.uWireframe.value = false;
                            material.needsUpdate = true;
                        }
                    });
                }
            });
        }

        this.syncWireframeButton();
    }

    resetWireframeState() {
        this.state.wireframeInitialized = false;
        this.disableWireframe();
    }

    showTexture() {
        this.disableWireframe();
        if (this.model) {
            this.model.traverse((child) => {
                if (child.isMesh) {
                    const originalMaterialEntry = this.getMaterialStoreEntry(child, this.originalMaterials);
                    const nextMaterialEntry = getMaterialArray(originalMaterialEntry).map((materialSnapshot) => {
                        const nextMaterial = new THREE.MeshBasicMaterial({
                            map: materialSnapshot?.map || null,
                            color: materialSnapshot?.color?.clone ? materialSnapshot.color.clone() : undefined,
                            opacity: materialSnapshot?.opacity ?? 1,
                            transparent: materialSnapshot?.transparent ?? false,
                            side: materialSnapshot?.side ?? THREE.FrontSide,
                        });
                        nextMaterial.needsUpdate = true;
                        return nextMaterial;
                    });

                    child.material = Array.isArray(originalMaterialEntry) ? nextMaterialEntry : nextMaterialEntry[0];
                    this.applyWireframeSupportToMaterialEntry(child.material);
                }
            });
            this.requestRender();
        }
    }

    showMesh() {
        this.disableWireframe();
        if (this.model) {
            this.model.traverse((child) => {
                if (child.isMesh) {
                    const originalMaterialEntry = this.getMaterialStoreEntry(child, this.originalMaterials);
                    const nextMaterialEntry = getMaterialArray(originalMaterialEntry).map((originalMaterial) => {
                        const newMaterialProps = {
                            color: 0xffffff,
                            map: null,
                            envMap: this.gradTexture,
                            envMapIntensity: originalMaterial?.envMapIntensity ?? 1.0,
                            roughness: 1,
                            metalness: 1,
                        };

                        if (originalMaterial?.vertexColors) {
                            newMaterialProps.vertexColors = true;
                        }

                        const material = new THREE.MeshStandardMaterial(newMaterialProps);
                        material.needsUpdate = true;
                        return material;
                    });

                    child.material = Array.isArray(originalMaterialEntry) ? nextMaterialEntry : nextMaterialEntry[0];
                    this.applyWireframeSupportToMaterialEntry(child.material);
                }
            });
            this.requestRender();
        }
    }

    /**
     * Adds barycentric coordinates to a BufferGeometry if not already present.
     * @param {THREE.BufferGeometry} geometry - The geometry to modify.
     */
    addBarycentricCoordinates(geometry) {
        if (geometry.attributes.barycentric) return;

        const position = geometry.attributes.position;
        const count = position.count;
        const barycentric = new Float32Array(count * 3);

        if (geometry.index) {
            const index = geometry.index;
            for (let i = 0; i < index.count; i += 3) {
                const a = index.array[i];
                const b = index.array[i + 1];
                const c = index.array[i + 2];
                barycentric[a * 3] = 1; barycentric[a * 3 + 1] = 0; barycentric[a * 3 + 2] = 0;
                barycentric[b * 3] = 0; barycentric[b * 3 + 1] = 1; barycentric[b * 3 + 2] = 0;
                barycentric[c * 3] = 0; barycentric[c * 3 + 1] = 0; barycentric[c * 3 + 2] = 1;
            }
        } else {
            for (let i = 0; i < count; i += 3) {
                barycentric[i * 3] = 1; barycentric[i * 3 + 1] = 0; barycentric[i * 3 + 2] = 0;
                barycentric[(i + 1) * 3] = 0; barycentric[(i + 1) * 3 + 1] = 1; barycentric[(i + 1) * 3 + 2] = 0;
                barycentric[(i + 2) * 3] = 0; barycentric[(i + 2) * 3 + 1] = 0; barycentric[(i + 2) * 3 + 2] = 1;
            }
        }

        geometry.setAttribute('barycentric', new THREE.BufferAttribute(barycentric, 3));
    }

    /**
     * Modifies a material to support wireframe overlay using barycentric coordinates.
     * @param {THREE.Material} material - The material to modify.
     */
    modifyMaterialForWireframe(material) {
        if (!material || material.userData.hasWireframeShader) {
            return;
        }

        material.onBeforeCompile = (shader) => {
            shader.uniforms.uWireframe = { value: this.state.isWireframeOn };

            shader.vertexShader = `
                attribute vec3 barycentric;
                varying vec3 vBarycentric;
                ${shader.vertexShader}
            `.replace(
                '#include <begin_vertex>',
                `
                #include <begin_vertex>
                vBarycentric = barycentric;
                `
            );

            let fragmentShader = `
                uniform bool uWireframe;
                varying vec3 vBarycentric;
                ${shader.fragmentShader}
            `;

            const wireframeInjection = `
                if (uWireframe) {
                    vec3 bary = vBarycentric;
                    vec3 d = fwidth(bary);
                    vec3 a3 = smoothstep(vec3(0.0), d * 0.5, bary);
                    float edgeFactor = min(min(a3.x, a3.y), a3.z);
                    float wireframeAlpha = 1.0 - edgeFactor;
                    vec4 wireframeColor = vec4(0.08, 0.1, 0.14, 0.85);
                    gl_FragColor.rgb = mix(gl_FragColor.rgb, wireframeColor.rgb, wireframeAlpha);
                    gl_FragColor.a = mix(gl_FragColor.a, wireframeColor.a, wireframeAlpha);
                }
            `;

            const fragmentAnchors = [
                '#include <dithering_fragment>',
                '#include <opaque_fragment>',
                '#include <output_fragment>',
            ];

            let injected = false;
            for (const anchor of fragmentAnchors) {
                if (fragmentShader.includes(anchor)) {
                    fragmentShader = fragmentShader.replace(
                        anchor,
                        `
                        ${anchor}
                        ${wireframeInjection}
                        `
                    );
                    injected = true;
                    break;
                }
            }

            if (!injected) {
                fragmentShader = fragmentShader.replace(
                    /}\s*$/,
                    `
                    ${wireframeInjection}
                    }
                    `
                );
            }

            shader.fragmentShader = fragmentShader;

            material.userData.shader = shader;
        };
        material.userData.hasWireframeShader = true;
        material.needsUpdate = true;
    }

    showWireframe() {
        if (!this.model) return;

        this.model.traverse((child) => {
            if (child.isMesh) {
                if (!this.state.wireframeInitialized && child.geometry.index && !child.geometry.userData.isNonIndexed) {
                    child.geometry = child.geometry.toNonIndexed();
                    child.geometry.userData.isNonIndexed = true;
                }
                if (!this.state.wireframeInitialized) {
                    this.addBarycentricCoordinates(child.geometry);
                }
                this.applyWireframeSupportToMaterialEntry(child.material);
                getMaterialArray(child.material).forEach((material) => {
                    material.needsUpdate = true;
                });
            }
        });

        this.state.wireframeInitialized = true;
        this.state.isWireframeOn = !this.state.isWireframeOn;

        this.model.traverse((child) => {
            if (child.isMesh) {
                getMaterialArray(child.material).forEach((material) => {
                    if (material?.userData?.shader?.uniforms?.uWireframe) {
                        material.userData.shader.uniforms.uWireframe.value = this.state.isWireframeOn;
                        material.needsUpdate = true;
                    }
                });
            }
        });

        this.syncWireframeButton();
        this.requestRender();
    }

    showNormal() {
        this.disableWireframe();
        if (this.model) {
            this.model.traverse((child) => {
                if (child.isMesh) {
                    const originalMaterialEntry = this.getMaterialStoreEntry(child, this.originalMaterials);
                    const nextMaterialEntry = getMaterialArray(originalMaterialEntry).map(() => {
                        const material = new THREE.MeshNormalMaterial();
                        material.needsUpdate = true;
                        return material;
                    });

                    child.material = Array.isArray(originalMaterialEntry) ? nextMaterialEntry : nextMaterialEntry[0];
                    this.applyWireframeSupportToMaterialEntry(child.material);
                }
            });
            this.requestRender();
        }
    }

    applyEnvironmentAttributes(options = {}) {
        const environmentUrl = this.getAttribute('environment-url');
        const environment = this.getAttribute('environment');

        if (environmentUrl) {
            return this.setEnvironment(environmentUrl, {
                ...options,
                syncAttribute: false,
            });
        }

        if (environment === CUSTOM_ENVIRONMENT_ID) {
            return this.clearEnvironment({
                ...options,
                syncAttribute: false,
            });
        }

        if (environment && environment !== 'none') {
            return this.setEnvironment(environment, {
                ...options,
                syncAttribute: false,
            });
        }

        return this.clearEnvironment({
            ...options,
            syncAttribute: false,
        });
    }

    loadEnvironmentTexture(url, options = {}) {
        const {
            environmentId = CUSTOM_ENVIRONMENT_ID,
            source = 'api',
            syncAttribute = false,
            emitEvent = true,
            revokeObjectUrl = false,
        } = options;

        this.disableWireframe();
        const loadToken = ++this.environmentLoadToken;
        const rgbeLoader = new RGBELoader();
        this.state.viewMode = 'default';
        this.refreshUiFromState({ syncTextureUi: false });

        return new Promise((resolve, reject) => {
            rgbeLoader.load(url, (texture) => {
            if (loadToken !== this.environmentLoadToken) {
                texture.dispose();
                if (revokeObjectUrl) {
                    URL.revokeObjectURL(url);
                }
                resolve(null);
                return;
            }

            texture.minFilter = THREE.LinearFilter;
            texture.magFilter = THREE.LinearFilter;
            texture.mapping = THREE.EquirectangularReflectionMapping;
            this.disposeEnvironmentTexture();
            this.currentEnvironmentTexture = texture;
            this.applyEnvironmentPresentation();

            if (this.model) {
                this.model.traverse((child) => {
                    if (child.isMesh) {
                        const originalMaterialEntry = this.getMaterialStoreEntry(child, this.originalMaterials);
                        const nextMaterialEntry = getMaterialArray(originalMaterialEntry).map((originalMaterial) => {
                            const isStandardMaterial = originalMaterial instanceof THREE.MeshStandardMaterial;
                            const newMaterialProps = {
                                color: originalMaterial?.color?.clone ? originalMaterial.color.clone() : new THREE.Color(0xffffff),
                                map: originalMaterial?.map || null,
                                emissive: originalMaterial?.emissive?.clone ? originalMaterial.emissive.clone() : new THREE.Color(0x000000),
                                emissiveIntensity: originalMaterial?.emissiveIntensity ?? 1,
                                emissiveMap: originalMaterial?.emissiveMap || null,
                                envMap: texture,
                                envMapIntensity: this.getEffectiveEnvironmentIntensity(originalMaterial?.envMapIntensity ?? 1.0),
                                roughness: isStandardMaterial && originalMaterial.roughness !== undefined ? originalMaterial.roughness : 0.5,
                                metalness: isStandardMaterial && originalMaterial.metalness !== undefined ? originalMaterial.metalness : 0.5,
                                roughnessMap: isStandardMaterial ? originalMaterial.roughnessMap : null,
                                metalnessMap: isStandardMaterial ? originalMaterial.metalnessMap : null,
                                normalMap: originalMaterial?.normalMap || null,
                                normalScale: originalMaterial?.normalScale?.clone ? originalMaterial.normalScale.clone() : new THREE.Vector2(1, 1),
                                aoMap: originalMaterial?.aoMap || null,
                                opacity: originalMaterial?.opacity ?? 1,
                                side: originalMaterial?.side ?? THREE.FrontSide,
                                transparent: originalMaterial?.transparent ?? false,
                                vertexColors: !!originalMaterial?.vertexColors,
                            };

                            const material = new THREE.MeshStandardMaterial(newMaterialProps);
                            if (material.map) {
                                material.map.encoding = THREE.sRGBEncoding;
                            }
                            if (material.emissiveMap) {
                                material.emissiveMap.encoding = THREE.sRGBEncoding;
                            }
                            material.needsUpdate = true;
                            return material;
                        });

                        child.material = Array.isArray(originalMaterialEntry) ? nextMaterialEntry : nextMaterialEntry[0];
                        this.applyWireframeSupportToMaterialEntry(child.material);
                    }
                });
            }
            this.state.environment = environmentId;
            this.state.environmentUrl = environmentId === CUSTOM_ENVIRONMENT_ID && !revokeObjectUrl ? url : null;
            this.setLight(false);
            this.updateEnvButtons();

            if (syncAttribute) {
                this.reflectAttribute('environment', environmentId);
                this.reflectAttribute('environment-url', environmentId === CUSTOM_ENVIRONMENT_ID && !revokeObjectUrl ? url : null);
            }

            this.requestRender();
            if (revokeObjectUrl) {
                URL.revokeObjectURL(url);
            }
            if (emitEvent) {
                this.emitEnvironmentChange(source, 'set');
            }
            resolve(texture);
        }, undefined, (err) => {
            console.error('Skybox err:', err);
            if (revokeObjectUrl) {
                URL.revokeObjectURL(url);
            }
            this.emitViewerError('set-environment', err, { url });
            reject(err);
        });
        });
    }

    setBackground1() {
        return this.setEnvironment('env1');
    }

    setBackground2() {
        return this.setEnvironment('env2');
    }

    setBackground3() {
        return this.setEnvironment('env3');
    }

    clearEnvironment(options = {}) {
        const {
            source = 'api',
            syncAttribute = false,
            emitEvent = true,
        } = options;

        this.environmentLoadToken += 1;
        this.disposeEnvironmentTexture();
        this.applyEnvironmentPresentation();
        this.state.environment = null;
        this.state.environmentUrl = null;
        this.state.viewMode = 'default';
        this.setDefaultMat();
        this.setLight(this.state.lightsOn);
        this.refreshUiFromState({ syncTextureUi: false });

        if (syncAttribute) {
            this.reflectAttribute('environment', null);
            this.reflectAttribute('environment-url', null);
        }

        if (emitEvent) {
            this.emitEnvironmentChange(source, 'clear');
        }

        this.requestRender();
    }

    setDefaultEnv() {
        return this.clearEnvironment({
            source: 'internal',
            syncAttribute: false,
            emitEvent: false,
        });
    }

    setDefaultMat() {
        this.disableWireframe();
        if (this.model) {
            this.model.traverse((child) => {
                if (child.isMesh) {
                    const originalMaterialEntry = this.getMaterialStoreEntry(child, this.originalMaterials);
                    const nextMaterialEntry = getMaterialArray(originalMaterialEntry).map((materialSnapshot) => {
                        const material = this.createDisplayMaterialFromSnapshot(materialSnapshot);
                        material.needsUpdate = true;
                        return material;
                    });

                    child.material = Array.isArray(originalMaterialEntry) ? nextMaterialEntry : nextMaterialEntry[0];
                    this.applyWireframeSupportToMaterialEntry(child.material);
                }
            });
            this.syncMaterialEditorControls();
            this.requestRender();
        }
    }

    setEnvironment(urlOrPreset, options = {}) {
        const normalizedValue = typeof urlOrPreset === 'string' ? urlOrPreset.trim() : '';

        if (!normalizedValue || normalizedValue === 'none') {
            return this.clearEnvironment(options);
        }

        if (ENVIRONMENT_URLS[normalizedValue]) {
            return this.loadEnvironmentTexture(ENVIRONMENT_URLS[normalizedValue], {
                ...options,
                environmentId: normalizedValue,
            });
        }

        return this.loadEnvironmentTexture(normalizedValue, {
            ...options,
            environmentId: CUSTOM_ENVIRONMENT_ID,
        });
    }

    getMeshIntersectionFromEvent(event) {
        if (!this.model || !this.meshParts.length || this.transformControls?.dragging) {
            return null;
        }

        const bounds = this.renderer.domElement.getBoundingClientRect();
        if (!bounds.width || !bounds.height) {
            return null;
        }

        this.pointerNdc.x = ((event.clientX - bounds.left) / bounds.width) * 2 - 1;
        this.pointerNdc.y = -(((event.clientY - bounds.top) / bounds.height) * 2 - 1);
        this.raycaster.setFromCamera(this.pointerNdc, this.camera);

        const intersections = this.raycaster.intersectObjects(
            this.meshParts.filter((mesh) => this.isObjectEffectivelyVisible(mesh)),
            false
        );

        return intersections.find((entry) => this.isSelectableMesh(entry.object)) || null;
    }

    handleCanvasPointerDown(event) {
        if (event.button !== 0) {
            return;
        }

        this.renderer.domElement.focus();
        this.canvasPointerDown = {
            x: event.clientX,
            y: event.clientY,
        };
    }

    handleCanvasPointerMove(event) {
        if (!this.isSelectionChannelEnabled('canvas')) {
            this.clearHoverState();
            return;
        }

        if (this.transformControls?.dragging) {
            this.clearHoverState();
            return;
        }

        if (this.canvasPointerDown) {
            const movedX = Math.abs(event.clientX - this.canvasPointerDown.x);
            const movedY = Math.abs(event.clientY - this.canvasPointerDown.y);
            if (movedX > POINTER_DRAG_THRESHOLD || movedY > POINTER_DRAG_THRESHOLD) {
                this.clearHoverState();
                return;
            }
        }

        const intersection = this.getMeshIntersectionFromEvent(event);
        this.setHoveredMeshPart(intersection?.object || null);
    }

    handleCanvasPointerLeave() {
        this.canvasPointerDown = null;
        this.clearHoverState();
    }

    handleCanvasClick(event) {
        if (event.button !== 0 || !this.isSelectionChannelEnabled('canvas')) {
            return;
        }

        if (this.transformControls?.dragging) {
            this.canvasPointerDown = null;
            return;
        }

        if (this.canvasPointerDown) {
            const movedX = Math.abs(event.clientX - this.canvasPointerDown.x);
            const movedY = Math.abs(event.clientY - this.canvasPointerDown.y);
            this.canvasPointerDown = null;

            if (movedX > POINTER_DRAG_THRESHOLD || movedY > POINTER_DRAG_THRESHOLD) {
                return;
            }
        }

        const intersection = this.getMeshIntersectionFromEvent(event);
        if (!intersection?.object) {
            this.clearSelection({
                source: 'canvas-clear',
            });
            return;
        }

        this.selectMeshPartInSceneGraph(intersection.object, null, {
            source: 'canvas',
            channel: 'canvas',
        });
    }

    handleDragEnter(event) {
        event.preventDefault();
        this.dropHoverDepth += 1;
        this.setDropHintVisible(true);
    }

    handleDragOver(event) {
        event.preventDefault();
        this.setDropHintVisible(true);
    }

    handleDragLeave(event) {
        event.preventDefault();
        this.dropHoverDepth = Math.max(0, this.dropHoverDepth - 1);
        if (this.dropHoverDepth === 0) {
            this.setDropHintVisible(false);
        }
    }

    handleDrop(event) {
        event.preventDefault();
        this.dropHoverDepth = 0;
        this.setDropHintVisible(false);

        const files = Array.from(event.dataTransfer?.files || []);
        const modelFile = files.find((file) => isSupportedModelFileName(file.name));
        if (!modelFile) {
            this.showStatus('Drop a supported model file: .glb, .gltf, .obj, .fbx, or .ply.', 'error', 3500);
            return;
        }

        void this.loadModelFromFile(modelFile)
            .then(() => {
                this.shadowRoot.querySelector('#fileInputContainer').style.display = 'none';
                this.showStatus(`Loaded ${modelFile.name} from drag-and-drop.`, 'success', 2500);
            })
            .catch((error) => {
                this.emitViewerError('load-model', error, { fileName: modelFile.name });
            });
    }

    handleComponentKeyDown(event) {
        if (isTextEntryElement(event.target)) {
            return;
        }

        if (event.key === 'Escape') {
            if (!this.selectedMeshPart) {
                return;
            }

            event.preventDefault();
            event.stopPropagation();
            this.clearSelection({
                source: 'keyboard',
            });
            return;
        }

        if (event.code === 'KeyF') {
            event.preventDefault();
            this.fitCameraToModel({
                source: 'keyboard',
                syncAttribute: true,
            });
            return;
        }

        if (event.code === 'KeyR') {
            event.preventDefault();
            this.resetView({
                source: 'keyboard',
                syncAttribute: true,
            });
            return;
        }

        if (event.code === 'Space' && this.animationActions.length > 0) {
            event.preventDefault();
            if (this.state.isAnimationPlaying) {
                this.pauseAnimation();
            } else {
                this.runAnimation();
            }
        }
    }

    handleCanvasDoubleClick(event) {
        if (event.button !== 0) {
            return;
        }

        const intersection = this.getMeshIntersectionFromEvent(event);
        if (!intersection?.object) {
            return;
        }

        this.frameObject(intersection.object, {
            source: 'double-click',
            syncAttribute: true,
        });
    }

    setCameraOrbit(value, options = {}) {
        const {
            source = 'api',
            syncAttribute = false,
            emitEvent = true,
            saveAsDefault = false,
        } = options;
        const position = parseVector3String(value);
        if (!position) {
            return false;
        }

        this.camera.position.copy(position);
        this.camera.lookAt(this.controls.target);
        this.isApplyingCameraState = true;
        try {
            this.controls.update();
            this.requestRender();
        } finally {
            this.isApplyingCameraState = false;
        }

        if (syncAttribute) {
            this.syncCameraAttributes();
        }

        if (saveAsDefault) {
            this.captureCurrentCameraStateAsDefault();
        }

        if (emitEvent) {
            this.emitCameraChange(source);
        }

        return true;
    }

    setCameraTarget(value, options = {}) {
        const {
            source = 'api',
            syncAttribute = false,
            emitEvent = true,
            saveAsDefault = false,
        } = options;
        const target = parseVector3String(value);
        if (!target) {
            return false;
        }

        this.controls.target.copy(target);
        this.camera.lookAt(target);
        this.isApplyingCameraState = true;
        try {
            this.controls.update();
            this.requestRender();
        } finally {
            this.isApplyingCameraState = false;
        }

        if (syncAttribute) {
            this.syncCameraAttributes();
        }

        if (saveAsDefault) {
            this.captureCurrentCameraStateAsDefault();
        }

        if (emitEvent) {
            this.emitCameraChange(source);
        }

        return true;
    }

    setCameraUp(value, options = {}) {
        const {
            source = 'api',
            syncAttribute = false,
            emitEvent = true,
            saveAsDefault = false,
        } = options;
        const up = parseVector3String(value);
        if (!up) {
            return false;
        }

        this.camera.up.copy(up.normalize());
        this.camera.lookAt(this.controls.target);
        this.isApplyingCameraState = true;
        try {
            this.controls.update();
            this.requestRender();
        } finally {
            this.isApplyingCameraState = false;
        }

        if (syncAttribute) {
            this.syncCameraAttributes();
        }

        if (saveAsDefault) {
            this.captureCurrentCameraStateAsDefault();
        }

        if (emitEvent) {
            this.emitCameraChange(source);
        }

        return true;
    }

    frameObject(object, options = {}) {
        const {
            padding = 1.5,
            source = 'api',
            emitEvent = true,
            syncAttribute = false,
            transitionDuration = 260,
        } = options;

        if (!object) {
            return null;
        }

        const box = new THREE.Box3().setFromObject(object);
        if (box.isEmpty()) {
            return null;
        }

        const size = box.getSize(new THREE.Vector3());
        const center = box.getCenter(new THREE.Vector3());
        const maxDim = Math.max(size.x, size.y, size.z, 1);
        const fov = THREE.MathUtils.degToRad(this.camera.fov);
        const distance = ((maxDim / 2) / Math.tan(fov / 2)) * padding;

        const direction = this.camera.position.clone().sub(this.controls.target);
        if (direction.lengthSq() === 0) {
            direction.set(0, 0, 1);
        }

        direction.normalize().multiplyScalar(distance);
        const targetSnapshot = this.cloneCameraState(this.getCameraStateSnapshot());
        targetSnapshot.target = {
            x: center.x,
            y: center.y,
            z: center.z,
        };
        targetSnapshot.position = {
            x: center.x + direction.x,
            y: center.y + direction.y,
            z: center.z + direction.z,
        };
        targetSnapshot.near = Math.max(0.01, maxDim / 100);
        targetSnapshot.far = Math.max(1000, distance * 10);

        this.applyCameraStateSnapshot(targetSnapshot, {
            source,
            emitEvent,
            syncAttribute,
            transitionDuration,
        });

        return {
            center,
            size,
            maxDim,
            distance,
        };
    }

    fitCameraToModel(options = {}) {
        return this.frameObject(this.model, options);
    }

    frameSelected(options = {}) {
        if (!this.selectedMeshPart) {
            return null;
        }

        return this.frameObject(this.selectedMeshPart, options);
    }

    resetView(options = {}) {
        const {
            source = 'api',
            emitEvent = true,
            syncAttribute = false,
            transitionDuration = 260,
        } = options;
        const targetState = this.model && this.modelDefaultCameraState
            ? this.modelDefaultCameraState
            : this.defaultCameraState;

        return this.applyCameraStateSnapshot(targetState, {
            source,
            emitEvent,
            syncAttribute,
            transitionDuration,
        });
    }

    getSelectedMaterialIndex(mesh = this.selectedMeshPart) {
        const materialSlotSelector = this.shadowRoot.querySelector('#materialSlotSelector');
        const parsedIndex = parseInt(materialSlotSelector?.value ?? `${this.selectedMaterialIndex}`, 10);
        const maxIndex = Math.max(0, this.getMaterialSlotCountForMesh(mesh) - 1);

        if (!Number.isFinite(parsedIndex)) {
            return 0;
        }

        return THREE.MathUtils.clamp(parsedIndex, 0, maxIndex);
    }

    getMaterialSelection() {
        const partSelector = this.shadowRoot.querySelector('#texturePartSelector');
        const typeSelector = this.shadowRoot.querySelector('#textureTypeSelector');
        const selectedPartIndex = parseInt(partSelector.value, 10);

        if (isNaN(selectedPartIndex) || selectedPartIndex < 0 || selectedPartIndex >= this.meshParts.length) {
            return null;
        }

        const mesh = this.meshParts[selectedPartIndex];
        const materialIndex = this.getSelectedMaterialIndex(mesh);
        const material = getMaterialEntryAt(mesh.material, materialIndex);
        const storedMaterial = this.getMaterialStoreMaterial(mesh, materialIndex, this.originalMaterials);
        const initialMaterial = this.getMaterialStoreMaterial(mesh, materialIndex, this.initialMaterials);

        if (!mesh || !material || !storedMaterial) {
            return null;
        }

        return {
            mesh,
            material,
            storedMaterial,
            initialMaterial,
            materialIndex,
            mapType: typeSelector.value,
            selectedPartIndex,
        };
    }

    getTextureSelection() {
        return this.getMaterialSelection();
    }

    getTextureForPreview(selection) {
        if (!selection) {
            return null;
        }

        return selection.storedMaterial[selection.mapType]
            || selection.material[selection.mapType]
            || selection.initialMaterial?.[selection.mapType]
            || null;
    }

    setMeshMaterialAt(mesh, materialIndex, nextMaterial) {
        if (!mesh || !nextMaterial) {
            return;
        }

        if (Array.isArray(mesh.material)) {
            mesh.material[materialIndex] = nextMaterial;
        } else {
            mesh.material = nextMaterial;
        }
    }

    createDisplayMaterialFromSnapshot(materialSnapshot) {
        if (this.noPBR) {
            return new THREE.MeshBasicMaterial({
                map: materialSnapshot?.map || null,
                color: materialSnapshot?.color?.clone ? materialSnapshot.color.clone() : undefined,
                opacity: materialSnapshot?.opacity ?? 1,
                transparent: materialSnapshot?.transparent ?? false,
                side: materialSnapshot?.side ?? THREE.FrontSide,
            });
        }

        const material = materialSnapshot.clone();
        if ('envMapIntensity' in material) {
            material.envMapIntensity = this.getEffectiveEnvironmentIntensity(materialSnapshot?.envMapIntensity ?? 1);
        }
        if (this.currentEnvironmentTexture) {
            material.envMap = this.currentEnvironmentTexture;
        }
        return material;
    }

    applyWireframeSupportToMaterialEntry(materialEntry) {
        if (Array.isArray(materialEntry)) {
            materialEntry.forEach((material) => this.modifyMaterialForWireframe(material));
            return;
        }

        this.modifyMaterialForWireframe(materialEntry);
    }

    applyTextureEncoding(texture, mapType) {
        if (!texture) {
            return;
        }

        texture.flipY = false;
        texture.encoding = mapType === 'map' || mapType === 'emissiveMap'
            ? THREE.sRGBEncoding
            : THREE.LinearEncoding;
    }

    normalizeMeshMaterial(mesh) {
        const materials = getMaterialArray(mesh?.material);
        if (!materials.length) {
            return null;
        }

        const normalizedMaterials = materials.map((material) => createStandardMaterialFromMaterial(material));
        normalizedMaterials.forEach((normalizedMaterial) => {
            if (normalizedMaterial.map) {
                normalizedMaterial.map.encoding = THREE.sRGBEncoding;
            }
            if (normalizedMaterial.emissiveMap) {
                normalizedMaterial.emissiveMap.encoding = THREE.sRGBEncoding;
            }
            if (normalizedMaterial.normalMap) {
                normalizedMaterial.normalMap.encoding = THREE.LinearEncoding;
            }
            if (normalizedMaterial.aoMap) {
                normalizedMaterial.aoMap.encoding = THREE.LinearEncoding;
                if (mesh.geometry.attributes.uv && !mesh.geometry.attributes.uv2) {
                    mesh.geometry.setAttribute('uv2', mesh.geometry.attributes.uv);
                }
            }
            normalizedMaterial.needsUpdate = true;
        });

        mesh.material = Array.isArray(mesh.material) ? normalizedMaterials : normalizedMaterials[0];

        return mesh.material;
    }

    resetAnimationState(options = {}) {
        const { emitEvent = true, source = 'reset' } = options;
        const hadAnimationState = this.currentAction || this.animationActions.length > 0 || this.state.animationSelection !== 'none';

        if (this.mixer) {
            this.mixer.removeEventListener('finished', this.handleAnimationMixerFinished);
            this.mixer.stopAllAction();
            this.mixer = null;
        }

        if (this.animationActions) {
            this.animationActions.forEach(action => action.stop());
        }

        if (this.currentAction) {
            this.currentAction.stop();
        }

        this.animationActions = [];
        this.currentAction = null;
        this.state.isAnimationPlaying = false;
        this.state.animationSelection = 'none';
        this.state.animationSpeed = DEFAULT_ANIMATION_SPEED;
        this.state.animationLoopMode = DEFAULT_ANIMATION_LOOP;
        this.isScrubbingAnimationTimeline = false;
        this.populateAnimationSelector();
        this.refreshUiFromState({ syncTextureUi: false });

        if (emitEvent && hadAnimationState) {
            this.emitAnimationChange(source, 'reset');
        }
    }

    clearSelectionState(options = {}) {
        const { emitEvent = true, source = 'reset' } = options;
        const hadSelection = !!this.selectedMeshPart;

        if (this.selectedSceneGraphLabel) {
            this.selectedSceneGraphLabel.classList.remove('selected');
        }

        this.selectedSceneGraphLabel = null;
        this.selectedMeshPart = null;
        this.selectedMeshPartIndex = -1;
        this.selectedMaterialIndex = 0;
        this.syncMaterialEditorControls();
        this.updateCameraActionButtons();
        this.updateSelectionHelpers();
        this.requestRender();

        if (emitEvent && hadSelection) {
            this.emitSelectionChange(source);
        }
    }

    disposeCurrentModel() {
        if (!this.model) {
            return;
        }

        this.scene.remove(this.model);
        const disposedTextures = new Set();

        this.model.traverse((child) => {
            if (!child.isMesh) {
                return;
            }

            if (child.geometry) {
                child.geometry.dispose();
            }

            this.disposeMaterialResources(child.material, disposedTextures);
        });

        this.transformControls.detach();
        this.model = null;
    }

    resetModelUiState(showLoading = true) {
        this.shadowRoot.querySelector('#modelInfo').innerHTML = showLoading
            ? '<strong>[Model Info]</strong> loading...'
            : '<strong>[Model Info]</strong> No model loaded';
        this.shadowRoot.querySelector('#posX').value = 0;
        this.shadowRoot.querySelector('#posY').value = 0;
        this.shadowRoot.querySelector('#posZ').value = 0;
        this.shadowRoot.querySelector('#rotX').value = 0;
        this.shadowRoot.querySelector('#rotY').value = 0;
        this.shadowRoot.querySelector('#rotZ').value = 0;
        this.shadowRoot.querySelector('#scale').value = 1;
        this.shadowRoot.querySelector('#baseColorInput').value = '#ffffff';
        this.shadowRoot.querySelector('#emissiveColorInput').value = '#000000';
        this.shadowRoot.querySelector('#emissiveIntensityInput').value = 1;
        this.shadowRoot.querySelector('#opacityInput').value = 1;
        this.shadowRoot.querySelector('#transparentToggle').checked = false;
        this.shadowRoot.querySelector('#doubleSidedToggle').checked = false;
        this.shadowRoot.querySelector('#roughness').value = 0.5;
        this.shadowRoot.querySelector('#metalness').value = 0.5;
        this.shadowRoot.querySelector('#normalScaleXInput').value = 1;
        this.shadowRoot.querySelector('#normalScaleYInput').value = 1;
        this.shadowRoot.querySelector('#envMapIntensityInput').value = 1;
        this.shadowRoot.querySelector('#environmentUrlInput').value = '';
        this.shadowRoot.querySelector('#environmentIntensityInput').value = DEFAULT_ENVIRONMENT_INTENSITY;
        this.shadowRoot.querySelector('#environmentRotationInput').value = DEFAULT_ENVIRONMENT_ROTATION;
        this.shadowRoot.querySelector('#environmentBackgroundToggle').checked = DEFAULT_ENVIRONMENT_BACKGROUND_VISIBLE;
        this.shadowRoot.querySelector('#exposureInput').value = this.state.exposure;
        this.shadowRoot.querySelector('#uvRepeatXInput').value = 1;
        this.shadowRoot.querySelector('#uvRepeatYInput').value = 1;
        this.shadowRoot.querySelector('#uvOffsetXInput').value = 0;
        this.shadowRoot.querySelector('#uvOffsetYInput').value = 0;
        this.shadowRoot.querySelector('#uvRotationInput').value = 0;
        this.shadowRoot.querySelector('#sceneGraphTree').innerHTML = '';
        this.sceneGraphLabelByMeshUuid.clear();
        this.shadowRoot.querySelector('#texturePartSelector').innerHTML = '';
        this.shadowRoot.querySelector('#materialSlotSelector').innerHTML = '';
        this.shadowRoot.querySelector('#texturePreview').innerHTML = '';
        this.shadowRoot.querySelector('#texturePreview').textContent = 'None';
        Object.assign(this.shadowRoot.querySelector('#texturePreview').style, PREVIEW_TEXT_STYLE);
        this.shadowRoot.querySelector('#textureMetaInfo').textContent = 'No texture selected';

        const historySelector = this.shadowRoot.querySelector('#textureHistorySelector');
        if (historySelector) {
            historySelector.innerHTML = '<option value="-1">Current</option>';
        }

        const explodeFieldset = this.shadowRoot.querySelector('#explode-fieldset');
        if (explodeFieldset) {
            explodeFieldset.remove();
        }

        this.shadowRoot.querySelector('#anim_description').style.display = 'none';
        this.updateRecordingStatus('Idle');
        this.setMaterialEditorControlsEnabled(false);
        this.clearHoverState();
    }

    discardModel() {
        if (this.model || Object.keys(this.originalMaterials).length > 0) {
            this.resetModelSession({ showLoading: false });
            this.reflectAttribute('src', null);

            const fileInputContainer = this.shadowRoot.querySelector('#fileInputContainer');
            fileInputContainer.style.display = 'block';

            this.requestRender();
        }
    }

    loadModelFromUrl(url) {
        if (!url) {
            return Promise.reject(new Error('A model URL is required.'));
        }

        const fileName = url.split('/').pop()?.split('?')[0] || 'model';
        this.reflectAttribute('src', url);
        return this.loadModel(url, fileName);
    }

    loadModel(url, fileName) {
        return new Promise((resolve, reject) => {
        const progressBar = this.shadowRoot.querySelector('#loadingProgressBar');
        progressBar.style.display = 'block';
        progressBar.style.width = '0%';
        const shouldRevokeObjectUrl = typeof url === 'string' && url.startsWith('blob:');
        this.resetModelSession({ showLoading: true });
        this.showStatus(`Loading ${fileName}...`, 'info');

        const fileExtension = fileName.split('.').pop().toLowerCase();
        switch (fileExtension) {
            case 'gltf':
            case 'glb':
                this.loader = this.gltfLoader;
                break;
            case 'obj':
                this.loader = this.objLoader;
                break;
            case 'fbx':
                this.loader = this.fbxLoader;
                break;
            case 'ply':
                this.loader = this.plyLoader;
                break;
            default:
                const unsupportedFormatError = new Error(`Unsupported file format: ${fileExtension}`);
                console.error('Unsupported file format:', fileExtension);
                progressBar.style.display = 'none';
                if (shouldRevokeObjectUrl) {
                    URL.revokeObjectURL(url);
                }
                this.emitViewerError('load-model', unsupportedFormatError, { src: url, fileName });
                reject(unsupportedFormatError);
                return;
        }

        this.loader.load(url, (object) => {
            switch (fileExtension) {
                case 'gltf':
                case 'glb':
                    this.model = object.scene;
                    break;
                case 'fbx':
                case 'obj':
                    this.model = object;
                    break;
                case 'ply':
                    // PLYLoader는 BufferGeometry를 반환.
                    const geometry = object;
                    // 정점 노멀이 없는 경우 계산.
                    if (!geometry.attributes.normal) {
                        geometry.computeVertexNormals();
                    }
                    // 정점 색상 정보가 있는지 확인하여 재질을 설정.
                    const material = new THREE.MeshStandardMaterial({ vertexColors: geometry.hasAttribute('color') });
                    this.model = new THREE.Mesh(geometry, material);
                    break;
            }

            const box = new THREE.Box3().setFromObject(this.model);
            const size = box.getSize(new THREE.Vector3());
            const maxDim = Math.max(size.x, size.y, size.z);
            let scaleFactor = 1;
            
            this.modelMaxDim = maxDim;

            if (maxDim > 0) {
                const targetSize = 10;
                scaleFactor = targetSize / maxDim;
            }

            this.model.scale.set(scaleFactor, scaleFactor, scaleFactor);
            this.modelSize = scaleFactor;
            this.shadowRoot.querySelector('#scale').value = this.modelSize;

            const updatedBox  = new THREE.Box3().setFromObject(this.model);
            const center = updatedBox.getCenter(new THREE.Vector3());
            this.model.position.sub(center);

            if (object.animations && object.animations.length > 0) {
                this.mixer = new THREE.AnimationMixer(this.model);
                this.mixer.addEventListener('finished', this.handleAnimationMixerFinished);
                this.animationActions = object.animations.map(clip => this.mixer.clipAction(clip));
                this.animationActions.forEach((action) => this.configureAnimationAction(action));

                this.currentAction = null;
                this.populateAnimationSelector();
                this.refreshUiFromState({ syncTextureUi: false });
            }

            const updatedsize = updatedBox.getSize(new THREE.Vector3());
            const updatedMaxDim = Math.max(updatedsize.x, updatedsize.y, updatedsize.z);
            const fov = this.camera.fov * (Math.PI / 180);
            let cameraZ = Math.abs(updatedMaxDim / 2 / Math.tan(fov / 2));

            this.gridHelper.position.set(center.x, - (updatedsize.y/2), center.z);

            const cameraOrbit = this.getAttribute('camera-orbit');
            if (cameraOrbit) {
                this.setCameraOrbit(cameraOrbit, {
                    source: 'attribute',
                    emitEvent: false,
                    saveAsDefault: true,
                });
            } else {
                this.camera.position.set(0, 0, cameraZ * 1.5);
                this.controls.target.set(0, 0, 0);
                this.camera.lookAt(this.controls.target);
            }

            const sceneGraphTreeUI = this.shadowRoot.querySelector('#sceneGraphTree');
            sceneGraphTreeUI.innerHTML = ''; // Clear the scene graph tree
            this.sceneGraphLabelByMeshUuid.clear();
            this.generateSceneGraphTree(this.model, sceneGraphTreeUI);

            let vertexCount = 0, faceCount = 0;
            this.standardMaterials = [];
            this.meshParts = [];
            this.meshPartTextureInfo = [];

            const modelBbox = new THREE.Box3().setFromObject(this.model);
            this.modelCenter = modelBbox.getCenter(new THREE.Vector3());


            // Mesh Parts and Textures
            this.model.traverse((child) => {
                if (child.isMesh && child.geometry) {
                    // for no-normal geometry
                    if (!child.geometry.attributes.normal) {
                        child.geometry.computeVertexNormals();
                    }
                    
                    vertexCount += child.geometry.attributes.position.count;
                    faceCount += child.geometry.index ? child.geometry.index.count / 3 : child.geometry.attributes.position.count / 3;

                    child.userData.originalPosition = child.position.clone();
                    this.meshParts.push(child);
                    const materialEntry = this.normalizeMeshMaterial(child);
                    if (!materialEntry) {
                        return;
                    }

                    this.applyWireframeSupportToMaterialEntry(materialEntry);
                    this.originalMaterials[child.uuid] = cloneMaterialEntry(materialEntry);
                    this.initialMaterials[child.uuid] = cloneMaterialEntry(materialEntry);
                    this.standardMaterials.push(...getMaterialArray(materialEntry));
                }
            });

            this.populateTextureMapSelector();
            this.canAdjustRoughnessMetalness = this.standardMaterials.length > 0;

            this.meshParts.forEach((mesh, index) => {
                const firstMaterial = getMaterialEntryAt(mesh.material, 0);
                const partTextureInfo = {
                    meshPartIndex: index,
                    diffuseMap: firstMaterial?.map || null,
                    roughnessMap: firstMaterial?.roughnessMap || null,
                    metalnessMap: firstMaterial?.metalnessMap || null,
                    normalMap: firstMaterial?.normalMap || null,
                    aoMap: firstMaterial?.aoMap || null,
                    emissiveMap: firstMaterial?.emissiveMap || null,
                };
                this.meshPartTextureInfo.push(partTextureInfo);
            });

            this.shadowRoot.querySelector('#modelInfo').innerHTML = `<strong>[Model Info]</strong> Vertices: ${vertexCount}, Faces: ${faceCount}`;
            this.scene.add(this.model);
            this.currentModelSource = shouldRevokeObjectUrl ? null : url;
            this.currentModelFileName = fileName;

            this.createExplodeSlider();

            this.renderMode();
            this.refreshUiFromState();
            this.updateSelectionHelpers();
            if (this.getAttribute('camera-target')) {
                this.setCameraTarget(this.getAttribute('camera-target'), {
                    source: 'attribute',
                    emitEvent: false,
                    saveAsDefault: true,
                });
            }
            if (this.getAttribute('camera-up')) {
                this.setCameraUp(this.getAttribute('camera-up'), {
                    source: 'attribute',
                    emitEvent: false,
                    saveAsDefault: true,
                });
            }
            const animationAttribute = this.getAttribute('animation');
            if (animationAttribute) {
                this.applyAnimationSelection(animationAttribute, {
                    autoplay: this.hasAttribute('autoplay'),
                    source: 'attribute',
                    syncAttribute: false,
                    emitEvent: false,
                });
            } else if (this.hasAttribute('autoplay') && this.animationActions.length > 0) {
                this.applyAnimationSelection('0', {
                    autoplay: true,
                    source: 'attribute',
                    syncAttribute: false,
                    emitEvent: false,
                });
            }
            if (this.hasAttribute('animation-speed')) {
                this.setAnimationSpeed(parseFloat(this.getAttribute('animation-speed')), {
                    source: 'attribute',
                    syncAttribute: false,
                    emitEvent: false,
                });
            }
            if (this.hasAttribute('animation-loop')) {
                this.setAnimationLoopMode(this.getAttribute('animation-loop'), {
                    source: 'attribute',
                    syncAttribute: false,
                    emitEvent: false,
                });
            }
            this.captureCurrentCameraStateAsDefault({ scope: 'model' });
            progressBar.style.display = 'none';
            this.updateDiscardButtonVisibility();
            if (shouldRevokeObjectUrl) {
                URL.revokeObjectURL(url);
            }
            const loadDetail = {
                src: this.currentModelSource,
                fileName,
                format: fileExtension,
                meshCount: this.meshParts.length,
                animationCount: this.animationActions.length,
                vertexCount,
                faceCount,
            };
            this.emitEvent('viewer-load', loadDetail);
            this.showStatus(`Loaded ${fileName}.`, 'success', 2500);
            resolve(loadDetail);
        }, (xhr) => {
            if (xhr.lengthComputable) {
                const percentComplete = xhr.loaded / xhr.total * 100;
                progressBar.style.width = `${percentComplete}%`;
            }
        }, (error) => {
            console.error('Loading Error:', error);
            progressBar.style.display = 'none';
            if (shouldRevokeObjectUrl) {
                URL.revokeObjectURL(url);
            }
            this.emitViewerError('load-model', error, { src: url, fileName });
            reject(error);
        });
        });
    }

    runAnimation() {
        if (!this.currentAction && this.animationActions.length > 0) {
            this.applyAnimationSelection(this.getAttribute('animation') || this.state.animationSelection || '0', {
                autoplay: true,
                source: 'ui',
                syncAttribute: true,
            });
            return;
        }

        if (this.currentAction) {
            const duration = this.getAnimationDuration();
            if (duration > 0 && this.currentAction.time >= duration - 0.001) {
                this.currentAction.reset();
            }
            this.configureAnimationAction(this.currentAction);
            this.currentAction.play();
            this.currentAction.paused = false;
            this.state.isAnimationPlaying = true;
            this.refreshUiFromState({ syncTextureUi: false });
            this.emitAnimationChange('ui', 'play');
        }
    }

    pauseAnimation() {
        if (this.currentAction) {
            this.currentAction.paused = true;
            this.state.isAnimationPlaying = false;
            this.refreshUiFromState({ syncTextureUi: false });
            this.emitAnimationChange('ui', 'pause');
        }
    }

    setAnimation(selection, options = {}) {
        return this.applyAnimationSelection(selection, options);
    }

    playAnimation() {
        this.runAnimation();
    }

    setAnimationSpeed(speed, options = {}) {
        const {
            source = 'api',
            syncAttribute = false,
            emitEvent = true,
        } = options;
        const normalizedSpeed = Number.isFinite(speed)
            ? THREE.MathUtils.clamp(speed, 0.1, 3)
            : DEFAULT_ANIMATION_SPEED;

        this.state.animationSpeed = normalizedSpeed;
        this.animationActions.forEach((action) => {
            action.setEffectiveTimeScale(normalizedSpeed);
        });

        if (syncAttribute) {
            this.reflectAttribute('animation-speed', normalizedSpeed);
        }

        this.refreshUiFromState({ syncTextureUi: false });
        this.requestRender();

        if (emitEvent) {
            this.emitAnimationChange(source, 'speed');
        }

        return true;
    }

    setAnimationLoopMode(loopMode, options = {}) {
        const {
            source = 'api',
            syncAttribute = false,
            emitEvent = true,
        } = options;
        const normalizedLoopMode = VALID_ANIMATION_LOOP_MODES.has(loopMode)
            ? loopMode
            : DEFAULT_ANIMATION_LOOP;

        this.state.animationLoopMode = normalizedLoopMode;
        this.animationActions.forEach((action) => this.configureAnimationAction(action));

        if (syncAttribute) {
            this.reflectAttribute('animation-loop', normalizedLoopMode);
        }

        this.refreshUiFromState({ syncTextureUi: false });
        this.requestRender();

        if (emitEvent) {
            this.emitAnimationChange(source, 'loop');
        }

        return true;
    }

    setAnimationTime(time, options = {}) {
        const {
            source = 'api',
            emitEvent = true,
        } = options;

        if (!this.currentAction && this.animationActions.length > 0) {
            this.applyAnimationSelection(this.state.animationSelection !== 'none' ? this.state.animationSelection : '0', {
                autoplay: false,
                source,
                syncAttribute: true,
                emitEvent: false,
            });
        }

        if (!this.currentAction || !this.mixer) {
            return false;
        }

        const duration = this.getAnimationDuration();
        const targetTime = THREE.MathUtils.clamp(Number.isFinite(time) ? time : 0, 0, duration || 0);
        this.currentAction.time = targetTime;
        this.mixer.setTime(targetTime);
        this.refreshUiFromState({ syncTextureUi: false });
        this.requestRender();

        if (emitEvent) {
            this.emitAnimationChange(source, 'scrub');
        }

        return true;
    }

    resolveAnimationSelection(selection) {
        if (!this.animationActions.length) {
            return null;
        }

        const normalizedValue = selection === null || selection === undefined
            ? 'none'
            : `${selection}`.trim();

        if (!normalizedValue || normalizedValue === 'none') {
            return {
                index: -1,
                value: 'none',
                clip: null,
            };
        }

        if (/^\d+$/.test(normalizedValue)) {
            const index = parseInt(normalizedValue, 10);
            const action = this.animationActions[index];
            if (action) {
                return {
                    index,
                    value: `${index}`,
                    clip: action.getClip(),
                };
            }
        }

        const normalizedName = normalizedValue.toLowerCase();
        const matchedIndex = this.animationActions.findIndex((action) => {
            const clipName = action.getClip()?.name || '';
            return clipName === normalizedValue || clipName.toLowerCase() === normalizedName;
        });

        if (matchedIndex === -1) {
            return null;
        }

        return {
            index: matchedIndex,
            value: `${matchedIndex}`,
            clip: this.animationActions[matchedIndex].getClip(),
        };
    }

    applyAnimationSelection(selection, options = {}) {
        const {
            autoplay = this.hasAttribute('autoplay'),
            source = 'api',
            syncAttribute = false,
            emitEvent = true,
        } = options;
        const selectionValue = selection === null || selection === undefined ? 'none' : `${selection}`.trim();
        const previousSelectionValue = this.state.animationSelection;
        this.state.animationSelection = selectionValue || 'none';

        if (!this.animationActions.length) {
            this.refreshUiFromState({ syncTextureUi: false });
            return false;
        }

        const resolvedSelection = this.resolveAnimationSelection(selectionValue);
        if (!resolvedSelection) {
            this.state.animationSelection = previousSelectionValue;
            this.emitViewerError('set-animation', new Error(`Unknown animation selection: ${selectionValue}`), {
                selection: selectionValue,
            });
            return false;
        }

        const previousAction = this.currentAction;

        if (resolvedSelection.index === -1) {
            if (previousAction) {
                previousAction.stop();
            }
            this.currentAction = null;
            this.state.isAnimationPlaying = false;
            this.state.animationSelection = 'none';
        } else {
            this.currentAction = this.animationActions[resolvedSelection.index];
            const isChangingClip = previousAction && previousAction !== this.currentAction;

            if (isChangingClip && !autoplay) {
                previousAction.stop();
            }

            this.configureAnimationAction(this.currentAction);
            this.currentAction.reset();
            this.currentAction.setEffectiveWeight(1);
            if (autoplay) {
                if (isChangingClip) {
                    previousAction.paused = false;
                    previousAction.enabled = true;
                    previousAction.play();
                }
                this.currentAction.play();
                if (isChangingClip) {
                    this.currentAction.crossFadeFrom(previousAction, ANIMATION_CROSS_FADE_DURATION, true);
                }
                this.currentAction.paused = false;
            } else {
                this.currentAction.play();
                this.currentAction.paused = true;
            }
            this.state.isAnimationPlaying = autoplay;
            this.state.animationSelection = resolvedSelection.clip?.name || resolvedSelection.value;
        }

        if (syncAttribute) {
            this.reflectAttribute('animation', this.state.animationSelection === 'none' ? null : this.state.animationSelection);
        }

        this.refreshUiFromState({ syncTextureUi: false });
        this.requestRender();

        if (emitEvent) {
            this.emitAnimationChange(source, 'select');
        }

        return true;
    }

    applyAutoplayState(enabled, options = {}) {
        const { source = 'api' } = options;
        if (!this.animationActions.length) {
            return false;
        }

        if (this.currentAction) {
            if (enabled) {
                this.currentAction.play();
                this.currentAction.paused = false;
                this.state.isAnimationPlaying = true;
            } else {
                this.currentAction.paused = true;
                this.state.isAnimationPlaying = false;
            }
            this.refreshUiFromState({ syncTextureUi: false });
            this.emitAnimationChange(source, 'autoplay');
            return true;
        }

        if (!enabled && this.state.animationSelection === 'none') {
            return false;
        }

        const selection = this.state.animationSelection !== 'none'
            ? this.state.animationSelection
            : '0';
        return this.applyAnimationSelection(selection, {
            autoplay: enabled,
            source,
            syncAttribute: false,
            emitEvent: true,
        });
    }

    setMaterialEditorControlsEnabled(enabled) {
        [
            '#texturePartSelector',
            '#materialSlotSelector',
            '#baseColorInput',
            '#emissiveColorInput',
            '#emissiveIntensityInput',
            '#opacityInput',
            '#transparentToggle',
            '#doubleSidedToggle',
            '#roughness',
            '#metalness',
            '#normalScaleXInput',
            '#normalScaleYInput',
            '#envMapIntensityInput',
            '#uvRotationInput',
            '#uvRepeatXInput',
            '#uvRepeatYInput',
            '#uvOffsetXInput',
            '#uvOffsetYInput',
            '#textureTypeSelector',
            '#textureHistorySelector',
            '#replaceTextureBtn',
            '#removeTextureBtn',
            '#resetTextureBtn',
            '#copyTextureSourceBtn',
        ].forEach((selector) => {
            const element = this.shadowRoot.querySelector(selector);
            if (element) {
                element.disabled = !enabled;
            }
        });
    }

    populateMaterialSlotSelector(mesh, preferredIndex = 0) {
        const materialSlotSelector = this.shadowRoot.querySelector('#materialSlotSelector');
        const slotCount = this.getMaterialSlotCountForMesh(mesh);

        materialSlotSelector.innerHTML = '';
        for (let index = 0; index < slotCount; index += 1) {
            const option = document.createElement('option');
            option.value = index;
            option.textContent = slotCount > 1 ? `Material ${index + 1}` : 'Material 1';
            materialSlotSelector.appendChild(option);
        }

        if (!slotCount) {
            this.selectedMaterialIndex = 0;
            return;
        }

        this.selectedMaterialIndex = THREE.MathUtils.clamp(preferredIndex, 0, slotCount - 1);
        materialSlotSelector.value = `${this.selectedMaterialIndex}`;
    }

    updateTextureMetaInfo(texture) {
        const metaElement = this.shadowRoot.querySelector('#textureMetaInfo');
        const copySourceButton = this.shadowRoot.querySelector('#copyTextureSourceBtn');

        if (!metaElement) {
            return;
        }

        if (!texture) {
            metaElement.textContent = 'No texture assigned';
            if (copySourceButton) {
                copySourceButton.disabled = true;
            }
            return;
        }

        const source = getTextureSource(texture);
        const image = texture.image || texture.source?.data || null;
        const width = image?.width || image?.videoWidth || image?.naturalWidth || null;
        const height = image?.height || image?.videoHeight || image?.naturalHeight || null;
        const details = [
            source ? `Source: ${source}` : 'Source: embedded or unavailable',
            width && height ? `Size: ${width} x ${height}` : null,
            texture.name ? `Name: ${texture.name}` : null,
        ].filter(Boolean);

        metaElement.textContent = details.join('\n');
        if (copySourceButton) {
            copySourceButton.disabled = !source || !navigator.clipboard?.writeText;
        }
    }

    syncMaterialEditorControls() {
        const selection = this.getMaterialSelection();
        const previewElement = this.shadowRoot.querySelector('#texturePreview');

        if (!selection) {
            this.selectedMaterialIndex = 0;
            this.populateMaterialSlotSelector(null);
            this.setMaterialEditorControlsEnabled(false);
            this.updateTextureMetaInfo(null);
            if (previewElement) {
                previewElement.innerHTML = '';
                previewElement.textContent = this.meshParts.length === 0 ? 'None' : 'Select';
                Object.assign(previewElement.style, PREVIEW_TEXT_STYLE);
            }
            return;
        }

        const { storedMaterial, materialIndex } = selection;
        this.selectedMaterialIndex = materialIndex;
        this.populateMaterialSlotSelector(selection.mesh, materialIndex);
        this.setMaterialEditorControlsEnabled(true);

        this.shadowRoot.querySelector('#baseColorInput').value = `#${storedMaterial.color.getHexString()}`;
        this.shadowRoot.querySelector('#emissiveColorInput').value = `#${storedMaterial.emissive.getHexString()}`;
        this.shadowRoot.querySelector('#emissiveIntensityInput').value = storedMaterial.emissiveIntensity ?? 1;
        this.shadowRoot.querySelector('#opacityInput').value = storedMaterial.opacity ?? 1;
        this.shadowRoot.querySelector('#transparentToggle').checked = !!storedMaterial.transparent;
        this.shadowRoot.querySelector('#doubleSidedToggle').checked = storedMaterial.side === THREE.DoubleSide;
        this.shadowRoot.querySelector('#roughness').value = storedMaterial.roughness ?? 0.5;
        this.shadowRoot.querySelector('#metalness').value = storedMaterial.metalness ?? 0.5;
        this.shadowRoot.querySelector('#normalScaleXInput').value = storedMaterial.normalScale?.x ?? 1;
        this.shadowRoot.querySelector('#normalScaleYInput').value = storedMaterial.normalScale?.y ?? 1;
        this.shadowRoot.querySelector('#envMapIntensityInput').value = storedMaterial.envMapIntensity ?? 1;

        const selectedTexture = this.getTextureForPreview(selection);
        const uvTexture = selectedTexture
            || storedMaterial.map
            || storedMaterial.emissiveMap
            || storedMaterial.normalMap
            || storedMaterial.roughnessMap
            || storedMaterial.metalnessMap
            || storedMaterial.aoMap
            || null;
        this.shadowRoot.querySelector('#uvRepeatXInput').value = uvTexture?.repeat?.x ?? 1;
        this.shadowRoot.querySelector('#uvRepeatYInput').value = uvTexture?.repeat?.y ?? 1;
        this.shadowRoot.querySelector('#uvOffsetXInput').value = uvTexture?.offset?.x ?? 0;
        this.shadowRoot.querySelector('#uvOffsetYInput').value = uvTexture?.offset?.y ?? 0;
        this.shadowRoot.querySelector('#uvRotationInput').value = uvTexture?.rotation ?? 0;

        this.updateTextureMapDisplay();
        this.updateHistorySelector();
    }

    refreshMaterialsAfterEdit(mapType = null) {
        if (this.state.viewMode === 'default') {
            this.setDefaultMat();
            return;
        }

        if (this.state.viewMode === 'diffuse' && mapType === 'map') {
            this.showTexture();
            return;
        }

        this.requestRender();
    }

    emitMaterialEditEvent(action, selection, extra = {}) {
        const materialSnapshot = selection?.storedMaterial || null;
        this.emitEvent('viewer-material-change', {
            source: extra.source || 'ui',
            action,
            material: {
                meshName: selection?.mesh?.name || null,
                meshIndex: selection?.selectedPartIndex ?? -1,
                materialIndex: selection?.materialIndex ?? 0,
                mapType: selection?.mapType || null,
                roughness: materialSnapshot?.roughness ?? null,
                metalness: materialSnapshot?.metalness ?? null,
                opacity: materialSnapshot?.opacity ?? null,
                transparent: materialSnapshot?.transparent ?? null,
                doubleSided: materialSnapshot ? materialSnapshot.side === THREE.DoubleSide : null,
                envMapIntensity: materialSnapshot?.envMapIntensity ?? null,
                ...extra.material,
            },
        });
    }

    applyMaterialPropertiesFromEditor(options = {}) {
        const selection = this.getMaterialSelection();
        if (!selection) {
            return false;
        }

        const {
            storedMaterial,
            mesh,
            materialIndex,
        } = selection;
        const normalScaleX = parseFloat(this.shadowRoot.querySelector('#normalScaleXInput').value);
        const normalScaleY = parseFloat(this.shadowRoot.querySelector('#normalScaleYInput').value);
        const opacity = THREE.MathUtils.clamp(parseFloat(this.shadowRoot.querySelector('#opacityInput').value), 0, 1);
        const transparentEnabled = this.shadowRoot.querySelector('#transparentToggle').checked || opacity < 1;

        storedMaterial.color.set(this.shadowRoot.querySelector('#baseColorInput').value);
        storedMaterial.emissive.set(this.shadowRoot.querySelector('#emissiveColorInput').value);
        storedMaterial.emissiveIntensity = Math.max(0, parseFloat(this.shadowRoot.querySelector('#emissiveIntensityInput').value) || 0);
        storedMaterial.opacity = Number.isFinite(opacity) ? opacity : 1;
        storedMaterial.transparent = transparentEnabled;
        storedMaterial.side = this.shadowRoot.querySelector('#doubleSidedToggle').checked ? THREE.DoubleSide : THREE.FrontSide;
        storedMaterial.roughness = THREE.MathUtils.clamp(parseFloat(this.shadowRoot.querySelector('#roughness').value) || 0.5, 0, 1);
        storedMaterial.metalness = THREE.MathUtils.clamp(parseFloat(this.shadowRoot.querySelector('#metalness').value) || 0.5, 0, 1);
        storedMaterial.envMapIntensity = Math.max(0, parseFloat(this.shadowRoot.querySelector('#envMapIntensityInput').value) || 0);
        storedMaterial.normalScale = storedMaterial.normalScale || new THREE.Vector2(1, 1);
        storedMaterial.normalScale.set(
            Number.isFinite(normalScaleX) ? normalScaleX : 1,
            Number.isFinite(normalScaleY) ? normalScaleY : 1
        );
        storedMaterial.needsUpdate = true;

        const currentMaterial = getMaterialEntryAt(mesh.material, materialIndex);
        if (this.state.viewMode === 'default' && currentMaterial?.isMeshStandardMaterial) {
            currentMaterial.color.copy(storedMaterial.color);
            currentMaterial.emissive.copy(storedMaterial.emissive);
            currentMaterial.emissiveIntensity = storedMaterial.emissiveIntensity;
            currentMaterial.opacity = storedMaterial.opacity;
            currentMaterial.transparent = storedMaterial.transparent;
            currentMaterial.side = storedMaterial.side;
            currentMaterial.roughness = storedMaterial.roughness;
            currentMaterial.metalness = storedMaterial.metalness;
            currentMaterial.envMapIntensity = this.getEffectiveEnvironmentIntensity(storedMaterial.envMapIntensity);
            currentMaterial.normalScale.copy(storedMaterial.normalScale);
            currentMaterial.needsUpdate = true;
        }

        this.refreshMaterialsAfterEdit();
        this.emitMaterialEditEvent(options.action || 'update-material-properties', selection, {
            source: options.source || 'ui',
        });
        this.syncMaterialEditorControls();
        return true;
    }

    applyUvTransformFromEditor(options = {}) {
        const selection = this.getMaterialSelection();
        if (!selection) {
            return false;
        }

        const texture = this.getTextureForPreview(selection);
        if (!texture) {
            this.updateTextureMetaInfo(null);
            return false;
        }

        texture.repeat.set(
            parseFloat(this.shadowRoot.querySelector('#uvRepeatXInput').value) || 1,
            parseFloat(this.shadowRoot.querySelector('#uvRepeatYInput').value) || 1
        );
        texture.offset.set(
            parseFloat(this.shadowRoot.querySelector('#uvOffsetXInput').value) || 0,
            parseFloat(this.shadowRoot.querySelector('#uvOffsetYInput').value) || 0
        );
        texture.rotation = parseFloat(this.shadowRoot.querySelector('#uvRotationInput').value) || 0;
        texture.needsUpdate = true;

        this.refreshMaterialsAfterEdit(selection.mapType);
        this.emitMaterialEditEvent(options.action || 'update-uv-transform', selection, {
            source: options.source || 'ui',
            material: {
                uv: {
                    repeat: { x: texture.repeat.x, y: texture.repeat.y },
                    offset: { x: texture.offset.x, y: texture.offset.y },
                    rotation: texture.rotation,
                },
            },
        });
        this.updateTextureMapDisplay();
        return true;
    }

    initTextureMapUI() {
        const partSelector = this.shadowRoot.querySelector('#texturePartSelector');
        const materialSlotSelector = this.shadowRoot.querySelector('#materialSlotSelector');
        const typeSelector = this.shadowRoot.querySelector('#textureTypeSelector');
        const replaceTextureButton = this.shadowRoot.querySelector('#replaceTextureBtn');
        const removeTextureButton = this.shadowRoot.querySelector('#removeTextureBtn');
        const resetTextureButton = this.shadowRoot.querySelector('#resetTextureBtn');
        const copyTextureSourceButton = this.shadowRoot.querySelector('#copyTextureSourceBtn');
        const historySelector = this.shadowRoot.querySelector('#textureHistorySelector');

        Object.entries(TEXTURE_INPUT_IDS).forEach(([mapType, inputId]) => {
            this.shadowRoot.querySelector(`#${inputId}`).addEventListener('change', (event) => {
                this.handleTextureFileChange(event, mapType);
            });
        });

        partSelector.addEventListener('change', () => {
            const selection = this.getTextureSelection();
            if (selection) {
                this.selectMeshPartInSceneGraph(selection.mesh, null, {
                    force: true,
                    source: 'ui',
                });
            }
            this.syncMaterialEditorControls();
        });

        materialSlotSelector.addEventListener('change', () => {
            this.selectedMaterialIndex = this.getSelectedMaterialIndex(this.meshParts[parseInt(partSelector.value, 10)]);
            this.syncMaterialEditorControls();
        });

        typeSelector.addEventListener('change', () => {
            this.syncMaterialEditorControls();
        });

        replaceTextureButton.addEventListener('click', () => {
            const selection = this.getTextureSelection();
            if (!selection) {
                return;
            }

            const fileInputId = TEXTURE_INPUT_IDS[selection.mapType];
            if (!fileInputId) {
                return;
            }

            const fileInput = this.shadowRoot.querySelector(`#${fileInputId}`);
            fileInput.dataset.meshPartIndex = selection.selectedPartIndex;
            fileInput.dataset.materialIndex = selection.materialIndex;
            fileInput.value = '';
            fileInput.click();
        });

        historySelector.addEventListener('change', () => {
            this.applyHistoryTexture();
        });

        removeTextureButton.addEventListener('click', () => {
            this.removeTexture();
        });

        resetTextureButton.addEventListener('click', () => {
            this.resetTextureToOriginal();
        });

        copyTextureSourceButton.addEventListener('click', async () => {
            const selection = this.getTextureSelection();
            const texture = this.getTextureForPreview(selection);
            const source = getTextureSource(texture);
            if (!source || !navigator.clipboard?.writeText) {
                return;
            }

            try {
                await navigator.clipboard.writeText(source);
            } catch (error) {
                console.warn('Failed to copy texture source.', error);
            }
        });

        [
            '#baseColorInput',
            '#emissiveColorInput',
            '#emissiveIntensityInput',
            '#opacityInput',
            '#transparentToggle',
            '#doubleSidedToggle',
            '#roughness',
            '#metalness',
            '#normalScaleXInput',
            '#normalScaleYInput',
            '#envMapIntensityInput',
        ].forEach((selector) => {
            this.shadowRoot.querySelector(selector).addEventListener('input', () => {
                this.applyMaterialPropertiesFromEditor();
            });
        });

        [
            '#uvRepeatXInput',
            '#uvRepeatYInput',
            '#uvOffsetXInput',
            '#uvOffsetYInput',
            '#uvRotationInput',
        ].forEach((selector) => {
            this.shadowRoot.querySelector(selector).addEventListener('input', () => {
                this.applyUvTransformFromEditor();
            });
        });

        this.setMaterialEditorControlsEnabled(false);
    }

    handleTextureFileChange(event, mapType) {
        if (!event.target) {
            console.error('event.target is null');
            return;
        }

        const fileInput = event.target;
        const file = fileInput.files[0];
        if (!file) return;

        const textureURL = URL.createObjectURL(file);
        const texture = this.textureLoader.load(textureURL, () => {
            const selectedMeshPartIndex = parseInt(fileInput.dataset.meshPartIndex);
            const materialIndex = parseInt(fileInput.dataset.materialIndex || '0', 10);
            if (isNaN(selectedMeshPartIndex)) {
                console.error("Mesh part index is not set on the file input.");
                URL.revokeObjectURL(textureURL);
                return;
            }

            const mesh = this.meshParts[selectedMeshPartIndex];
            const material = this.getMaterialStoreMaterial(mesh, materialIndex, this.originalMaterials);
            if (!mesh || !material) {
                console.error("Mesh or material not found for index:", selectedMeshPartIndex);
                URL.revokeObjectURL(textureURL);
                return;
            }

            this.applyTextureEncoding(texture, mapType);
            texture.name = file.name;
            this.saveTextureToHistory(mesh, materialIndex, mapType, material[mapType]);

            material[mapType] = texture;
            material.needsUpdate = true;
            this.refreshMaterialsAfterEdit(mapType);

            this.emitMaterialEditEvent('replace-texture', {
                mesh,
                selectedPartIndex: selectedMeshPartIndex,
                materialIndex,
                mapType,
                storedMaterial: material,
            });
            this.syncMaterialEditorControls();
            URL.revokeObjectURL(textureURL);
        }, undefined, (error) => {
            console.error('Texture loading error:', error);
            alert('Failed to load texture.');
            URL.revokeObjectURL(textureURL);
        });
    }

    saveTextureToHistory(mesh, materialIndex, mapType, texture) {
        const historyKey = this.getTextureHistoryKey(mesh, materialIndex);
        if (!this.textureHistory.has(historyKey)) {
            this.textureHistory.set(historyKey, new Map());
        }
        const typeHistory = this.textureHistory.get(historyKey);
        if (!typeHistory.has(mapType)) {
            typeHistory.set(mapType, []);
        }
        const historyArray = typeHistory.get(mapType);
        if (texture) {
            historyArray.push(cloneTexture(texture));
        }
    }

    populateTextureMapSelector() {
        const partSelector = this.shadowRoot.querySelector('#texturePartSelector');
        const historySelector = this.shadowRoot.querySelector('#textureHistorySelector');

        partSelector.innerHTML = '';
        this.meshParts.forEach((mesh, index) => {
            const option = document.createElement('option');
            option.value = index;
            const materialCount = this.getMaterialSlotCountForMesh(mesh);
            option.textContent = mesh.name || `Part ${index + 1}`;
            if (materialCount > 1) {
                option.textContent += ` (${materialCount} materials)`;
            }
            partSelector.appendChild(option);
        });

        partSelector.selectedIndex = this.meshParts.length > 0 ? 0 : -1;
        this.selectedMaterialIndex = 0;
        this.populateMaterialSlotSelector(this.meshParts[0] || null, 0);
        historySelector.value = '-1';

        this.syncMaterialEditorControls();
    }

    updateHistorySelector() {
        const historySelector = this.shadowRoot.querySelector('#textureHistorySelector');
        historySelector.innerHTML = '<option value="-1">Current</option>';
        const selection = this.getTextureSelection();

        if (!selection) {
            return;
        }

        const historyKey = this.getTextureHistoryKey(selection.mesh, selection.materialIndex);
        if (this.textureHistory.has(historyKey) && this.textureHistory.get(historyKey).has(selection.mapType)) {
            const historyArray = this.textureHistory.get(historyKey).get(selection.mapType);
            historyArray.forEach((texture, index) => {
                const option = document.createElement('option');
                option.value = index;
                option.textContent = `History ${index + 1}`;
                historySelector.appendChild(option);
            });
        }
    }

    applyHistoryTexture() {
        const historySelector = this.shadowRoot.querySelector('#textureHistorySelector');
        const selection = this.getTextureSelection();
        const historyIndex = parseInt(historySelector.value, 10);

        if (!selection) {
            return;
        }

        const { mesh, storedMaterial, initialMaterial, mapType, materialIndex } = selection;
        const historyKey = this.getTextureHistoryKey(mesh, materialIndex);
        let selectedTexture = null;

        if (historyIndex === -1) {
            selectedTexture = storedMaterial[mapType] || initialMaterial?.[mapType] || null;
        } else if (this.textureHistory.has(historyKey) && this.textureHistory.get(historyKey).has(mapType)) {
            const historyArray = this.textureHistory.get(historyKey).get(mapType);
            if (historyIndex >= 0 && historyIndex < historyArray.length) {
                selectedTexture = historyArray[historyIndex];
            }
        }

        if (!selectedTexture) {
            this.syncMaterialEditorControls();
            return;
        }

        storedMaterial[mapType] = cloneTexture(selectedTexture);
        storedMaterial.needsUpdate = true;

        this.refreshMaterialsAfterEdit(mapType);
        this.emitMaterialEditEvent('apply-texture-history', selection, {
            source: 'ui',
            material: {
                historyIndex,
            },
        });
        this.syncMaterialEditorControls();
    }

    removeTexture() {
        const selection = this.getTextureSelection();
        if (!selection) {
            return false;
        }

        const currentTexture = selection.storedMaterial[selection.mapType];
        if (currentTexture) {
            this.saveTextureToHistory(selection.mesh, selection.materialIndex, selection.mapType, currentTexture);
        }

        selection.storedMaterial[selection.mapType] = null;
        selection.storedMaterial.needsUpdate = true;
        this.refreshMaterialsAfterEdit(selection.mapType);
        this.emitMaterialEditEvent('remove-texture', selection, {
            source: 'ui',
        });
        this.syncMaterialEditorControls();
        return true;
    }

    resetTextureToOriginal() {
        const selection = this.getTextureSelection();
        if (!selection) {
            return false;
        }

        const currentTexture = selection.storedMaterial[selection.mapType];
        if (currentTexture) {
            this.saveTextureToHistory(selection.mesh, selection.materialIndex, selection.mapType, currentTexture);
        }

        selection.storedMaterial[selection.mapType] = cloneTexture(selection.initialMaterial?.[selection.mapType] || null);
        selection.storedMaterial.needsUpdate = true;
        this.refreshMaterialsAfterEdit(selection.mapType);
        this.emitMaterialEditEvent('reset-texture', selection, {
            source: 'ui',
        });
        this.syncMaterialEditorControls();
        return true;
    }

    updateTextureMapDisplay() {
        const previewElement = this.shadowRoot.querySelector('#texturePreview');
        const selection = this.getTextureSelection();

        if (!selection) {
            previewElement.innerHTML = '';
            previewElement.textContent = this.meshParts.length === 0 ? 'None' : 'Error';
            Object.assign(previewElement.style, PREVIEW_TEXT_STYLE);
            this.updateTextureMetaInfo(null);
            return;
        }

        const selectedTexture = this.getTextureForPreview(selection);

        this.drawPreview(selectedTexture, previewElement);
        this.updateTextureMetaInfo(selectedTexture);
    }

    drawPreview(selectedTexture, previewElement){
        previewElement.innerHTML = '';
        if (selectedTexture) {
            if (selectedTexture.image instanceof ImageBitmap) {
                this.setImageBitmapPreview(selectedTexture.image, previewElement);
            } else if (selectedTexture.image) {
                const imageSource = selectedTexture.image.currentSrc || selectedTexture.image.src;
                const img = document.createElement('img');
                img.src = imageSource;
                img.alt = 'Texture preview';
                previewElement.textContent = '';
                previewElement.appendChild(img);
            } else {
                previewElement.textContent = 'Preview Unavailable';
                Object.assign(previewElement.style, PREVIEW_TEXT_STYLE);
            }
        } else {
            previewElement.textContent = 'None';
            Object.assign(previewElement.style, PREVIEW_TEXT_STYLE);
        }
    }

    setImageBitmapPreview(imageBitmap, previewElement) {
        const canvas = document.createElement('canvas');
        canvas.width = imageBitmap.width;
        canvas.height = imageBitmap.height;
        const ctx = canvas.getContext('2d');

        if (!ctx) {
            previewElement.textContent = 'Cannot Preview';
            previewElement.style.lineHeight = '1.4';
            previewElement.style.textAlign = 'center';
            console.error('Canvas context is null, cannot generate ImageBitmap preview.');
            return;
        }

        try {
            ctx.drawImage(imageBitmap, 0, 0, imageBitmap.width, imageBitmap.height);
            previewElement.innerHTML = '';
            previewElement.appendChild(canvas);

        } catch (error) {
            console.error('Error drawing ImageBitmap on canvas:', error);
            previewElement.textContent = 'Preview Error';
            previewElement.style.lineHeight = '1.4';
            previewElement.style.textAlign = 'center';
        }
    }

    updateMaterialProperties() {
        return this.applyMaterialPropertiesFromEditor({
            source: 'ui',
            action: 'update-surface',
        });
    }

    serializeState() {
        const selection = this.selectedMeshPart ? {
            index: this.selectedMeshPartIndex,
            name: this.selectedMeshPart.name || null,
            uuid: this.selectedMeshPart.uuid,
        } : null;

        return {
            schemaVersion: STATE_SCHEMA_VERSION,
            src: this.currentModelSource || this.getAttribute('src') || null,
            fileName: this.currentModelFileName,
            viewMode: this.state.viewMode,
            environment: this.state.environment === CUSTOM_ENVIRONMENT_ID ? null : this.state.environment,
            environmentUrl: this.state.environmentUrl,
            environmentState: {
                preset: this.state.environment === CUSTOM_ENVIRONMENT_ID ? null : this.state.environment,
                url: this.state.environmentUrl,
                intensity: this.state.environmentIntensity,
                rotation: this.state.environmentRotation,
                backgroundVisible: this.state.environmentBackgroundVisible,
            },
            backgroundColor: this.state.backgroundColor,
            exposure: this.state.exposure,
            autoRotate: this.autoRotate,
            anglePerSecond: this.anglePerSecond,
            animation: this.state.animationSelection,
            isAnimationPlaying: this.state.isAnimationPlaying,
            animationState: this.getAnimationStateSnapshot(),
            selectionMode: this.state.selectionMode,
            performanceMode: this.state.performanceMode,
            selection,
            camera: this.getCameraStateSnapshot(),
            materialOverrides: this.serializeMaterialOverrides(),
            modelTransform: this.model ? {
                position: {
                    x: this.model.position.x,
                    y: this.model.position.y,
                    z: this.model.position.z,
                },
                rotation: {
                    x: this.model.rotation.x,
                    y: this.model.rotation.y,
                    z: this.model.rotation.z,
                },
                scale: {
                    x: this.model.scale.x,
                    y: this.model.scale.y,
                    z: this.model.scale.z,
                },
            } : null,
        };
    }

    exportState() {
        const state = this.serializeState();
        this.emitEvent('viewer-state-export', {
            source: 'api',
            state,
        });
        return state;
    }

    serializeTextureOverride(texture) {
        if (!texture) {
            return null;
        }

        return {
            hasTexture: true,
            source: getTextureSource(texture) || null,
            name: texture.name || null,
            repeat: texture.repeat ? { x: texture.repeat.x, y: texture.repeat.y } : null,
            offset: texture.offset ? { x: texture.offset.x, y: texture.offset.y } : null,
            rotation: texture.rotation ?? 0,
        };
    }

    serializeMaterialOverrides() {
        return this.meshParts.map((mesh, meshIndex) => {
            const materialEntry = this.getMaterialStoreEntry(mesh, this.originalMaterials);
            return getMaterialArray(materialEntry).map((material, materialIndex) => ({
                meshIndex,
                meshName: mesh.name || null,
                materialIndex,
                color: material?.color ? `#${material.color.getHexString()}` : null,
                emissive: material?.emissive ? `#${material.emissive.getHexString()}` : null,
                emissiveIntensity: material?.emissiveIntensity ?? 1,
                opacity: material?.opacity ?? 1,
                transparent: !!material?.transparent,
                side: material?.side === THREE.DoubleSide ? 'double' : 'front',
                roughness: material?.roughness ?? 0.5,
                metalness: material?.metalness ?? 0.5,
                envMapIntensity: material?.envMapIntensity ?? 1,
                normalScale: material?.normalScale ? { x: material.normalScale.x, y: material.normalScale.y } : null,
                textures: Object.fromEntries(
                    Object.keys(TEXTURE_INPUT_IDS).map((mapType) => [mapType, this.serializeTextureOverride(material?.[mapType] || null)])
                ),
            }));
        }).flat();
    }

    async loadTextureOverride(textureState, mapType) {
        if (!textureState?.source) {
            return null;
        }

        return new Promise((resolve, reject) => {
            this.textureLoader.load(textureState.source, (texture) => {
                this.applyTextureEncoding(texture, mapType);
                texture.name = textureState.name || '';
                if (textureState.repeat) {
                    texture.repeat.set(textureState.repeat.x, textureState.repeat.y);
                }
                if (textureState.offset) {
                    texture.offset.set(textureState.offset.x, textureState.offset.y);
                }
                texture.rotation = textureState.rotation ?? 0;
                texture.needsUpdate = true;
                resolve(texture);
            }, undefined, reject);
        });
    }

    async applyMaterialOverrides(overrides = []) {
        if (!Array.isArray(overrides) || !this.model) {
            return false;
        }

        const textureLoads = [];
        overrides.forEach((override) => {
            const mesh = this.meshParts[override.meshIndex]
                || this.meshParts.find((candidate) => candidate.name && candidate.name === override.meshName)
                || null;
            const storedMaterial = this.getMaterialStoreMaterial(mesh, override.materialIndex, this.originalMaterials);
            if (!mesh || !storedMaterial) {
                return;
            }

            if (override.color) {
                storedMaterial.color.set(override.color);
            }
            if (override.emissive) {
                storedMaterial.emissive.set(override.emissive);
            }
            storedMaterial.emissiveIntensity = override.emissiveIntensity ?? storedMaterial.emissiveIntensity;
            storedMaterial.opacity = override.opacity ?? storedMaterial.opacity;
            storedMaterial.transparent = override.transparent ?? storedMaterial.transparent;
            storedMaterial.side = override.side === 'double' ? THREE.DoubleSide : THREE.FrontSide;
            storedMaterial.roughness = override.roughness ?? storedMaterial.roughness;
            storedMaterial.metalness = override.metalness ?? storedMaterial.metalness;
            storedMaterial.envMapIntensity = override.envMapIntensity ?? storedMaterial.envMapIntensity;
            if (override.normalScale) {
                storedMaterial.normalScale = storedMaterial.normalScale || new THREE.Vector2(1, 1);
                storedMaterial.normalScale.set(override.normalScale.x, override.normalScale.y);
            }

            Object.entries(override.textures || {}).forEach(([mapType, textureState]) => {
                if (textureState === null) {
                    storedMaterial[mapType] = null;
                    return;
                }

                if (!textureState?.hasTexture) {
                    return;
                }

                if (textureState.source) {
                    textureLoads.push(
                        this.loadTextureOverride(textureState, mapType)
                            .then((texture) => {
                                storedMaterial[mapType] = texture;
                            })
                            .catch((error) => {
                                this.emitViewerError('import-texture-override', error, {
                                    mapType,
                                    source: textureState.source,
                                });
                            })
                    );
                    return;
                }

                const existingTexture = storedMaterial[mapType];
                if (existingTexture) {
                    if (textureState.repeat) {
                        existingTexture.repeat.set(textureState.repeat.x, textureState.repeat.y);
                    }
                    if (textureState.offset) {
                        existingTexture.offset.set(textureState.offset.x, textureState.offset.y);
                    }
                    existingTexture.rotation = textureState.rotation ?? existingTexture.rotation ?? 0;
                    existingTexture.needsUpdate = true;
                }
            });

            storedMaterial.needsUpdate = true;
        });

        await Promise.all(textureLoads);
        this.renderMode();
        this.syncMaterialEditorControls();
        this.requestRender();
        return true;
    }

    async importState(state) {
        if (!state || typeof state !== 'object') {
            return false;
        }

        if (state.src) {
            await this.loadModelFromUrl(state.src);
        }

        if (state.backgroundColor) {
            this.setBackgroundColor(state.backgroundColor, {
                source: 'import',
                syncAttribute: true,
            });
        }

        if (state.exposure !== undefined) {
            this.setExposure(state.exposure, {
                source: 'import',
                syncAttribute: true,
            });
        }

        if (state.environmentState?.intensity !== undefined) {
            this.setEnvironmentIntensity(state.environmentState.intensity, {
                source: 'import',
                syncAttribute: true,
                emitEvent: false,
            });
        }

        if (state.environmentState?.rotation !== undefined) {
            this.setEnvironmentRotation(state.environmentState.rotation, {
                source: 'import',
                syncAttribute: true,
                emitEvent: false,
            });
        }

        if (state.environmentState?.backgroundVisible !== undefined) {
            this.setEnvironmentBackgroundVisible(state.environmentState.backgroundVisible, {
                source: 'import',
                syncAttribute: true,
                emitEvent: false,
            });
        }

        if (state.selectionMode) {
            this.applySelectionMode(state.selectionMode, {
                source: 'import',
                syncAttribute: true,
            });
        }

        if (state.performanceMode) {
            this.applyPerformanceMode(state.performanceMode, {
                source: 'import',
                syncAttribute: true,
            });
        }

        if (state.environmentUrl) {
            await this.setEnvironment(state.environmentUrl, {
                source: 'import',
                syncAttribute: true,
            });
        } else if (state.environment) {
            await this.setEnvironment(state.environment, {
                source: 'import',
                syncAttribute: true,
            });
        } else {
            this.clearEnvironment({
                source: 'import',
                syncAttribute: true,
            });
        }

        if (state.camera?.target) {
            this.setCameraTarget(new THREE.Vector3(
                state.camera.target.x,
                state.camera.target.y,
                state.camera.target.z
            ), {
                source: 'import',
                syncAttribute: true,
                emitEvent: false,
            });
        }

        if (state.camera?.up) {
            this.setCameraUp(new THREE.Vector3(
                state.camera.up.x,
                state.camera.up.y,
                state.camera.up.z
            ), {
                source: 'import',
                syncAttribute: true,
                emitEvent: false,
            });
        }

        if (state.camera?.position) {
            this.setCameraOrbit(new THREE.Vector3(
                state.camera.position.x,
                state.camera.position.y,
                state.camera.position.z
            ), {
                source: 'import',
                syncAttribute: true,
                emitEvent: false,
            });
        }

        if (state.camera?.fov !== undefined) {
            this.updateCameraFov(state.camera.fov, {
                source: 'import',
                emitEvent: false,
            });
        }

        if (state.camera?.near !== undefined) {
            this.updateCameraNear(state.camera.near, {
                source: 'import',
                emitEvent: false,
            });
        }

        if (state.camera?.far !== undefined) {
            this.updateCameraFar(state.camera.far, {
                source: 'import',
                emitEvent: false,
            });
        }

        if (state.viewMode) {
            this.state.viewMode = state.viewMode;
            this.renderMode();
        }

        if (state.autoRotate !== undefined) {
            this.autoRotate = !!state.autoRotate;
            this.reflectBooleanAttribute('auto-rotate', this.autoRotate);
        }

        if (state.anglePerSecond !== undefined) {
            this.anglePerSecond = state.anglePerSecond;
            this.reflectAttribute('angle-per-second', state.anglePerSecond);
        }

        if (state.animationState?.speed !== undefined) {
            this.setAnimationSpeed(state.animationState.speed, {
                source: 'import',
                syncAttribute: true,
                emitEvent: false,
            });
        }

        if (state.animationState?.loopMode) {
            this.setAnimationLoopMode(state.animationState.loopMode, {
                source: 'import',
                syncAttribute: true,
                emitEvent: false,
            });
        }

        if (state.modelTransform && this.model) {
            this.model.position.set(
                state.modelTransform.position.x,
                state.modelTransform.position.y,
                state.modelTransform.position.z
            );
            this.model.rotation.set(
                state.modelTransform.rotation.x,
                state.modelTransform.rotation.y,
                state.modelTransform.rotation.z
            );
            this.model.scale.set(
                state.modelTransform.scale.x,
                state.modelTransform.scale.y,
                state.modelTransform.scale.z
            );
            this.updateControlPanel();
            this.requestRender();
        }

        if (state.materialOverrides) {
            await this.applyMaterialOverrides(state.materialOverrides);
        }

        const importedAnimationSelection = state.animationState?.selected ?? state.animation;
        const importedAnimationPlaying = state.animationState?.isPlaying ?? state.isAnimationPlaying;
        if (importedAnimationSelection && importedAnimationSelection !== 'none') {
            this.applyAnimationSelection(importedAnimationSelection, {
                autoplay: !!importedAnimationPlaying,
                source: 'import',
                syncAttribute: true,
            });
            if (state.animationState?.time !== undefined) {
                this.setAnimationTime(state.animationState.time, {
                    source: 'import',
                    emitEvent: false,
                });
            }
        } else if (importedAnimationPlaying === false) {
            this.applyAnimationSelection('none', {
                autoplay: false,
                source: 'import',
                syncAttribute: true,
            });
        }

        if (state.selection?.name) {
            this.selectMeshByName(state.selection.name);
        } else if (typeof state.selection?.index === 'number') {
            this.selectMeshByIndex(state.selection.index);
        }

        this.captureCurrentCameraStateAsDefault();
        this.emitCameraChange('import');
        return true;
    }

    generateSceneGraphTree(object, parentElement) {
        const ul = document.createElement('ul');

        object.children.forEach(child => {
            const li = document.createElement('li');
            const label = document.createElement('label');
            const toggleId = `material-toggle-${child.uuid}`;
            const checkbox = document.createElement('input');
            checkbox.type = 'checkbox';
            checkbox.id = toggleId;
            checkbox.checked = child.visible;

            if (child.isMesh) {
                this.sceneGraphLabelByMeshUuid.set(child.uuid, label);
            }

            checkbox.addEventListener('change', (e) => {
                child.visible = e.target.checked;
                if (!this.isObjectEffectivelyVisible(child)) {
                    if (this.selectedMeshPart === child) {
                        this.clearSelectionState({
                            source: 'visibility-change',
                            emitEvent: true,
                        });
                    }
                    if (this.hoveredMeshPart === child) {
                        this.clearHoverState();
                    }
                }
                this.requestRender();
            });


            let name = child.name || child.type;
            if (name === '') name = 'unnamed';
            const nameSpan = document.createElement('span');
            nameSpan.textContent = name;
            label.appendChild(nameSpan);
            label.appendChild(checkbox);

            label.addEventListener('click', (event) => {
                event.stopPropagation();
                this.selectMeshPartInSceneGraph(child, label, {
                    source: 'ui',
                    channel: 'scene-graph',
                });
            });

            li.appendChild(label);
            ul.appendChild(li);

            if (child.children.length > 0) {
                this.generateSceneGraphTree(child, li);
            }
        });
        parentElement.appendChild(ul);
    }

    animate(time) {
        if (!this.isConnectedToDom) {
            return;
        }

        if (!this.lastTime) this.lastTime = 0;
        const deltaTime = (time - this.lastTime) / 1000;
        this.lastTime = time;

        if (this.autoRotate && this.model) {
            const rotationSpeed = THREE.MathUtils.degToRad(this.anglePerSecond);
            this.model.rotation.y += rotationSpeed * deltaTime;
        }

        if (this.mixer) {
            this.mixer.update(deltaTime);
        }

        // idle animation for no model
        if (!this.model) {
            if (!this.isIdleAnimationRunning) {
                this.initIdleAnimation();
                if (typeof TWEEN !== 'undefined') {
                    TWEEN.update();
                }
            }
            if (this.tweenGroup) {
                this.tweenGroup.update(time);
            }
            if (this.animationMesh && this.isIdleAnimationRunning) {
                const rotationSpeedy = Math.PI / 6; // 30 deg
                const rotationSpeedz = Math.PI / 3; // 60 deg
                const rotationSpeedx = Math.PI / 9;
                this.animationMesh.rotation.y += rotationSpeedy * deltaTime;
                this.animationMesh.rotation.z += rotationSpeedz * deltaTime;
                this.animationMesh.rotation.x += rotationSpeedx * deltaTime;
            }

        } else if (this.isIdleAnimationRunning) {
            this.scene.remove(this.animationMesh);
            if (this.animationGeometry) {
                this.animationGeometry.dispose();
            }
            this.animationMesh = null;
            this.tweenGroup = null;
            this.isIdleAnimationRunning = false;
        }

        this.controls.update();
        this.updateAnimationUi();
        this.updateSelectionHelpers();
        this.renderer.render(this.scene, this.camera);
    }

    resizeRenderer() {
        const host = this.shadowRoot.host;
        const width = host.clientWidth;
        const height = host.clientHeight || Math.max(320, Math.round(width * 0.6));

        if (!width || !height) {
            return;
        }

        this.renderer.setSize(width, height);
        this.camera.aspect = width / height;
        this.camera.updateProjectionMatrix();
    }

    createToonMaterial(originalTexture = null) {
        const toonMaterial = new THREE.ShaderMaterial({
            uniforms: {
                lightDirection: { value: new THREE.Vector3(0.5, 0.5, 1).normalize() },
                outlineColor: { value: new THREE.Color(0x000000) },
                toonColors: { value: [new THREE.Color(0xffffff), new THREE.Color(0xc0c0c0), new THREE.Color(0x808080)] },
                toonSteps: { value: [0.8, 0.5] },
                originalTexture: { value: originalTexture },
                textureBlendFactor: { value: 0.8 },
                outlineThickness: { value: 0.05 },
                rimColor: { value: new THREE.Color(0xaaaaaa) },
                rimPower: { value: 2.0 }
            },
            vertexShader: /*glsl*/`
            varying vec3 vNormal;
            varying vec3 vWorldPosition;
            varying vec2 vUv;

            void main() {
                vNormal = normalize(normalMatrix * normal);
                vec4 worldPosition = modelMatrix * vec4(position, 1.0);
                vWorldPosition = worldPosition.xyz;
                vUv = uv;
                gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
            }
        `,
            fragmentShader: /*glsl*/`
            uniform vec3 lightDirection;
            uniform vec3 outlineColor;
            uniform vec3 toonColors[3];
            uniform float toonSteps[2];
            uniform sampler2D originalTexture;
            uniform float textureBlendFactor;
            uniform float outlineThickness;

            varying vec3 vNormal;
            varying vec3 vWorldPosition;
            varying vec2 vUv;

            uniform vec3 rimColor;
            uniform float rimPower;

            void main() {
                float diffuseIntensity = max(0.0, dot(vNormal, lightDirection));
                vec3 toonColor = toonColors[0];
                if (diffuseIntensity < toonSteps[0]) toonColor = toonColors[1];
                if (diffuseIntensity < toonSteps[1]) toonColor = toonColors[2];

                vec3 viewDir = normalize(cameraPosition - vWorldPosition);
                float outlineFactor = 1.0 - max(0.0, dot(vNormal, viewDir));
                float outlineThreshold = 0.7;
                float outlineMix = smoothstep(outlineThreshold - outlineThickness, outlineThreshold + outlineThickness, outlineFactor);

                vec3 finalToonColor = mix(toonColor, outlineColor, outlineMix);
                vec4 originalTexColor = texture2D(originalTexture, vUv);

                float rimFactor = 1.0 - max(0.0, dot(vNormal, viewDir));
                rimFactor = pow(rimFactor, rimPower); // curvature effect
                vec3 rimLighting = rimColor * rimFactor;

                vec3 finalColor = mix(finalToonColor, originalTexColor.rgb, textureBlendFactor) + rimLighting;
                gl_FragColor = vec4(finalColor, 1.0);
            }
        `
        });
        if (originalTexture) {
            originalTexture.encoding = THREE.sRGBEncoding;
        }
        return toonMaterial;
    }

    enableToonShading() {
        if (!this.model) return;
        this.toonMaterial = this.toonMaterial || this.createToonMaterial();
        this.model.traverse((child) => {
            if (child.isMesh) {
                this.toonMaterialBackups.set(child.uuid, child.material);
                child.material = this.toonMaterial;
                const backupMaterial = this.toonMaterialBackups.get(child.uuid);
                const sourceMaterial = getMaterialEntryAt(backupMaterial, 0);
                if (sourceMaterial?.map) {
                    this.toonMaterial.uniforms.originalTexture.value = sourceMaterial.map;
                    this.toonMaterial.uniforms.originalTexture.needsUpdate = true; // Texture uniform update
                } else {
                    this.toonMaterial.uniforms.originalTexture.value = this.whiteTexture; // White texture as default
                    this.toonMaterial.uniforms.originalTexture.needsUpdate = true;
                }
            }
        });
    }

    disableToonShading() {
        if (!this.model) return;
        this.model.traverse((child) => {
            if (child.isMesh && this.toonMaterialBackups.has(child.uuid)) {
                child.material = this.toonMaterialBackups.get(child.uuid);
            }
        });
        this.toonMaterialBackups.clear();
    }

    createGlowMaterial() {
        return new THREE.ShaderMaterial({
            uniforms: {
                glowColor: { value: new THREE.Color(0x00ff00) },
                glowIntensity: { value: 1.5 },
                baseOpacity: { value: 0.2 }
            },
            vertexShader: /*glsl*/`
                varying vec3 vNormal;
                varying vec3 vWorldPosition;
                void main() {
                    vNormal = normalize(normalMatrix * normal);
                    vWorldPosition = (modelMatrix * vec4(position, 1.0)).xyz;
                    gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
                }
            `,
            fragmentShader: /*glsl*/`
                uniform vec3 glowColor;
                uniform float glowIntensity;
                uniform float baseOpacity;

                varying vec3 vNormal;
                varying vec3 vWorldPosition;

                void main() {
                    vec3 viewDir = normalize(cameraPosition - vWorldPosition);
                    float edgeFactor = 1.0 - abs(dot(vNormal, viewDir));
                    float glow = pow(edgeFactor, 2.0) * glowIntensity;
                    vec3 finalColor = glowColor * glow;
                    gl_FragColor = vec4(finalColor, baseOpacity + glow);
                }
            `,
            transparent: true
        });
    }


    selectMeshByName(name) {
        if (!name) {
            return null;
        }

        const exactMatch = this.meshParts.find((mesh) => mesh.name === name);
        const normalizedName = `${name}`.toLowerCase();
        const mesh = exactMatch || this.meshParts.find((entry) => (entry.name || '').toLowerCase() === normalizedName);

        if (!mesh) {
            return null;
        }

        this.selectMeshPartInSceneGraph(mesh, null, {
            force: true,
            source: 'api',
        });
        return mesh;
    }

    selectMeshByIndex(index) {
        const mesh = this.meshParts[index];
        if (!mesh) {
            return null;
        }

        this.selectMeshPartInSceneGraph(mesh, null, {
            force: true,
            source: 'api',
        });
        return mesh;
    }

    clearSelection(options = {}) {
        const { source = 'api', emitEvent = true } = options;
        this.clearSelectionState({
            source,
            emitEvent,
        });
    }

    selectMeshPartInSceneGraph(mesh, labelElement, options = {}) {
        const {
            force = false,
            source = 'ui',
            channel = 'scene-graph',
        } = options;
        if (!this.isSelectableMesh(mesh)) {
            return null;
        }

        if (!force && !this.isSelectionChannelEnabled(channel)) {
            return null;
        }

        const nextLabel = labelElement || this.getSceneGraphLabelForMesh(mesh);
        if (this.selectedMeshPart === mesh) {
            this.setSelectedSceneGraphLabel(nextLabel);
            this.updateSelectionHelpers();
            this.requestRender();
            return mesh;
        }

        this.selectedMeshPart = mesh;
        this.selectedMeshPartIndex = this.meshParts.indexOf(mesh);
        const partSelector = this.shadowRoot.querySelector('#texturePartSelector');
        if (this.selectedMeshPartIndex >= 0 && partSelector.value !== `${this.selectedMeshPartIndex}`) {
            partSelector.value = `${this.selectedMeshPartIndex}`;
        }
        this.populateMaterialSlotSelector(mesh, 0);

        this.setSelectedSceneGraphLabel(nextLabel);

        this.syncMaterialEditorControls();
        this.updateCameraActionButtons();
        this.updateSelectionHelpers();
        this.requestRender();

        this.emitSelectionChange(source);
        return mesh;
    }

    createExplodeSlider() {
        const editTabContent = this.shadowRoot.querySelector("#render-tab-content");
        if (!editTabContent) {
          console.warn("Could not find the 'Edit' tab to add the explode slider.");
          return;
        }
    
        // Prevent adding multiple sliders if a model is reloaded without discarding
        if (this.shadowRoot.querySelector("#explode-fieldset")) {
          return;
        }
    
        const fieldset = document.createElement("fieldset");
        fieldset.id = "explode-fieldset"; // For easy selection/removal later
        fieldset.style.marginTop = "0.5rem";
    
        const legend = document.createElement("legend");
        legend.style.fontSize = "0.8rem";
        legend.innerHTML = "<strong>Explode</strong>";
        fieldset.appendChild(legend);
    
        const sliderContainer = document.createElement("div");
        sliderContainer.style.display = "flex";
        sliderContainer.style.alignItems = "center";
        sliderContainer.style.justifyContent = "center";
        sliderContainer.style.margin = "5px 0";
    
        const label = document.createElement("span");
        label.textContent = "Amount:";
        label.style.marginRight = "10px";
        label.style.fontWeight = "bold";
    
        const slider = document.createElement("input");
        slider.type = "range";
        slider.min = "0";
        slider.max = "1";
        slider.step = "0.01";
        slider.value = "0";
        slider.style.width = "100%";
    
        slider.oninput = (event) => {
          const explodeAmount = parseFloat(event.target.value);
          this.applyExplodeEffect(explodeAmount);
        };
    
        sliderContainer.appendChild(label);
        sliderContainer.appendChild(slider);
        fieldset.appendChild(sliderContainer);
    
        // Add the new fieldset to the edit tab
        editTabContent.appendChild(fieldset);
    }
    
    applyExplodeEffect(explodeAmount) {
        if (!this.model || !this.modelCenter) return;
    
        // A multiplier to make the explosion visually significant.
        // Using half of the model's max dimension provides a good scale.
        const explosionFactor = this.modelMaxDim * 1.5;
    
        this.model.traverse((part) => {
          if (part.isMesh) {
            // The original position should have been stored in loadModel.
            if (!part.userData.originalPosition) {
              console.warn("Original position not found for part:", part.name, "- Storing now.");
              part.userData.originalPosition = part.position.clone();
            }
    
            const bbox = new THREE.Box3().setFromObject(part);
            const part_center = bbox.getCenter(new THREE.Vector3());
            
            // Direction is from the center of the whole model to the center of the part
            const direction = part_center.clone().sub(this.modelCenter).normalize();
            direction.x *= 2
            direction.z *= 2
    
            const originalPosition = part.userData.originalPosition;
            const offset = direction.multiplyScalar(explodeAmount * explosionFactor);
            const newPosition = originalPosition.clone().add(offset);
    
            // Apply the new calculated position
            part.position.copy(newPosition);
          }
        });
    }
}

customElements.define('simple-model-viewer', SimpleModelViewer);
export { SimpleModelViewer };
