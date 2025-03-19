import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';
import { OBJLoader } from 'three/addons/loaders/OBJLoader.js';
import { RGBELoader } from 'three/addons/loaders/RGBELoader.js';
import { DRACOLoader } from 'three/addons/loaders/DRACOLoader.js';
import { TransformControls } from 'three/addons/controls/TransformControls.js';

class SimpleModelViewer extends HTMLElement {
    constructor() {
        super();
        this.attachShadow({ mode: 'open' });
        this.shadowRoot.innerHTML = /*html*/`
            <style>
                :host {
                    display: block;
                    border-radius: 1px;
                    min-height: 300px;
                    background-color:rgba(200, 200, 200, 0.5);
                    font-family: Verdana, Geneva, Arial, sans-serif;
                    position: relative; /* Required for absolute positioning of panels */
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
                    margin: 5px;
                    position: absolute; /* Make controls container positioned relative to :host */
                    top: 0.5rem;
                    right: 0;
                    z-index: 1000; /* Ensure it's above canvas */
                }

                button {
                    background-color: #444444;
                    border: none;
                    color: white;
                    padding: 5px 10px;
                    border-radius: 2px;
                    cursor: pointer;
                    text-align: center;
                    text-decoration: none;
                    display: inline-block;
                    font-size: 0.8rem;
                    z-index: 1001;
                    margin-right: 0px;
                    margin-top: 0.1rem;
                    margin-bottom: 0.1rem;
                    min-width: 4rem;
                    width: 32.5%;
                }

                button:hover {
                    background-color: #3e8e41
                }

                button.toggled-off {
                    background-color: #3e8e41;
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
                    background-color: #444444;
                    border: none;
                    color: white;
                    padding: 5px 10px;
                    border-radius: 2px;
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
                    background-color: #3e8e41;
                }

                .transform-button:hover {
                    background-color: #3e8e41
                }

                .right-ui-panel { /* Renamed and unified panel */
                    position: absolute;
                    top: 0.5rem;
                    right: 0.5rem; /* Positioned to the right */
                    font-size: 0.7rem;
                    background-color: rgba(200, 200, 200, 0.5);
                    padding: 0.5rem;
                    border-radius: 5px;
                    display: flex;
                    flex-direction: column;
                    gap: 3px;
                    z-index: 1000;
                    max-width: 25rem;
                }

                .right-ui-panel label {
                    display: flex;
                    justify-content: space-between;
                    align-items: center;

                }

                .right-ui-panel input {
                    width: 3rem;
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
                    background-color: #444444;
                    height: 5px;
                    border-radius: 4px;
                }

                input[type="range"]::-moz-range-track {
                    background-color: #444444;
                    height: 5px;
                    border-radius: 4px;
                }

                input[type="range"]::-webkit-slider-thumb {
                    -webkit-appearance: none;
                    appearance: none;
                    background-color: #444444;
                    border: none;
                    height: 16px;
                    width: 16px;
                    border-radius: 50%;
                    margin-top: -5.5px;
                }

                input[type="range"]::-moz-range-thumb {
                    -moz-appearance: none;
                    appearance: none;
                    background-color: #444444;
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
                    width: 18rem;
                    height: 18rem;
                    border: 1px solid #ccc;
                    background-color: #eee;
                    margin-right: 5px;
                    display: inline-block;
                    vertical-align: middle;
                }

                ul {
                    left: 0;
                    padding-inline-start: 0.75rem;
                    font-size: 0.65rem;
                }

                .texture-button-group {
                    display: flex;
                    gap: 5px;
                    align-items: center;
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
                }

                .tab-button {
                    background-color: #aaaaaa;
                    border: none;
                    padding: 8px 16px;
                    cursor: pointer;
                    border-radius: 3px 3px 0 0;
                    font-size: 0.8rem;
                    margin-right: 2px;
                    color: black;
                    width: 8.3rem;
                }

                .tab-button.active {
                    background-color: rgba(200, 200, 200, 0); /* Active tab background */
                }

                .tab-button:hover {
                    background-color: #ccc;
                }

                .tab-content {
                    padding: 0.5rem;
                    border-radius: 0 0 5px 5px;
                    /* background-color: rgba(200, 200, 200, 0.5); Already set in .right-ui-panel */
                }

                fieldset {
                    max-width: 23rem;
                }
            </style>
            <div class="controls">
                <div class="right-ui-panel">
                    <button id="togglePanelBtn" style="width:100%"><i class="bi bi-caret-right"></i></button>
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
                                    <button id="setBgBtn1">Env1</button>
                                    <button id="setBgBtn2">Env2</button>
                                    <button id="setBgBtn3">Env3</button>
                                </fieldset>

                                <fieldset style="margin-top: 0.5rem;">
                                    <legend style="font-size: 0.8rem;"><strong>Util</strong></legend>
                                    <button id="autoRotateBtn">Auto-Rotate</button>
                                    <button id="screenshotBtn">Screenshot</button>
                                    <button id="discardModelBtn" style="background-color: red">Discard Model</button>
                                    <button id="runAnimationBtn" style="display: none; background-color: #149ddd">Run</button>
                                    <button id="pauseAnimationBtn" style="display: none; background-color: #777777">Pause</button>
                                    <div id="anim_description" style="display: none; margin-bottom: 0.5rem;">
                                        <strong>Actions:</strong>
                                    </div>
                                </fieldset>
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
                                    <label>FOV: <input type="number" id="cameraFov" step="1" value="50"></label>
                                    <label>Near: <input type="number" id="cameraNear" step="0.1" value="0.1"></label>
                                    <label>Far: <input type="number" id="cameraFar" step="100" value="1000"></label>
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
                                    <div style="display: none;">Scale: <input type="range" id="scale" style="width: 17rem; background-color: #d3d9de; color: #d3d9de;" min="1" max="20" step="0.1" value="8"></div>
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
                                    <legend style="font-size: 0.8rem;"><strong>Texture</strong></legend>
                                    <div>Roughness: <input type="range" id="roughness" style="font-size: 0.8rem; width: 17rem;" min="0" max="1" step="0.01" value="0.5"></div>
                                    <div>Metalness: <input type="range" id="metalness" style="font-size: 0.8rem; width: 17rem;" min="0" max="1" step="0.01" value="0.5"></div>
                                    <hr/>
                                    <div style="display: flex; align-items: center; gap: 10px; margin-bottom: 5px;">
                                        <label for="texturePartSelector" style="font-size: 0.8rem;">Part:</label>
                                        <select id="texturePartSelector" style="font-size: 0.8rem; max-width: 9rem;">
                                            <!-- Part options will be populated here -->
                                        </select>

                                        <label for="textureTypeSelector" style="font-size: 0.8rem;">Type:</label>
                                        <select id="textureTypeSelector" style="font-size: 0.8rem; max-width: 9rem;">
                                            <option value="map">Diffuse</option>
                                            <option value="roughnessMap">Roughness</option>
                                            <option value="metalnessMap">Metalness</option>
                                            <option value="normalMap">Normal</option>
                                            <option value="aoMap">AO</option>
                                            <option value="emissiveMap">Emissive</option>
                                            <!-- Texture type options -->
                                        </select>
                                    </div>

                                    <div style="display: flex; align-items: center; gap: 10px;">
                                        <div id="texturePreview" class="texture-preview">
                                            <!-- Texture preview will be displayed here -->
                                        </div>
                                        <div class="texture-button-group">
                                            <button id="replaceTextureBtn" style="padding: 3px 6px; font-size: 0.6rem; min-width: 3rem">Replace</button>
                                        </div>
                                    </div>
                                </fieldset>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
            <div id="canvas-container" style='text-align: center'>
                <div id="loadingProgressBar"></div>
                <div id="fileInputContainer" style="display: none;">
                    <input type="file" id="fileInput" accept=".glb,.gltf, .obj">
                    <p style="font-size: 0.8rem; margin-top: 5px;"><strong>Select a GLTF/GLB/OBJ file</strong></p>
                </div>
            </div>

            <!-- Hidden file inputs for texture replacement -->
            <input type="file" id="diffuseMapInput" style="display: none;" accept="image/*">
            <input type="file" id="roughnessMapInput" style="display: none;" accept="image/*">
            <input type="file" id="metalnessMapInput" style="display: none;" accept="image/*">
            <input type="file" id="normalMapInput" style="display: none;" accept="image/*">
            <input type="file" id="aoMapInput" style="display: none;" accept="image/*">
            <input type="file" id="emissiveMapInput" style="display: none;" accept="image/*">
        `;

        this.scene = new THREE.Scene();
        this.camera = new THREE.PerspectiveCamera(50, 1, 0.1, 1000);
        this.renderer = new THREE.WebGLRenderer({
            antialias: true,
            alpha: true,
            preserveDrawingBuffer: true // screenshot
        });
        this.renderer.setPixelRatio(window.devicePixelRatio);
        this.renderer.outputEncoding = THREE.sRGBEncoding;
        this.renderer.setClearColor(0xeeeeee, 1); // (light gray)
        this.renderer.shadowMap.enabled = true;
        this.renderer.toneMapping = THREE.ACESFilmicToneMapping;
        this.renderer.toneMappingExposure = 1;

        this.animationGeometry = null;
        this.animationMesh = null;
        this.tweenGroup = null;
        this.isIdleAnimationRunning = false;

        this.gridHelper = new THREE.GridHelper(10, 10, 0x888888, 0x444444);
        this.gridHelper.visible = false;
        this.scene.add(this.gridHelper);

        this.shadowRoot.querySelector('#canvas-container').appendChild(this.renderer.domElement);

        this.state = {
            lightsOn: true,
            viewMode: 'default', // 'default', 'diffuse', 'geometry', 'normal'
            wireframeInitialized: false,
            isWireframeOn: false,
            environment: null, // null, 'env1', 'env2', 'env3'
            isAnimationPlaying: false,
            wireframeInitialized: false,
            isWireframeOn: false,
            transformMode: 'none',
        };

        this.mixer = null;
        this.animationActions = [];
        this.currentAction = null;

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
        this.controls.addEventListener('change', () => this.updateControlPanel());

        this.textureLoader = new THREE.TextureLoader();
        this.whiteTexture = this.textureLoader.load('https://huggingface.co/spaces/hhhwan/custom_gs/resolve/main/glbs/white.jpg');
        this.whiteTexture.mapping = THREE.EquirectangularReflectionMapping;

        this.gradTexture = this.textureLoader.load('https://huggingface.co/spaces/hhhwan/custom_gs/resolve/main/glbs/gradient.jpg');
        this.gradTexture.mapping = THREE.EquirectangularReflectionMapping;

        this.dracoLoader = new DRACOLoader();
        this.objLoader = new OBJLoader();
        this.dracoLoader.setDecoderPath( 'https://www.gstatic.com/draco/versioned/decoders/1.5.7/' );
        // this.dracoLoader.setDecoderPath( './draco/' );
        this.gltfLoader = new GLTFLoader();
        this.gltfLoader.setDRACOLoader(this.dracoLoader);
        this.model = null;
        this.originalMaterials = {};
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
        this.standardMaterials = [];

        this.showLightHelpers = false; // Light Helper visiblity - default true, changed to true initially for better UX
        this.canAdjustRoughnessMetalness = false;
        this.meshParts = [];

        this.selectedSceneGraphLabel = null;
        this.selectedMeshPart = null;
        this.selectedMeshPartIndex = -1;
        this.glowMaterial = this.createGlowMaterial(); // Glow Material
        this.previousSelectedMeshPart = null; // Previous selected mesh part
        this.previousMeshPartOriginalMaterial = null;

        // TransformControls instance generation
        this.transformControls = new TransformControls(this.camera, this.renderer.domElement);
        this.transformControls.addEventListener('change', () => {
            // this.render(); // TransformControls render
            this.renderer.render(this.scene, this.camera);
            this.updateControlPanel(); // TransformControls update
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

        this.renderer.setAnimationLoop((time) => this.animate(time));
    }

    initIdleAnimation() {
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
    
        if (typeof TWEEN === 'undefined') {
            console.error('TWEEN is not defined. Ensure Tween.js is loaded before simple-model-viewer.js.');
            return;
        }
    
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
        this.resizeRenderer();
        if (!this.getAttribute('src')) {
            const fileInputContainer = this.shadowRoot.querySelector('#fileInputContainer');
            fileInputContainer.style.display = 'block';
        }
        this.animate(0);
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

    static get observedAttributes() {
        return [
            'src', // source for mesh file
            'auto-rotate', // auto-rotate option
            'angle-per-second', // animation angle per sec
            'camera-orbit',  // init camera orbit
            'hide-control-ui', // hide ui
            'light-off', // turn off basic light
            'no-pbr', // turn off light, default as diffuse mode
            'view-mode', // 'default', 'diffuse', 'geometry', 'normal'
            'ambient-light', // 0x color, intensity
            'direct-light', // x,y,z,intensity
        ];
    }

    attributeChangedCallback(name, oldValue, newValue) {
        if (name === 'src' && newValue) {
            this.loadModel(newValue, newValue);
            const fileInputContainer = this.shadowRoot.querySelector('#fileInputContainer');
            if (fileInputContainer) {
                fileInputContainer.style.display = 'none';
            }
        } else if (name === 'auto-rotate') {
            this.autoRotate = newValue !== null;
        } else if (name === 'angle-per-second') {
            this.anglePerSecond = parseFloat(newValue) || 30;
        } else if (name === 'camera-orbit') {
            this.setCameraOrbit(newValue);
        } else if (name === 'hide-control-ui') {
            const controlsDiv = this.shadowRoot.querySelector('.controls');
            if (newValue !== null) {
                controlsDiv.style.display = 'none';
            } else {
                controlsDiv.style.display = 'block';
            }
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
        this.updateViewModeButtons();
    }


    setAmbientLight(value) {
        const [color, intensity] = value.split(' ');
        const colorValue = parseInt(color, 16);
        const intensityValue = parseFloat(intensity);
        if (!isNaN(colorValue) && !isNaN(intensityValue)) {
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
                this.setBackground1();
                this.state.environment = 'env1';
                this.setLight(false); // Hide helpers when environment changes
            } else {
                this.setDefaultEnv();
                this.setDefaultMat();
                this.state.environment = null;
                this.state.viewMode = 'default';
                this.setLight(this.state.lightsOn);
            }
            this.updateEnvButtons();
        });

        this.shadowRoot.querySelector('#setBgBtn2').addEventListener('click', () => {
            if (this.state.environment !== 'env2') {
                this.setBackground2();
                this.state.environment = 'env2';
                this.setLight(false); // Hide helpers when environment changes
            } else {
                this.setDefaultEnv();
                this.setDefaultMat();
                this.state.environment = null;
                this.state.viewMode = 'default';
                this.setLight(this.state.lightsOn);
            }
            this.updateEnvButtons();
        });

        this.shadowRoot.querySelector('#setBgBtn3').addEventListener('click', () => {
            if (this.state.environment !== 'env3') {
                this.setBackground3();
                this.state.environment = 'env3';
                this.ambientLight.visible = false; // Hide helpers when environment changes
            } else {
                this.setDefaultEnv();
                this.setDefaultMat();
                this.state.environment = null;
                this.state.viewMode = 'default';
                this.setLight(this.state.lightsOn);
            }
            this.updateEnvButtons();
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
            rotateButton.textContent = this.autoRotate ? 'Auto-Rotate Off' : 'Auto-Rotate';
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
                controls.style.width = '25rem';
                content.style.display = `block`;
                button.innerHTML = `<i class="bi bi-caret-left"></i>`;
            } else {
                button.innerHTML = `<i class="bi bi-caret-right"></i>`;
                controls.style.width = '4rem';
                content.style.display = `none`;
            }
        });


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
        fileInput.addEventListener('change', (e) => {
            const file = e.target.files[0];
            if (file) {
                const fileName = file.name;
                this.loadModel(URL.createObjectURL(file), fileName);
                const fileInputContainer = this.shadowRoot.querySelector('#fileInputContainer');
                fileInputContainer.style.display = 'none';
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
            this.setBackgroundColor(event.target.value);
        });

        this.shadowRoot.querySelector('#screenshotBtn').addEventListener('click', () => {
            this.takeScreenshotToClipboard();
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
    }

    setTransformMode(mode) {
        const translateBtn = this.shadowRoot.querySelector('#translateBtn');
        const rotateBtn = this.shadowRoot.querySelector('#rotateBtn');

        if (this.model){
            if (this.state.transformMode === mode) {
                // If the same button is clicked again, deactivate TransformControls
                this.state.transformMode = null; // or 'none', or any value to indicate no active mode
                this.transformControls.detach();
                this.transformControls.visible = false;
                this.controls.enabled = true; // OrbitControls
                translateBtn.classList.remove('active');
                rotateBtn.classList.remove('active');
            } else {
                // Activate TransformControls for the selected mode
                this.state.transformMode = mode;
                this.transformControls.setMode(mode);
                this.transformControls.attach(this.model);
                this.transformControls.visible = true;
                this.controls.enabled = false; // OrbitControls 비활성화
                if (mode === 'translate') {
                    translateBtn.classList.add('active');
                    rotateBtn.classList.remove('active');
                } else if (mode === 'rotate') {
                    rotateBtn.classList.add('active');
                    translateBtn.classList.remove('active');
                }
            }
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
        // this.render();
    }

    initDiscardButton() {
        const discardButton = this.shadowRoot.querySelector('#discardModelBtn');
        discardButton.style.display = 'none';
    }


    updateCameraFov(fov) {
        this.camera.fov = fov;
        this.camera.updateProjectionMatrix();
        this.shadowRoot.querySelector('#cameraFov').value = this.camera.fov;
    }

    updateCameraNear(near) {
        this.camera.near = near;
        this.camera.updateProjectionMatrix();
        this.shadowRoot.querySelector('#cameraNear').value = this.camera.near;
    }

    updateCameraFar(far) {
        this.camera.far = far;
        this.camera.updateProjectionMatrix();
        this.shadowRoot.querySelector('#cameraFar').value = this.camera.far;
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
        const newLight = new THREE.DirectionalLight(0xffffff, 1);
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
        // if (this.directionalLights.length <= 1) {
        //     alert('At least one directional light is needed.');
        //     return;
        // }

        const lightToRemove = this.directionalLights[this.selectedDirectionalLightIndex];
        const helperToRemove = this.directionalLightHelpers[this.selectedDirectionalLightIndex];

        this.scene.remove(lightToRemove);
        this.scene.remove(helperToRemove);

        this.directionalLights.splice(this.selectedDirectionalLightIndex, 1);
        this.directionalLightHelpers.splice(this.selectedDirectionalLightIndex, 1);

        this.selectedDirectionalLightIndex = Math.max(0, this.selectedDirectionalLightIndex - 1);
        this.updateDirectionalLightUIValues();
        this.populateDirectionalLightList(); // Update the list after removal
    }

    populateDirectionalLightList() {
        const lightList = this.shadowRoot.querySelector('#directionalLightList');
        lightList.innerHTML = '';

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


    setBackgroundColor(color) {
        this.renderer.setClearColor(color, 1);
    }

    takeScreenshotToClipboard() {
        const canvas = this.renderer.domElement;

        canvas.toBlob(async (blob) => {
            if (!blob) {
                console.error("Failed to create blob from canvas");
                alert("Failed to create screenshot.");
                return;
            }

            try {
                await navigator.clipboard.write([
                    new ClipboardItem({
                        'image/png': blob
                    })
                ]);
                alert('Screenshot copied to clipboard!');
            } catch (err) {
                console.error('Failed to copy to clipboard:', err);
                alert('Screenshot to clipboard failed. Please check browser permissions or try again.');
            }
        }, 'image/png');
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
        }
    }

    showTexture() {
        if (this.model) {
            this.model.traverse((child) => {
                if (child.isMesh) {
                    child.material = new THREE.MeshBasicMaterial({ map: this.originalMaterials[child.uuid].map });
                    this.modifyMaterialForWireframe(child.material);
                    child.material.needsUpdate = true;
                }
            });
            this.renderer.render(this.scene, this.camera);
        }
    }

    showMesh() {
        if (this.model) {
            this.model.traverse((child) => {
                if (child.isMesh) {
                    child.material = new THREE.MeshStandardMaterial({
                        color: 0xffffff,
                        map: null,
                        envMap: this.gradTexture,
                        envMapIntensity: 1.0,
                        roughness: 1,
                        metalness: 1
                    });
                    this.modifyMaterialForWireframe(child.material);
                    child.material.needsUpdate = true;
                }
            });
            this.renderer.render(this.scene, this.camera);
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
        material.onBeforeCompile = (shader) => {
            shader.uniforms.uWireframe = { value: false };

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

            shader.fragmentShader = `
                uniform bool uWireframe;
                varying vec3 vBarycentric;
                ${shader.fragmentShader}
            `.replace(
                '#include <output_fragment>',
                `
                #include <output_fragment>
                if (uWireframe) {
                    vec3 bary = vBarycentric;
                    vec3 d = fwidth(bary);
                    vec3 a3 = smoothstep(vec3(0.0), d * 0.5, bary);
                    float edgeFactor = min(min(a3.x, a3.y), a3.z);
                    float wireframeAlpha = 1.0 - edgeFactor;
                    vec4 wireframeColor = vec4(0.6, 0.6, 0.6, 0.7); // #transparent grey
                    gl_FragColor.rgb = mix(gl_FragColor.rgb, wireframeColor.rgb, wireframeAlpha);
                gl_FragColor.a = mix(gl_FragColor.a, wireframeColor.a, wireframeAlpha);
                }
                `
            );

            material.userData.shader = shader;
            material.needsUpdate = true;
        };
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
                // this.modifyMaterialForWireframe(child.material);
            }
        });

        this.state.isWireframeOn = !this.state.isWireframeOn;

        this.model.traverse((child) => {
            if (child.isMesh && child.material.userData.shader) {
                child.material.userData.shader.uniforms.uWireframe.value = this.state.isWireframeOn;
                child.material.needsUpdate = true;
            }
        });

        const wireframeBtn = this.shadowRoot.querySelector('#wireframeBtn');
        // wireframeBtn.textContent = this.state.isWireframeOn ? 'Wireframe Off' : 'Wireframe';
        wireframeBtn.classList.toggle('toggled-off', this.state.isWireframeOn);
    }

    showNormal() {
        if (this.model) {
            this.model.traverse((child) => {
                if (child.isMesh) {
                    child.material = new THREE.MeshNormalMaterial();
                    child.material.needsUpdate = true;
                }
            });
            this.renderer.render(this.scene, this.camera);
        }
    }

    set_bg(url, rgbeLoader) {
        rgbeLoader.load(url, (texture) => {
            texture.minFilter = THREE.LinearFilter;
            texture.magFilter = THREE.LinearFilter;
            texture.mapping = THREE.EquirectangularReflectionMapping;
            this.scene.background = texture;
            this.scene.environment = texture;

            if (this.model) {
                this.model.traverse((child) => {
                    if (child.isMesh) {
                        const originalMaterial = this.originalMaterials[child.uuid];
                        const isStandardMaterial = originalMaterial instanceof THREE.MeshStandardMaterial;

                        child.material = new THREE.MeshStandardMaterial({
                            map: originalMaterial.map ? originalMaterial.map.clone() : null,
                            envMap: texture,
                            envMapIntensity: 1.0,
                            roughness: isStandardMaterial && originalMaterial.roughness !== undefined ? originalMaterial.roughness : 0.5,
                            metalness: isStandardMaterial && originalMaterial.metalness !== undefined ? originalMaterial.metalness : 0.5,
                            roughnessMap: isStandardMaterial && originalMaterial.roughnessMap !== undefined ? originalMaterial.roughnessMap : null,
                            metalnessMap: isStandardMaterial && originalMaterial.metalnessMap !== undefined ? originalMaterial.metalnessMap : null,
                            normalMap: originalMaterial.normalMap ? originalMaterial.normalMap : null,
                            emissiveMap: originalMaterial.emissiveMap ? originalMaterial.emissiveMap : null,
                        });
                        if (child.material.map) {
                            child.material.map.encoding = THREE.sRGBEncoding;
                        }
                        if (child.material.emissiveMap) {
                            child.material.emissiveMap.encoding = THREE.sRGBEncoding;
                        }
                        this.modifyMaterialForWireframe(child.material);
                        child.material.needsUpdate = true;
                    }
                });

                // UI update
                let roughnessSum = 0, metalnessSum = 0, materialCount = 0;
                this.standardMaterials.forEach(material => {
                    roughnessSum += material.roughness || 0.5;
                    metalnessSum += material.metalness || 0.5;
                    materialCount++;
                });
                const initialRoughness = materialCount > 0 ? roughnessSum / materialCount : 0.5;
                const initialMetalness = materialCount > 0 ? metalnessSum / materialCount : 0.5;

                const roughnessInput = this.shadowRoot.querySelector('#roughness');
                const metalnessInput = this.shadowRoot.querySelector('#metalness');
                if (roughnessInput) roughnessInput.value = initialRoughness;
                if (metalnessInput) metalnessInput.value = initialMetalness;

                roughnessInput.disabled = !this.canAdjustRoughnessMetalness;
                metalnessInput.disabled = !this.canAdjustRoughnessMetalness;
            }
            this.renderer.render(this.scene, this.camera);
        }, undefined, (err) => {
            console.error('Skybox err:', err);
            alert('Cannot load Skybox Image');
        });
        this.state.viewMode = 'default';
        this.updateViewModeButtons();
    }

    setBackground1() {
        const url = 'https://huggingface.co/spaces/hhhwan/custom_gs/resolve/main/glbs/spruit_sunrise_1k_HDR.hdr';
        const rgbeLoader = new RGBELoader();
        this.set_bg(url, rgbeLoader);
    }

    setBackground2() {
        const url = 'https://huggingface.co/spaces/hhhwan/custom_gs/resolve/main/glbs/aircraft_workshop_01_1k.hdr';
        const rgbeLoader = new RGBELoader();
        this.set_bg(url, rgbeLoader);
    }

    setBackground3() {
        const url = 'https://huggingface.co/spaces/hhhwan/custom_gs/resolve/main/glbs/lebombo_1k.hdr';
        const rgbeLoader = new RGBELoader();
        this.set_bg(url, rgbeLoader);
    }

    setDefaultEnv() {
        this.scene.background = null;
        this.scene.environment = null;
    }

    setDefaultMat() {
        if (this.model) {
            this.model.traverse((child) => {
                if (child.isMesh) {
                    const originalMaterial = this.originalMaterials[child.uuid];

                    if (this.noPBR) {
                        child.material = new THREE.MeshBasicMaterial({ map: this.originalMaterials[child.uuid].map });
                    } else {
                        child.material = originalMaterial.clone();
                    }
                    this.modifyMaterialForWireframe(child.material);
                    child.material.needsUpdate = true;

                    if (this.canAdjustRoughnessMetalness) {
                        const roughnessInput = this.shadowRoot.querySelector('#roughness');
                        const metalnessInput = this.shadowRoot.querySelector('#metalness');
                        roughnessInput.value = originalMaterial.roughness || 0.5;
                        metalnessInput.value = originalMaterial.metalness || 0.5;
                    }
                }
            });
        }
    }

    setCameraOrbit(value) {
        const [x, y, z] = value.split(' ').map(parseFloat);
        if (!isNaN(x) && !isNaN(y) && !isNaN(z)) {
            this.camera.position.set(x, y, z);
            this.camera.lookAt(0, 0, 0);
        }
    }

    discardModel() {
        if (this.model) {
            this.scene.remove(this.model);

            // dispose
            this.model.traverse((child) => {
                if (child.isMesh) {
                    child.geometry.dispose();
                    if (child.material) {
                        if (Array.isArray(child.material)) {
                            child.material.forEach(mat => {
                                mat.dispose();
                            });
                        } else {
                            child.material.dispose();
                        }
                    }
                }
            });

            this.transformControls.detach();

            this.model = null;
            this.shadowRoot.querySelector('#modelInfo').innerHTML = `<strong>[Model Info]</strong> loading...`;

            const fileInputContainer = this.shadowRoot.querySelector('#fileInputContainer');
            fileInputContainer.style.display = 'block';

            const discardButton = this.shadowRoot.querySelector('#discardModelBtn');
            discardButton.style.display = 'none';

            this.shadowRoot.querySelector('#posX').value = 0;
            this.shadowRoot.querySelector('#posY').value = 0;
            this.shadowRoot.querySelector('#posZ').value = 0;
            this.shadowRoot.querySelector('#rotX').value = 0;
            this.shadowRoot.querySelector('#rotY').value = 0;
            this.shadowRoot.querySelector('#rotZ').value = 0;
            this.shadowRoot.querySelector('#scale').value = 8;
            this.shadowRoot.querySelector('#roughness').value = 0.5;
            this.shadowRoot.querySelector('#metalness').value = 0.5;

            const sceneGraphTreeUI = this.shadowRoot.querySelector('#sceneGraphTree');
            sceneGraphTreeUI.innerHTML = ''; // scene graph ui init

            const partSelector = this.shadowRoot.querySelector('#texturePartSelector');
            partSelector.innerHTML = ''; // part selector init
            const previewElement = this.shadowRoot.querySelector('#texturePreview');
            previewElement.textContent = ''; // preview init
            previewElement.style.backgroundImage = '';

            this.selectedSceneGraphLabel = null;
            this.selectedMeshPart = null;
            this.selectedMeshPartIndex = -1;

            if (this.mixer) {
                this.mixer.stopAllAction();
                this.mixer = null;
                const animationSelector = this.shadowRoot.querySelector('#animationSelector');
                if (animationSelector) {
                    animationSelector.remove();
                }
            }
            if (this.animationActions) {
                this.animationActions.forEach(action => action.stop());
                this.animationActions = [];
            }
            if (this.currentAction) {
                this.currentAction.stop();
                this.currentAction = null;
            }

            this.updateAnimationButtons();
            this.shadowRoot.querySelector('#anim_description').style.display = 'none';

            this.renderer.render(this.scene, this.camera);
        }
    }

    loadModel(url, fileName) {
        const progressBar = this.shadowRoot.querySelector('#loadingProgressBar');
        progressBar.style.display = 'block';
        progressBar.style.width = '0%';

        if (this.mixer) {
            this.mixer.stopAllAction();
            this.mixer = null;
        }
        this.animationActions = [];
        this.currentAction = null;
        const existingSelector = this.shadowRoot.querySelector('#animationSelector');
        if (existingSelector) {
            existingSelector.remove();
        }

        const fileExtension = fileName.split('.').pop().toLowerCase();
        const loader = fileExtension === 'obj' ? this.objLoader : this.gltfLoader;


        loader.load(url, (object) => {
            if (this.model) {
                this.scene.remove(this.model);
            }
            this.model = fileExtension === 'obj' ? object : object.scene;

            // this.model = gltf.scene;
            const box = new THREE.Box3().setFromObject(this.model);
            const size = box.getSize(new THREE.Vector3());
            const maxDim = Math.max(size.x, size.y, size.z);
            let scaleFactor = 1;

            if (maxDim > 0) {
                const targetSize = 10;
                scaleFactor = targetSize / maxDim;
            }

            this.model.scale.set(scaleFactor, scaleFactor, scaleFactor);
            this.modelSize = scaleFactor * this.modelSize;
            this.shadowRoot.querySelector('#scale').value = this.modelSize;

            const updatedBox  = new THREE.Box3().setFromObject(this.model);
            const center = updatedBox.getCenter(new THREE.Vector3());
            this.model.position.sub(center);

            if (fileExtension !== 'obj' && object.animations.length > 0) {
                this.mixer = new THREE.AnimationMixer(this.model);
                this.animationActions = object.animations.map(clip => this.mixer.clipAction(clip));

                this.shadowRoot.querySelector('#anim_description').style.display = 'block';

                const animationSelector = document.createElement('select');
                animationSelector.id = 'animationSelector';
                animationSelector.style.width = '100%';


                const noneOption = document.createElement('option');
                noneOption.value = 'none';
                noneOption.textContent = 'None';
                animationSelector.appendChild(noneOption);

                object.animations.forEach((clip, index) => {
                    const option = document.createElement('option');
                    option.value = index.toString();
                    option.textContent = clip.name || `Animation ${index + 1}`;
                    animationSelector.appendChild(option);
                });

                animationSelector.value = 'none';
                this.currentAction = null;

                animationSelector.addEventListener('change', (event) => {
                    const value = event.target.value;
                    if (this.currentAction) {
                        this.currentAction.stop();
                        this.state.isAnimationPlaying = false;
                    }
                    if (value !== 'none') {
                        const index = parseInt(value, 10);
                        this.currentAction = this.animationActions[index];
                        this.currentAction.play();
                        this.state.isAnimationPlaying = true;
                    } else {
                        this.currentAction = null;
                    }
                    this.updateAnimationButtons();
                });

                const utilFieldset = this.shadowRoot.querySelector('#render-tab-content fieldset:nth-of-type(4)');
                utilFieldset.appendChild(animationSelector);

                this.shadowRoot.querySelector('#pauseAnimationBtn').disabled = false;
                this.shadowRoot.querySelector('#runAnimationBtn').disabled = false;
                this.updateAnimationButtons();
            }

            const updatedsize = updatedBox.getSize(new THREE.Vector3());
            const updatedMaxDim = Math.max(updatedsize.x, updatedsize.y, updatedsize.z);
            const fov = this.camera.fov * (Math.PI / 180);
            let cameraZ = Math.abs(updatedMaxDim / 2 / Math.tan(fov / 2));

            this.gridHelper.position.set(center.x, - (updatedsize.y/2), center.z);

            const cameraOrbit = this.getAttribute('camera-orbit');
            if (cameraOrbit) {
                this.setCameraOrbit(cameraOrbit);
            } else {
                this.camera.position.set(0, 0, cameraZ * 1.5);
                this.camera.lookAt(0, 0, 0);
            }

            const sceneGraphTreeUI = this.shadowRoot.querySelector('#sceneGraphTree');
            sceneGraphTreeUI.innerHTML = ''; // Clear the scene graph tree
            this.generateSceneGraphTree(this.model, sceneGraphTreeUI);

            let vertexCount = 0, faceCount = 0;
            this.standardMaterials = [];
            this.meshParts = [];
            this.meshPartTextureInfo = [];


            // Mesh Parts and Textures
            this.model.traverse((child) => {
                if (child.isMesh && child.geometry) {
                    vertexCount += child.geometry.attributes.position.count;
                    faceCount += child.geometry.index ? child.geometry.index.count / 3 : child.geometry.attributes.position.count / 3;

                    this.modifyMaterialForWireframe(child.material); // Wireframe overlay
                    this.originalMaterials[child.uuid] = child.material.clone();
                    this.meshParts.push(child);

                    if (child.material instanceof THREE.MeshStandardMaterial) {
                        this.standardMaterials.push(child.material);
                        const material = child.material;

                        if (material.map) {
                            material.map.encoding = THREE.sRGBEncoding;
                        }
                        if (material.emissiveMap) {
                            material.emissiveMap.encoding = THREE.sRGBEncoding;
                        }
                        if (material.metallicRoughnessMap) {
                            material.metallicRoughnessMap.encoding = THREE.LinearEncoding;
                        }
                        if (material.normalMap) {
                            material.normalMap.encoding = THREE.LinearEncoding;
                        }
                        if (material.aoMap) {
                            material.aoMap.encoding = THREE.LinearEncoding;
                            if (child.geometry.attributes.uv && !child.geometry.attributes.uv2) {
                                child.geometry.setAttribute('uv2', child.geometry.attributes.uv);
                            }
                        }
                        material.needsUpdate = true;
                    } else {
                        child.material = new THREE.MeshStandardMaterial({
                            map: child.material.map,
                            roughness: child.material.roughness !== undefined ? child.material.roughness : 0.5,
                            metalness: child.material.metalness !== undefined ? child.material.metalness : 0.5
                        });
                        if (child.material.map) {
                            child.material.map.encoding = THREE.sRGBEncoding;
                            child.material.map.flipY = false;
                        }
                        child.material.needsUpdate = true;
                        this.standardMaterials.push(child.material);
                    }

                }
            });

            this.populateTextureMapSelector();

            // Roughness/Metalness Adjustable
            this.canAdjustRoughnessMetalness = this.meshParts.length === 1 &&
                !this.standardMaterials[0].roughnessMap &&
                !this.standardMaterials[0].metalnessMap;
            console.log('Adjustable:', this.canAdjustRoughnessMetalness);

            const roughnessInput = this.shadowRoot.querySelector('#roughness');
            const metalnessInput = this.shadowRoot.querySelector('#metalness');
            if (this.canAdjustRoughnessMetalness) {
                roughnessInput.disabled = false;
                metalnessInput.disabled = false;
                roughnessInput.value = this.standardMaterials[0].roughness || 0.5;
                metalnessInput.value = this.standardMaterials[0].metalness || 0.5;

                roughnessInput.oninput = () => this.updateMaterialProperties();
                metalnessInput.oninput = () => this.updateMaterialProperties();
            } else {
                roughnessInput.disabled = true;
                metalnessInput.disabled = true;
                roughnessInput.value = 0.5;
                metalnessInput.value = 0.5;
            }

            this.meshParts.forEach((mesh, index) => {
                // Texture Map Controls UI
                const partTextureInfo = {
                    meshPartIndex: index,
                    diffuseMap: mesh.material.map,
                    roughnessMap: mesh.material.roughnessMap,
                    metalnessMap: mesh.material.metalnessMap,
                    normalMap: mesh.material.normalMap,
                    aoMap: mesh.material.aoMap,
                    emissiveMap: mesh.material.emissiveMap,
                };
                this.meshPartTextureInfo.push(partTextureInfo);
            });

            this.shadowRoot.querySelector('#modelInfo').innerHTML = `<strong>[Model Info]</strong> Vertices: ${vertexCount}, Faces: ${faceCount}`;
            this.scene.add(this.model);

            this.updateControlPanel();
            this.renderMode();
            this.updateLightsButtonUI();
            progressBar.style.display = 'none';
            const discardButton = this.shadowRoot.querySelector('#discardModelBtn');
            discardButton.style.display = 'inline-block';
        }, (xhr) => {
            if (xhr.lengthComputable) {
                const percentComplete = xhr.loaded / xhr.total * 100;
                progressBar.style.width = `${percentComplete}%`;
            }
        }, (error) => {
            console.error('Loading Error:', error);
        });
    }

    runAnimation() {
        if (this.currentAction) {
            this.currentAction.paused = false;
            this.state.isAnimationPlaying = true;
            this.updateAnimationButtons();
        }
    }

    pauseAnimation() {
        if (this.currentAction && this.state.isAnimationPlaying) {
            this.currentAction.paused = true;
            this.state.isAnimationPlaying = false;
            this.updateAnimationButtons();
        }
    }

    updateAnimationButtons() {
        if (this.currentAction) {
            // this.shadowRoot.querySelector('#runAnimationBtn').disabled = this.state.isAnimationPlaying;
            // this.shadowRoot.querySelector('#pauseAnimationBtn').disabled = !this.state.isAnimationPlaying;

            this.shadowRoot.querySelector('#runAnimationBtn').style.display = this.state.isAnimationPlaying ? 'none' : 'inline-block';
            this.shadowRoot.querySelector('#pauseAnimationBtn').style.display = this.state.isAnimationPlaying ? 'inline-block' : 'none';
        } else {
            // this.shadowRoot.querySelector('#runAnimationBtn').disabled = true;
            // this.shadowRoot.querySelector('#pauseAnimationBtn').disabled = true;

            this.shadowRoot.querySelector('#runAnimationBtn').style.display = 'none';
            this.shadowRoot.querySelector('#pauseAnimationBtn').style.display = 'none';
        }
    }

    initTextureMapUI() {
        const diffuseMapInput = this.shadowRoot.querySelector('#diffuseMapInput');
        const roughnessMapInput = this.shadowRoot.querySelector('#roughnessMapInput');
        const metalnessMapInput = this.shadowRoot.querySelector('#metalnessMapInput');
        const normalMapInput = this.shadowRoot.querySelector('#normalMapInput');
        const AOMapInput = this.shadowRoot.querySelector('#aoMapInput');
        const emissiveMapInput = this.shadowRoot.querySelector('#emissiveMapInput');
        const partSelector = this.shadowRoot.querySelector('#texturePartSelector');

        diffuseMapInput.addEventListener('change', (e) => this.handleTextureFileChange(e, 'map'));
        roughnessMapInput.addEventListener('change', (e) => this.handleTextureFileChange(e, 'roughnessMap'));
        metalnessMapInput.addEventListener('change', (e) => this.handleTextureFileChange(e, 'metalnessMap'));
        normalMapInput.addEventListener('change', (e) => this.handleTextureFileChange(e, 'normalMap'));
        AOMapInput.addEventListener('change', (e) => this.handleTextureFileChange(e, 'aoMap'));
        emissiveMapInput.addEventListener('change', (e) => this.handleTextureFileChange(e, 'emissiveMap'));

        partSelector.addEventListener('change', () => {
            const selectedPartIndex = parseInt(partSelector.value);
            if (!isNaN(selectedPartIndex) && selectedPartIndex >= 0 && selectedPartIndex < this.meshParts.length) {
                const selectedMeshPart = this.meshParts[selectedPartIndex];
                this.selectMeshPartInSceneGraph(selectedMeshPart, null);
            }
        });
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
            texture.encoding = THREE.sRGBEncoding;
            texture.flipY = false;

            const selectedMeshPartIndex = parseInt(fileInput.dataset.meshPartIndex);
            if (isNaN(selectedMeshPartIndex)) {
                console.error("Mesh part index is not set on the file input.");
                return;
            }

            const mesh = this.meshParts[selectedMeshPartIndex];
            if (!mesh || !mesh.material) {
                console.error("Mesh or material not found for index:", selectedMeshPartIndex);
                return;
            }

            if (mesh.material.isShared) {
                mesh.material = mesh.material.clone();
            }

            mesh.material[mapType] = texture;
            this.originalMaterials[mesh.uuid][mapType] = texture.clone();
            mesh.material.needsUpdate = true;

            this.updateTextureMapDisplay();
            this.renderer.render(this.scene, this.camera);
        }, undefined, (error) => {
            console.error('Texture loading error:', error);
            alert('Failed to load texture.');
        });
    }

    populateTextureMapSelector() {
        const partSelector = this.shadowRoot.querySelector('#texturePartSelector');
        const typeSelector = this.shadowRoot.querySelector('#textureTypeSelector');
        const replaceTextureButton = this.shadowRoot.querySelector('#replaceTextureBtn');

        partSelector.innerHTML = '';
        this.meshParts.forEach((mesh, index) => {
            const option = document.createElement('option');
            option.value = index;
            option.textContent = mesh.name || `Part ${index + 1}`;
            partSelector.appendChild(option);
        });

        partSelector.selectedIndex = 0;

        partSelector.addEventListener('change', () => {
            this.updateTextureMapDisplay(); //
        });

        typeSelector.addEventListener('change', () => {
            this.updateTextureMapDisplay(); //
        });

        replaceTextureButton.addEventListener('click', () => {
            const selectedType = typeSelector.value;
            let fileInputId = '';
            if (selectedType === 'map') fileInputId = 'diffuseMapInput';
            else if (selectedType === 'roughnessMap') fileInputId = 'roughnessMapInput';
            else if (selectedType === 'metalnessMap') fileInputId = 'metalnessMapInput';
            else if (selectedType === 'normalMap') fileInputId = 'normalMapInput';
            else if (selectedType === 'aoMap') fileInputId = 'aoMapInput';
            else if (selectedType === 'emissiveMap') fileInputId = 'emissiveMapInput';

            if (fileInputId) {
                const fileInput = this.shadowRoot.querySelector(`#${fileInputId}`);
                const selectedPartIndex = parseInt(partSelector.value);
                fileInput.dataset.meshPartIndex = selectedPartIndex;
                fileInput.click();
            }
        });

        this.updateTextureMapDisplay();
    }

    updateTextureMapDisplay() {
        const partSelector = this.shadowRoot.querySelector('#texturePartSelector');
        const typeSelector = this.shadowRoot.querySelector('#textureTypeSelector');
        const previewElement = this.shadowRoot.querySelector('#texturePreview');
        const selectedPartIndex = parseInt(partSelector.value);
        const selectedType = typeSelector.value;

        if (isNaN(selectedPartIndex) || selectedPartIndex < 0 || selectedPartIndex >= this.meshParts.length) {
            console.error('Invalid selected part index:', selectedPartIndex);
            previewElement.textContent = 'Error';
            previewElement.style.backgroundImage = '';
            return;
        }

        const selectedMesh = this.meshParts[selectedPartIndex];
        const selectedMaterial = this.originalMaterials[selectedMesh.uuid];
        // const selectedMaterial = selectedMesh.material;
        let selectedTexture = selectedMaterial[selectedType];

        if (selectedTexture) {
            if (selectedTexture.image instanceof ImageBitmap) {
                this.setImageBitmapPreview(selectedTexture.image, previewElement);
            } else if (selectedTexture.image) {
                const imageSource = selectedTexture.image.currentSrc || selectedTexture.image.src;
                previewElement.style.backgroundImage = `url(${imageSource})`;
                previewElement.style.backgroundSize = 'cover';
                previewElement.textContent = '';
            } else {
                previewElement.textContent = 'Preview Unavailable';
                previewElement.style.backgroundImage = '';
                previewElement.style.lineHeight = '150px';
                previewElement.style.textAlign = 'center';
            }
        } else {
            previewElement.textContent = 'None';
            previewElement.style.backgroundImage = '';
            previewElement.style.lineHeight = '150px';
            previewElement.style.textAlign = 'center';
        }
    }

    setImageBitmapPreview(imageBitmap, previewElement) {
        const canvas = document.createElement('canvas');
        canvas.width = 1024;  // Match preview size
        canvas.height = 1024;
        const ctx = canvas.getContext('2d');

        if (!ctx) {
            previewElement.textContent = 'Cannot Preview';
            previewElement.style.lineHeight = '1024';
            previewElement.style.textAlign = 'center';
            console.error('Canvas context is null, cannot generate ImageBitmap preview.');
            return;
        }

        try {
            ctx.drawImage(imageBitmap, 0, 0, 1024, 1024);
            previewElement.innerHTML = ''; //
            previewElement.appendChild(canvas); //

        } catch (error) {
            console.error('Error drawing ImageBitmap on canvas:', error);
            previewElement.textContent = 'Preview Error';
            previewElement.style.lineHeight = '1024px';
            previewElement.style.textAlign = 'center';
        }
    }

    updateMaterialProperties() {
        if (this.canAdjustRoughnessMetalness) {
            const roughnessValue = parseFloat(this.shadowRoot.querySelector('#roughness').value);
            const metalnessValue = parseFloat(this.shadowRoot.querySelector('#metalness').value);

            if (this.model) {
                this.model.traverse((child) => {
                    if (child.isMesh) {
                        const originalMaterial = this.originalMaterials[child.uuid];

                        child.material = originalMaterial.clone();
                        child.material.roughness = roughnessValue;
                        child.material.metalness = metalnessValue;
                        child.material.needsUpdate = true;
                    }
                });
            }
            this.renderer.render(this.scene, this.camera);
        }
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

            checkbox.addEventListener('change', (e) => {
                child.visible = e.target.checked;
                this.renderer.render(this.scene, this.camera);
            });


            let name = child.name || child.type;
            if (name === '') name = 'unnamed';
            const nameSpan = document.createElement('span');
            nameSpan.textContent = name;
            label.appendChild(nameSpan);
            label.appendChild(checkbox);

            label.addEventListener('click', (event) => {
                event.stopPropagation();
                this.selectMeshPartInSceneGraph(child, label);
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

        // 모델이 없을 때만 idle 애니메이션 실행
        if (!this.model) {
            if (!this.isIdleAnimationRunning) {
                this.initIdleAnimation();
                TWEEN.update();
            }
            this.tweenGroup.update(time); // Use tweenGroup.update(time)
            if (this.animationMesh && this.isIdleAnimationRunning) {
                const rotationSpeed = Math.PI / 6; // 초당 30도 회전 (라디안)
                this.animationMesh.rotation.y += rotationSpeed * deltaTime;
            }

        } else if (this.isIdleAnimationRunning) {
            this.scene.remove(this.animationMesh);
            this.animationGeometry.dispose();
            this.animationMesh = null;
            this.tweenGroup = null;
            this.isIdleAnimationRunning = false;
        }

        this.controls.update();
        this.renderer.render(this.scene, this.camera);
    }

    resizeRenderer() {
        const host = this.shadowRoot.host;
        const metaDiv = this.shadowRoot.querySelector('#meta');
        const metaHeight = this.shadowRoot.querySelector('.tab-buttons').offsetHeight + this.shadowRoot.querySelector('#meta').offsetHeight; // consider tab height
        const width = host.clientWidth;
        const height = host.clientHeight;

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
                this.originalMaterials[child.uuid] = child.material;
                child.material = this.toonMaterial;
                if (this.originalMaterials[child.uuid].map) {
                    this.toonMaterial.uniforms.originalTexture.value = this.originalMaterials[child.uuid].map;
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
            if (child.isMesh && this.originalMaterials[child.uuid]) {
                child.material = this.originalMaterials[child.uuid];
            }
        });
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


    selectMeshPartInSceneGraph(mesh, labelElement) {
        if (this.selectedMeshPart === mesh) {
            return;
        }

        if (this.glowTimeoutId) {
            clearTimeout(this.glowTimeoutId);
            this.glowTimeoutId = null;
        }
        if (this.previousSelectedMeshPart && this.previousMeshPartOriginalMaterial) {
            this.previousSelectedMeshPart.material = this.previousMeshPartOriginalMaterial;
            this.previousSelectedMeshPart.material.needsUpdate = true;
        }

        this.selectedMeshPart = mesh;
        this.selectedMeshPartIndex = this.meshParts.indexOf(mesh);

        if (labelElement) {
            if (this.selectedSceneGraphLabel) {
                this.selectedSceneGraphLabel.classList.remove('selected');
            }
            this.selectedSceneGraphLabel = labelElement;
            this.selectedSceneGraphLabel.classList.add('selected');
        }

        this.previousMeshPartOriginalMaterial = mesh.material;
        this.previousSelectedMeshPart = mesh;
        mesh.material = this.glowMaterial;
        mesh.material.needsUpdate = true;

        this.updateTextureMapDisplay();
        this.renderer.render(this.scene, this.camera);

        this.glowTimeoutId = setTimeout(() => {
            if (this.selectedMeshPart === mesh) {
                mesh.material = this.previousMeshPartOriginalMaterial; // original mat
                if (mesh.material) {
                    mesh.material.needsUpdate = true;
                }
                this.previousMeshPartOriginalMaterial = null;
                this.previousSelectedMeshPart = null;
                this.glowTimeoutId = null;
                this.renderer.render(this.scene, this.camera);
            }
        }, 5000); // 1000ms = 1sec
    }
}

customElements.define('simple-model-viewer', SimpleModelViewer);
export { SimpleModelViewer };