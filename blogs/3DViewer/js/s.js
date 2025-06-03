import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';
import { OBJLoader } from 'three/addons/loaders/OBJLoader.js';
import { RGBELoader } from 'three/addons/loaders/RGBELoader.js';
import { DRACOLoader } from 'three/addons/loaders/DRACOLoader.js';
import { TransformControls } from 'three/addons/controls/TransformControls.js';
// If using Tween.js, make sure it's loaded globally or imported
// import * as TWEEN from '@tweenjs/tween.js'; // Example import

// Simple debounce function
function debounce(func, wait, immediate) {
    var timeout;
    return function() {
        var context = this, args = arguments;
        var later = function() {
            timeout = null;
            if (!immediate) func.apply(context, args);
        };
        var callNow = immediate && !timeout;
        clearTimeout(timeout);
        timeout = setTimeout(later, wait);
        if (callNow) func.apply(context, args);
    };
};


class SimpleModelViewer extends HTMLElement {
    constructor() {
        super();
        this.attachShadow({ mode: 'open' });
        this.shadowRoot.innerHTML = /*html*/`
            <style>
                /* ... (Existing CSS) ... */
                :host([fullscreen]) #canvas-container {
                    position: fixed;
                    top: 0;
                    left: 0;
                    width: 100vw; /* Use vw/vh for true fullscreen */
                    height: 100vh;
                    z-index: 9999; /* Ensure it's on top */
                }
                :host([fullscreen]) .controls { /* Hide controls in fullscreen maybe? */
                   /* display: none; */
                   /* Or adjust position/opacity */
                   z-index: 10000; /* Keep controls on top if visible */
                }
                #fullscreenBtn {
                    /* Style your fullscreen button */
                    margin-left: 5px; /* Example */
                }
                 #errorDisplay {
                    position: absolute;
                    bottom: 10px;
                    left: 10px;
                    background-color: rgba(255, 0, 0, 0.7);
                    color: white;
                    padding: 5px 10px;
                    border-radius: 3px;
                    font-size: 0.8rem;
                    z-index: 1001;
                    display: none; /* Hidden by default */
                 }
            </style>
            <div class="controls">
                 <div class="right-ui-panel">
                    <button id="togglePanelBtn" style="width:100%" title="Toggle Panel"><i class="bi bi-caret-right"></i></button>
                    <div id="panelContent" style="display: none;">
                        <!-- ... (Tabs: Render, Control, Edit) ... -->
                         <div id="render-tab-content" class="tab-content" style="display: block;">
                             <!-- ... -->
                            <fieldset style="margin-top: 0.5rem;">
                                <legend style="font-size: 0.8rem;"><strong>Util</strong></legend>
                                <!-- ... other buttons ... -->
                                <button id="fullscreenBtn" title="Toggle Fullscreen (F)">
                                     <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" fill="currentColor" class="bi bi-fullscreen" viewBox="0 0 16 16">
                                        <path d="M1.5 1a.5.5 0 0 0-.5.5v4a.5.5 0 0 1-1 0v-4A1.5 1.5 0 0 1 1.5 0h4a.5.5 0 0 1 0 1zM10 .5a.5.5 0 0 1 .5-.5h4A1.5 1.5 0 0 1 16 1.5v4a.5.5 0 0 1-1 0v-4a.5.5 0 0 0-.5-.5h-4a.5.5 0 0 1-.5-.5M.5 10a.5.5 0 0 1 .5.5v4a.5.5 0 0 0 .5.5h4a.5.5 0 0 1 0 1h-4A1.5 1.5 0 0 1 0 14.5v-4a.5.5 0 0 1 .5-.5m15 0a.5.5 0 0 1 .5.5v4a1.5 1.5 0 0 1-1.5 1.5h-4a.5.5 0 0 1 0-1h4a.5.5 0 0 0 .5-.5v-4a.5.5 0 0 1 .5-.5"/>
                                     </svg>
                                </button>
                                <!-- ... other buttons ... -->
                            </fieldset>
                            <!-- ... -->
                         </div>
                         <!-- ... (Other Tabs) ... -->
                    </div>
                 </div>
            </div>
            <div id="canvas-container" style='text-align: center'>
                <div id="loadingProgressBar"></div>
                 <div id="errorDisplay"></div> {/* Error message area */}
                <div id="fileInputContainer" style="display: none;">
                    <!-- ... -->
                </div>
            </div>
            <!-- ... (Video Modal, Hidden Inputs) ... -->
        `;

        this.scene = new THREE.Scene();
        this.camera = new THREE.PerspectiveCamera(50, 1, 0.1, 1000); // Aspect ratio will be set in resize
        this.renderer = new THREE.WebGLRenderer({
            antialias: true,
            alpha: true, // Allows transparent background if CSS is set
            preserveDrawingBuffer: true // Needed for screenshots
        });
        this.renderer.setPixelRatio(window.devicePixelRatio);
        this.renderer.outputEncoding = THREE.sRGBEncoding; // Correct color space
        this.renderer.toneMapping = THREE.ACESFilmicToneMapping;
        this.renderer.toneMappingExposure = 1;
        this.renderer.shadowMap.enabled = true;
        // this.renderer.shadowMap.type = THREE.PCFSoftShadowMap; // Optional: softer shadows

        this.renderer.setClearColor(0xeeeeee, 1); // Default background

        this.shadowRoot.querySelector('#canvas-container').appendChild(this.renderer.domElement);
        this.errorDisplay = this.shadowRoot.querySelector('#errorDisplay');

        // --- Resource Tracking ---
        this.textureObjectURLs = new Set(); // Keep track of Object URLs for textures

        // --- Debounced Functions ---
        this.debouncedUpdateDirectLightIntensity = debounce((intensity, index) => this.updateDirectLightIntensity(intensity, index), 150);
        this.debouncedUpdateAmbientLightIntensity = debounce((intensity) => this.updateAmbientLightIntensity(intensity), 150);
        this.debouncedUpdateDirectLightPosition = debounce((x, y, z, index) => this.updateDirectLightPosition(x, y, z, index), 150);
        // Add more for other sliders (roughness, metalness, camera near/far/fov, scale)


        // ... (rest of the constructor properties like loaders, lights, controls, state, etc.)
        this.gridHelper = new THREE.GridHelper(10, 10, 0x888888, 0x444444);
        this.gridHelper.visible = false;
        this.scene.add(this.gridHelper);

        this.state = {
            lightsOn: true,
            viewMode: 'default', // 'default', 'diffuse', 'geometry', 'normal'
            // wireframeInitialized: false, // Let's remove this
            isWireframeOn: false,
            environment: null, // null, 'env1', 'env2', 'env3'
            isAnimationPlaying: false,
            transformMode: null, // 'translate', 'rotate', or null
            isFullscreen: false,
        };

        // ... (rest of constructor logic, light setup, controls setup)

        this.dracoLoader = new DRACOLoader();
        this.objLoader = new OBJLoader();
        this.dracoLoader.setDecoderPath( 'https://www.gstatic.com/draco/versioned/decoders/1.5.7/' );
        this.gltfLoader = new GLTFLoader()
        this.gltfLoader.setDRACOLoader(this.dracoLoader);
        this.model = null;
        this.originalMaterials = new Map(); // Use Map for easier management
        this.wireframeMeshes = []; // This might be obsolete with the barycentric approach
        this.modelSize = 8; // Initial scale multiplier target
        this.autoRotate = false;
        this.anglePerSecond = 30;
        this.lastTime = 0;
        this.toonEnabled = false; // Toon shading state flag
        this.noPBR = false;

        // Lights Setup (ensure defaults if attributes not set)
        this.ambientLight = new THREE.AmbientLight(0x404040, 3);
        this.scene.add(this.ambientLight);
        this.directionalLights = [];
        this.directionalLightHelpers = [];
        if (!this.hasAttribute('direct-light')) {
             this.addDirectionalLight(); // Add a default light if none specified
             this.selectedDirectionalLightIndex = 0;
        }

        // Controls
        this.controls = new OrbitControls(this.camera, this.renderer.domElement);
        this.controls.enableDamping = true;
        this.controls.dampingFactor = 0.05;
        this.controls.addEventListener('change', () => this.updateControlPanel()); // Update UI on orbit change

        // Transform Controls
        this.transformControls = new TransformControls(this.camera, this.renderer.domElement);
        this.transformControls.addEventListener('change', () => {
             // Render is handled by animation loop, just update UI here
             this.updateControlPanel();
        });
        this.transformControls.addEventListener('dragging-changed', (event) => {
             // Disable OrbitControls while dragging
             this.controls.enabled = !event.value;
        });
        this.transformControls.visible = false;
        this.transformControls.enabled = false; // Should be enabled when attached
        this.scene.add(this.transformControls);

        // Textures & Loaders
        this.textureLoader = new THREE.TextureLoader();
        // Consider loading these only when needed, or provide defaults
        this.whiteTexture = this.textureLoader.load('https://huggingface.co/spaces/hhhwan/custom_gs/resolve/main/glbs/white.jpg');
        this.whiteTexture.mapping = THREE.EquirectangularReflectionMapping;
        this.gradTexture = this.textureLoader.load('https://huggingface.co/spaces/hhhwan/custom_gs/resolve/main/glbs/gradient.jpg');
        this.gradTexture.mapping = THREE.EquirectangularReflectionMapping;


        // Scene Graph Selection
        this.selectedSceneGraphLabel = null;
        this.selectedMeshPart = null;
        this.selectedMeshPartIndex = -1;
        this.glowMaterial = this.createGlowMaterial();
        this.previousSelectedMeshPart = null;
        this.previousMeshPartOriginalMaterial = null;
        this.glowTimeoutId = null; // Store timeout ID for clearing

        // Texture Editing
        this.textureHistory = new Map(); // Map<meshUUID, Map<mapType, Texture[]>>
        this.meshParts = [];
        this.meshPartTextureInfo = []; // Maybe integrate this with meshParts better
        this.canAdjustRoughnessMetalness = false; // Determined after loading

        // Animation
        this.mixer = null;
        this.animationActions = [];
        this.currentAction = null;

        // Recording
        this.mediaRecorder = null;
        this.recordedChunks = [];
        this.videoBlob = null;
        this.stream = null;
        this.startRecording = this.startRecording.bind(this);
        this.stopRecording = this.stopRecording.bind(this);
        this.downloadVideo = this.downloadVideo.bind(this);
        this.closeModal = this.closeModal.bind(this);

        // Idle Animation
        this.animationGeometry = null;
        this.animationMesh = null;
        // this.tweenGroup = null; // Use global TWEEN or a dedicated group if needed
        this.isIdleAnimationRunning = false;
        // Ensure TWEEN is loaded if you use it


        // --- Init ---
        this.initEventListeners(); // Break this down further if needed
        this.initLightUIValues();
        this.initCameraUIValues();
        this.updateDirectionalLightHelpersVisibility();
        this.initDiscardButton();
        this.initTextureMapUI();
        this.initTabSwitching();
        this.setupKeyboardShortcuts(); // Add keyboard shortcuts setup

        this.renderer.setAnimationLoop((time) => this.animate(time));

        // Initial resize
        this.resizeObserver = new ResizeObserver(() => this.resizeRenderer());

    } // --- End Constructor ---


    // --- Lifecycle Callbacks ---

    connectedCallback() {
        // Observe the host element for resize
        this.resizeObserver.observe(this.shadowRoot.host);

        // Initial setup based on attributes
        if (!this.getAttribute('src')) {
            this.showFileInput();
        } else {
             this.hideFileInput();
        }
        // Apply initial attribute settings
        this.applyInitialAttributes();

        // Trigger initial resize and render
        this.resizeRenderer();
        // this.animate(0); // setAnimationLoop handles rendering
    }

    disconnectedCallback() {
         // Stop observing when disconnected
         this.resizeObserver.disconnect();
         // Clean up animation loop
         this.renderer.setAnimationLoop(null);
         // Remove global event listeners
         window.removeEventListener('keydown', this._handleKeyDown);
         document.removeEventListener('fullscreenchange', this._handleFullscreenChange);
         // Stop any ongoing processes (like recording)
         if (this.mediaRecorder && this.mediaRecorder.state === "recording") {
             this.stopRecording();
         }
         // Dispose of resources
         this.discardModel(true); // Pass flag to indicate component removal
         this.disposeResources(); // Dispose scene-independent resources
    }

     disposeResources() {
         console.log("Disposing global resources...");
         // Dispose default textures if loaded
         this.whiteTexture?.dispose();
         this.gradTexture?.dispose();
         // Dispose glow material
         this.glowMaterial?.dispose();
         // Dispose idle animation resources
         this.animationGeometry?.dispose();
         if (this.animationMesh?.material) {
             if (Array.isArray(this.animationMesh.material)) {
                 this.animationMesh.material.forEach(m => m.dispose());
             } else {
                 this.animationMesh.material.dispose();
             }
         }
         // Dispose renderer resources (check Three.js docs for full cleanup if needed)
         this.renderer.dispose();
         // Clear Object URLs tracked
         this.textureObjectURLs.forEach(url => URL.revokeObjectURL(url));
         this.textureObjectURLs.clear();
         // Remove transform controls event listeners if added manually (Three.js might handle this)
         // Remove orbit controls event listeners if added manually (Three.js might handle this)

     }


    static get observedAttributes() {
        // Consolidated list
        return [
            'src', 'auto-rotate', 'angle-per-second', 'camera-orbit',
            'hide-control-ui', 'ui', 'light-off', 'no-pbr', 'view-mode',
            'ambient-light', 'direct-light', 'background-color' // Added background-color
        ];
    }

    attributeChangedCallback(name, oldValue, newValue) {
        if (oldValue === newValue) return; // No change

        switch (name) {
            case 'src':
                if (newValue) {
                    this.loadModel(newValue, newValue); // Assume URL is filename for now
                    this.hideFileInput();
                } else {
                    this.discardModel(); // Remove model if src is removed
                    this.showFileInput();
                }
                break;
            case 'auto-rotate':
                this.autoRotate = newValue !== null;
                this.updateAutoRotateButton(); // Update UI
                break;
            case 'angle-per-second':
                this.anglePerSecond = parseFloat(newValue) || 30;
                break;
            case 'camera-orbit':
                this.setCameraOrbit(newValue);
                break;
            case 'hide-control-ui':
                this.shadowRoot.querySelector('.controls').style.display = newValue !== null ? 'none' : ''; // Use empty string to reset
                break;
            case 'ui': // Toggle panel visibility based on 'ui' attribute presence
                const panelContent = this.shadowRoot.querySelector('#panelContent');
                const toggleButton = this.shadowRoot.querySelector('#togglePanelBtn');
                const shouldShow = newValue !== null;
                panelContent.style.display = shouldShow ? 'block' : 'none';
                // Adjust button icon and panel width accordingly (call toggle function maybe)
                this.updatePanelToggleState(shouldShow);
                break;
            case 'light-off':
                 this.state.lightsOn = newValue === null; // light-off attribute means lights ARE off
                 this.updateLightVisibility();
                 this.updateLightsButtonUI();
                 break;
            case 'no-pbr':
                 this.noPBR = newValue !== null;
                 this.state.lightsOn = !this.noPBR; // If no-pbr, lights are off
                 this.shadowRoot.querySelector('#lightControls').style.display = this.noPBR ? 'none' : '';
                 this.shadowRoot.querySelector('#toggleLightsBtn').style.display = this.noPBR ? 'none' : '';
                 this.shadowRoot.querySelector('#textureBtn').style.display = this.noPBR ? 'none' : ''; // Hide diffuse if no PBR
                 this.updateLightVisibility();
                 if (this.noPBR && this.model) {
                     this.state.viewMode = 'diffuse'; // Force diffuse if no PBR
                     this.renderMode();
                 } else if (!this.noPBR && this.model) {
                     this.setDefaultMat(); // Revert to default if no-pbr removed
                     this.renderMode();
                 }
                 break;
            case 'view-mode':
                if (['default', 'diffuse', 'geometry', 'normal'].includes(newValue)) {
                    this.state.viewMode = newValue;
                    if(this.model) this.renderMode(); // Apply mode if model exists
                }
                break;
            case 'ambient-light':
                this.setAmbientLight(newValue);
                this.initLightUIValues(); // Update UI
                break;
            case 'direct-light':
                 // Remove existing directional lights before setting new one from attribute
                 this.removeAllDirectionalLights();
                 this.setDirectLight(newValue); // Sets the first light
                 this.initLightUIValues(); // Update UI
                 break;
             case 'background-color':
                 this.setBackgroundColor(newValue);
                 this.shadowRoot.querySelector('#bgColorPicker').value = newValue; // Update UI
                 break;
        }
    }

    // Helper to apply initial attributes on connect
    applyInitialAttributes() {
        SimpleModelViewer.observedAttributes.forEach(attr => {
            if (this.hasAttribute(attr)) {
                this.attributeChangedCallback(attr, null, this.getAttribute(attr));
            }
        });
         // Ensure default state if attributes conflicting (e.g., no-pbr and light-off) are handled
         if (this.noPBR) {
             this.state.lightsOn = false;
             this.updateLightVisibility();
             this.renderMode(); // Apply no-pbr mode
         }
    }


    // --- UI Initialization & Updates ---

    initEventListeners() {
        // Break down listeners by section
        this.setupGeneralControls();
        this.setupRenderTabControls();
        this.setupControlTabControls();
        this.setupEditTabControls();
        this.setupRecordingControls();

         // Window resize is handled by ResizeObserver now
    }

     setupGeneralControls() {
         const fileInput = this.shadowRoot.querySelector('#fileInput');
         const urlInput = this.shadowRoot.querySelector('#urlInput');
         const loadUrlButton = this.shadowRoot.querySelector('#loadUrlButton');
         const togglePanelBtn = this.shadowRoot.querySelector('#togglePanelBtn');

         fileInput.addEventListener('change', (e) => {
             const file = e.target.files[0];
             if (file) {
                 const objectURL = URL.createObjectURL(file);
                 this.loadModel(objectURL, file.name);
                 this.hideFileInput();
                 // No need to track this specific ObjectURL here, loadModel handles it
             }
         });

         loadUrlButton.addEventListener('click', () => {
             const url = urlInput.value.trim();
             if (url) {
                 this.loadModel(url, url); // Use URL as name guess
                 this.hideFileInput();
                 urlInput.value = '';
             } else {
                 this.showError("Please enter a valid URL.");
             }
         });

         urlInput.addEventListener('keypress', (e) => {
             if (e.key === 'Enter') {
                 loadUrlButton.click();
             }
         });

         togglePanelBtn.addEventListener('click', () => {
            const content = this.shadowRoot.querySelector('#panelContent');
            const isHidden = content.style.display === 'none';
            this.updatePanelToggleState(!isHidden); // Toggle state
             // Also update 'ui' attribute? Optional, depends on desired behavior
             // if (isHidden) this.setAttribute('ui', ''); else this.removeAttribute('ui');
         });
     }

     updatePanelToggleState(show) {
         const controls = this.shadowRoot.querySelector('.right-ui-panel');
         const content = this.shadowRoot.querySelector('#panelContent');
         const button = this.shadowRoot.querySelector('#togglePanelBtn');
         if (show) {
             controls.style.width = '25rem'; // Or use a CSS class
             content.style.display = 'block';
             button.innerHTML = `<svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" fill="currentColor" class="bi bi-caret-right-fill" viewBox="0 0 16 16"><path d="m12.14 8.753-5.482 4.796c-.646.566-1.658.106-1.658-.753V3.204a1 1 0 0 1 1.659-.753l5.48 4.796a1 1 0 0 1 0 1.506z"/></svg>`; // Pointing Left
         } else {
             controls.style.width = '4rem'; // Or use a CSS class
             content.style.display = 'none';
             button.innerHTML = `<svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" fill="currentColor" class="bi bi-caret-left-fill" viewBox="0 0 16 16"><path d="m3.86 8.753 5.482 4.796c.646.566 1.658.106 1.658-.753V3.204a1 1 0 0 0-1.659-.753l-5.48 4.796a1 1 0 0 0 0 1.506z"/></svg>`; // Pointing Right
         }
     }

     setupRenderTabControls() {
         this.shadowRoot.querySelector('#bgColorPicker').addEventListener('input', (event) => this.setAttribute('background-color', event.target.value));
         this.shadowRoot.querySelector('#toggleGridBtn').addEventListener('click', () => this.toggleGrid());
         this.shadowRoot.querySelector('#textureBtn').addEventListener('click', () => this.toggleViewMode('diffuse'));
         this.shadowRoot.querySelector('#meshBtn').addEventListener('click', () => this.toggleViewMode('geometry'));
         this.shadowRoot.querySelector('#normalBtn').addEventListener('click', () => this.toggleViewMode('normal'));
         this.shadowRoot.querySelector('#wireframeBtn').addEventListener('click', () => this.toggleWireframe());
         this.shadowRoot.querySelector('#setBgBtn1').addEventListener('click', () => this.toggleEnvironment('env1'));
         this.shadowRoot.querySelector('#setBgBtn2').addEventListener('click', () => this.toggleEnvironment('env2'));
         this.shadowRoot.querySelector('#setBgBtn3').addEventListener('click', () => this.toggleEnvironment('env3'));
         this.shadowRoot.querySelector('#autoRotateBtn').addEventListener('click', () => this.toggleAutoRotate());
         this.shadowRoot.querySelector('#screenshotBtn').addEventListener('click', () => this.takeScreenshotToClipboard());
         this.shadowRoot.querySelector('#fullscreenBtn').addEventListener('click', () => this.toggleFullscreen());
         this.shadowRoot.querySelector('#runAnimationBtn').addEventListener('click', () => this.runAnimation());
         this.shadowRoot.querySelector('#pauseAnimationBtn').addEventListener('click', () => this.pauseAnimation());
         this.shadowRoot.querySelector('#discardModelBtn').addEventListener('click', () => this.discardModel());
         // Toon shading button might need special handling if added back
     }

     setupControlTabControls() {
         // Lights
         this.shadowRoot.querySelector('#toggleLightsBtn').addEventListener('click', () => this.toggleLights());
         this.shadowRoot.querySelector('#toggleLightHelpersBtn').addEventListener('click', () => this.toggleLightHelpers());
         this.shadowRoot.querySelector('#ambientColorPicker').addEventListener('input', (event) => this.updateAmbientLightColor(event.target.value));
         this.shadowRoot.querySelector('#ambientIntensity').addEventListener('input', (event) => this.debouncedUpdateAmbientLightIntensity(parseFloat(event.target.value))); // Debounced
         this.shadowRoot.querySelector('#addLightBtn').addEventListener('click', () => this.addDirectionalLight());
         this.shadowRoot.querySelector('#removeLightBtn').addEventListener('click', () => this.removeDirectionalLight());
         this.shadowRoot.querySelector('#directionalLightList').addEventListener('change', (event) => {
             this.selectedDirectionalLightIndex = parseInt(event.target.value);
             this.updateDirectionalLightUIValues();
         });
         this.shadowRoot.querySelector('#directColorPicker').addEventListener('input', (event) => {
             if (this.directionalLights[this.selectedDirectionalLightIndex]) {
                 this.updateDirectLightColor(event.target.value, this.selectedDirectionalLightIndex);
             }
         });
         // Debounce position/intensity inputs
         this.shadowRoot.querySelector('#directPosX').addEventListener('input', (event) => this.handleDirectLightPosInput('x', event.target.value));
         this.shadowRoot.querySelector('#directPosY').addEventListener('input', (event) => this.handleDirectLightPosInput('y', event.target.value));
         this.shadowRoot.querySelector('#directPosZ').addEventListener('input', (event) => this.handleDirectLightPosInput('z', event.target.value));
         this.shadowRoot.querySelector('#directIntensity').addEventListener('input', (event) => {
             if (this.directionalLights[this.selectedDirectionalLightIndex]) {
                 this.debouncedUpdateDirectLightIntensity(parseFloat(event.target.value), this.selectedDirectionalLightIndex);
             }
         });

         // Camera
         // Add debounce if needed, though camera changes might be less frequent/performance-intensive
         this.shadowRoot.querySelector('#cameraFov').addEventListener('input', (event) => this.updateCameraFov(parseFloat(event.target.value)));
         this.shadowRoot.querySelector('#cameraNear').addEventListener('input', (event) => this.updateCameraNear(parseFloat(event.target.value)));
         this.shadowRoot.querySelector('#cameraFar').addEventListener('click', (event) => this.updateCameraFar(parseFloat(event.target.value))); // Changed to click, assuming it's not meant to be input

         // Transform
         this.shadowRoot.querySelector('#translateBtn').addEventListener('click', () => this.setTransformMode('translate'));
         this.shadowRoot.querySelector('#rotateBtn').addEventListener('click', () => this.setTransformMode('rotate'));
         this.shadowRoot.querySelector('#scale').addEventListener('input', (e) => { // Debounce if needed
             this.updateModelScale(parseFloat(e.target.value));
         });
         // Position/Rotation inputs are updated by controls/transformControls change events, no need for listeners here usually
    }

     // Helper for debounced position updates
     handleDirectLightPosInput(axis, value) {
         if (this.directionalLights[this.selectedDirectionalLightIndex]) {
             const pos = { x: null, y: null, z: null };
             pos[axis] = parseFloat(value);
             this.debouncedUpdateDirectLightPosition(pos.x, pos.y, pos.z, this.selectedDirectionalLightIndex);
         }
     }


     setupEditTabControls() {
         this.shadowRoot.querySelector('#roughness').addEventListener('input', () => this.updateMaterialProperties()); // Debounce?
         this.shadowRoot.querySelector('#metalness').addEventListener('input', () => this.updateMaterialProperties()); // Debounce?

         const partSelector = this.shadowRoot.querySelector('#texturePartSelector');
         const typeSelector = this.shadowRoot.querySelector('#textureTypeSelector');
         const replaceTextureButton = this.shadowRoot.querySelector('#replaceTextureBtn');
         const historySelector = this.shadowRoot.querySelector('#textureHistorySelector'); // Assuming it's added in initTextureMapUI

         partSelector.addEventListener('change', () => {
             const selectedPartIndex = parseInt(partSelector.value);
             if (!isNaN(selectedPartIndex) && selectedPartIndex >= 0 && selectedPartIndex < this.meshParts.length) {
                 const selectedMeshPart = this.meshParts[selectedPartIndex];
                 // Find the corresponding label element to pass - might need to store mapping during graph generation
                 // For now, pass null. Selection highlighting will still work.
                 this.selectMeshPartInSceneGraph(selectedMeshPart, null); // Select in graph
                 //this.updateTextureMapDisplay(); // This should be called by selectMeshPartInSceneGraph now
                 this.updateHistorySelector();
             }
         });

         typeSelector.addEventListener('change', () => {
             this.updateTextureMapDisplay();
             this.updateHistorySelector();
         });

         replaceTextureButton.addEventListener('click', () => {
             const selectedType = typeSelector.value;
             const fileInputId = this.getTextureInputId(selectedType); // Helper function
             if (fileInputId) {
                 const fileInput = this.shadowRoot.querySelector(`#${fileInputId}`);
                 const selectedPartIndex = parseInt(partSelector.value);
                 if (!isNaN(selectedPartIndex)) {
                     fileInput.dataset.meshPartIndex = selectedPartIndex.toString(); // Store index
                     fileInput.click();
                 } else {
                     this.showError("No mesh part selected.");
                 }
             }
         });

         if (historySelector) {
             historySelector.addEventListener('change', () => {
                 this.applyHistoryTexture(); // Apply selected history texture
             });
         }

          // Scene graph interaction is handled within generateSceneGraphTree's listeners
     }

     getTextureInputId(mapType) {
         switch (mapType) {
             case 'map': return 'diffuseMapInput';
             case 'roughnessMap': return 'roughnessMapInput';
             case 'metalnessMap': return 'metalnessMapInput';
             case 'normalMap': return 'normalMapInput';
             case 'aoMap': return 'aoMapInput';
             case 'emissiveMap': return 'emissiveMapInput';
             default: return null;
         }
     }

     setupRecordingControls() {
        this.recordBtn = this.shadowRoot.querySelector('#recordBtn');
        this.stopBtn = this.shadowRoot.querySelector('#stopBtn');
        this.videoModal = this.shadowRoot.querySelector('#videoModal');
        this.videoPreview = this.shadowRoot.querySelector('#videoPreview');
        this.downloadBtn = this.shadowRoot.querySelector('#downloadBtn');
        this.closeModalBtn = this.shadowRoot.querySelector('#closeModalBtn');
        this.closeModal(); // Ensure hidden initially

        if (this.recordBtn) this.recordBtn.addEventListener('click', this.startRecording);
        if (this.stopBtn) this.stopBtn.addEventListener('click', this.stopRecording);
        if (this.downloadBtn) this.downloadBtn.addEventListener('click', this.downloadVideo);
        if (this.closeModalBtn) this.closeModalBtn.addEventListener('click', this.closeModal);

        if (this.videoModal) {
            this.videoModal.addEventListener('click', (event) => {
                if (event.target === this.videoModal) {
                    this.closeModal();
                }
            });
        }
     }


    // --- Error Handling ---
    showError(message, duration = 5000) {
        console.error("Viewer Error:", message); // Keep console log
        if (!this.errorDisplay) return;
        this.errorDisplay.textContent = message;
        this.errorDisplay.style.display = 'block';
        // Clear previous timeout if any
        if (this.errorTimeout) clearTimeout(this.errorTimeout);
        // Set new timeout
        this.errorTimeout = setTimeout(() => {
            this.errorDisplay.style.display = 'none';
            this.errorTimeout = null;
        }, duration);
    }

    clearError() {
         if (this.errorTimeout) clearTimeout(this.errorTimeout);
         this.errorDisplay.style.display = 'none';
    }

    // --- Resource Management ---

    // Enhanced discardModel
    discardModel(isDisconnecting = false) {
        this.clearError(); // Clear any errors
        if (this.model) {
            console.log("Discarding model...");
            // 1. Detach transform controls
            if (this.transformControls.object) {
                this.transformControls.detach();
            }
            // 2. Stop animations
            if (this.mixer) {
                this.mixer.stopAllAction();
                this.mixer = null; // Allow garbage collection
            }
            this.animationActions = [];
            this.currentAction = null;

            // 3. Remove model from scene
            this.scene.remove(this.model);

            // 4. Dispose geometries, materials, textures
            this.model.traverse((child) => {
                if (child.isMesh) {
                    child.geometry?.dispose();
                    if (child.material) {
                        const materials = Array.isArray(child.material) ? child.material : [child.material];
                        materials.forEach(mat => {
                            // Dispose textures attached to this specific material instance
                            for (const key in mat) {
                                if (mat[key] instanceof THREE.Texture) {
                                    mat[key].dispose();
                                }
                            }
                            mat.dispose(); // Dispose the material itself
                        });
                    }
                }
            });

             // 5. Dispose textures stored in originalMaterials Map
             this.originalMaterials.forEach((material) => {
                 // We assume original materials were cloned, so disposing here is safe
                 for (const key in material) {
                     if (material[key] instanceof THREE.Texture) {
                         material[key].dispose();
                     }
                 }
                 material.dispose(); // Dispose the cloned original material
             });
             this.originalMaterials.clear(); // Clear the map

             // 6. Dispose textures stored in textureHistory Map
             this.textureHistory.forEach((typeMap) => {
                 typeMap.forEach((historyArray) => {
                     historyArray.forEach(texture => texture.dispose());
                 });
             });
             this.textureHistory.clear();

             // 7. Clean up Object URLs created for *this model's* textures
             // This requires tracking which URLs belong to which model if multiple loads happen,
             // or clearing all tracked URLs on discard (simpler but might revoke unrelated URLs if not careful)
             // Let's clear all tracked texture URLs on discard for simplicity here:
             this.textureObjectURLs.forEach(url => URL.revokeObjectURL(url));
             this.textureObjectURLs.clear();


             // 8. Reset state variables
             this.model = null;
             this.meshParts = [];
             this.meshPartTextureInfo = []; // Reset derived info
             this.standardMaterials = [];
             this.canAdjustRoughnessMetalness = false;
             this.selectedMeshPart = null;
             this.selectedMeshPartIndex = -1;
             this.previousSelectedMeshPart = null;
             this.previousMeshPartOriginalMaterial = null;
             if (this.glowTimeoutId) clearTimeout(this.glowTimeoutId);
             this.selectedSceneGraphLabel = null; // Reset scene graph selection highlight
             this.state.isWireframeOn = false; // Reset wireframe state

             // 9. Reset UI elements
             this.resetUI();

             // 10. Conditionally show file input (unless component is disconnecting)
             if (!isDisconnecting) {
                 this.showFileInput();
             }
             console.log("Model discarded.");
        } else if (!isDisconnecting){
             // If no model exists but discard is called (e.g., from button), ensure UI is reset
             this.resetUI();
             this.showFileInput();
        }
         // Ensure idle animation starts if no model and not disconnecting
         if (!this.model && !this.isIdleAnimationRunning && !isDisconnecting && typeof TWEEN !== 'undefined') {
             this.initIdleAnimation();
         }
    }

    // UI Reset Helper
    resetUI() {
         this.shadowRoot.querySelector('#modelInfo').innerHTML = `<strong>[Model Info]</strong> No model loaded.`;
         this.shadowRoot.querySelector('#discardModelBtn').style.display = 'none';
         // Reset transform UI
         this.shadowRoot.querySelector('#posX').value = 0;
         this.shadowRoot.querySelector('#posY').value = 0;
         this.shadowRoot.querySelector('#posZ').value = 0;
         this.shadowRoot.querySelector('#rotX').value = 0;
         this.shadowRoot.querySelector('#rotY').value = 0;
         this.shadowRoot.querySelector('#rotZ').value = 0;
         this.shadowRoot.querySelector('#scale').value = 8; // Reset to default scale target
         // Reset material UI
         this.shadowRoot.querySelector('#roughness').value = 0.5;
         this.shadowRoot.querySelector('#metalness').value = 0.5;
         this.shadowRoot.querySelector('#roughness').disabled = true;
         this.shadowRoot.querySelector('#metalness').disabled = true;
         // Clear Scene Graph
         this.shadowRoot.querySelector('#sceneGraphTree').innerHTML = '';
         // Clear Texture Editor UI
         this.shadowRoot.querySelector('#texturePartSelector').innerHTML = '';
         this.shadowRoot.querySelector('#textureTypeSelector').value = 'map'; // Reset type
         const previewElement = this.shadowRoot.querySelector('#texturePreview');
         previewElement.textContent = '';
         previewElement.style.backgroundImage = '';
         const historySelector = this.shadowRoot.querySelector('#textureHistorySelector');
         if(historySelector) historySelector.innerHTML = '<option value="-1">Current</option>';
         // Reset animation UI
         const animSelector = this.shadowRoot.querySelector('#animationSelector');
         if (animSelector) animSelector.remove();
         this.shadowRoot.querySelector('#anim_description').style.display = 'none';
         this.updateAnimationButtons(); // Ensure play/pause are hidden/disabled
         // Reset view mode buttons
         this.state.viewMode = 'default';
         this.updateViewModeButtons();
         // Reset env buttons
         this.state.environment = null;
         this.updateEnvButtons();
         // Reset wireframe button
         this.state.isWireframeOn = false;
         this.updateWireframeButton();
         // Reset grid button
         this.gridHelper.visible = false;
         this.updateGridButton();
          // Reset transform mode buttons
         this.setTransformMode(null); // Deactivate any active transform mode
    }


    // Helper to manage texture Object URLs
    trackObjectURL(url) {
        this.textureObjectURLs.add(url);
    }

    revokeObjectURL(url) {
        if (this.textureObjectURLs.has(url)) {
            URL.revokeObjectURL(url);
            this.textureObjectURLs.delete(url);
        }
    }

    // Update texture loading to use tracking
    handleTextureFileChange(event, mapType) {
        const fileInput = event.target;
        const file = fileInput.files[0];
        if (!file) return;

        const textureURL = URL.createObjectURL(file);
        this.trackObjectURL(textureURL); // Track the URL

        const selectedMeshPartIndex = parseInt(fileInput.dataset.meshPartIndex);
        if (isNaN(selectedMeshPartIndex) || !this.meshParts[selectedMeshPartIndex]) {
             this.showError("Invalid mesh part selected for texture replacement.");
             this.revokeObjectURL(textureURL); // Revoke if invalid
             return;
        }
        const mesh = this.meshParts[selectedMeshPartIndex];

        this.textureLoader.load(textureURL, (texture) => {
             // Success
             texture.encoding = mapType === 'map' || mapType === 'emissiveMap' ? THREE.sRGBEncoding : THREE.LinearEncoding; // Correct encoding
             texture.flipY = false; // GLTF standard

             // Save old texture to history *before* replacing
             this.saveTextureToHistory(mesh, mapType, mesh.material[mapType]);

             // Ensure material is unique before modification
             if (mesh.material.isShared) { // Check if material is shared
                 console.warn(`Material for ${mesh.name || 'part'} was shared. Cloning.`);
                 mesh.material = mesh.material.clone();
                 // Update originalMaterials map if the original was also the shared one
                 // This might require more complex tracking if multiple parts share the same original material.
                 // For simplicity, assume originalMaterials holds clones already or update it here.
                 this.originalMaterials.set(mesh.uuid, mesh.material.clone());
             }

             // Dispose old texture *if* it's not potentially used elsewhere (e.g., in history or another part)
             // This is tricky. A safer approach is to rely on discardModel for bulk disposal.
             // Let's skip disposing the old texture here to avoid complexity.
             // mesh.material[mapType]?.dispose();

             mesh.material[mapType] = texture;
             // Update the corresponding texture in the 'original' map *if* this should be the new base
             // This depends on the desired behavior. Let's assume replacing updates the current state,
             // but originalMaterials keeps the *initially loaded* state unless specifically overwritten.
             // So, we don't update originalMaterials here. User can revert via history or discard/reload.

             mesh.material.needsUpdate = true;

             this.updateTextureMapDisplay(); // Update preview for the current selection
             this.updateHistorySelector();   // Update history dropdown

             // No need to revoke URL here, texture loader keeps reference. Revoke on discardModel.
        },
        undefined, // Progress callback (optional)
        (error) => { // Error callback
             this.showError(`Failed to load texture: ${file.name}`);
             console.error('Texture loading error:', error);
             this.revokeObjectURL(textureURL); // Revoke URL on loading error
        });
    }

    // --- Fullscreen ---
    toggleFullscreen() {
        if (!document.fullscreenElement) {
            // Enter fullscreen on the container of the canvas, or the host element
            const container = this.shadowRoot.querySelector('#canvas-container');
            container.requestFullscreen().catch(err => {
                this.showError(`Error entering fullscreen: ${err.message}`);
            });
        } else {
            document.exitFullscreen();
        }
    }

    _handleFullscreenChange = () => { // Use arrow function for correct 'this'
        this.state.isFullscreen = !!document.fullscreenElement;
        this.shadowRoot.host.toggleAttribute('fullscreen', this.state.isFullscreen);
        // Update button appearance if needed
        const fsButton = this.shadowRoot.querySelector('#fullscreenBtn');
         // Simple toggle: change icon or style
        fsButton.innerHTML = this.state.isFullscreen ? `
            <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" fill="currentColor" class="bi bi-fullscreen-exit" viewBox="0 0 16 16">
              <path d="M5.5 0a.5.5 0 0 1 .5.5v4a.5.5 0 0 1-1 0v-4a.5.5 0 0 1 .5-.5m5 0a.5.5 0 0 1 .5.5v4a.5.5 0 0 1-1 0v-4a.5.5 0 0 1 .5-.5m-5 10a.5.5 0 0 1 .5.5v4a.5.5 0 0 1-1 0v-4a.5.5 0 0 1 .5-.5m5 0a.5.5 0 0 1 .5.5v4a.5.5 0 0 1-1 0v-4a.5.5 0 0 1 .5-.5"/>
            </svg>` : `
            <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" fill="currentColor" class="bi bi-fullscreen" viewBox="0 0 16 16">
               <path d="M1.5 1a.5.5 0 0 0-.5.5v4a.5.5 0 0 1-1 0v-4A1.5 1.5 0 0 1 1.5 0h4a.5.5 0 0 1 0 1zM10 .5a.5.5 0 0 1 .5-.5h4A1.5 1.5 0 0 1 16 1.5v4a.5.5 0 0 1-1 0v-4a.5.5 0 0 0-.5-.5h-4a.5.5 0 0 1-.5-.5M.5 10a.5.5 0 0 1 .5.5v4a.5.5 0 0 0 .5.5h4a.5.5 0 0 1 0 1h-4A1.5 1.5 0 0 1 0 14.5v-4a.5.5 0 0 1 .5-.5m15 0a.5.5 0 0 1 .5.5v4a1.5 1.5 0 0 1-1.5 1.5h-4a.5.5 0 0 1 0-1h4a.5.5 0 0 0 .5-.5v-4a.5.5 0 0 1 .5-.5"/>
            </svg>`;
        // Crucially, trigger a resize after entering/exiting fullscreen
        // Use setTimeout to allow the browser to settle dimensions
        setTimeout(() => this.resizeRenderer(), 100);
    }

    // --- Keyboard Shortcuts ---
    setupKeyboardShortcuts() {
        // Bind the handler once to keep the correct 'this' context
        this._handleKeyDown = this._handleKeyDown.bind(this);
        // Add listener to the window or a specific container
        window.addEventListener('keydown', this._handleKeyDown);
        // Listen for fullscreen changes globally
        document.addEventListener('fullscreenchange', this._handleFullscreenChange);
    }

    _handleKeyDown(event) {
        // Ignore shortcuts if user is typing in an input field
        if (event.target.tagName === 'INPUT' || event.target.tagName === 'SELECT' || event.target.tagName === 'TEXTAREA') {
            return;
        }
         // Ignore if modifier keys are pressed (unless specifically needed)
         if (event.ctrlKey || event.altKey || event.metaKey) {
             return;
         }

        switch (event.key.toLowerCase()) {
            case 't': // Translate
                this.setTransformMode('translate');
                event.preventDefault(); // Prevent default browser action for 't' if any
                break;
            case 'r': // Rotate
                this.setTransformMode('rotate');
                 event.preventDefault();
                break;
             // Add Scale ('s') if implemented
             // case 's':
             //     this.setTransformMode('scale');
             //     event.preventDefault();
             //     break;
            case 'g': // Grid
                this.toggleGrid();
                event.preventDefault();
                break;
            case 'f': // Fullscreen
                this.toggleFullscreen();
                event.preventDefault();
                break;
            case 'l': // Toggle Lights
                 this.toggleLights();
                 event.preventDefault();
                 break;
             case 'h': // Toggle Light Helpers
                 this.toggleLightHelpers();
                 event.preventDefault();
                 break;
             case 'w': // Toggle Wireframe
                 this.toggleWireframe();
                 event.preventDefault();
                 break;
             case 'Escape': // Escape key
                 // Deselect transform mode
                 if (this.state.transformMode) {
                     this.setTransformMode(null); // Pass null to deactivate
                     event.preventDefault();
                 }
                 // Exit fullscreen handled by browser, listener updates state
                 // Could potentially deselect scene graph item here too
                 break;
             // Add more shortcuts as needed
        }
    }

     // --- Toggles and State Updates ---
     toggleGrid() {
         this.gridHelper.visible = !this.gridHelper.visible;
         this.updateGridButton();
     }
     updateGridButton() {
         const btn = this.shadowRoot.querySelector('#toggleGridBtn');
         btn.textContent = this.gridHelper.visible ? 'Hide Grid' : 'Show Grid';
         btn.classList.toggle('toggled-off', this.gridHelper.visible);
     }

     toggleLights() {
         this.state.lightsOn = !this.state.lightsOn;
         this.updateLightVisibility();
         this.updateLightsButtonUI();
     }
     updateLightVisibility() {
          // Apply state.lightsOn AND consider noPBR override
          const lightsActuallyOn = this.state.lightsOn && !this.noPBR;
          this.ambientLight.visible = lightsActuallyOn;
          this.directionalLights.forEach(light => {
              light.visible = lightsActuallyOn;
          });
          this.updateDirectionalLightHelpersVisibility(); // Update helpers based on new visibility
     }
      updateLightsButtonUI() {
         const btn = this.shadowRoot.querySelector('#toggleLightsBtn');
         btn.textContent = this.state.lightsOn ? 'Lights Off' : 'Lights On';
         btn.classList.toggle('toggled-off', this.state.lightsOn);
     }


     toggleLightHelpers() {
         this.showLightHelpers = !this.showLightHelpers;
         this.updateDirectionalLightHelpersVisibility();
         // No need to call updateLightVisibility here, just helper visibility
     }
     updateDirectionalLightHelpersVisibility() {
         // Visibility depends on toggle AND if lights are actually on
         const helpersActuallyVisible = this.showLightHelpers && this.state.lightsOn && !this.noPBR;
         this.directionalLightHelpers.forEach(helper => {
             helper.visible = helpersActuallyVisible;
         });
         const btn = this.shadowRoot.querySelector('#toggleLightHelpersBtn');
         btn.textContent = this.showLightHelpers ? 'Hide Helpers' : 'Show Helpers'; // Shorter text
         btn.classList.toggle('toggled-off', this.showLightHelpers); // Toggle based on the switch itself
     }


      toggleAutoRotate() {
         this.autoRotate = !this.autoRotate;
         this.updateAutoRotateButton();
     }
     updateAutoRotateButton() {
         const btn = this.shadowRoot.querySelector('#autoRotateBtn');
         // Use SVG inside button - maybe change fill or add class?
         btn.classList.toggle('toggled-off', this.autoRotate);
         // Add title for clarity
         btn.title = this.autoRotate ? 'Stop Auto-Rotation' : 'Start Auto-Rotation';
     }


     toggleViewMode(mode) {
         if (this.state.viewMode === mode) {
             this.state.viewMode = 'default'; // Toggle off back to default
         } else {
             this.state.viewMode = mode;
         }
         this.renderMode();
     }

     toggleEnvironment(env) {
         if (this.state.environment === env) {
             this.state.environment = null; // Toggle off back to default
         } else {
             this.state.environment = env;
         }
         this.renderMode(); // Apply environment change
     }

     toggleWireframe() {
          if (!this.model) return;
          this.state.isWireframeOn = !this.state.isWireframeOn;
          this.applyWireframeState(); // Apply the state change
          this.updateWireframeButton();
     }
     updateWireframeButton() {
          const btn = this.shadowRoot.querySelector('#wireframeBtn');
          // btn.textContent = this.state.isWireframeOn ? 'Wireframe Off' : 'Wireframe';
          btn.classList.toggle('toggled-off', this.state.isWireframeOn);
     }

     // Apply wireframe state to materials
     applyWireframeState() {
         if (!this.model) return;
         this.model.traverse((child) => {
             if (child.isMesh && child.material?.userData?.shader) {
                 child.material.userData.shader.uniforms.uWireframe.value = this.state.isWireframeOn;
                 child.material.needsUpdate = true; // Might not be necessary if only uniform changes
             } else if (child.isMesh && child.material && !child.material.userData.shader) {
                 // Ensure material is prepared if wireframe is toggled on for the first time after load/material change
                 if (this.state.isWireframeOn) {
                      console.warn("Wireframe toggled for material without pre-compiled shader. Re-applying shader modification.");
                      this.addBarycentricCoordinates(child.geometry); // Ensure coords exist
                      this.modifyMaterialForWireframe(child.material); // Re-apply modification
                      // Need to wait for compile or force compile? Usually happens on next render.
                      // Set uniform after potential recompile (might be async)
                      if (child.material.userData.shader) {
                           child.material.userData.shader.uniforms.uWireframe.value = true;
                      }
                 }
             }
         });
     }

    // --- Core Logic (Loading, Rendering, etc.) ---

     loadModel(url, fileName) {
         // 1. Discard existing model first
         this.discardModel();
         this.clearError(); // Clear any previous error messages

         const progressBar = this.shadowRoot.querySelector('#loadingProgressBar');
         progressBar.style.display = 'block';
         progressBar.style.width = '0%';

         // Determine loader based on extension
         const fileExtension = fileName.split('.').pop().toLowerCase();
         const loader = fileExtension === 'obj' ? this.objLoader : this.gltfLoader;

         // Track ObjectURL if it's a blob URL
         let isObjectURL = url.startsWith('blob:');
         if (isObjectURL) {
             // We already tracked it in the file input handler, but good to double check
             this.trackObjectURL(url);
         }

         loader.load(url,
             (object) => { // --- Success Callback ---
                 progressBar.style.display = 'none';
                 this.model = fileExtension === 'obj' ? object : object.scene;

                 // Stop idle animation if running
                 if (this.isIdleAnimationRunning) {
                     this.scene.remove(this.animationMesh);
                     this.animationGeometry?.dispose(); // Dispose geometry
                     this.animationMesh?.material?.dispose(); // Dispose material
                     this.animationMesh = null;
                     // Stop TWEEN? Depends on implementation
                     // TWEEN.removeAll(); // If using global TWEEN
                     this.isIdleAnimationRunning = false;
                 }


                 // Store original materials and prepare scene graph data
                 this.originalMaterials.clear(); // Clear previous originals
                 this.meshParts = [];

                 let vertexCount = 0;
                 let faceCount = 0;
                 let hasStandardMaterial = false; // Check if we have any standard mats

                 this.model.traverse((child) => {
                     if (child.isMesh) {
                         this.meshParts.push(child); // Add to list for texture editing etc.

                         // Geometry processing
                         vertexCount += child.geometry.attributes.position.count;
                         faceCount += child.geometry.index ? child.geometry.index.count / 3 : child.geometry.attributes.position.count / 3;
                         // Add barycentric coordinates for wireframe overlay
                         this.addBarycentricCoordinates(child.geometry);

                         // Material processing
                         const materials = Array.isArray(child.material) ? child.material : [child.material];
                         const processedMaterials = materials.map(mat => {
                             let originalMat = mat.clone(); // Clone original for reference
                             let workingMat = mat; // Start with the loaded material

                              // Ensure it's MeshStandardMaterial if needed (e.g., for env maps, PBR features)
                             // If noPBR is active, we might skip this or use MeshBasicMaterial
                             if (!this.noPBR && !(mat instanceof THREE.MeshStandardMaterial) && !(mat instanceof THREE.MeshPhysicalMaterial)) {
                                 console.warn(`Converting non-standard material (${mat.type}) on ${child.name || 'mesh'} to MeshStandardMaterial.`);
                                 workingMat = new THREE.MeshStandardMaterial({
                                     map: mat.map, // Copy basic map if exists
                                     color: mat.color || new THREE.Color(0xffffff), // Copy color
                                     // roughness/metalness defaults used by StandardMaterial
                                 });
                                 if (mat.map) {
                                     workingMat.map.encoding = THREE.sRGBEncoding; // Assume sRGB for basic maps
                                     workingMat.map.flipY = false; // GLTF standard
                                 }
                                 workingMat.needsUpdate = true;
                             } else if (mat instanceof THREE.MeshStandardMaterial || mat instanceof THREE.MeshPhysicalMaterial) {
                                  hasStandardMaterial = true;
                                 // Apply correct encodings for GLTF standard materials
                                 if (mat.map) mat.map.encoding = THREE.sRGBEncoding;
                                 if (mat.emissiveMap) mat.emissiveMap.encoding = THREE.sRGBEncoding;
                                 // Rough/Metal/AO/Normal should be Linear
                                 if (mat.roughnessMap) mat.roughnessMap.encoding = THREE.LinearEncoding;
                                 if (mat.metalnessMap) mat.metalnessMap.encoding = THREE.LinearEncoding;
                                 if (mat.aoMap) {
                                     mat.aoMap.encoding = THREE.LinearEncoding;
                                      // Ensure uv2 exists for aoMap
                                      if (child.geometry.attributes.uv && !child.geometry.attributes.uv2) {
                                           console.log("Applying uv as uv2 for aoMap on", child.name);
                                           child.geometry.setAttribute('uv2', new THREE.BufferAttribute(child.geometry.attributes.uv.array, 2));
                                      }
                                 }
                                 if (mat.normalMap) mat.normalMap.encoding = THREE.LinearEncoding;
                             }

                             // Store the *cloned* original material
                             // Use UUID as key. If multiple materials, maybe store array?
                             // For simplicity, let's assume single material or handle first one.
                             if (!Array.isArray(child.material)) {
                                 this.originalMaterials.set(child.uuid, originalMat);
                             } else {
                                 // How to handle multiple original materials? Maybe store array?
                                 // Or use a combined UUID/index key? Let's store the array for now.
                                 if (!this.originalMaterials.has(child.uuid)) {
                                     this.originalMaterials.set(child.uuid, []);
                                 }
                                 this.originalMaterials.get(child.uuid).push(originalMat);
                             }


                             // Apply wireframe shader modification
                             this.modifyMaterialForWireframe(workingMat);

                             return workingMat;
                         });

                         // Assign back the processed materials
                         child.material = Array.isArray(child.material) ? processedMaterials : processedMaterials[0];
                         child.castShadow = true; // Enable shadows by default
                         child.receiveShadow = true;
                     }
                 });

                 // Calculate bounding box and center/scale model
                 const box = new THREE.Box3().setFromObject(this.model);
                 const size = box.getSize(new THREE.Vector3());
                 const center = box.getCenter(new THREE.Vector3());
                 const maxDim = Math.max(size.x, size.y, size.z);

                 let scaleFactor = 1;
                 const targetSize = 10; // Target size for normalization
                 if (maxDim > 0) {
                     scaleFactor = targetSize / maxDim;
                 }

                 this.model.scale.set(scaleFactor, scaleFactor, scaleFactor);
                 this.model.position.sub(center.multiplyScalar(scaleFactor)); // Center scaled model

                 this.shadowRoot.querySelector('#scale').value = scaleFactor; // Update UI scale slider


                 // Position camera based on scaled model size
                 const fov = this.camera.fov * (Math.PI / 180);
                 const cameraDist = (targetSize / 2) / Math.tan(fov / 2); // Distance based on target size
                 const cameraOrbitAttr = this.getAttribute('camera-orbit');
                 if (cameraOrbitAttr) {
                     this.setCameraOrbit(cameraOrbitAttr); // Apply attribute orbit
                 } else {
                      // Default camera position slightly above and back
                     this.camera.position.set(0, targetSize * 0.2, cameraDist * 1.5);
                     this.camera.lookAt(this.model.position); // Look at the centered model position
                     this.controls.target.copy(this.model.position); // Set orbit controls target
                     this.controls.update();
                 }

                 // Position grid helper under the model
                 this.gridHelper.position.set(this.model.position.x, box.min.y * scaleFactor + this.model.position.y, this.model.position.z);
                 this.gridHelper.scale.set(scaleFactor, scaleFactor, scaleFactor); // Scale grid if needed

                 // Update Model Info display
                 this.shadowRoot.querySelector('#modelInfo').innerHTML = `<strong>[Model Info]</strong> ${fileName}<br> Vertices: ${vertexCount.toLocaleString()}, Faces: ${faceCount.toLocaleString()}`;

                 // Generate Scene Graph UI
                 const sceneGraphTreeUI = this.shadowRoot.querySelector('#sceneGraphTree');
                 sceneGraphTreeUI.innerHTML = ''; // Clear previous tree
                 this.generateSceneGraphTree(this.model, sceneGraphTreeUI); // Generate new tree

                 // Setup Texture Editing UI
                 this.populateTextureMapSelector(); // Populate dropdowns
                 this.updateTextureMapDisplay();   // Show initial preview


                 // Determine if roughness/metalness can be adjusted globally
                 // Simplification: Allow if only one mesh part OR all parts lack rough/metal maps
                 const canAdjust = this.meshParts.length > 0 && this.meshParts.every(mesh =>
                     !mesh.material.roughnessMap && !mesh.material.metalnessMap
                 );
                 this.canAdjustRoughnessMetalness = canAdjust;
                 this.updateRoughnessMetalnessUI();


                 // Handle Animations
                 this.setupAnimations(fileExtension === 'obj' ? null : object.animations);

                 // Add model to scene
                 this.scene.add(this.model);

                 // Initial UI updates
                 this.updateControlPanel(); // Update transform inputs
                 this.updateLightsButtonUI();
                 this.updateViewModeButtons();
                 this.updateEnvButtons();
                 this.updateWireframeButton();
                 this.shadowRoot.querySelector('#discardModelBtn').style.display = 'inline-block'; // Show discard button
                 this.applyWireframeState(); // Ensure wireframe is applied based on current toggle state

             },
             (xhr) => { // --- Progress Callback ---
                 if (xhr.lengthComputable) {
                     const percentComplete = xhr.loaded / xhr.total * 100;
                     progressBar.style.width = `${Math.round(percentComplete)}%`;
                 }
             },
             (error) => { // --- Error Callback ---
                 progressBar.style.display = 'none';
                 const message = `Failed to load model: ${fileName || url}`;
                 this.showError(message);
                 console.error(message, error);
                  // Clean up object URL if it was created for this failed load
                  if (isObjectURL) {
                      this.revokeObjectURL(url);
                  }
                 this.showFileInput(); // Show file input again on error
             }
         );
     }

     setupAnimations(animations) {
         // Clear previous animation setup
         if (this.mixer) {
             this.mixer.stopAllAction();
             this.mixer = null;
         }
         this.animationActions = [];
         this.currentAction = null;
         const existingSelector = this.shadowRoot.querySelector('#animationSelector');
         if (existingSelector) existingSelector.remove();
         this.shadowRoot.querySelector('#anim_description').style.display = 'none';


         if (animations && animations.length > 0) {
             this.mixer = new THREE.AnimationMixer(this.model);
             this.animationActions = animations.map(clip => this.mixer.clipAction(clip));

             this.shadowRoot.querySelector('#anim_description').style.display = 'block';

             const animationSelector = document.createElement('select');
             animationSelector.id = 'animationSelector';
             animationSelector.style.width = '100%';
             animationSelector.title = 'Select Animation Clip';

             const noneOption = document.createElement('option');
             noneOption.value = '-1'; // Use -1 for none
             noneOption.textContent = 'None';
             animationSelector.appendChild(noneOption);

             animations.forEach((clip, index) => {
                 const option = document.createElement('option');
                 option.value = index.toString();
                 option.textContent = clip.name || `Animation ${index + 1}`;
                 animationSelector.appendChild(option);
             });

             animationSelector.value = '-1'; // Default to None

             animationSelector.addEventListener('change', (event) => {
                 const index = parseInt(event.target.value, 10);
                 if (this.currentAction) {
                      // Fade out previous action smoothly
                      this.currentAction.fadeOut(0.3);
                      // Don't stop immediately, let fadeOut complete
                 }
                 if (index >= 0 && this.animationActions[index]) {
                      this.currentAction = this.animationActions[index];
                      this.currentAction.reset().setEffectiveTimeScale(1).setEffectiveWeight(1).fadeIn(0.3).play();
                      this.state.isAnimationPlaying = true;
                 } else {
                      this.currentAction = null;
                      this.state.isAnimationPlaying = false; // Explicitly set false if 'None' selected
                 }
                 this.updateAnimationButtons();
             });

             // Append selector to the correct fieldset
             const utilFieldset = this.shadowRoot.querySelector('#render-tab-content fieldset:nth-of-type(4)'); // Util fieldset
             if (utilFieldset) {
                 utilFieldset.appendChild(animationSelector);
             }
         }
         this.updateAnimationButtons(); // Update buttons based on whether animations exist
     }


     renderMode() {
         let needsEnvMapUpdate = false;
         let targetEnvMap = null;
         let lightsShouldBeOn = this.state.lightsOn && !this.noPBR; // Base light state

         // Handle Environment Change
         if (this.state.environment) {
             lightsShouldBeOn = false; // Turn off scene lights when env map is active
             const envURL = this.getEnvironmentURL(this.state.environment);
             if (envURL) {
                  // Load environment only if it changed or wasn't loaded
                  if (!this.scene.environment || this.scene.environment.userData?.url !== envURL) {
                      needsEnvMapUpdate = true;
                      targetEnvMap = this.loadEnvironmentMap(envURL); // Returns Promise<Texture> or null
                  } else {
                      // Env already loaded and correct, just use it
                      targetEnvMap = this.scene.environment;
                  }
             }
         } else {
              // No environment selected, remove existing env map if present
              if (this.scene.environment) {
                  this.scene.background = null; // Remove background too
                  this.scene.environment = null;
                  needsEnvMapUpdate = true; // Need to update materials to remove envmap
              }
              targetEnvMap = null;
         }

         // Update Lights Visibility *before* material changes
         this.ambientLight.visible = lightsShouldBeOn;
         this.directionalLights.forEach(light => light.visible = lightsShouldBeOn);
         this.updateDirectionalLightHelpersVisibility();


         // Apply View Mode & Environment Map (potentially async)
         Promise.resolve(targetEnvMap).then(loadedEnvMap => {
             if (this.model) {
                 this.model.traverse((child) => {
                     if (child.isMesh) {
                          const originalMat = this.originalMaterials.get(child.uuid); // Get the stored original
                         let newMat = null;

                         // Choose material based on view mode
                         switch (this.state.viewMode) {
                             case 'diffuse':
                                 newMat = new THREE.MeshBasicMaterial({
                                      map: originalMat?.map, // Use original map
                                      color: originalMat?.map ? 0xffffff : (originalMat?.color || 0xffffff) // White if map, else original color
                                 });
                                 break;
                             case 'geometry':
                                 newMat = new THREE.MeshStandardMaterial({
                                     color: 0xffffff, // White base
                                     roughness: 0.8, metalness: 0.1, // Non-shiny default
                                     envMap: loadedEnvMap, // Use loaded env map
                                     envMapIntensity: loadedEnvMap ? 0.5 : 0 // Low intensity for geometry view
                                 });
                                 break;
                             case 'normal':
                                 newMat = new THREE.MeshNormalMaterial();
                                 break;
                             case 'default':
                             default:
                                 if (this.noPBR) {
                                      // Basic material if noPBR is forced
                                      newMat = new THREE.MeshBasicMaterial({
                                          map: originalMat?.map,
                                          color: originalMat?.map ? 0xffffff : (originalMat?.color || 0xffffff)
                                      });
                                 } else if (originalMat) {
                                      // Clone original for default PBR view
                                      newMat = originalMat.clone();
                                      // Apply environment map if loaded
                                      if (loadedEnvMap) {
                                           newMat.envMap = loadedEnvMap;
                                           newMat.envMapIntensity = 1.0; // Standard intensity
                                      } else {
                                          newMat.envMap = null; // Ensure no envmap if none loaded
                                      }
                                 } else {
                                     // Fallback if original somehow missing
                                     newMat = new THREE.MeshStandardMaterial({color: 0xcccccc});
                                 }
                                 break;
                         }

                         if (newMat) {
                             // Dispose old material if it's different from the original AND the new one
                              if (child.material && child.material !== originalMat && child.material !== newMat) {
                                   // Be careful here - might dispose shared materials unintentionally
                                   // Let's rely on discardModel for bulk cleanup instead of disposing here.
                                   // child.material.dispose();
                              }
                             // Apply the new material
                             child.material = newMat;
                             // IMPORTANT: Re-apply wireframe shader modification to the new material
                             this.modifyMaterialForWireframe(child.material);
                             child.material.needsUpdate = true;
                         }
                     }
                 });

                  // Update scene background/environment *after* traversing materials
                  this.scene.environment = loadedEnvMap;
                  this.scene.background = loadedEnvMap || (this.hasAttribute('background-color') ? new THREE.Color(this.getAttribute('background-color')) : new THREE.Color(0xeeeeee));

                  // Ensure wireframe state is applied correctly to the new materials
                  this.applyWireframeState();
                  this.updateRoughnessMetalnessUI(); // Update UI based on new material state
             }
             // Update UI buttons after changes applied
             this.updateViewModeButtons();
             this.updateEnvButtons();

         }).catch(error => {
              this.showError("Error applying rendering mode or environment.");
              console.error("RenderMode Error:", error);
              // Revert to a safe default state?
              this.state.viewMode = 'default';
              this.state.environment = null;
              this.scene.environment = null;
              this.scene.background = (this.hasAttribute('background-color') ? new THREE.Color(this.getAttribute('background-color')) : new THREE.Color(0xeeeeee));
              if(this.model) this.setDefaultMat(); // Revert model materials
              this.updateViewModeButtons();
              this.updateEnvButtons();
         });
     }

      getEnvironmentURL(envName) {
         switch (envName) {
             case 'env1': return 'https://huggingface.co/spaces/hhhwan/custom_gs/resolve/main/glbs/spruit_sunrise_1k_HDR.hdr';
             case 'env2': return 'https://huggingface.co/spaces/hhhwan/custom_gs/resolve/main/glbs/aircraft_workshop_01_1k.hdr';
             case 'env3': return 'https://huggingface.co/spaces/hhhwan/custom_gs/resolve/main/glbs/lebombo_1k.hdr';
             default: return null;
         }
     }

     // Cache loaded environments to avoid reloading
     _envCache = new Map();
     loadEnvironmentMap(url) {
          if (this._envCache.has(url)) {
              return Promise.resolve(this._envCache.get(url)); // Return cached texture
          }

          const rgbeLoader = new RGBELoader();
          return new Promise((resolve, reject) => {
              rgbeLoader.load(url, (texture) => {
                   texture.mapping = THREE.EquirectangularReflectionMapping;
                   texture.userData = { url: url }; // Store URL for comparison
                   this._envCache.set(url, texture); // Cache it
                   resolve(texture);
              },
              undefined, // Progress
              (error) => {
                   console.error(`Failed to load environment map: ${url}`, error);
                   reject(error); // Reject the promise on error
              });
          });
     }


    animate(time) {
        // Calculate deltaTime
        const deltaTime = (time - (this.lastTime || 0)) / 1000;
        this.lastTime = time;

        // Update controls
        this.controls.update(); // Required if enableDamping is true

        // Auto-rotate model
        if (this.autoRotate && this.model) {
            const rotationSpeed = THREE.MathUtils.degToRad(this.anglePerSecond);
            this.model.rotation.y += rotationSpeed * deltaTime;
        }

        // Update animation mixer
        if (this.mixer) {
            this.mixer.update(deltaTime);
        }

        // Update idle animation (if active)
        if (this.isIdleAnimationRunning && typeof TWEEN !== 'undefined') {
             TWEEN.update(time); // Update TWEEN animations
             if (this.animationMesh) { // Rotate the idle animation mesh
                 const rotationSpeedY = Math.PI / 6;
                 const rotationSpeedZ = Math.PI / 3;
                 const rotationSpeedX = Math.PI / 9;
                 this.animationMesh.rotation.y += rotationSpeedY * deltaTime;
                 this.animationMesh.rotation.z += rotationSpeedZ * deltaTime;
                 this.animationMesh.rotation.x += rotationSpeedX * deltaTime;
             }
        }

        // Render the scene
        this.renderer.render(this.scene, this.camera);
    }

    resizeRenderer() {
        const container = this.shadowRoot.querySelector('#canvas-container');
        const host = this.shadowRoot.host;
        // Use container dimensions if available, otherwise host
        const width = container.clientWidth || host.clientWidth;
        const height = container.clientHeight || host.clientHeight;

        if(width === 0 || height === 0) return; // Avoid resizing to zero

        this.camera.aspect = width / height;
        this.camera.updateProjectionMatrix();
        this.renderer.setSize(width, height);
    }

    // --- Helper Functions ---

    showFileInput() {
        this.shadowRoot.querySelector('#fileInputContainer').style.display = 'block';
    }

    hideFileInput() {
        this.shadowRoot.querySelector('#fileInputContainer').style.display = 'none';
    }

    // --- Other methods like setCameraOrbit, updateControlPanel, setBackground*, etc. ---
    // Review these for potential improvements based on the suggestions above.
    // For example, updateMaterialProperties should check canAdjustRoughnessMetalness.

    updateRoughnessMetalnessUI() {
         const roughnessInput = this.shadowRoot.querySelector('#roughness');
         const metalnessInput = this.shadowRoot.querySelector('#metalness');

         roughnessInput.disabled = !this.canAdjustRoughnessMetalness;
         metalnessInput.disabled = !this.canAdjustRoughnessMetalness;

         if (this.canAdjustRoughnessMetalness && this.meshParts.length > 0) {
              // Assuming all materials are the same if adjustable
             const mat = this.meshParts[0].material;
             roughnessInput.value = mat.roughness ?? 0.5; // Use nullish coalescing
             metalnessInput.value = mat.metalness ?? 0.5;
         } else {
             // Reset to default values if not adjustable
             roughnessInput.value = 0.5;
             metalnessInput.value = 0.5;
         }
    }

     updateModelScale(scale) {
         if (this.model) {
             this.model.scale.set(scale, scale, scale);
             // If you have a direct link between modelSize property and scale, update it too
             // this.modelSize = scale; // Assuming modelSize was intended to be the scale factor
         }
     }

     setCameraOrbit(value) {
        if (!value) return;
        const parts = value.split(' ').map(parseFloat);
        if (parts.length === 3 && !parts.some(isNaN)) {
            const [x, y, z] = parts;
            this.camera.position.set(x, y, z);
            // Also update controls target, assuming orbit looks at origin or model center
            const target = this.model ? this.model.position : new THREE.Vector3(0, 0, 0);
            this.camera.lookAt(target);
            this.controls.target.copy(target);
            this.controls.update(); // Apply changes immediately
        } else {
            console.warn("Invalid camera-orbit attribute value:", value);
        }
    }

     setTransformMode(mode) { // mode can be 'translate', 'rotate', 'scale', or null/undefined
         const translateBtn = this.shadowRoot.querySelector('#translateBtn');
         const rotateBtn = this.shadowRoot.querySelector('#rotateBtn');
         // Add scale button if you implement scale transform control

         // Deactivate previous mode
         translateBtn.classList.remove('active');
         rotateBtn.classList.remove('active');
         // scaleBtn?.classList.remove('active');

         if (mode && this.model) {
             // Activate new mode
             this.state.transformMode = mode;
             this.transformControls.setMode(mode);
             this.transformControls.attach(this.model);
             this.transformControls.enabled = true;
             this.transformControls.visible = true;
             this.controls.enabled = false; // Disable OrbitControls

             // Update button state
             if (mode === 'translate') translateBtn.classList.add('active');
             if (mode === 'rotate') rotateBtn.classList.add('active');
             // if (mode === 'scale') scaleBtn.classList.add('active');
         } else {
             // Deactivate all
             this.state.transformMode = null;
             if (this.transformControls.object) {
                 this.transformControls.detach();
             }
             this.transformControls.enabled = false;
             this.transformControls.visible = false;
             this.controls.enabled = true; // Re-enable OrbitControls
         }
         this.updateTransformInputVisibility(); // Show/hide relevant inputs
     }

      updateTransformInputVisibility() {
         // Show/hide relevant inputs based on this.state.transformMode
         // Example: Show position inputs for translate, rotation for rotate
         const showPos = this.state.transformMode === 'translate';
         const showRot = this.state.transformMode === 'rotate';
         this.shadowRoot.querySelector('#posX').parentElement.style.display = showPos ? 'flex': 'none';
         this.shadowRoot.querySelector('#posY').parentElement.style.display = showPos ? 'flex': 'none';
         this.shadowRoot.querySelector('#posZ').parentElement.style.display = showPos ? 'flex': 'none';
         this.shadowRoot.querySelector('#rotX').parentElement.style.display = showRot ? 'flex': 'none';
         this.shadowRoot.querySelector('#rotY').parentElement.style.display = showRot ? 'flex': 'none';
         this.shadowRoot.querySelector('#rotZ').parentElement.style.display = showRot ? 'flex': 'none';
         this.shadowRoot.querySelector('#scale').parentElement.style.display = this.state.transformMode ? 'flex' : 'none'; // Show scale slider always when transforming?
     }


     // Barycentric coordinates need to be added only once per geometry.
     addBarycentricCoordinates(geometry) {
         if (!geometry || geometry.getAttribute('barycentric')) {
             return; // Already added or no geometry
         }
         // Ensure geometry is non-indexed for simpler barycentric assignment per triangle
         if (geometry.index) {
             console.log("Converting geometry to non-indexed for barycentric coordinates.");
             geometry = geometry.toNonIndexed(); // This returns a *new* geometry
              // PROBLEM: How to replace the original geometry on the mesh?
              // This is complex. A better approach might be to calculate based on index if available.
              // Let's stick to the indexed approach calculation if possible.
             // Reverting to the simpler approach: calculate based on index if present
             geometry = geometry; // Use original geometry
         }

         const position = geometry.attributes.position;
         const count = position.count; // Number of vertices
         const barycentric = new Float32Array(count * 3);

         if (geometry.index) {
             // Indexed geometry: Assign barycentric per-vertex based on its role in triangles
             const index = geometry.index.array;
             // Initialize all to zero
             barycentric.fill(0);
             // Iterate through triangles
             for (let i = 0; i < index.length; i += 3) {
                 const iA = index[i + 0];
                 const iB = index[i + 1];
                 const iC = index[i + 2];
                 // Set barycentric coordinates: A=(1,0,0), B=(0,1,0), C=(0,0,1)
                 barycentric[iA * 3 + 0] = 1; // Vertex A is (1,0,0)
                 barycentric[iB * 3 + 1] = 1; // Vertex B is (0,1,0)
                 barycentric[iC * 3 + 2] = 1; // Vertex C is (0,0,1)
                  // Note: If a vertex is shared, the last triangle writing to it wins.
                  // This simple method works okay for basic wireframe rendering.
             }
         } else {
             // Non-indexed geometry: Each set of 3 vertices is a triangle
             for (let i = 0; i < count; i += 3) {
                 barycentric[(i + 0) * 3 + 0] = 1; // Vertex A: (1,0,0)
                 barycentric[(i + 0) * 3 + 1] = 0;
                 barycentric[(i + 0) * 3 + 2] = 0;

                 barycentric[(i + 1) * 3 + 0] = 0; // Vertex B: (0,1,0)
                 barycentric[(i + 1) * 3 + 1] = 1;
                 barycentric[(i + 1) * 3 + 2] = 0;

                 barycentric[(i + 2) * 3 + 0] = 0; // Vertex C: (0,0,1)
                 barycentric[(i + 2) * 3 + 1] = 0;
                 barycentric[(i + 2) * 3 + 2] = 1;
             }
         }
         geometry.setAttribute('barycentric', new THREE.BufferAttribute(barycentric, 3));
     }

     // Wireframe shader modification (keep as is, seems okay)
    modifyMaterialForWireframe(material) {
        if (!material || material.userData.hasWireframeShader) return; // Already modified or no material

        material.onBeforeCompile = (shader) => {
            // Check if uniforms already exist (might happen if material is cloned)
            if (shader.uniforms.uWireframe) {
                 material.userData.shader = shader; // Store shader reference
                 return;
            }
            shader.uniforms.uWireframe = { value: this.state.isWireframeOn }; // Init with current state

            shader.vertexShader = `
                attribute vec3 barycentric;
                varying vec3 vBarycentric;
                ${shader.vertexShader}
            `.replace(
                '#include <common>', // Inject after common includes
                `
                #include <common>
                vBarycentric = barycentric;
                `
            ).replace( // Ensure it's not added twice if #include <common> not present early
                 'void main() {',
                 'varying vec3 vBarycentric;\nvoid main() {'
            );


            shader.fragmentShader = `
                uniform bool uWireframe;
                varying vec3 vBarycentric;
                ${shader.fragmentShader}
            `.replace(
                '#include <dithering_fragment>', // Inject before final dithering
                `
                #include <dithering_fragment> // Include existing dithering logic

                if (uWireframe) {
                    // Calculate edge factor based on barycentric coordinates
                    vec3 d = fwidth(vBarycentric); // Use fwidth for consistent line thickness
                    vec3 a3 = smoothstep(vec3(0.0), d * 1.5, vBarycentric); // Wider smoothing
                    float edgeFactor = min(min(a3.x, a3.y), a3.z);

                    // Define wireframe color (e.g., black, semi-transparent)
                    vec4 wireframeColor = vec4(0.0, 0.0, 0.0, 0.5); // Black, 50% opacity

                    // Blend the wireframe color based on edgeFactor
                    // Mix based on wireframe alpha: if edgeFactor is near 0 (on edge), use more wireframe color.
                    gl_FragColor.rgb = mix(wireframeColor.rgb, gl_FragColor.rgb, edgeFactor);
                    // Blend alpha as well, making edges more opaque if base is transparent
                    gl_FragColor.a = mix(wireframeColor.a, gl_FragColor.a, edgeFactor);
                }
                `
            );
            material.userData.shader = shader; // Store shader reference for later uniform updates
        };
         // Flag that this material instance has been processed
         material.userData.hasWireframeShader = true;
         // Request recompilation
         // This might be necessary if the material was already used/compiled
         material.needsUpdate = true;
    }


     // Ensure all necessary methods are defined...
     updateControlPanel() {
         if (!this.model) return;
         // Update position inputs
         this.shadowRoot.querySelector('#posX').value = this.model.position.x.toFixed(2);
         this.shadowRoot.querySelector('#posY').value = this.model.position.y.toFixed(2);
         this.shadowRoot.querySelector('#posZ').value = this.model.position.z.toFixed(2);
         // Update rotation inputs (in degrees)
         this.shadowRoot.querySelector('#rotX').value = THREE.MathUtils.radToDeg(this.model.rotation.x).toFixed(1);
         this.shadowRoot.querySelector('#rotY').value = THREE.MathUtils.radToDeg(this.model.rotation.y).toFixed(1);
         this.shadowRoot.querySelector('#rotZ').value = THREE.MathUtils.radToDeg(this.model.rotation.z).toFixed(1);
         // Update scale input
         // Assuming uniform scaling for simplicity
         this.shadowRoot.querySelector('#scale').value = this.model.scale.x.toFixed(2);
     }

     // ... (add definitions for other methods like showTexture, showMesh, showNormal, set_bg, setBackgroundColor, takeScreenshotToClipboard, updateViewModeButtons, updateEnvButtons, updateModelTransform, generateSceneGraphTree, selectMeshPartInSceneGraph, createGlowMaterial, initIdleAnimation, etc. Make sure they are consistent with the refactored logic and state management.)

     // Example: Updating initLightUIValues
     initLightUIValues() {
         if (this.ambientLight) {
             this.shadowRoot.querySelector('#ambientColorPicker').value = `#${this.ambientLight.color.getHexString()}`;
             this.shadowRoot.querySelector('#ambientIntensity').value = this.ambientLight.intensity;
         }
         this.populateDirectionalLightList(); // Ensure list is populated
         if (this.directionalLights.length > 0 && this.directionalLights[this.selectedDirectionalLightIndex]) {
              this.updateDirectionalLightUIValues(); // Update UI for the selected light
         } else {
              // Reset directional UI if no lights exist
              this.shadowRoot.querySelector('#directColorPicker').value = '#ffffff';
              this.shadowRoot.querySelector('#directPosX').value = 0;
              this.shadowRoot.querySelector('#directPosY').value = 0;
              this.shadowRoot.querySelector('#directPosZ').value = 0;
              this.shadowRoot.querySelector('#directIntensity').value = 1;
         }
     }

      // Example: Updating addDirectionalLight
      addDirectionalLight() {
         const lightCount = this.directionalLights.length;
         const newLight = new THREE.DirectionalLight(0xffffff, 1); // Default intensity 1
         // Position new lights slightly differently
         newLight.position.set(5 + lightCount * 2, 5, 5 - lightCount * 2);
         newLight.castShadow = true;
         // Configure shadow properties if needed
         // newLight.shadow.mapSize.width = 1024;
         // newLight.shadow.mapSize.height = 1024;
         // newLight.shadow.camera.near = 0.5;
         // newLight.shadow.camera.far = 50;

         this.directionalLights.push(newLight);
         this.scene.add(newLight);

         const helper = new THREE.DirectionalLightHelper(newLight, 1, 0xffaa00); // Orange helper
         this.directionalLightHelpers.push(helper);
         this.scene.add(helper);

         // Select the newly added light
         this.selectedDirectionalLightIndex = this.directionalLights.length - 1;

         // Update UI
         this.populateDirectionalLightList(); // Rebuild dropdown
         this.updateDirectionalLightUIValues(); // Update inputs for the new light
         this.updateDirectionalLightHelpersVisibility(); // Update helper visibility
         this.updateLightVisibility(); // Update main light visibility based on global state
     }

      removeDirectionalLight() {
         if (this.directionalLights.length === 0) return; // No lights to remove

         const indexToRemove = this.selectedDirectionalLightIndex;
         const lightToRemove = this.directionalLights[indexToRemove];
         const helperToRemove = this.directionalLightHelpers[indexToRemove];

         // Remove from scene
         if (lightToRemove) this.scene.remove(lightToRemove);
         if (helperToRemove) this.scene.remove(helperToRemove);
         // Dispose shadows? (If applicable)
         // lightToRemove.shadow?.dispose(); // Check if shadow exists before disposing

         // Remove from arrays
         this.directionalLights.splice(indexToRemove, 1);
         this.directionalLightHelpers.splice(indexToRemove, 1);

         // Adjust selected index
         this.selectedDirectionalLightIndex = Math.max(0, indexToRemove - 1);
          if (this.directionalLights.length === 0) {
               this.selectedDirectionalLightIndex = -1; // Indicate no selection
          }


         // Update UI
         this.populateDirectionalLightList();
         this.updateDirectionalLightUIValues(); // Update inputs (will reset if no lights left)
     }

      removeAllDirectionalLights() {
           while(this.directionalLights.length > 0) {
               this.selectedDirectionalLightIndex = 0; // Always remove the first one
               this.removeDirectionalLight();
           }
      }


} // --- End Class ---

customElements.define('simple-model-viewer', SimpleModelViewer);
export { SimpleModelViewer };