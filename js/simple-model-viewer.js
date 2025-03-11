import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';
import { RGBELoader } from 'three/addons/loaders/RGBELoader.js';

class SimpleModelViewer extends HTMLElement {
    constructor() {
        super();
        this.attachShadow({ mode: 'open' });
        this.shadowRoot.innerHTML = `
            <style>
                :host { display: block; border: 2px solid #ccc; border-radius: 8px;}
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
                #canvas-container { width: 100%; height: 100%; position: relative; }
                canvas { width: 100%; height: 100%; }
                .controls { margin: 5px; }
                button {
                    background-color: #04AA6D;
                    border: none;
                    color: white;
                    padding: 5px 10px;
                    border-radius: 4px;
                    cursor: pointer;
                    text-align: center;
                    text-decoration: none;
                    display: inline-block;
                    font-size: 0.8rem;
                }
                button:hover { background-color: #3e8e41 }
                .transform-panel {
                    position: absolute;
                    top: 0.5rem;
                    right: 0.5rem;
                    font-size: 0.8rem;
                    background-color: rgba(200, 200, 200, 0.8);
                    padding: 0.5rem;
                    border-radius: 5px;
                    display: flex;
                    flex-direction: column;
                    gap: 3px;
                    z-index: 1000;
                }
                .transform-panel label {
                    display: flex;
                    justify-content: space-between;
                    align-items: center;
                }
                .transform-panel input { width: 3rem; }
            </style>
            <div class="controls">
                <div id='meta'>
                    <button id="textureBtn">Diffuse</button>
                    <button id="meshBtn">Geometry</button>
                    <button id="normalBtn">Normal</button>
                    <button id="wireframeBtn">Wireframe</button>
                    <button id="autoRotateBtn">Auto-Rotate</button>
                    <button id="toonShadingBtn">Toon Shading</button>
                    <button id="setBgBtn1">Env1</button>
                    <button id="setBgBtn2">Env2</button>
                    <button id="setBgBtn3">Env3</button>
                    <button id="removeBgBtn">Remove Env</button>
                    <div id="modelInfo" style='padding-left: 0.1rem; font-size:0.8rem'><strong>[Model Info]</strong> loading...</div>
                </div>
                <div id="transform-container" style="position: relative;">
                    <div class="transform-panel">
                        <button id="togglePanelBtn"><i class="bi bi-caret-left"></i></button>
                        <div id="transformControls" style="display: block;">
                            <label>Position X: <input type="number" id="posX" step="0.1" value="0"></label>
                            <label>Position Y: <input type="number" id="posY" step="0.1" value="0"></label>
                            <label>Position Z: <input type="number" id="posZ" step="0.1" value="0"></label>
                            <label>Rotation X (deg): <input type="number" id="rotX" step="1" value="0"></label>
                            <label>Rotation Y (deg): <input type="number" id="rotY" step="1" value="0"></label>
                            <label>Rotation Z (deg): <input type="number" id="rotZ" step="1" value="0"></label>
                            <div>Scale: <input type="range" id="scale" style="width: 7rem;" min="1" max="20" step="0.1" value="8"></div>
                            <div>Roughness: <input type="range" id="roughness" style="width: 7rem;" min="0" max="1" step="0.01" value="0.5"></div> 
                            <div>Metalness: <input type="range" id="metalness" style="width: 7rem;" min="0" max="1" step="0.01" value="0.5"></div> 
                        </div>
                    </div>
                </div>
            </div>
            <div id="canvas-container" style='text-align: center'>
                <div id="loadingProgressBar"></div>
            </div>
        `;

        this.scene = new THREE.Scene();
        this.camera = new THREE.PerspectiveCamera(50, 1, 0.1, 1000);
        this.renderer = new THREE.WebGLRenderer({ 
            antialias: true, 
            alpha: true,
            preserveDrawingBuffer: true
        });
        this.renderer.setPixelRatio(window.devicePixelRatio);
        // this.renderer.outputEncoding = THREE.LinearEncoding;
        this.renderer.outputEncoding = THREE.sRGBEncoding;
        this.renderer.setClearColor(0xeeeeee, 1);
        this.renderer.shadowMap.enabled = true;
        this.renderer.toneMapping = THREE.ACESFilmicToneMapping;
        this.renderer.toneMappingExposure = 1.0;

        this.shadowRoot.querySelector('#canvas-container').appendChild(this.renderer.domElement);

        // const ambient = new THREE.AmbientLight(0x404040, 0.5);      
        // const directional = new THREE.DirectionalLight(0xffffff, 1); 
        // directional.position.set(5, 10, 7.5);
        // directional.castShadow = true;
        // this.scene.add(ambient, directional);

        this.controls = new OrbitControls(this.camera, this.renderer.domElement);
        this.controls.enableDamping = true;
        this.controls.dampingFactor = 0.05;
        this.controls.addEventListener('change', () => this.updateControlPanel()); 

        this.textureLoader = new THREE.TextureLoader();
        this.whiteTexture = this.textureLoader.load('https://huggingface.co/spaces/hhhwan/custom_gs/resolve/main/glbs/white.jpg');
        this.whiteTexture.mapping = THREE.EquirectangularReflectionMapping;

        this.gradTexture = this.textureLoader.load('https://huggingface.co/spaces/hhhwan/custom_gs/resolve/main/glbs/gradient.jpg');
        this.gradTexture.mapping = THREE.EquirectangularReflectionMapping;

        this.loader = new GLTFLoader();
        this.model = null;
        this.originalMaterials = {};
        this.wireframeMeshes = [];
        this.modelSize = 8;
        this.autoRotate = false;
        this.anglePerSecond = 30;
        this.lastTime = 0;
        this.toonEnabled = false; 
        this.toonMaterial = null;
        this.standardMaterials = [];

        this.initEventListeners();
    }

    connectedCallback() {
        this.resizeRenderer();
        this.animate(0);
    }

    static get observedAttributes() {
        return ['src', 'auto-rotate', 'angle-per-second', 'camera-orbit', 'hide-control-ui'];
    }

    attributeChangedCallback(name, oldValue, newValue) {
        if (name === 'src' && newValue) {
            this.loadModel(newValue);
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
        }
    }

    initEventListeners() {
        this.shadowRoot.querySelector('#textureBtn').addEventListener('click', () => this.showTexture());
        this.shadowRoot.querySelector('#meshBtn').addEventListener('click', () => this.showMesh());
        this.shadowRoot.querySelector('#wireframeBtn').addEventListener('click', () => this.showWireframe());
        this.shadowRoot.querySelector('#normalBtn').addEventListener('click', () => this.showNormal());
        this.shadowRoot.querySelector('#setBgBtn1').addEventListener('click', () => this.setBackground1());
        this.shadowRoot.querySelector('#setBgBtn2').addEventListener('click', () => this.setBackground2());
        this.shadowRoot.querySelector('#setBgBtn3').addEventListener('click', () => this.setBackground3());
        this.shadowRoot.querySelector('#removeBgBtn').addEventListener('click', () => this.removeBG());

        this.shadowRoot.querySelector('#posX').addEventListener('input', () => this.updateModelTransform());
        this.shadowRoot.querySelector('#posY').addEventListener('input', () => this.updateModelTransform());
        this.shadowRoot.querySelector('#posZ').addEventListener('input', () => this.updateModelTransform());
        this.shadowRoot.querySelector('#rotX').addEventListener('input', () => this.updateModelTransform());
        this.shadowRoot.querySelector('#rotY').addEventListener('input', () => this.updateModelTransform());
        this.shadowRoot.querySelector('#rotZ').addEventListener('input', () => this.updateModelTransform());

        this.shadowRoot.querySelector('#roughness').addEventListener('input', () => this.updateMaterialProperties());
        this.shadowRoot.querySelector('#metalness').addEventListener('input', () => this.updateMaterialProperties());

        this.shadowRoot.querySelector('#scale').addEventListener('input', (e) => {
            this.modelSize = parseFloat(e.target.value);
            if (this.model) this.model.scale.set(this.modelSize, this.modelSize, this.modelSize);
        });

        this.shadowRoot.querySelector('#autoRotateBtn').addEventListener('click', () => {
            this.autoRotate = !this.autoRotate;
            this.shadowRoot.querySelector('#autoRotateBtn').textContent = this.autoRotate ? 'Auto-Rotate On' : 'Auto-Rotate Off';
        });

        this.shadowRoot.querySelector('#togglePanelBtn').addEventListener('click', () => {
            const controls = this.shadowRoot.querySelector('#transformControls');
            const button = this.shadowRoot.querySelector('#togglePanelBtn');
            if (controls.style.display === 'none') {
                controls.style.display = 'block';
                button.innerHTML = '<i class="bi bi-caret-left-fill"></i>';
            } else {
                controls.style.display = 'none';
                button.innerHTML = '<i class="bi bi-caret-left"></i>';
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
                }
            });
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
                }
            });
        }
    }

    showWireframe() {
        if (this.model) {
            if (this.wireframeMeshes.length > 0) {
                this.wireframeMeshes.forEach((mesh) => this.model.remove(mesh));
                this.wireframeMeshes = [];
            } else {
                this.model.traverse((child) => {
                    if (child.isMesh) {
                        const wireframeMesh = new THREE.Mesh(child.geometry, new THREE.MeshBasicMaterial({
                            wireframe: true,
                            color: 0xaaaaaa,
                            depthTest: true,
                            transparent: true,
                            opacity: 0.8
                        }));
                        this.model.add(wireframeMesh);
                        this.wireframeMeshes.push(wireframeMesh);
                    }
                });
            }
        }
    }

    showNormal() {
        if (this.model) {
            this.model.traverse((child) => {
                if (child.isMesh) {
                    child.material = new THREE.MeshNormalMaterial();
                }
            });
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
                let initialRoughness = 0.5, initialMetalness = 0.5;
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
                            normalMap: originalMaterial.normalMap ? originalMaterial.normalMap : null,
                            emissiveMap: originalMaterial.emissiveMap ? originalMaterial.emissiveMap : null,
                        });
                        if (child.material.map) {
                            child.material.map.encoding = THREE.sRGBEncoding; 
                        }
                        if (child.material.emissiveMap) {
                            child.material.emissiveMap.encoding = THREE.sRGBEncoding;
                        }
                        child.material.needsUpdate = true;
    
                        // ui init
                        if (!initialRoughness && !initialMetalness && isStandardMaterial) {
                            initialRoughness = child.material.roughness;
                            initialMetalness = child.material.metalness;
                        }
                    }
                });
                const roughnessInput = this.shadowRoot.querySelector('#roughness');
                const metalnessInput = this.shadowRoot.querySelector('#metalness');
                if (roughnessInput) roughnessInput.value = initialRoughness;
                if (metalnessInput) metalnessInput.value = initialMetalness;
            }
            this.renderer.render(this.scene, this.camera);
        }, undefined, (err) => {
            console.error('Skybox err:', err);
            alert('Cannot load Skybox Image');
        });
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

    removeBG() {
        this.scene.background = null;
        this.scene.environment = null;
        if (this.model) {
            this.model.traverse((child) => {
                if (child.isMesh) {
                    child.material = new THREE.MeshBasicMaterial({ map: this.originalMaterials[child.uuid].map, envMap: null });
                    child.material.needsUpdate = true;
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

    loadModel(url) {
        const progressBar = this.shadowRoot.querySelector('#loadingProgressBar');
        progressBar.style.display = 'block'; 
        progressBar.style.width = '0%'

        this.loader.load(url, (gltf) => {
            if (this.model) {
                this.scene.remove(this.model);
            }
            this.model = gltf.scene;
            this.model.scale.set(this.modelSize, this.modelSize, this.modelSize);

            const box = new THREE.Box3().setFromObject(this.model);
            const center = box.getCenter(new THREE.Vector3());
            this.model.position.sub(center);

            const size = box.getSize(new THREE.Vector3());
            const maxDim = Math.max(size.x, size.y, size.z);
            const fov = this.camera.fov * (Math.PI / 180);
            let cameraZ = Math.abs(maxDim / 2 / Math.tan(fov / 2));

            const cameraOrbit = this.getAttribute('camera-orbit');
            if (cameraOrbit) {
                this.setCameraOrbit(cameraOrbit);
            } else {
                this.camera.position.set(0, 0, cameraZ * 1.5);
                this.camera.lookAt(0, 0, 0);
            }

            let vertexCount = 0, faceCount = 0;
            this.standardMaterials = []; 
            let initialRoughness = 0.5; 
            let initialMetalness = 0.5;  
            let standardMaterialFound = false;

            this.model.traverse((child) => {
                if (child.isMesh && child.geometry) {
                    vertexCount += child.geometry.attributes.position.count;
                    faceCount += child.geometry.index ? child.geometry.index.count / 3 : child.geometry.attributes.position.count / 3;

                    this.originalMaterials[child.uuid] = child.material;
                    if (child.material instanceof THREE.MeshStandardMaterial) {
                        this.standardMaterials.push(child.material);
                        if (!standardMaterialFound) { 
                            initialRoughness = child.material.roughness || 0.5;
                            initialMetalness = child.material.metalness || 0.5;
                            standardMaterialFound = true;

                            const roughnessInput = this.shadowRoot.querySelector('#roughness');
                            const metalnessInput = this.shadowRoot.querySelector('#metalness');
                            if (roughnessInput) roughnessInput.value = initialRoughness;
                            if (metalnessInput) metalnessInput.value = initialMetalness;
                        }
                        const material = child.material;
                        if (material.map) {
                            material.map.encoding = THREE.sRGBEncoding;
                        }
                        if (material.emissiveMap) {
                            material.emissiveMap.encoding = THREE.sRGBEncoding;
                        }
                        material.needsUpdate = true;
                    } else {
                        // not Standard
                        child.material = new THREE.MeshStandardMaterial({
                            map: child.material.map,
                            roughness: 0.5,
                            metalness: 0.5
                        });
                        if (child.material.map) {
                            child.material.map.encoding = THREE.sRGBEncoding;
                        }
                        child.material.needsUpdate = true;
                        this.standardMaterials.push(child.material);
                    }
                }
            });
            this.shadowRoot.querySelector('#roughness').value = initialRoughness;
            this.shadowRoot.querySelector('#metalness').value = initialMetalness;

            this.shadowRoot.querySelector('#modelInfo').innerHTML = `<strong>[Model Info]</strong> Vertices: ${vertexCount}, Faces: ${faceCount}`;
            this.scene.add(this.model);

            this.showTexture();
            this.updateControlPanel(); 

            progressBar.style.display = 'none'; // hide progress bar
            }, (xhr) => { // onProgress call back
                if (xhr.lengthComputable) {
                    const percentComplete = xhr.loaded / xhr.total * 100;
                    progressBar.style.width = `${percentComplete}%`; // bar update
                }
        }, undefined, (error) => {
            console.error('Loading Error:', error);
        });
    }

    updateMaterialProperties() { 
        const roughnessValue = parseFloat(this.shadowRoot.querySelector('#roughness').value);
        const metalnessValue = parseFloat(this.shadowRoot.querySelector('#metalness').value);

        this.standardMaterials.forEach(material => { 
            material.roughness = roughnessValue;
            material.metalness = metalnessValue;
            material.needsUpdate = true;
        });
        if (this.model) {
            this.model.traverse((child) => {
                if (child.isMesh) {
                    const originalMaterial = this.originalMaterials[child.uuid];
                    const isStandardMaterial = originalMaterial instanceof THREE.MeshStandardMaterial;

                    child.material = new THREE.MeshStandardMaterial({
                        map: originalMaterial.map ? originalMaterial.map.clone() : null,
                        envMap: this.scene.environment || this.scene.background,
                        envMapIntensity: 1.0,
                        roughness: roughnessValue,
                        metalness: metalnessValue,
                        normalMap: originalMaterial.normalMap ? originalMaterial.normalMap : null, 
                        emissiveMap: originalMaterial.emissiveMap ? originalMaterial.emissiveMap : null,
                    });
                    if (child.material.map) {
                        child.material.map.encoding = THREE.sRGBEncoding; 
                    }
                    if (child.material.emissiveMap) {
                        child.material.emissiveMap.encoding = THREE.sRGBEncoding;
                    }
                    child.material.needsUpdate = true;
                }
            });
        }
        this.renderer.render(this.scene, this.camera);
    }

    animate(time) {
        requestAnimationFrame((t) => this.animate(t));
        const deltaTime = (time - this.lastTime) / 1000;
        this.lastTime = time;

        if (this.autoRotate && this.model) {
            const rotationSpeed = THREE.MathUtils.degToRad(this.anglePerSecond);
            this.model.rotation.y += rotationSpeed * deltaTime;
        }

        this.controls.update();
        this.renderer.render(this.scene, this.camera);
    }

    resizeRenderer() {
        const host = this.shadowRoot.host;
        const metaDiv = this.shadowRoot.querySelector('#meta');
        const metaHeight = metaDiv ? metaDiv.offsetHeight : 0;
        const width = host.clientWidth * 0.99;
        const height = host.clientHeight - metaHeight;

        this.renderer.setSize(width, height * 0.96);
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
            vertexShader: `
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
            fragmentShader: `
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
}

customElements.define('simple-model-viewer', SimpleModelViewer);
export { SimpleModelViewer };