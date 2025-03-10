import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';
import { RGBELoader } from 'three/addons/loaders/RGBELoader.js';

class SimpleModelViewer extends HTMLElement {
    constructor() {
        super();
        // Shadow DOM을 사용해 내부 스타일과 구조를 캡슐화
        this.attachShadow({ mode: 'open' });
        this.shadowRoot.innerHTML = `
            <style>
                :host { display: block; border: 2px solid #ccc; border-radius: 8px;}
                #canvas-container { width: 100%; height: 100%; }
                canvas { width: 100%; height: 100%; }
                .controls { margin: 5px; }
                button {
                    background-color: #04AA6D; /* Green */
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
                button:hover{
                    background-color: #3e8e41
                }
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
                }
                
                .transform-panel label {
                    display: flex;
                    justify-content: space-between; 
                    align-items: center;
                }
                
                .transform-panel input {
                    width: 3rem; 
                }
            </style>
            <div class="controls">
                <div id='meta'>
                    <button id="textureBtn">Diffuse</button>
                    <button id="meshBtn">Geometry</button>
                    <button id="normalBtn">Normal</button>
                    <button id="wireframeBtn">Wireframe</button>
                    <button id="setBgBtn1">Env1</button>
                    <button id="setBgBtn2">Env2</button>
                    <button id="setBgBtn3">Env3</button>
                    <button id="removeBgBtn">Remove Env</button>

                    <div id="modelInfo" style='padding-left: 0.1rem; font-size:0.8rem'><strong>[Model Info]</strong> loading...</div>
                </div>
                <div id="canvas-container" style="position: relative;">
                    <div class="transform-panel">
                        <label>Position X: <input type="number" id="posX" step="0.1" value="0"></label>
                        <label>Position Y: <input type="number" id="posY" step="0.1" value="0"></label>
                        <label>Position Z: <input type="number" id="posZ" step="0.1" value="0"></label>
                        <label>Rotation X (deg): <input type="number" id="rotX" step="1" value="0"></label>
                        <label>Rotation Y (deg): <input type="number" id="rotY" step="1" value="0"></label>
                        <label>Rotation Z (deg): <input type="number" id="rotZ" step="1" value="0"></label>
                        <div>Scale: <input type="range" id="scale" style='width: 7rem; ' min="1" max="20" step="0.1" value="8"></div>
                    </div>
                </div>
            </div>
            <div id="canvas-container" style='text-align: center'></div>
        `;

        // Three.js settings
        this.scene = new THREE.Scene();
        this.camera = new THREE.PerspectiveCamera(50, 1, 0.1, 1000); //
        this.renderer = new THREE.WebGLRenderer({ antialias: true });
        this.renderer.setClearColor(0xeeeeee, 1);
        this.shadowRoot.querySelector('#canvas-container').appendChild(this.renderer.domElement);

        // OrbitControls for mouse control
        this.controls = new OrbitControls(this.camera, this.renderer.domElement);
        this.controls.enableDamping = true;
        this.controls.dampingFactor = 0.05;

        // Init texture and loader
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

        this.initEventListeners();
    }

    // DOM
    connectedCallback() {
        this.resizeRenderer(); // init resizing
        this.animate();
    }

    static get observedAttributes() {
        return ['src'];
    }

    // src track
    attributeChangedCallback(name, oldValue, newValue) {
        if (name === 'src' && newValue) {
            this.loadModel(newValue);
        }
    }

    // model load
    loadModel(url) {
        this.loader.load(url, (gltf) => {
            if (this.model) {
                this.scene.remove(this.model);
            }
            this.model = gltf.scene;
            this.model.scale.set(this.modelSize, this.modelSize, this.modelSize);
            this.model.position.set(0, -3, 0);

            // model info
            let vertexCount = 0, faceCount = 0;
            this.model.traverse((child) => {
                if (child.isMesh && child.geometry) {
                    vertexCount += child.geometry.attributes.position.count;
                    faceCount += child.geometry.index ? child.geometry.index.count / 3 : child.geometry.attributes.position.count / 3;
                    this.originalMaterials[child.uuid] = child.material;
                }
            });
            this.shadowRoot.querySelector('#modelInfo').innerHTML = `<strong>[Model Info]</strong> Vertices: ${vertexCount}, Faces: ${faceCount}`;
            this.scene.add(this.model);

            this.showTexture();

            this.camera.position.set(0, 5, 15);
            this.camera.lookAt(0, 0, 0);
        }, undefined, (error) => {
            console.error('Loading Error:', error);
        });
    }

    // Button Event Listner
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

        this.shadowRoot.querySelector('#scale').addEventListener('input', (e) => {
            this.modelSize = parseFloat(e.target.value);
            if (this.model) this.model.scale.set(this.modelSize, this.modelSize, this.modelSize);
        });
    }

    updateModelTransform() {
        if (this.model) {
            const posX = parseFloat(this.shadowRoot.querySelector('#posX').value);
            const posY = parseFloat(this.shadowRoot.querySelector('#posY').value);
            const posZ = parseFloat(this.shadowRoot.querySelector('#posZ').value);
            this.model.position.set(posX, posY-3, posZ);

            const rotX = THREE.MathUtils.degToRad(parseFloat(this.shadowRoot.querySelector('#rotX').value));
            const rotY = THREE.MathUtils.degToRad(parseFloat(this.shadowRoot.querySelector('#rotY').value));
            const rotZ = THREE.MathUtils.degToRad(parseFloat(this.shadowRoot.querySelector('#rotZ').value));
            this.model.rotation.set(rotX, rotY, rotZ);
        }
    }

    // Show Diffuse 
    showTexture() {
        if (this.model) {
            this.model.traverse((child) => {
                if (child.isMesh) {
                    child.material = new THREE.MeshBasicMaterial({
                        map: this.originalMaterials[child.uuid].map,
                    });
                }
            });
        }
    }

    // Mesh Geometry
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

    // Wireframe toggle
    showWireframe() {
        if (this.model) {
            if (this.wireframeMeshes.length > 0) {
                this.wireframeMeshes.forEach((mesh) => {
                    this.model.remove(mesh); // remove wireframe
                });
                this.wireframeMeshes = [];
            } 
            // make wireframe
            else {
                this.model.traverse((child) => {
                    if (child.isMesh) {
                        const wireframeMesh = new THREE.Mesh(child.geometry, new THREE.MeshBasicMaterial({
                            wireframe: true,
                            color: 0xaaaaaa, 
                            depthTest: true,
                            transparent: true,
                            opacity: 0.8 // transparent
                        }));
                        this.model.add(wireframeMesh); // add as child 
                        this.wireframeMeshes.push(wireframeMesh); 
                    }
                });
            }
        }
    }

    // Normal Material
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
            texture.mapping = THREE.EquirectangularReflectionMapping;
            this.scene.background = texture;
            // this.scene.environment = texture;
            if (this.model) {
                this.model.traverse((child) => {
                    if (child.isMesh) {
                        child.material = new THREE.MeshStandardMaterial({
                            map: this.originalMaterials[child.uuid].map,
                            envMap: texture,
                            envMapIntensity: 1.0,
                            roughness: 0.99,
                            metalness: 0.01
                        });
                        child.material.needsUpdate = true;
                    }
                });
            }
        }, undefined, (err) => {
            console.error('Skybox err:', err);
            alert('Cannot load Skybox Image');
        });
    }

    // Skybox
    setBackground1() {
        const url = 'https://huggingface.co/spaces/hhhwan/custom_gs/resolve/main/glbs/spruit_sunrise_1k_HDR.hdr';
        const rgbeLoader = new RGBELoader();
        this.set_bg(url, rgbeLoader)
    }
    setBackground2() {
        const url = 'https://huggingface.co/spaces/hhhwan/custom_gs/resolve/main/glbs/aircraft_workshop_01_1k.hdr';
        const rgbeLoader = new RGBELoader();
        this.set_bg(url, rgbeLoader)
    }

    setBackground3() {
        const url = 'https://huggingface.co/spaces/hhhwan/custom_gs/resolve/main/glbs/lebombo_1k.hdr';
        const rgbeLoader = new RGBELoader();
        this.set_bg(url, rgbeLoader)
    }


    // Remove BG
    removeBG() {
        this.scene.background = null;
        this.scene.environment = null;
        if (this.model) {
            this.model.traverse((child) => {
                if (child.isMesh) {
                    child.material = new THREE.MeshBasicMaterial({
                        map: this.originalMaterials[child.uuid].map,
                        envMap: null,
                    });
                    child.material.needsUpdate = true;
                }
            });
        }
    }

    animate() {
        requestAnimationFrame(() => this.animate());
        this.controls.update();
        this.renderer.render(this.scene, this.camera);
    }

    // Resizing for the window
    resizeRenderer() {
        const host = this.shadowRoot.host;
        const metaDiv = this.shadowRoot.querySelector('#meta');
        const metaHeight = metaDiv ? metaDiv.offsetHeight : 0; // meta div
        const width = host.clientWidth * 0.99; 
        const height = host.clientHeight - metaHeight; // rest

        // this.shadowRoot.host.style.setProperty('--meta-height', `${metaHeight}px`);

        this.renderer.setSize(width, height * 0.96);
        this.camera.aspect = width / height;
        this.camera.updateProjectionMatrix();
    }
}

// custom element 
customElements.define('simple-model-viewer', SimpleModelViewer);
export { SimpleModelViewer };