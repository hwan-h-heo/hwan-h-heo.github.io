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
        this.renderer = new THREE.WebGLRenderer({ antialias: true });
        this.renderer.setClearColor(0xeeeeee, 1);
        this.shadowRoot.querySelector('#canvas-container').appendChild(this.renderer.domElement);

        this.controls = new OrbitControls(this.camera, this.renderer.domElement);
        this.controls.enableDamping = true;
        this.controls.dampingFactor = 0.05;

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
    }

    updateModelTransform() {
        if (this.model) {
            const posX = parseFloat(this.shadowRoot.querySelector('#posX').value);
            const posY = parseFloat(this.shadowRoot.querySelector('#posY').value);
            const posZ = parseFloat(this.shadowRoot.querySelector('#posZ').value);
            this.model.position.set(posX, posY - 3, posZ);

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
            texture.mapping = THREE.EquirectangularReflectionMapping;
            this.scene.background = texture;
            if (this.model) {
                this.model.traverse((child) => {
                    if (child.isMesh) {
                        child.material = new THREE.MeshStandardMaterial({
                            map: this.originalMaterials[child.uuid].map,
                            envMap: texture,
                            envMapIntensity: 1.0,
                            roughness: this.originalMaterials[child.uuid].roughness,
                            metalness: this.originalMaterials[child.uuid].metalness,
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
            progressBar.style.display = 'none'; // hide
            }, (xhr) => { // onProgress call back
                if (xhr.lengthComputable) {
                    const percentComplete = xhr.loaded / xhr.total * 100;
                    progressBar.style.width = `${percentComplete}%`; // bar update
                }
        }, undefined, (error) => {
            console.error('Loading Error:', error);
        });
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
}

customElements.define('simple-model-viewer', SimpleModelViewer);
export { SimpleModelViewer };