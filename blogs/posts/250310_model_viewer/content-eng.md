title: 3D Model Viewer in Web
date: March 10, 2025
author: Hwan Heo
--- 여기부터 실제 콘텐츠 ---

<!-- 2. TOC(목차) 추가 -->
<nav class="toc">
    <ul>
        <li><a href="#h2-3"> Introduction </a><ul></ul></li>
        <li><a href="#h2-1"> Google Model Viewer </a>
            <ul>
                <li><a href="#h3-1"> Basic Usage </a></li>
                <li><a href="#h3-2"> Geometry Rendering </a></li>
            </ul>
        </li>
        <li><a href="#h2-2"> Threejs-Based Custom Viewer </a>
            <ul>
                <li><a href="#h3-3"> Basic Usage </a></li>
                <li><a href="#h3-4"> Custom Viewer </a></li>
                <li><a href="#h3-5"> Key Takeaways </a></li>
            </ul>
        </li>
        <li><a href='#conclusion'>Conclusion</a></li>
    </ul>
</nav>

<button id="copyButton">
    <i class="bi bi-share-fill"></i>
</button>

<div id="myshare_modal" class="share_modal">
    <div class="share_modal-content">
        <span class="share_modal_close">×</span>
        <p><strong>Link Copied!</strong></p>
        <div class="copy_indicator-container">
        <div class="copy_indicator" id="share_modalIndicator"></div>
        </div>
    </div>
</div>

<hr/>

<!-- <img src='./250310_model_viewer/assets/image.gif' width=100%> -->


<h2 id="h2-3" > Introduction </h2>
<p id="p-1" class='lang eng'> Since my work tends to revolve around the 3D domain,
    I occasionally find myself needing to render 3D models on web pages—whether it’s for a quick demo to present internally or to build a project page for a research paper. </p>
<p id="p-2" class='lang eng'> For someone like me who isn’t deeply familiar with web development, there are two straightforward options: </p>
<ol id="ol-1" class="lang eng"> <li>Use Google Model Viewer, or</li> <li id="li-1">Build a simple viewer from scratch using Three.js.</li> </ol>
<p id="p-3" class='lang eng'> In this post, I’ll dive into my personal experiences with both approaches, exploring their usage, pros, and cons.
    I’ll also share details about a custom Three.js-based <code>SimpleModelViewer</code> I’ve developed. If you’re only interested in the custom viewer itself, feel free to jump straight to
    <a href="#h4-3">this section</a>!
</p>



<h2 id="h2-1" > Google Model Viewer </h2>
<p id="p-5" class='lang eng'> <a id="a-1" href='https://modelviewer.dev/'>Model Viewer</a> is a handy 3D model viewing package provided by Google. It’s a breeze to use whenever I need a simple viewer, and I’ve even employed it for the 3D model viewer on the <a id="a-1" href='https://ncsoft.github.io/CaPa/'>CaPa</a> project page. </p>

<h3 id="h3-1" > Basic Usage </h3>
<p id="p-7" class='lang eng'> You can easily import it into an HTML file via CDN like this: </p>

<pre id="pre-2" ><code id="model-viewer-cdn" style="font-size: 1rem"  class="language-html">&lt;script type=&quot;module&quot; src=&quot;https://unpkg.com/@google/model-viewer/dist/model-viewer.min.js&quot;&gt;&lt;/script&gt;
</code></pre>
<p id="p-8" class='lang eng'> Then, just declare <code>model-viewer</code> and specify the 3D model path in the <code>src</code> attribute to embed an interactive 3D model directly into your webpage: </p>

<pre id="pre-3" ><code style="font-size: 1rem"  class='language-html'>&lt;model-viewer 
    src=&quot;your_3d_model.glb&quot; >
&lt;/model-viewer&gt;
</code></pre>

<p id="p-9" class='lang eng'> Beyond that, it offers options like: </p>

<ul id="ul-1" >
  <li> <code>auto-rotate</code> </li>
  <li> <code>rotation-per-second</code> </li>
  <li> <code>camera-orbit</code>: $ (\theta, \phi, r) $</li>
  <li> <code>exposure</code></li>
  <li> <code>skybox-image</code></li>
</ul>
<p id="p-10" class='lang eng'> These allow you to tweak things like camera angles or environment settings during model loading, making it super convenient to achieve polished rendering with minimal effort. </p>
<p id="p-11" ><strong id="strong-1"><em>Example:</em></strong></p>
<pre id="pre-3" ><code style="font-size: 1rem"  class='language-html'>&lt;model-viewer 
    src=&quot;omni.glb&quot; 
    auto-rotate
    rotation-per-second=&quot;60deg&quot;
    camera-orbit=&quot;0deg 90deg 5m&quot; 
    exposure=&quot;3&quot;
    skybox-image=&quot;sunrise_1K_hdr.hdr&quot; >
&lt;/model-viewer&gt;
</code></pre>
<model-viewer 
    style="width: 100%; height: 700px;" 
    exposure="3" 
    id="model-ex" 
    src="https://huggingface.co/spaces/hhhwan/custom_gs/resolve/main/glbs/omni.glb" 
    alt="model sample" 
    lighting="none"  
    auto-rotate
    rotation-per-second="60deg"
    skybox-image="https://huggingface.co/spaces/hhhwan/custom_gs/resolve/main/glbs/spruit_sunrise_1k_HDR.hdr" >
</model-viewer>
<br/>

<h3 id="h3-2" > Geometry Rendering </h3>
<p id="p-12" class='lang eng'> For a basic 3D model viewer, it’s hard to beat Google Model Viewer. However, it’s tricky to modify its default shaders, and it doesn’t support alternative rendering types like Normal, Geometry, or Wireframe. This makes it tough to visually inspect detailed mesh elements beyond the base texture. </p>
<p id="p-13" class='lang eng'> That said, you can work around this limitation to some extent by removing the texture and using a gradient equirectangular image as the environment. This leverages PBR rendering to reflect the gradient, creating an effect that mimics mesh geometry rendering. </p>
<figure id="figure-1" >
  <img src='./250310_model_viewer/assets/gradient.jpg' alt='img alt' width='100%'>
  <figcaption style='text-align: center; font-size: 15px;'><strong>Figure: </strong> Gradient Map</figcaption>
</figure>

<p id="p-14" class='lang eng'> Here’s how you can set this up in JavaScript by using the gradient image as the environment and stripping the original texture: </p>


<pre id="pre-4" ><code id="code-mesh" style="font-size: 1rem"  class="language-javascript">var window_state = {};
function show_geometry(){
    let modelViewer = document.getElementById(&#39;model&#39;);
    if (modelViewer.model.materials[0].pbrMetallicRoughness.baseColorTexture.texture === null) return;
    window_state.textures = [];
    for (let i = 0; i < modelViewer.model.materials.length; i++) {
        window_state.textures.push(modelViewer.model.materials[i].pbrMetallicRoughness.baseColorTexture.texture);
    }
    window_state.exposure = modelViewer.exposure;
    modelViewer.environmentImage = &#39;&lt;img src=&quot;gradient.jpg&quot; width=&quot;100%&quot; /&gt;&#39;;
    for (let i = 0; i < modelViewer.model.materials.length; i++) {
        modelViewer.model.materials[i].pbrMetallicRoughness.baseColorTexture.setTexture(null);
    }
    modelViewer.exposure = 3;
}
</code></pre>
<p id="p-15" class='lang eng'> By saving the original texture and adding a function to restore it, you can map these to button UI elements. This lets you toggle between texture and geometry views, as shown below: </p>


<button class="btn btn-sm btn-primary" onclick="show_geometry()">Geometry</button>
<button class="btn btn-sm btn-secondary" onclick="show_texture()">Texture</button>
<model-viewer 
    style="width: 100%; height: 700px;" 
    exposure="3" 
    id="model-g" 
    src="https://huggingface.co/spaces/hhhwan/custom_gs/resolve/main/glbs/omni.glb" 
    alt="model sample" 
    lighting="none"  camera-controls>
</model-viewer>



<h2 id="h2-2" > Threejs-Based Custom Viewer </h2>
<p id="p-16" class='lang eng'> To implement rendering modes like Normal or Wireframe that Google Model Viewer can’t handle, you’ll need to roll up your sleeves and build a custom model viewer class using Three.js. </p>
<h3 id="h3-3" > Basic Usage </h3>
<pre id="pre-2" ><code style="font-size: 1rem"  class='language-html'>&lt;script type=&quot;importmap&quot;&gt;
    {
        &quot;imports&quot;: {
            &quot;three&quot;: &quot;https://unpkg.com/three@0.150.0/build/three.module.js&quot;,
            &quot;three/addons/&quot;: &quot;https://unpkg.com/three@0.157.0/examples/jsm/&quot;
        }
    }
&lt;/script&gt;
</code></pre>
<p id="p-17" class='lang eng'> Like Model Viewer, you can import Three.js via CDN, but setting it up is quite a bit more involved. </p>
<p id="p-18" class='lang eng'> You have to manually configure the <code>scene</code>, <code>camera</code>, <code>renderer</code>, and more. Start by importing the essential packages in JavaScript: </p>
<pre id="pre-3" ><code style="font-size: 1rem"  class='language-javascript'>import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';
import { RGBELoader } from 'three/addons/loaders/RGBELoader.js';
</code></pre>
<p id="p-19" class='lang eng'> Then define your scene and camera: </p>

<pre id="pre-4" ><code style="font-size: 1rem"  class='language-javascript'>scene = new THREE.Scene(); 
camera = new THREE.PerspectiveCamera(50, 1, 0.1, 1000);
renderer = new THREE.WebGLRenderer({ antialias: true });
controls = new OrbitControls(this.camera, this.renderer.domElement);
</code></pre>
<p id="p-20" class='lang eng'> Finally, load a GLTF 3D model and add it to the scene: </p>
<pre id="pre-5"><code style="font-size: 1rem"  class='language-javascript'>loader = new GLTFLoader();
loader.load('your_3d_model.glb', (gltf) => { scene.add(gltf.scene); }, undefined, (error) => { console.error('Loading Error:', error); });</code></pre>
<br/>


<h3 id="h3-4">Custom Viewer Implementation</h3>
<p id="p-21" class='lang eng'> A Three.js-based custom viewer offers far greater flexibility compared to Google Model Viewer. It allows you to implement diverse rendering modes like Diffuse, Mesh, Wireframe, and Normal with ease. </p>
<p id="p-22" class='lang eng'> In Three.js, mesh textures are defined as PBR materials using <code>MeshStandardMaterial</code>. Rendering modes like Normal, Geometry, or Wireframe can be achieved by simply adjusting the material mappings. </p>
<h4 id="h4-1" > Normal Map </h4>
<p id="p-23" class='lang eng'> For example, to render a Normal map, you can set the mesh material to <code>THREE.MeshNormalMaterial()</code>, which Three.js provides out of the box: </p>
<pre id="pre-5" ><code style="font-size: 1rem"  class="language-javascript">if (model) {
    model.traverse((child) => {
        if (child.isMesh) {
            child.material = new THREE.MeshNormalMaterial();
        }
    });
}
</code></pre>
<br/>


<h4 id="h4-2" > Wireframe </h4>
<p id="p-24" class='lang eng'> For Wireframe, setting <code>wireframe: true</code> in <code>THREE.MeshBasicMaterial()</code> will display the wireframe, but this hides the original texture and geometry, reducing visibility. </p>
<p id="p-25" class='lang eng'> A better approach is to create a separate wireframe mesh as a copy and add it as a child of the original mesh. This way, you can render the wireframe alongside the original texture and geometry: </p>
<pre id="pre-6" ><code style="font-size: 1rem" class="language-javascript">model.traverse((child) => {
    if (child.isMesh) {
        const wireframeMesh = new THREE.Mesh(child.geometry, new THREE.MeshBasicMaterial({
            wireframe: true,
            color: 0xaaaaaa, // light gray
            depthTest: true,
            transparent: true,
            opacity: 0.8 // transparency
        }));
        model.add(wireframeMesh); // add as child
    }
});
</code></pre>
<br/>
<h4 id="h4-3" > Custom Model Viewer </h4>
<p id="p-26" class='lang eng'> Reimplementing these features for every 3D model can get tedious, so I've built a lightweight <code>SimpleModelViewer</code> that brings together the best of both worlds: ease of use like Google Model Viewer, with the flexibility of Three.js. It includes pre-defined skybox environment map settings, the ability to remove them, and a UI panel for adjusting model position and rotation. </p>
<simple-model-viewer 
    src="https://huggingface.co/spaces/hhhwan/custom_gs/resolve/main/glbs/omni.glb" 
    view-mode="diffuse"
    style="width: 100%; height: 600px;">
</simple-model-viewer>
<p class="lang eng"> Check out the source code for the viewer below: </p>
<div id="div-1"  class="accordion accordion-flush">
    <div class="accordion-item">
      <h2 class="accordion-header">
        <button class="accordion-button collapsed" type="button" data-bs-toggle="collapse" data-bs-target="#flush-collapseOne" aria-expanded="false" aria-controls="flush-collapseOne">
          <strong><em>simple-model-viewer.js</em></strong>
        </button>
      </h2>
      <div id="flush-collapseOne" class="accordion-collapse collapse" data-bs-parent="#accordionFlushExample">
        <div class="accordion-body">
          <pre><code style="font-size: 1rem;" class="language-javascript">import * as THREE from &#39;three&#39;;
import { OrbitControls } from &#39;three/addons/controls/OrbitControls.js&#39;;
import { GLTFLoader } from &#39;three/addons/loaders/GLTFLoader.js&#39;;
import { RGBELoader } from &#39;three/addons/loaders/RGBELoader.js&#39;;

class SimpleModelViewer extends HTMLElement {
    constructor() {
        super();
        this.attachShadow({ mode: &#39;open&#39; });
        this.shadowRoot.innerHTML = `
            &lt;style&gt;
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
            &lt;/style&gt;
            &lt;div class=&quot;controls&quot;&gt;
                &lt;div id=&#39;meta&#39;&gt;
                    &lt;button id=&quot;textureBtn&quot;&gt;Diffuse&lt;/button&gt;
                    &lt;button id=&quot;meshBtn&quot;&gt;Geometry&lt;/button&gt;
                    &lt;button id=&quot;normalBtn&quot;&gt;Normal&lt;/button&gt;
                    &lt;button id=&quot;wireframeBtn&quot;&gt;Wireframe&lt;/button&gt;
                    &lt;button id=&quot;autoRotateBtn&quot;&gt;Auto-Rotate&lt;/button&gt;
                    &lt;button id=&quot;toonShadingBtn&quot;&gt;Toon Shading&lt;/button&gt;
                    &lt;button id=&quot;setBgBtn1&quot;&gt;Env1&lt;/button&gt;
                    &lt;button id=&quot;setBgBtn2&quot;&gt;Env2&lt;/button&gt;
                    &lt;button id=&quot;setBgBtn3&quot;&gt;Env3&lt;/button&gt;
                    &lt;button id=&quot;removeBgBtn&quot;&gt;Remove Env&lt;/button&gt;
                    &lt;div id=&quot;modelInfo&quot; style=&#39;padding-left: 0.1rem; font-size:0.8rem&#39;&gt;&lt;strong&gt;[Model Info]&lt;/strong&gt; loading...&lt;/div&gt;
                &lt;/div&gt;
                &lt;div id=&quot;transform-container&quot; style=&quot;position: relative;&quot;&gt;
                    &lt;div class=&quot;transform-panel&quot;&gt;
                        &lt;button id=&quot;togglePanelBtn&quot;&gt;&lt;i class=&quot;bi bi-caret-left&quot;&gt;&lt;/i&gt;&lt;/button&gt;
                        &lt;div id=&quot;transformControls&quot; style=&quot;display: block;&quot;&gt;
                            &lt;label&gt;Position X: &lt;input type=&quot;number&quot; id=&quot;posX&quot; step=&quot;0.1&quot; value=&quot;0&quot;&gt;&lt;/label&gt;
                            &lt;label&gt;Position Y: &lt;input type=&quot;number&quot; id=&quot;posY&quot; step=&quot;0.1&quot; value=&quot;0&quot;&gt;&lt;/label&gt;
                            &lt;label&gt;Position Z: &lt;input type=&quot;number&quot; id=&quot;posZ&quot; step=&quot;0.1&quot; value=&quot;0&quot;&gt;&lt;/label&gt;
                            &lt;label&gt;Rotation X (deg): &lt;input type=&quot;number&quot; id=&quot;rotX&quot; step=&quot;1&quot; value=&quot;0&quot;&gt;&lt;/label&gt;
                            &lt;label&gt;Rotation Y (deg): &lt;input type=&quot;number&quot; id=&quot;rotY&quot; step=&quot;1&quot; value=&quot;0&quot;&gt;&lt;/label&gt;
                            &lt;label&gt;Rotation Z (deg): &lt;input type=&quot;number&quot; id=&quot;rotZ&quot; step=&quot;1&quot; value=&quot;0&quot;&gt;&lt;/label&gt;
                            &lt;div&gt;Scale: &lt;input type=&quot;range&quot; id=&quot;scale&quot; style=&quot;width: 7rem;&quot; min=&quot;1&quot; max=&quot;20&quot; step=&quot;0.1&quot; value=&quot;8&quot;&gt;&lt;/div&gt;
                            &lt;div&gt;Roughness: &lt;input type=&quot;range&quot; id=&quot;roughness&quot; style=&quot;width: 7rem;&quot; min=&quot;0&quot; max=&quot;1&quot; step=&quot;0.01&quot; value=&quot;0.5&quot;&gt;&lt;/div&gt; 
                            &lt;div&gt;Metalness: &lt;input type=&quot;range&quot; id=&quot;metalness&quot; style=&quot;width: 7rem;&quot; min=&quot;0&quot; max=&quot;1&quot; step=&quot;0.01&quot; value=&quot;0.5&quot;&gt;&lt;/div&gt; 
                        &lt;/div&gt;
                    &lt;/div&gt;
                &lt;/div&gt;
            &lt;/div&gt;
            &lt;div id=&quot;canvas-container&quot; style=&#39;text-align: center&#39;&gt;
                &lt;div id=&quot;loadingProgressBar&quot;&gt;&lt;/div&gt;
            &lt;/div&gt;
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

        this.shadowRoot.querySelector(&#39;#canvas-container&#39;).appendChild(this.renderer.domElement);

        // const ambient = new THREE.AmbientLight(0x404040, 0.5);      
        // const directional = new THREE.DirectionalLight(0xffffff, 1); 
        // directional.position.set(5, 10, 7.5);
        // directional.castShadow = true;
        // this.scene.add(ambient, directional);

        this.controls = new OrbitControls(this.camera, this.renderer.domElement);
        this.controls.enableDamping = true;
        this.controls.dampingFactor = 0.05;
        this.controls.addEventListener(&#39;change&#39;, () => this.updateControlPanel()); 

        this.textureLoader = new THREE.TextureLoader();
        this.whiteTexture = this.textureLoader.load(&#39;https://huggingface.co/spaces/hhhwan/custom_gs/resolve/main/glbs/white.jpg&#39;);
        this.whiteTexture.mapping = THREE.EquirectangularReflectionMapping;

        this.gradTexture = this.textureLoader.load(&#39;https://huggingface.co/spaces/hhhwan/custom_gs/resolve/main/glbs/gradient.jpg&#39;);
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
        return [&#39;src&#39;, &#39;auto-rotate&#39;, &#39;angle-per-second&#39;, &#39;camera-orbit&#39;, &#39;hide-control-ui&#39;];
    }

    attributeChangedCallback(name, oldValue, newValue) {
        if (name === &#39;src&#39; && newValue) {
            this.loadModel(newValue);
        } else if (name === &#39;auto-rotate&#39;) {
            this.autoRotate = newValue !== null;
        } else if (name === &#39;angle-per-second&#39;) {
            this.anglePerSecond = parseFloat(newValue) || 30;
        } else if (name === &#39;camera-orbit&#39;) {
            this.setCameraOrbit(newValue);
        } else if (name === &#39;hide-control-ui&#39;) {
            const controlsDiv = this.shadowRoot.querySelector(&#39;.controls&#39;);
            if (newValue !== null) {
                controlsDiv.style.display = &#39;none&#39;;
            } else {
                controlsDiv.style.display = &#39;block&#39;;
            }
        }
    }

    initEventListeners() {
        this.shadowRoot.querySelector(&#39;#textureBtn&#39;).addEventListener(&#39;click&#39;, () => this.showTexture());
        this.shadowRoot.querySelector(&#39;#meshBtn&#39;).addEventListener(&#39;click&#39;, () => this.showMesh());
        this.shadowRoot.querySelector(&#39;#wireframeBtn&#39;).addEventListener(&#39;click&#39;, () => this.showWireframe());
        this.shadowRoot.querySelector(&#39;#normalBtn&#39;).addEventListener(&#39;click&#39;, () => this.showNormal());
        this.shadowRoot.querySelector(&#39;#setBgBtn1&#39;).addEventListener(&#39;click&#39;, () => this.setBackground1());
        this.shadowRoot.querySelector(&#39;#setBgBtn2&#39;).addEventListener(&#39;click&#39;, () => this.setBackground2());
        this.shadowRoot.querySelector(&#39;#setBgBtn3&#39;).addEventListener(&#39;click&#39;, () => this.setBackground3());
        this.shadowRoot.querySelector(&#39;#removeBgBtn&#39;).addEventListener(&#39;click&#39;, () => this.removeBG());

        this.shadowRoot.querySelector(&#39;#posX&#39;).addEventListener(&#39;input&#39;, () => this.updateModelTransform());
        this.shadowRoot.querySelector(&#39;#posY&#39;).addEventListener(&#39;input&#39;, () => this.updateModelTransform());
        this.shadowRoot.querySelector(&#39;#posZ&#39;).addEventListener(&#39;input&#39;, () => this.updateModelTransform());
        this.shadowRoot.querySelector(&#39;#rotX&#39;).addEventListener(&#39;input&#39;, () => this.updateModelTransform());
        this.shadowRoot.querySelector(&#39;#rotY&#39;).addEventListener(&#39;input&#39;, () => this.updateModelTransform());
        this.shadowRoot.querySelector(&#39;#rotZ&#39;).addEventListener(&#39;input&#39;, () => this.updateModelTransform());

        this.shadowRoot.querySelector(&#39;#roughness&#39;).addEventListener(&#39;input&#39;, () => this.updateMaterialProperties());
        this.shadowRoot.querySelector(&#39;#metalness&#39;).addEventListener(&#39;input&#39;, () => this.updateMaterialProperties());

        this.shadowRoot.querySelector(&#39;#scale&#39;).addEventListener(&#39;input&#39;, (e) => {
            this.modelSize = parseFloat(e.target.value);
            if (this.model) this.model.scale.set(this.modelSize, this.modelSize, this.modelSize);
        });

        this.shadowRoot.querySelector(&#39;#autoRotateBtn&#39;).addEventListener(&#39;click&#39;, () => {
            this.autoRotate = !this.autoRotate;
            this.shadowRoot.querySelector(&#39;#autoRotateBtn&#39;).textContent = this.autoRotate ? &#39;Auto-Rotate On&#39; : &#39;Auto-Rotate Off&#39;;
        });

        this.shadowRoot.querySelector(&#39;#togglePanelBtn&#39;).addEventListener(&#39;click&#39;, () => {
            const controls = this.shadowRoot.querySelector(&#39;#transformControls&#39;);
            const button = this.shadowRoot.querySelector(&#39;#togglePanelBtn&#39;);
            if (controls.style.display === &#39;none&#39;) {
                controls.style.display = &#39;block&#39;;
                button.innerHTML = &#39;&lt;i class=&quot;bi bi-caret-left-fill&quot;&gt;&lt;/i&gt;&#39;;
            } else {
                controls.style.display = &#39;none&#39;;
                button.innerHTML = &#39;&lt;i class=&quot;bi bi-caret-left&quot;&gt;&lt;/i&gt;&#39;;
            }
        });

        this.shadowRoot.querySelector(&#39;#toonShadingBtn&#39;).addEventListener(&#39;click&#39;, () => { 
            this.toonEnabled = !this.toonEnabled;
            if (this.toonEnabled) {
                this.enableToonShading();
                this.shadowRoot.querySelector(&#39;#toonShadingBtn&#39;).textContent = &#39;Toon Shading Off&#39;;
            } else {
                this.disableToonShading();
                this.shadowRoot.querySelector(&#39;#toonShadingBtn&#39;).textContent = &#39;Toon Shading On&#39;;
            }
        });
    }

    updateControlPanel() { 
        if (this.model) {
            this.shadowRoot.querySelector(&#39;#posX&#39;).value = this.model.position.x.toFixed(1);
            this.shadowRoot.querySelector(&#39;#posY&#39;).value = this.model.position.y.toFixed(1);
            this.shadowRoot.querySelector(&#39;#posZ&#39;).value = this.model.position.z.toFixed(1);

            this.shadowRoot.querySelector(&#39;#rotX&#39;).value = THREE.MathUtils.radToDeg(this.model.rotation.x).toFixed(0);
            this.shadowRoot.querySelector(&#39;#rotY&#39;).value = THREE.MathUtils.radToDeg(this.model.rotation.y).toFixed(0);
            this.shadowRoot.querySelector(&#39;#rotZ&#39;).value = THREE.MathUtils.radToDeg(this.model.rotation.z).toFixed(0);
        }
    }

    updateModelTransform() {
        if (this.model) {
            const posX = parseFloat(this.shadowRoot.querySelector(&#39;#posX&#39;).value);
            const posY = parseFloat(this.shadowRoot.querySelector(&#39;#posY&#39;).value);
            const posZ = parseFloat(this.shadowRoot.querySelector(&#39;#posZ&#39;).value);
            this.model.position.set(posX, posY, posZ);

            const rotX = THREE.MathUtils.degToRad(parseFloat(this.shadowRoot.querySelector(&#39;#rotX&#39;).value));
            const rotY = THREE.MathUtils.degToRad(parseFloat(this.shadowRoot.querySelector(&#39;#rotY&#39;).value));
            const rotZ = THREE.MathUtils.degToRad(parseFloat(this.shadowRoot.querySelector(&#39;#rotZ&#39;).value));
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
                const roughnessInput = this.shadowRoot.querySelector(&#39;#roughness&#39;);
                const metalnessInput = this.shadowRoot.querySelector(&#39;#metalness&#39;);
                if (roughnessInput) roughnessInput.value = initialRoughness;
                if (metalnessInput) metalnessInput.value = initialMetalness;
            }
            this.renderer.render(this.scene, this.camera);
        }, undefined, (err) => {
            console.error(&#39;Skybox err:&#39;, err);
            alert(&#39;Cannot load Skybox Image&#39;);
        });
    }

    setBackground1() {
        const url = &#39;https://huggingface.co/spaces/hhhwan/custom_gs/resolve/main/glbs/spruit_sunrise_1k_HDR.hdr&#39;;
        const rgbeLoader = new RGBELoader();
        this.set_bg(url, rgbeLoader);
    }

    setBackground2() {
        const url = &#39;https://huggingface.co/spaces/hhhwan/custom_gs/resolve/main/glbs/aircraft_workshop_01_1k.hdr&#39;;
        const rgbeLoader = new RGBELoader();
        this.set_bg(url, rgbeLoader);
    }

    setBackground3() {
        const url = &#39;https://huggingface.co/spaces/hhhwan/custom_gs/resolve/main/glbs/lebombo_1k.hdr&#39;;
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
        const [x, y, z] = value.split(&#39; &#39;).map(parseFloat);
        if (!isNaN(x) && !isNaN(y) && !isNaN(z)) {
            this.camera.position.set(x, y, z);
            this.camera.lookAt(0, 0, 0);
        }
    }

    loadModel(url) {
        const progressBar = this.shadowRoot.querySelector(&#39;#loadingProgressBar&#39;);
        progressBar.style.display = &#39;block&#39;; 
        progressBar.style.width = &#39;0%&#39;

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

            const cameraOrbit = this.getAttribute(&#39;camera-orbit&#39;);
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

                            const roughnessInput = this.shadowRoot.querySelector(&#39;#roughness&#39;);
                            const metalnessInput = this.shadowRoot.querySelector(&#39;#metalness&#39;);
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
            this.shadowRoot.querySelector(&#39;#roughness&#39;).value = initialRoughness;
            this.shadowRoot.querySelector(&#39;#metalness&#39;).value = initialMetalness;

            this.shadowRoot.querySelector(&#39;#modelInfo&#39;).innerHTML = `&lt;strong&gt;[Model Info]&lt;/strong&gt; Vertices: ${vertexCount}, Faces: ${faceCount}`;
            this.scene.add(this.model);

            this.showTexture();
            this.updateControlPanel(); 

            progressBar.style.display = &#39;none&#39;; // hide progress bar
            }, (xhr) => { // onProgress call back
                if (xhr.lengthComputable) {
                    const percentComplete = xhr.loaded / xhr.total * 100;
                    progressBar.style.width = `${percentComplete}%`; // bar update
                }
        }, undefined, (error) => {
            console.error(&#39;Loading Error:&#39;, error);
        });
    }

    updateMaterialProperties() { 
        const roughnessValue = parseFloat(this.shadowRoot.querySelector(&#39;#roughness&#39;).value);
        const metalnessValue = parseFloat(this.shadowRoot.querySelector(&#39;#metalness&#39;).value);

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
        const metaDiv = this.shadowRoot.querySelector(&#39;#meta&#39;);
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

customElements.define(&#39;simple-model-viewer&#39;, SimpleModelViewer);
export { SimpleModelViewer };
          </code></pre>
        </div>
      </div>
    </div>
  </div>

<p id="p-27" class='lang eng'> By loading this JS file into an HTML file that imports Three.js, you can configure <code>simple-model-viewer</code> as follows. It works independently across multiple 3D models, much like Google Model Viewer: </p>
<pre id="pre-12" ><code id="code-NaN" style="font-size:1rem"  class="language-javascript">&lt;simple-model-viewer 
    src=&quot;model.glb&quot; 
    camera-orbit=&quot;0 0 20&quot;
    angle-per-second=&quot;45&quot;
    style=&quot;width: 100%; height: 800px;&quot;>
&lt;/simple-model-viewer&gt;
    
&lt;script type=&quot;module&quot;>
    import { SimpleModelViewer } from &#39;simple-model-viewer.js&#39;;

        // Resizing
        window.addEventListener(&#39;resize&#39;, () => {
            const viewers = document.querySelectorAll(&#39;simple-model-viewer&#39;);
            viewers.forEach(viewer => viewer.resizeRenderer());
        }); 
&lt;/script&gt; </code></pre>
<p class="lang eng"> As another example, here’s a monster asset generated with <a href='https://ncsoft.github.io/CaPa/'>CaPa</a>, previously shared in <a href="../250302_3d_latent_diffusion/">Deep Dive into 3D Latent Diffusion</a>: </p>
<simple-model-viewer 
    src="https://huggingface.co/spaces/hhhwan/custom_gs/resolve/main/glbs/monster_ours.glb" 
    view-mode="diffuse"
    style="width: 100%; height: 600px;">
</simple-model-viewer>
<p id="p-28" class='lang eng'> With <code>auto-animate</code>, a basic animation to automatically change the mesh's rotation Y-angle can be added. Just like Model Viewer, this value is adjustable via <code>angle-per-second</code>, and the initial camera position is also configurable using <code>camera-orbit</code> option.</p> 
<p id="p-29" class='lang eng'> By incorporating toggleable settings for features like the <code>Control Panel</code>, I’ve developed a 3D Model Viewer that, while similar to Model Viewer, offers a broader range of functionalities. </p>
<br/>
<h3 id="h3-5">Key Takeaways</h3>
<br/>
<ul class='lang eng'>
    <li><strong>Google Model Viewer</strong>: Quick and easy setup is its strength, but it’s limited in rendering modes and customization.</li> 
    <li><strong>Three.js Custom Viewer</strong>: Offers flexibility and fine-grained control, though it requires more complex setup and time.</li>
    <li>Choose based on your project’s needs—simplicity or versatility.</li> 
</ul>
<br/>
<h2 id="conclusion">Conclusion</h2>

<p id="p-30" class='lang eng'> Google Model Viewer is a fantastic tool for quickly and easily embedding 3D models on the web. However, when you need finer control or a variety of rendering modes, a Three.js-based custom viewer is the way to go. </p> 
<p id="p-31" class='lang eng'> In this project, I used Three.js to build a custom viewer supporting Diffuse, Mesh, Wireframe, and Normal rendering modes, complete with a panel for adjusting model position and rotation. This gives users the flexibility to manipulate models and explore different visual effects. </p> 
<p id="p-32" class='lang eng'> Looking ahead, there’s room to expand with features like lighting controls or camera view saving. I hope this post helps you understand the ins and outs of 3D model viewers and pick the best approach for your next project! </p>

<hr/>
<p>
    You may also like, 
</p>
<ul>
    <li>
        <a href="/blogs/posts/?id=240917_3djs">
            <span style="text-decoration: underline;">Add Gaussian Splatting to your Website</span>
        </a>
    </li>
    <li>
        <a href="/blogs/posts/?id=250302_3d_latent_diffusion">
            <span style="text-decoration: underline;">A Deep Dive into 3D Latent Diffusion</span>
        </a>
    </li>
</ul>
<br/>