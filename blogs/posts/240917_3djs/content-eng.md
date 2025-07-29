title: Add Gaussian Splatting to Your Website
date: September 17, 2024
author: Hwan Heo
--- 여기부터 실제 콘텐츠 ---

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

<nav class="toc">
    <ul>
        <li>
            <a href="#tl-dr"> TL; DR</a>
        </li>
        <li><a href="#step"> Step-by-Step</a></li>
        <ul>
            <li><a href="#step1"> Step1: CDN setting</a></li>
            <li><a href="#step2"> Step2: HTML Structures</a></li>
            <li><a href="#step3"> Example Result</a></li>
        </ul>
    </ul>
</nav>

<br/>
<h2 id="tl-dr">TL; DR</h2>
<p class="lang eng"> 
    <em>(short article)</em> <br/>
    In this tutorial, I’ll guide you how to integrate 3D Gaussian splatting into your website. 
    Gaussian splatting is a rendering technique that represents point clouds as smooth, detailed "splats" rather than simple points, creating impressive visuals.
</p>
<p class="lang eng"> 
    Don’t worry if you're not a web expert—just follow along and copy-paste the code provided! 
</p>
<h3>
    About the Project
</h3>
<p class="lang eng">
    We’ll be using the <span style="text-decoration: underline;"><a href="https://github.com/mkkellogg/GaussianSplats3D"> 3D Gaussian Splatting for Three.js </a></span> by mkkellogg. 
    This package allows you to render point clouds as smooth, Gaussian splats in a 3D environment using Three.js. 
    We’ll use this package to integrate advanced point-based rendering into our website. 
    Now, let’s walk through the setup process.
</p>
<br/>

<h2 id="step">
    Step-by-Step
</h2>
<hr/>
<h3 id="step1">Step 1: Project Setup (CDN-based)</h3>
<p class="lang eng">
    For those not familiar with package management or setting up a local development environment, you can use a CDN to load the necessary libraries directly into your HTML file. Here’s how you can set up Three.js and GaussianSplats3D using CDN links:
</p>
<pre class="language-javascript" style="font-size: 16px;"><code>&lt;script type="importmap"&gt;
{
    "imports": {
        "three": "https://unpkg.com/three@0.150.0/build/three.module.js",
        "three/addons/": "https://unpkg.com/three@0.157.0/examples/jsm/",
        "GaussianSplats3D": "https://unpkg.com/@mkkellogg/gaussian-splats-3d@0.4.0/build/gaussian-splats-3d.module.js"
    }
}
&lt;/script&gt; </code></pre><br/>

<h3 id="step2">Step 2: HTML Structure</h3>
<p class="lang eng">
We will now initialize the WebGL renderer, set the camera’s position, and use the GaussianSplats3D viewer to load a 3D model (in .ply or .ksplat format, for example). 
</p>
<p class="lang eng">
<strong>Note: </strong>
If you follow the basic usage from the GitHub repo, the 3D content will be rendered to fill the entire page. 
Therefore, I’ve implemented this so the 3D scene appears only within a predefined canvas region. 
You can easily modify the canvas size to suit your needs.
<br/>
The code below handles all of these details:
</p>
<div class="accordion accordion-flush" id="accordionFlushExample">
<div class="accordion-item">
<h2 class="accordion-header">
<button class="accordion-button collapsed" type="button" data-bs-toggle="collapse" data-bs-target="#flush-collapseOne" aria-expanded="false" aria-controls="flush-collapseOne">
    <strong><em>Show Code!</em></strong>
</button>
</h2>
<div id="flush-collapseOne" class="accordion-collapse collapse" data-bs-parent="#accordionFlushExample">
<div class="accordion-body">

```javascript
<script type="module">
// Create a new THREE.js scene
import * as GaussianSplats3D from 'GaussianSplats3D';
import * as THREE from 'three';

// Set the desired render width and height
const renderWidth = 640;
const renderHeight = 360;

// Get the canvas container
const rootElement = document.getElementById('canvasContainer');
rootElement.style.width = renderWidth + 'px';
rootElement.style.height = renderHeight + 'px';

// Initialize WebGL renderer
const renderer = new THREE.WebGLRenderer({
    antialias: true
});
renderer.setSize(renderWidth, renderHeight);
renderer.setClearColor(0xf8f9fa, 1);
rootElement.appendChild(renderer.domElement);

// Initialize camera
const camera = new THREE.PerspectiveCamera(65, renderWidth / renderHeight, 0.1, 500);
camera.position.copy(new THREE.Vector3().fromArray([-1.5, -2, 3]));
camera.up = new THREE.Vector3().fromArray([0, -1, -0.6]).normalize();
camera.lookAt(new THREE.Vector3().fromArray([0, 3, 0]));
                
// Initialize the GaussianSplats3D viewer
const viewer = new GaussianSplats3D.Viewer({
    'selfDrivenMode': true,
    'renderer': renderer,
    'camera': camera,
    'useBuiltInControls': true,
    'ignoreDevicePixelRatio': false,
    'gpuAcceleratedSort': true,
    'enableSIMDInSort': true,
    'sharedMemoryForWorkers': false,
    'integerBasedSort': true,
    'halfPrecisionCovariancesOnGPU': true,
    'dynamicScene': false,
    'webXRMode': GaussianSplats3D.WebXRMode.None,
    'renderMode': GaussianSplats3D.RenderMode.OnChange,
    'sceneRevealMode': GaussianSplats3D.SceneRevealMode.Instant,
    'antialiased': false,
    'focalAdjustment': 1.0,
    'logLevel': GaussianSplats3D.LogLevel.None,
    'sphericalHarmonicsDegree': 0,
    'enableOptionalEffects': false,
    'plyInMemoryCompressionLevel': 2,
    'freeIntermediateSplatData': false
});
                
// Load a 3D scene (replace with the actual path to your model)
viewer.addSplatScene('<path/to/your/gs path>', {
    'position': [-0.7, -0.3, 0.9],
    'rotation': [0, 1, 0.2, 0.1],
    'scale': [3, 3, 3]
})
.then(() => {
    requestAnimationFrame(update);
});

// Update function to render the scene

function update() {
    requestAnimationFrame(update);
    viewer.update();
    viewer.render();
}
</script>;
```
</div>
</div>
</div>
<p>
    Tip.
</p>
<ul class="lang eng">
    <li>
        Set <code>'sharedMemoryForWorkers': false</code> for preventing CORS error.
    </li>
</ul>
<br/>
<h3 id="step3">
    Example Result of Custom GS Scene
</h3>
<div style="margin-bottom: -20vh;">
    <p class="lang eng">
        Here is the result of the above code. 
        The loaded Gaussian Splatting is my custom captured miniature guitar. 
        You can directly interact with the 3D Scene!
    </p>
</div>
<div id="canvasContainer" style="margin-bottom: 30vh;">
    <canvas id="threeCanvas"></canvas>
</div>

<h2>Closing</h2>
<p class="lang eng"> 
    For researchers, this topic might not be of direct interest to everyone. 
    However, in an era where personal projects and portfolios are increasingly important, 
    I believe that learning how to add a custom Gaussian splatting scene to personal webpage could be valuable for computer vision researchers who keep an eye on the field of neural rendering. 
    If you're curious about other research related to neural rendering, feel free to check out the other posts on my blog. 
    <br/><br/>
    That’s all for now!
</p>

<hr/>
<p>
    You may also like, 
</p>
<ul>
    <li>
        <a href="./?id=240805_gs/">
            <span style="text-decoration: underline;">A Comprehensive Analysis of Gaussian Splatting Rasterization</span>
        </a>
    </li>
    <li>
        <a href="./?id=240823_grt/">
            <span style="text-decoration: underline;">Don't Rasterize But Ray Trace Gaussian</span>
        </a>
    </li>
    <li>
        <a href="./?id=240602_2dgs/">
            <span style="text-decoration: underline;">Under the 3D: Geometrically Accurate 2D Gaussian Splatting </span>
        </a>
    </li>
</ul>
<br/>